import concurrent.futures
import dataclasses
import functools
import pathlib
import pickle
from typing import Sequence

import einops
from flax import struct
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import matplotlib.pyplot as plt
import numpy as np
import optax
import tqdm_loggable.auto as tqdm
import tyro
import wandb

import generate_data
import model as _model
import train_expert

WANDB_PROJECT = "dreamchunk-kinetix-bc"
LOG_DIR = pathlib.Path("logs-ebjepa")


@dataclasses.dataclass(frozen=True)
class Config:
    run_path: str
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    )
    batch_size: int = 1024
    sequence_length: int = 16
    num_epochs: int = 2
    seed: int = 0
    learning_rate: float = 2e-5
    grad_norm_clip: float = 10.0
    weight_decay: float = 1e-2
    lr_warmup_steps: int = 1000
    eval_fraction: float = 0.2
    eval_every_updates: int = 50
    prediction_loss_rolloutstep: int = 1
    wandb_name: str = "my-default-trainebjepa-name"
    model: _model.ModelConfig = dataclasses.field(
        default_factory=lambda: dataclasses.replace(_model.ModelConfig(), wm_type="ebjepa")
    )


@struct.dataclass
class EpochCarry:
    rng: jax.Array
    train_state: nnx.State
    graphdef: nnx.GraphDef[tuple[_model.DeterministicJepaWorldModel, nnx.Optimizer]]


def _save_learning_curve(
    output_dir: pathlib.Path,
    level_name: str,
    updates: list[int],
    epochs: list[float],
    train_loss_mean: list[float],
    train_loss_std: list[float],
    eval_loss_mean: list[float],
    eval_loss_std: list[float],
    train_pred_loss: list[float],
    eval_pred_loss: list[float],
    train_reg_loss: list[float],
    eval_reg_loss: list[float],
    grad_norms: list[float],
):
    output_dir.mkdir(parents=True, exist_ok=True)
    updates_arr = np.asarray(updates)
    np.savez(
        output_dir / f"{level_name}.npz",
        update=updates_arr,
        epoch=np.asarray(epochs),
        train_loss_mean=np.asarray(train_loss_mean),
        train_loss_std=np.asarray(train_loss_std),
        eval_loss_mean=np.asarray(eval_loss_mean),
        eval_loss_std=np.asarray(eval_loss_std),
        train_pred_loss=np.asarray(train_pred_loss),
        eval_pred_loss=np.asarray(eval_pred_loss),
        train_reg_loss=np.asarray(train_reg_loss),
        eval_reg_loss=np.asarray(eval_reg_loss),
        grad_norm=np.asarray(grad_norms),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.reshape(-1)
    panels = [
        ("Prediction Loss", train_pred_loss, eval_pred_loss),
        ("VC Regularizer", train_reg_loss, eval_reg_loss),
    ]
    for ax, (title, train_values, eval_values) in zip(axes[:2], panels, strict=True):
        ax.plot(updates_arr, train_values, marker="o", label="train")
        ax.plot(updates_arr, eval_values, marker="s", linestyle="--", label="eval")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.legend(frameon=False, fontsize=8)
    axes[2].plot(updates_arr, train_loss_mean, marker="o", label="train")
    axes[2].plot(updates_arr, eval_loss_mean, marker="s", linestyle="--", label="eval")
    axes[2].set_ylabel("Total Loss")
    axes[2].grid(alpha=0.3)
    axes[2].legend(frameon=False, fontsize=8)
    axes[3].plot(updates_arr, grad_norms, marker="o")
    axes[3].set_ylabel("Grad Norm")
    axes[3].grid(alpha=0.3)
    for ax in axes[2:]:
        ax.set_xlabel("update")
    fig.suptitle(level_name)
    fig.tight_layout()
    fig.savefig(output_dir / f"{level_name}.png", dpi=200)
    plt.close(fig)


def _make_batched_indices(indices: np.ndarray, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    num_batches = int(np.ceil(len(indices) / batch_size))
    padded = np.pad(indices, (0, num_batches * batch_size - len(indices)), mode="edge")
    mask = np.zeros(num_batches * batch_size, dtype=np.float32)
    mask[: len(indices)] = 1.0
    return padded.reshape(num_batches, batch_size), mask.reshape(num_batches, batch_size)


def _shuffle_batched_indices(indices: np.ndarray, batch_size: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    return _make_batched_indices(rng.permutation(indices), batch_size)


def main(config: Config):
    model_config = dataclasses.replace(
        config.model,
        prediction_loss_rolloutstep=config.prediction_loss_rolloutstep,
    )
    run_dir = LOG_DIR / config.wandb_name
    if run_dir.is_dir():
        epoch_dirs = [p for p in run_dir.iterdir() if p.is_dir() and p.name.isdigit()]
        if epoch_dirs and max(int(p.name) for p in epoch_dirs) >= config.num_epochs - 1:
            print(
                f"Skipping EB-JEPA training for {config.wandb_name}: "
                f"found existing checkpoint epoch {max(int(p.name) for p in epoch_dirs)} >= target {config.num_epochs - 1}"
            )
            return

    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(config.level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    mesh = jax.make_mesh((jax.local_device_count(),), ("level",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("level"))

    def load_data(level_path: str):
        level_name = level_path.replace("/", "_").replace(".json", "")
        print("Loading data for level:", level_name)
        return dict(np.load(pathlib.Path(config.run_path) / "data" / f"{level_name}.npz"))

    with concurrent.futures.ThreadPoolExecutor() as executor:
        data = list(executor.map(load_data, config.level_paths))
    with jax.default_device(jax.devices("cpu")[0]):
        data = jax.tree.map(lambda *x: einops.rearrange(jnp.stack(x), "l s e ... -> l (e s) ..."), *data)
        valid_steps = data["obs"].shape[1] - 1
        data = jax.tree.map(lambda x: x[:, : (valid_steps // config.batch_size) * config.batch_size + 1], data)
        data = jax.tree.map(
            lambda x: jax.make_array_from_single_device_arrays(
                x.shape,
                sharding,
                [jax.device_put(y, d) for y, d in zip(jnp.split(x, jax.local_device_count()), jax.local_devices(), strict=True)],
            ),
            data,
        )
    data: generate_data.Data = generate_data.Data(**data)

    num_samples = data.obs.shape[1] - config.sequence_length
    eval_size = max(1, int(num_samples * config.eval_fraction))
    train_size = num_samples - eval_size
    split_rng = np.random.default_rng(config.seed)
    shuffled_indices = split_rng.permutation(num_samples)
    eval_indices = shuffled_indices[:eval_size]
    train_indices = shuffled_indices[eval_size:]
    train_batch_indices, train_batch_mask = _make_batched_indices(train_indices, config.batch_size)
    eval_batch_indices, eval_batch_mask = _make_batched_indices(eval_indices, config.batch_size)

    obs_dim = data.obs.shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    @functools.partial(jax.jit, in_shardings=sharding, out_shardings=sharding)
    @jax.vmap
    def init(rng: jax.Array) -> EpochCarry:
        rng, key = jax.random.split(rng)
        world_model = _model.build_world_model(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=model_config,
            rngs=nnx.Rngs(key),
        )
        encoder_params = sum(x.size for x in jax.tree.leaves(nnx.state(world_model.encoder_wm, nnx.Param)))
        projector_params = sum(x.size for x in jax.tree.leaves(nnx.state(world_model.projector, nnx.Param)))
        action_params = sum(x.size for x in jax.tree.leaves(nnx.state(world_model.action_encoder, nnx.Param)))
        predictor_params = sum(x.size for x in jax.tree.leaves(nnx.state(world_model.predictor, nnx.Param)))
        norm_params = sum(x.size for x in jax.tree.leaves(nnx.state(world_model.latent_norm, nnx.Param)))
        print(
            "Trainable EB-JEPA params: "
            f"encoder_wm={encoder_params:,} projector={projector_params:,} "
            f"action_encoder={action_params:,} predictor={predictor_params:,} latent_norm={norm_params:,}"
        )
        optimizer = nnx.Optimizer(
            world_model,
            optax.chain(
                optax.clip_by_global_norm(config.grad_norm_clip),
                optax.adamw(
                    optax.warmup_constant_schedule(0, config.learning_rate, config.lr_warmup_steps),
                    weight_decay=config.weight_decay,
                ),
            ),
        )
        graphdef, train_state = nnx.split((world_model, optimizer))
        return EpochCarry(rng, train_state, graphdef)

    @functools.partial(jax.jit, donate_argnums=(0,), in_shardings=(sharding, sharding, None, None), out_shardings=(sharding, sharding))
    @functools.partial(jax.vmap, in_axes=(0, 0, None, None))
    def train_step(epoch_carry: EpochCarry, data: generate_data.Data, batch_idxs: jax.Array, batch_mask: jax.Array):
        del batch_mask
        world_model, optimizer = nnx.merge(epoch_carry.graphdef, epoch_carry.train_state)
        batch_windows = batch_idxs[:, None] + jnp.arange(config.sequence_length + 1)[None, :]
        action_windows = batch_idxs[:, None] + jnp.arange(config.sequence_length)[None, :]
        obs = data.obs[batch_windows]
        action = data.action[action_windows]
        done = data.done[action_windows]

        def compute_loss_info(model):
            return model.rollout_observed_sequence(obs, action, done, rng=None)

        def loss_fn(model):
            return compute_loss_info(model)["loss"]

        _, grads = nnx.value_and_grad(loss_fn)(world_model)
        loss_info = compute_loss_info(world_model)
        grad_norm = optax.global_norm(grads)
        optimizer.update(grads)
        _, train_state = nnx.split((world_model, optimizer))
        info = {
            "train_loss_mean": loss_info["loss"],
            "train_loss_std": 0.0,
            "train_pred_loss": loss_info["pred_loss"],
            "train_reg_loss": loss_info["reg_loss"],
            "grad_norm": grad_norm,
        }
        return EpochCarry(epoch_carry.rng, train_state, epoch_carry.graphdef), info

    @functools.partial(jax.jit, in_shardings=(sharding, sharding, None, None), out_shardings=sharding)
    @functools.partial(jax.vmap, in_axes=(0, 0, None, None))
    def eval_step(epoch_carry: EpochCarry, data: generate_data.Data, eval_batch_indices: jax.Array, eval_batch_mask: jax.Array):
        world_model, _ = nnx.merge(epoch_carry.graphdef, epoch_carry.train_state)

        def eval_minibatch(carry, minibatch):
            loss_sum, loss_sq_sum, pred_sum, reg_sum, count = carry
            batch_idxs, batch_mask = minibatch
            batch_windows = batch_idxs[:, None] + jnp.arange(config.sequence_length + 1)[None, :]
            action_windows = batch_idxs[:, None] + jnp.arange(config.sequence_length)[None, :]
            obs = data.obs[batch_windows]
            action = data.action[action_windows]
            done = data.done[action_windows]
            loss_info = world_model.rollout_observed_sequence(obs, action, done, rng=None)
            weight = jnp.sum(batch_mask)
            return (
                loss_sum + loss_info["loss"] * weight,
                loss_sq_sum + jnp.square(loss_info["loss"]) * weight,
                pred_sum + loss_info["pred_loss"] * weight,
                reg_sum + loss_info["reg_loss"] * weight,
                count + weight,
            ), None

        (eval_loss_sum, eval_loss_sq_sum, eval_pred_sum, eval_reg_sum, eval_count), _ = jax.lax.scan(
            eval_minibatch,
            (0.0, 0.0, 0.0, 0.0, 0.0),
            (jnp.asarray(eval_batch_indices), jnp.asarray(eval_batch_mask)),
        )
        eval_loss_mean = eval_loss_sum / (eval_count + 1e-8)
        eval_loss_var = eval_loss_sq_sum / (eval_count + 1e-8) - jnp.square(eval_loss_mean)
        return {
            "eval_loss_mean": eval_loss_mean,
            "eval_loss_std": jnp.sqrt(jnp.maximum(eval_loss_var, 0.0)),
            "eval_pred_loss": eval_pred_sum / (eval_count + 1e-8),
            "eval_reg_loss": eval_reg_sum / (eval_count + 1e-8),
        }

    wandb.init(project=WANDB_PROJECT, name=config.wandb_name)
    rng = jax.random.key(config.seed)
    epoch_carry = init(jax.random.split(rng, len(config.level_paths)))
    history = {
        level_path.replace("/", "_").replace(".json", ""): {
            "update": [],
            "epoch": [],
            "train_loss_mean": [],
            "train_loss_std": [],
            "eval_loss_mean": [],
            "eval_loss_std": [],
            "train_pred_loss": [],
            "eval_pred_loss": [],
            "train_reg_loss": [],
            "eval_reg_loss": [],
            "grad_norm": [],
        }
        for level_path in config.level_paths
    }
    train_rng = np.random.default_rng(config.seed)
    num_updates_per_epoch = int(np.ceil(len(train_indices) / config.batch_size))
    progress = tqdm.tqdm(total=config.num_epochs * num_updates_per_epoch)
    global_update = 0
    last_eval_info = None
    for epoch_idx in range(config.num_epochs):
        epoch_batch_indices, epoch_batch_mask = _shuffle_batched_indices(train_indices, config.batch_size, train_rng)
        for batch_idxs, batch_mask in zip(epoch_batch_indices, epoch_batch_mask, strict=True):
            epoch_carry, train_info = train_step(epoch_carry, data, jnp.asarray(batch_idxs), jnp.asarray(batch_mask))
            if last_eval_info is None or global_update % config.eval_every_updates == 0:
                last_eval_info = eval_step(epoch_carry, data, eval_batch_indices, eval_batch_mask)
            eval_info = last_eval_info
            progress.update(1)
            for i, level_path in enumerate(config.level_paths):
                level_name = level_path.replace("/", "_").replace(".json", "")
                level_train_info = {k: float(v[i]) for k, v in train_info.items()}
                level_eval_info = {k: float(v[i]) for k, v in eval_info.items()}
                for key, value in (
                    ("update", global_update),
                    ("epoch", epoch_idx + (global_update % num_updates_per_epoch) / num_updates_per_epoch),
                ):
                    history[level_name][key].append(value)
                for key, value in {**level_train_info, **level_eval_info}.items():
                    history[level_name][key].append(value)
                wandb.log({f"{level_name}/{k}": v for k, v in {**level_train_info, **level_eval_info}.items()}, step=global_update)
            global_update += 1

        log_dir = run_dir / str(epoch_idx)
        world_model_dir = log_dir / "world_models"
        world_model_dir.mkdir(parents=True, exist_ok=True)
        for i, level_path in enumerate(config.level_paths):
            level_name = level_path.replace("/", "_").replace(".json", "")
            level_train_state = jax.tree.map(lambda x: x[i], epoch_carry.train_state)
            with (world_model_dir / f"{level_name}.pkl").open("wb") as f:
                world_model, _ = nnx.merge(epoch_carry.graphdef, level_train_state)
                pickle.dump(nnx.state(world_model).to_pure_dict(), f)
    progress.close()

    curve_dir = run_dir / "learning_curves"
    for level_path in config.level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        _save_learning_curve(
            curve_dir,
            level_name,
            history[level_name]["update"],
            history[level_name]["epoch"],
            history[level_name]["train_loss_mean"],
            history[level_name]["train_loss_std"],
            history[level_name]["eval_loss_mean"],
            history[level_name]["eval_loss_std"],
            history[level_name]["train_pred_loss"],
            history[level_name]["eval_pred_loss"],
            history[level_name]["train_reg_loss"],
            history[level_name]["eval_reg_loss"],
            history[level_name]["grad_norm"],
        )


if __name__ == "__main__":
    tyro.cli(main)
