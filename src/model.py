import dataclasses
import functools
from typing import Literal, NamedTuple, TypeAlias, Self

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    wm_type: Literal["rssm", "lewm", "ebjepa"] = "rssm"
    prediction_loss_rolloutstep: int = 1
    prediction_loss_scale: float = 1.0
    channel_dim: int = 256
    channel_hidden_dim: int = 512
    token_hidden_dim: int = 64
    num_layers: int = 4
    action_chunk_size: int = 8
    simulated_delay: int | None = None
    rssm_deter_dim: int = 256
    rssm_stoch_dim: int = 32
    rssm_discrete_dim: int = 16
    rssm_hidden_dim: int = 512
    rssm_obs_layers: int = 1
    rssm_img_layers: int = 2
    rssm_dyn_layers: int = 1
    rssm_blocks: int = 8
    rssm_unimix_ratio: float = 0.01
    vector_encoder_hidden_dim: int = 512
    kl_free_nats: float = 1.0
    dyn_loss_scale: float = 1.0
    rep_loss_scale: float = 0.1
    barlow_loss_scale: float = 1.0
    barlow_lambda: float = 5e-3
    jepa_obs_hidden_dim: int = 512
    jepa_action_hidden_dim: int = 256
    jepa_predictor_hidden_dim: int = 512
    jepa_reg_weight: float = 0.1
    jepa_sigreg_knots: int = 17
    jepa_sigreg_num_proj: int = 1024
    jepa_vc_std_coeff: float = 1.0
    jepa_vc_cov_coeff: float = 1.0
    jepa_vc_sim_coeff_t: float = 0.0


def posemb_sincos(pos: jax.Array, embedding_dim: int, min_period: float, max_period: float) -> jax.Array:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


PrefixAttentionSchedule: TypeAlias = Literal["linear", "exp", "ones", "zeros"]


def get_prefix_weights(start: int, end: int, total: int, schedule: PrefixAttentionSchedule) -> jax.Array:
    """With start=2, end=6, total=10, the output will be:
    1  1  4/5 3/5 2/5 1/5 0  0  0  0
           ^              ^
         start           end
    `start` (inclusive) is where the chunk starts being allowed to change. `end` (exclusive) is where the chunk stops
    paying attention to the prefix. if start == 0, then the entire chunk is allowed to change. if end == total, then the
    entire prefix is attended to.

    `end` takes precedence over `start` in the sense that, if `end < start`, then `start` is pushed down to `end`. Thus,
    if `end` is 0, then the entire prefix will always be ignored.
    """
    start = jnp.minimum(start, end)
    if schedule == "ones":
        w = jnp.ones(total)
    elif schedule == "zeros":
        w = (jnp.arange(total) < start).astype(jnp.float32)
    elif schedule == "linear" or schedule == "exp":
        w = jnp.clip((start - 1 - jnp.arange(total)) / (end - start + 1) + 1, 0, 1)
        if schedule == "exp":
            w = w * jnp.expm1(w) / (jnp.e - 1)
    else:
        raise ValueError(f"Invalid schedule: {schedule}")
    return jnp.where(jnp.arange(total) >= end, 0, w)


class MLPMixerBlock(nnx.Module):
    def __init__(
        self, token_dim: int, token_hidden_dim: int, channel_dim: int, channel_hidden_dim: int, *, rngs: nnx.Rngs
    ):
        self.token_mix_in = nnx.Linear(token_dim, token_hidden_dim, use_bias=False, rngs=rngs)
        self.token_mix_out = nnx.Linear(token_hidden_dim, token_dim, use_bias=False, rngs=rngs)
        self.channel_mix_in = nnx.Linear(channel_dim, channel_hidden_dim, use_bias=False, rngs=rngs)
        self.channel_mix_out = nnx.Linear(channel_hidden_dim, channel_dim, use_bias=False, rngs=rngs)
        self.norm_1 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.norm_2 = nnx.LayerNorm(channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.adaln_1 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)
        self.adaln_2 = nnx.Linear(channel_dim, 3 * channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs)

    def __call__(self, x: jax.Array, adaln_cond: jax.Array) -> jax.Array:
        scale_1, shift_1, gate_1 = jnp.split(self.adaln_1(adaln_cond), 3, axis=-1)
        scale_2, shift_2, gate_2 = jnp.split(self.adaln_2(adaln_cond), 3, axis=-1)

        # token mix
        residual = x
        x = self.norm_1(x) * (1 + scale_1) + shift_1
        x = x.transpose(0, 2, 1)
        x = self.token_mix_in(x)
        x = nnx.gelu(x)
        x = self.token_mix_out(x)
        x = x.transpose(0, 2, 1)
        x = residual + gate_1 * x

        # channel mix
        residual = x
        x = self.norm_2(x) * (1 + scale_2) + shift_2
        x = self.channel_mix_in(x)
        x = nnx.gelu(x)
        x = self.channel_mix_out(x)
        x = residual + gate_2 * x
        return x


class BlockLinear(nnx.Module):
    def __init__(self, in_features: int, out_features: int, blocks: int, *, rngs: nnx.Rngs):
        if in_features % blocks != 0 or out_features % blocks != 0:
            raise ValueError(f"BlockLinear expects divisibility by blocks: {in_features=}, {out_features=}, {blocks=}")
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.weight = nnx.Param(jax.random.normal(rngs.params(), (out_features // blocks, in_features // blocks, blocks)) * 0.02)
        self.bias = nnx.Param(jnp.zeros((out_features,)))

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_shape = x.shape[:-1]
        x = x.reshape(*batch_shape, self.blocks, self.in_features // self.blocks)
        x = jnp.einsum("...gi,oig->...go", x, self.weight.value)
        x = x.reshape(*batch_shape, self.out_features)
        return x + self.bias.value


class RSSMHead(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, num_layers: int, *, rngs: nnx.Rngs):
        self.layers = [
            (
                nnx.Linear(in_dim if i == 0 else hidden_dim, hidden_dim, rngs=rngs),
                nnx.RMSNorm(hidden_dim, rngs=rngs),
            )
            for i in range(num_layers)
        ]
        self.out = nnx.Linear(hidden_dim if num_layers > 0 else in_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        for linear, norm in self.layers:
            x = nnx.swish(norm(linear(x)))
        return self.out(x)


class DeterTransition(nnx.Module):
    def __init__(
        self,
        deter_dim: int,
        flat_stoch_dim: int,
        action_dim: int,
        hidden_dim: int,
        blocks: int,
        dyn_layers: int,
        *,
        rngs: nnx.Rngs,
    ):
        if deter_dim % blocks != 0:
            raise ValueError(f"deter_dim must be divisible by blocks, got {deter_dim=} {blocks=}")
        self.deter_dim = deter_dim
        self.blocks = blocks
        self.hidden_dim = hidden_dim
        self.in_deter = nnx.Sequential(
            nnx.Linear(deter_dim, hidden_dim, rngs=rngs),
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.swish,
        )
        self.in_stoch = nnx.Sequential(
            nnx.Linear(flat_stoch_dim, hidden_dim, rngs=rngs),
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.swish,
        )
        self.in_action = nnx.Sequential(
            nnx.Linear(action_dim, hidden_dim, rngs=rngs),
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.swish,
        )
        in_dim = (3 * hidden_dim + deter_dim // blocks) * blocks
        self.hidden_layers = [
            (
                BlockLinear(in_dim if i == 0 else deter_dim, deter_dim, blocks, rngs=rngs),
                nnx.RMSNorm(deter_dim, rngs=rngs),
            )
            for i in range(dyn_layers)
        ]
        self.gru_proj = BlockLinear(deter_dim if dyn_layers > 0 else in_dim, 3 * deter_dim, blocks, rngs=rngs)

    def _flat2group(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-1], self.blocks, -1)

    def _group2flat(self, x: jax.Array) -> jax.Array:
        return x.reshape(*x.shape[:-2], -1)

    def __call__(self, stoch: jax.Array, deter: jax.Array, action: jax.Array) -> jax.Array:
        flat_stoch = stoch.reshape(*stoch.shape[:-2], -1)
        denom = jax.lax.stop_gradient(jnp.clip(jnp.abs(action), a_min=1.0))
        action = action / denom
        x0 = self.in_deter(deter)
        x1 = self.in_stoch(flat_stoch)
        x2 = self.in_action(action)
        x = jnp.concatenate([x0, x1, x2], axis=-1)
        x = jnp.broadcast_to(x[..., None, :], (*x.shape[:-1], self.blocks, x.shape[-1]))
        x = self._group2flat(jnp.concatenate([self._flat2group(deter), x], axis=-1))
        for linear, norm in self.hidden_layers:
            x = nnx.swish(norm(linear(x)))
        x = self.gru_proj(x)
        reset, cand, update = jnp.split(self._flat2group(x), 3, axis=-1)
        reset = self._group2flat(reset)
        cand = self._group2flat(cand)
        update = self._group2flat(update)
        reset = jax.nn.sigmoid(reset)
        cand = jnp.tanh(reset * cand)
        update = jax.nn.sigmoid(update - 1.0)
        return update * cand + (1.0 - update) * deter


class JepaMLP(nnx.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, *, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(in_dim, hidden_dim, rngs=rngs)
        self.in_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, out_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = nnx.swish(self.in_norm(self.in_proj(x)))
        return self.out_proj(x)


class JepaPredictor(nnx.Module):
    def __init__(self, latent_dim: int, action_dim: int, hidden_dim: int, *, rngs: nnx.Rngs):
        self.in_proj = nnx.Linear(latent_dim + action_dim, hidden_dim, rngs=rngs)
        self.in_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.hidden_proj = nnx.Linear(hidden_dim, hidden_dim, rngs=rngs)
        self.hidden_norm = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.out_proj = nnx.Linear(hidden_dim, latent_dim, rngs=rngs)
        self.out_norm = nnx.LayerNorm(latent_dim, rngs=rngs)

    def __call__(self, latent: jax.Array, action_embed: jax.Array) -> jax.Array:
        x = jnp.concatenate([latent, action_embed], axis=-1)
        x = nnx.swish(self.in_norm(self.in_proj(x)))
        x = nnx.swish(self.hidden_norm(self.hidden_proj(x)))
        return self.out_norm(self.out_proj(x) + latent)


def _jepa_sigreg_loss(latent: jax.Array, knots: int, num_proj: int, rng: jax.Array | None) -> jax.Array:
    if latent.ndim != 3:
        raise ValueError(f"SIGReg expects [B,T,D], got {latent.shape}")
    if rng is None:
        rng = jax.random.key(0)
    proj_dim = latent.shape[-1]
    A = jax.random.normal(rng, (proj_dim, num_proj), dtype=latent.dtype)
    A = A / jnp.clip(jnp.linalg.norm(A, axis=0, keepdims=True), a_min=1e-8)
    t = jnp.linspace(0.0, 3.0, knots, dtype=latent.dtype)
    dt = 3.0 / max(knots - 1, 1)
    weights = jnp.full((knots,), 2.0 * dt, dtype=latent.dtype)
    if knots >= 2:
        weights = weights.at[0].set(dt)
        weights = weights.at[-1].set(dt)
    phi = jnp.exp(-jnp.square(t) / 2.0)
    weights = weights * phi
    x_t = jnp.einsum("btd,dp,k->btpk", latent, A, t, precision=jax.lax.Precision.HIGHEST)
    err = jnp.square(jnp.mean(jnp.cos(x_t), axis=0) - phi[None, None, :]) + jnp.square(jnp.mean(jnp.sin(x_t), axis=0))
    statistic = jnp.einsum("tpk,k->tp", err, weights, precision=jax.lax.Precision.HIGHEST) * latent.shape[0]
    return jnp.mean(statistic)


def _jepa_vc_loss(
    latent: jax.Array,
    std_coeff: float,
    cov_coeff: float,
    sim_coeff_t: float,
    eps: float = 1e-4,
) -> tuple[jax.Array, dict[str, jax.Array]]:
    flat = latent.reshape((-1, latent.shape[-1]))
    std = jnp.sqrt(jnp.var(flat, axis=0) + eps)
    std_loss = jnp.mean(jax.nn.relu(1.0 - std))
    centered = flat - jnp.mean(flat, axis=0, keepdims=True)
    n = flat.shape[0]
    cov = (centered.T @ centered) / jnp.maximum(n - 1, 1)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.mean(jnp.square(off_diag))
    if latent.shape[1] < 2 or sim_coeff_t <= 0.0:
        sim_t_loss = jnp.array(0.0, dtype=latent.dtype)
    else:
        sim_t_loss = jnp.mean(jnp.square(latent[:, 1:] - latent[:, :-1]))
    total = std_coeff * std_loss + cov_coeff * cov_loss + sim_coeff_t * sim_t_loss
    return total, {
        "std_loss": std_loss,
        "cov_loss": cov_loss,
        "sim_t_loss": sim_t_loss,
    }


class FlowPolicy(nnx.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.simulated_delay = config.simulated_delay

        self.in_proj = nnx.Linear(action_dim + obs_dim, config.channel_dim, rngs=rngs)
        self.mlp_stack = [
            MLPMixerBlock(
                config.action_chunk_size,
                config.token_hidden_dim,
                config.channel_dim,
                config.channel_hidden_dim,
                rngs=rngs,
            )
            for _ in range(config.num_layers)
        ]
        self.time_mlp = nnx.Sequential(
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.channel_dim, config.channel_dim, rngs=rngs),
            nnx.swish,
        )
        self.final_norm = nnx.LayerNorm(config.channel_dim, use_scale=False, use_bias=False, rngs=rngs)
        self.final_adaln = nnx.Linear(
            config.channel_dim, 2 * config.channel_dim, kernel_init=nnx.initializers.zeros_init(), rngs=rngs
        )
        self.out_proj = nnx.Linear(config.channel_dim, action_dim, rngs=rngs)

    def __call__(self, obs: jax.Array, x_t: jax.Array, time: jax.Array) -> jax.Array:
        assert x_t.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_t.shape
        if time.ndim == 1:
            time = time[:, None]
        time = jnp.broadcast_to(time, (obs.shape[0], self.action_chunk_size))
        time_emb = jax.vmap(
            functools.partial(posemb_sincos, embedding_dim=self.channel_dim, min_period=4e-3, max_period=4.0)
        )(time)
        time_emb = self.time_mlp(time_emb)
        obs = einops.repeat(obs, "b e -> b c e", c=self.action_chunk_size)
        x = jnp.concatenate([x_t, obs], axis=-1)
        x = self.in_proj(x)
        for mlp in self.mlp_stack:
            x = mlp(x, time_emb)
        assert x.shape == (obs.shape[0], self.action_chunk_size, self.channel_dim), x.shape
        scale, shift = jnp.split(self.final_adaln(time_emb), 2, axis=-1)
        x = self.final_norm(x) * (1 + scale) + shift
        x = self.out_proj(x)
        return x

    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def bid_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        n_samples: int,
        # when below two are None, it becomes backwards loss only (i.e., rejection sampling)
        bid_weak_policy: Self | None = None,
        bid_k: int | None = None,
    ) -> jax.Array:
        obs = einops.repeat(obs, "b ... -> (n b) ...", n=n_samples)
        weights = get_prefix_weights(inference_delay, prefix_attention_horizon, self.action_chunk_size, "exp")

        def backward_loss(action_chunks: jax.Array):
            error = jnp.linalg.norm(action_chunks - prev_action_chunk, axis=-1)  # [n, b, h]
            return jnp.sum(error * weights[None, None, :], axis=-1)  # [n, b]

        strong_actions = einops.rearrange(self.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples)
        loss = backward_loss(strong_actions)  # [n, b]

        if bid_weak_policy is not None or bid_k is not None:
            assert bid_weak_policy is not None and bid_k is not None, (bid_weak_policy, bid_k)
            weak_actions = einops.rearrange(
                bid_weak_policy.action(rng, obs, num_steps), "(n b) h d -> n b h d", n=n_samples
            )
            weak_loss = backward_loss(weak_actions)  # [n, b]
            weak_idxs = jax.lax.top_k(-weak_loss.T, bid_k)[1].T  # [k, b]
            strong_idxs = jax.lax.top_k(-loss.T, bid_k)[1].T  # [k, b]
            a_plus = jnp.take_along_axis(strong_actions, strong_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            a_minus = jnp.take_along_axis(weak_actions, weak_idxs[:, :, None, None], axis=0)  # [k, b, h, d]
            # compute forward loss for each action in strong_actions
            forward_loss = jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_plus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            ) - jnp.sum(
                jnp.linalg.norm(strong_actions[:, None] - a_minus[None, :], axis=-1),  # [n, k, b, h]
                axis=(1, 3),  # [n, b]
            )
            loss += forward_loss / n_samples

        best_idxs = jnp.argmin(loss, axis=0)  # [b]
        return jnp.take_along_axis(strong_actions, best_idxs[None, :, None, None], axis=0).squeeze(0)  # [b, h, d]

    def realtime_action(
        self,
        rng: jax.Array,
        obs: jax.Array,
        num_steps: int,
        prev_action_chunk: jax.Array,  # [batch, horizon, action_dim]
        inference_delay: int,
        prefix_attention_horizon: int,
        prefix_attention_schedule: PrefixAttentionSchedule,
        max_guidance_weight: float,
    ) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry

            @functools.partial(jax.vmap, in_axes=(0, 0, 0, None))  # over batch
            def pinv_corrected_velocity(obs, x_t, y, t):
                def denoiser(x_t):
                    v_t = self(obs[None], x_t[None], t)[0]
                    return x_t + v_t * (1 - t), v_t

                x_1, vjp_fun, v_t = jax.vjp(denoiser, x_t, has_aux=True)
                weights = get_prefix_weights(
                    inference_delay, prefix_attention_horizon, self.action_chunk_size, prefix_attention_schedule
                )
                error = (y - x_1) * weights[:, None]
                pinv_correction = vjp_fun(error)[0]
                # method constants
                inv_r2 = (t**2 + (1 - t) ** 2) / ((1 - t) ** 2)
                c = jnp.nan_to_num((1 - t) / t, posinf=max_guidance_weight)
                guidance_weight = jnp.minimum(c * inv_r2, max_guidance_weight)
                return v_t + guidance_weight * pinv_correction

            if self.simulated_delay is not None:
                mask = jnp.arange(self.action_chunk_size)[None, :] < inference_delay
                x_t = jnp.where(mask[:, :, None], prev_action_chunk, x_t)
                time_chunk = jnp.where(mask, 1.0, time)
                v_t = self(obs, x_t, time_chunk)
            else:
                v_t = pinv_corrected_velocity(obs, x_t, prev_action_chunk, time)
            return (x_t + dt * v_t, time + dt), None

        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1

    def loss(self, rng: jax.Array, obs: jax.Array, action: jax.Array):
        assert action.dtype == jnp.float32
        assert action.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), action.shape
        noise_rng, time_rng, delay_rng = jax.random.split(rng, 3)
        time = jax.random.uniform(time_rng, (obs.shape[0],))
        noise = jax.random.normal(noise_rng, shape=action.shape)
        u_t = action - noise

        if self.simulated_delay is None:
            x_t = (1 - time[:, None, None]) * noise + time[:, None, None] * action
            pred = self(obs, x_t, time)
            return jnp.mean(jnp.square(pred - u_t))

        w = jnp.exp(jnp.arange(0, self.simulated_delay)[::-1])
        w = w / jnp.sum(w)
        delay = jax.random.choice(delay_rng, self.simulated_delay, (obs.shape[0],), p=w)
        mask = jnp.arange(self.action_chunk_size)[None, :] < delay[:, None]
        time = jnp.where(mask, 1.0, time[:, None])
        x_t = (1 - time[:, :, None]) * noise + time[:, :, None] * action
        pred = self(obs, x_t, time)
        loss = jnp.square(pred - u_t)
        loss_mask = jnp.logical_not(mask)[:, :, None]
        return jnp.sum(loss * loss_mask) / (jnp.sum(loss_mask) + 1e-8)


class LatentWorldModel(nnx.Module):
    class RSSMState(NamedTuple):
        deter: jax.Array
        stoch: jax.Array
        logit: jax.Array

    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.channel_dim = config.channel_dim
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.deter_dim = config.rssm_deter_dim
        self.stoch_dim = config.rssm_stoch_dim
        self.discrete_dim = config.rssm_discrete_dim
        self.flat_stoch_dim = self.stoch_dim * self.discrete_dim
        self.embed_dim = config.channel_dim
        self.kl_free_nats = config.kl_free_nats
        self.prediction_loss_rolloutstep = config.prediction_loss_rolloutstep
        self.prediction_loss_scale = config.prediction_loss_scale
        self.dyn_loss_scale = config.dyn_loss_scale
        self.rep_loss_scale = config.rep_loss_scale
        self.barlow_loss_scale = config.barlow_loss_scale
        self.barlow_lambda = config.barlow_lambda
        self.unimix_ratio = config.rssm_unimix_ratio

        self.encoder_wm = nnx.Sequential(
            # Vector encoder: [B, obs_dim] -> [B, embed_dim]
            nnx.Linear(obs_dim, config.vector_encoder_hidden_dim, rngs=rngs),
            nnx.RMSNorm(config.vector_encoder_hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.vector_encoder_hidden_dim, self.embed_dim, rngs=rngs),
        )
        self.latent_norm = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        # Official-style deterministic transition h_t = f(h_{t-1}, z_{t-1}, a_{t-1}).
        self.sequence_model = DeterTransition(
            self.deter_dim,
            self.flat_stoch_dim,
            action_dim,
            config.rssm_hidden_dim,
            config.rssm_blocks,
            config.rssm_dyn_layers,
            rngs=rngs,
        )
        # Prior logits p(z_t | h_t): [B, deter_dim] -> [B, stoch_dim, discrete_dim]
        self.prior_wm = RSSMHead(
            self.deter_dim,
            config.rssm_hidden_dim,
            self.flat_stoch_dim,
            config.rssm_img_layers,
            rngs=rngs,
        )
        # Posterior logits q(z_t | h_t, e_t): [B, deter_dim + embed_dim] -> [B, stoch_dim, discrete_dim]
        self.posterior_wm = RSSMHead(
            self.deter_dim + self.embed_dim,
            config.rssm_hidden_dim,
            self.flat_stoch_dim,
            config.rssm_obs_layers,
            rngs=rngs,
        )
        # Project latent features into encoder embedding space for the Barlow Twins loss:
        #   input  = concat(h_t, flatten(z_t))    [B, deter_dim + stoch_dim * discrete_dim]
        #   output = projected feature k_t        [B, embed_dim]
        self.projector = nnx.Linear(self.deter_dim + self.flat_stoch_dim, self.embed_dim, rngs=rngs)

    def initial_state(self, batch_size: int) -> "LatentWorldModel.RSSMState":
        """Returns the zero initial RSSM state for a batch."""
        deter = jnp.zeros((batch_size, self.deter_dim), dtype=jnp.float32)
        logit = jnp.zeros((batch_size, self.stoch_dim, self.discrete_dim), dtype=jnp.float32)
        stoch = jnp.zeros_like(logit)
        return self.RSSMState(deter=deter, stoch=stoch, logit=logit)

    def encode_vector(self, obs: jax.Array) -> jax.Array:
        """Encodes vector observations: [B, obs_dim] -> [B, embed_dim]."""
        return self.latent_norm(self.encoder_wm(obs))

    def _reshape_logits(self, logits: jax.Array) -> jax.Array:
        return logits.reshape(*logits.shape[:-1], self.stoch_dim, self.discrete_dim)

    def _apply_unimix(self, logits: jax.Array) -> jax.Array:
        probs = jax.nn.softmax(logits, axis=-1)
        uniform = self.unimix_ratio / self.discrete_dim
        probs = probs * (1.0 - self.unimix_ratio) + uniform
        return jnp.log(probs)

    def _sample_stoch(self, logits: jax.Array, rng: jax.Array | None) -> jax.Array:
        logits = self._apply_unimix(logits)
        if rng is None:
            one_hot = jax.nn.one_hot(jnp.argmax(logits, axis=-1), self.discrete_dim, dtype=logits.dtype)
            probs = jax.nn.softmax(logits, axis=-1)
            return probs + jax.lax.stop_gradient(one_hot - probs)
        gumbel = jax.random.gumbel(rng, logits.shape, dtype=logits.dtype)
        soft = jax.nn.softmax(logits + gumbel, axis=-1)
        one_hot = jax.nn.one_hot(jnp.argmax(soft, axis=-1), self.discrete_dim, dtype=soft.dtype)
        return soft + jax.lax.stop_gradient(one_hot - soft)

    def get_feat(self, state: "LatentWorldModel.RSSMState") -> jax.Array:
        """Returns Dreamer-style latent features s=(h,z) [B, deter_dim + stoch_dim * discrete_dim]."""
        flat_stoch = state.stoch.reshape(*state.stoch.shape[:-2], self.flat_stoch_dim)
        return jnp.concatenate([state.deter, flat_stoch], axis=-1)

    def observe_initial(
        self, obs: jax.Array, rng: jax.Array | None = None
    ) -> tuple["LatentWorldModel.RSSMState", jax.Array]:
        """Posterior state from the initial observation only."""
        return self.observe_from_prior(self.initial_state(obs.shape[0]), obs, rng=rng)

    def observe_from_prior(
        self,
        prior_state: "LatentWorldModel.RSSMState",
        obs: jax.Array,
        rng: jax.Array | None = None,
    ) -> tuple["LatentWorldModel.RSSMState", jax.Array]:
        """Observation update q(z_t | h_t, e_t) with fixed deterministic state h_t."""
        embed = self.encode_vector(obs)
        logit = self._reshape_logits(self.posterior_wm(jnp.concatenate([prior_state.deter, embed], axis=-1)))
        stoch = self._sample_stoch(logit, rng)
        return self.RSSMState(deter=prior_state.deter, stoch=stoch, logit=logit), embed

    def encode_state(self, obs: jax.Array) -> "LatentWorldModel.RSSMState":
        """Encodes a single observation into an RSSM posterior state."""
        return self.observe_initial(obs, rng=None)[0]

    def imagine_step(
        self,
        prev_state: "LatentWorldModel.RSSMState",
        action: jax.Array,
        rng: jax.Array | None = None,
    ) -> "LatentWorldModel.RSSMState":
        """One dynamics step p(h_t, z_t | h_{t-1}, z_{t-1}, a_t)."""
        deter = self.sequence_model(prev_state.stoch, prev_state.deter, action)
        logit = self._reshape_logits(self.prior_wm(deter))
        stoch = self._sample_stoch(logit, rng)
        return self.RSSMState(deter=deter, stoch=stoch, logit=logit)

    def _categorical_kl(self, lhs_logit: jax.Array, rhs_logit: jax.Array) -> jax.Array:
        lhs_logit = self._apply_unimix(lhs_logit)
        rhs_logit = self._apply_unimix(rhs_logit)
        lhs_prob = jax.nn.softmax(lhs_logit, axis=-1)
        return jnp.sum(lhs_prob * (lhs_logit - rhs_logit), axis=(-1, -2))

    def _barlow_loss(self, latent_feat: jax.Array, embed: jax.Array) -> jax.Array:
        """Barlow Twins loss between projected RSSM features and detached encoder embeddings."""
        proj = self.projector(latent_feat).reshape((-1, self.embed_dim))
        embed = jax.lax.stop_gradient(embed).reshape((-1, self.embed_dim))
        proj = (proj - jnp.mean(proj, axis=0, keepdims=True)) / (jnp.std(proj, axis=0, keepdims=True) + 1e-5)
        embed = (embed - jnp.mean(embed, axis=0, keepdims=True)) / (jnp.std(embed, axis=0, keepdims=True) + 1e-5)
        c = (proj.T @ embed) / proj.shape[0]
        invariance = jnp.sum(jnp.square(jnp.diag(c) - 1.0))
        redundancy = jnp.sum(jnp.square(c - jnp.diag(jnp.diag(c))))
        return invariance + self.barlow_lambda * redundancy

    def rollout_observed_sequence(
        self,
        obs: jax.Array,
        action: jax.Array,
        done: jax.Array,
        rng: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        """Runs RSSM posterior rollout on an observed sequence.

        obs:    [B, T + 1, obs_dim]
        action: [B, T, action_dim]
        done:   [B, T]
        """
        assert obs.ndim == 3 and action.ndim == 3 and done.ndim == 2, (obs.shape, action.shape, done.shape)
        assert obs.shape[1] == action.shape[1] + 1, (obs.shape, action.shape)
        batch_size, seq_len = action.shape[:2]
        init_rng = None if rng is None else jax.random.fold_in(rng, 0)
        init_state, init_embed = self.observe_initial(obs[:, 0], rng=init_rng)
        reset_state = self.initial_state(batch_size)

        def scan_step(carry, inputs):
            action_t, next_obs, done_t, step_idx = inputs
            step_rng = None if rng is None else jax.random.fold_in(rng, step_idx.astype(jnp.uint32) + 1)
            prior_state = self.imagine_step(carry, action_t, rng=step_rng)
            prior_for_post = jax.tree.map(
                lambda reset_x, prior_x: jnp.where(
                    done_t.reshape(done_t.shape + (1,) * (prior_x.ndim - done_t.ndim)),
                    reset_x,
                    prior_x,
                ),
                reset_state,
                prior_state,
            )
            post_state, embed = self.observe_from_prior(prior_for_post, next_obs, rng=step_rng)
            valid = 1.0 - done_t.astype(jnp.float32)
            dyn_kl = self._categorical_kl(jax.lax.stop_gradient(post_state.logit), prior_state.logit)
            rep_kl = self._categorical_kl(post_state.logit, jax.lax.stop_gradient(prior_state.logit))
            return post_state, {
                "deter": post_state.deter,
                "stoch": post_state.stoch,
                "post_feat": self.get_feat(post_state),
                "embed": embed,
                "dyn_kl": jnp.maximum(dyn_kl, self.kl_free_nats) * valid,
                "rep_kl": jnp.maximum(rep_kl, self.kl_free_nats) * valid,
                "valid": valid,
            }

        _, outputs = jax.lax.scan(
            scan_step,
            init_state,
            (
                action.swapaxes(0, 1),
                obs[:, 1:].swapaxes(0, 1),
                done.swapaxes(0, 1),
                jnp.arange(seq_len, dtype=jnp.int32),
            ),
        )
        post_feats = jnp.concatenate([self.get_feat(init_state)[:, None, :], outputs["post_feat"].swapaxes(0, 1)], axis=1)
        embeds = jnp.concatenate([init_embed[:, None, :], outputs["embed"].swapaxes(0, 1)], axis=1)
        post_deters = jnp.concatenate([init_state.deter[:, None, :], outputs["deter"].swapaxes(0, 1)], axis=1)
        post_stochs = jnp.concatenate([init_state.stoch[:, None, ...], outputs["stoch"].swapaxes(0, 1)], axis=1)
        valid = outputs["valid"].swapaxes(0, 1)
        dyn_loss = jnp.sum(outputs["dyn_kl"].swapaxes(0, 1)) / (jnp.sum(valid) + 1e-8)
        rep_loss = jnp.sum(outputs["rep_kl"].swapaxes(0, 1)) / (jnp.sum(valid) + 1e-8)
        pred_loss = self._rollout_prediction_loss(post_deters, post_stochs, post_feats, action, done)
        barlow_loss = self._barlow_loss(post_feats, embeds)
        total_loss = (
            self.prediction_loss_scale * pred_loss
            + self.dyn_loss_scale * dyn_loss
            + self.rep_loss_scale * rep_loss
            + self.barlow_loss_scale * barlow_loss
        )
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "dyn_loss": dyn_loss,
            "rep_loss": rep_loss,
            "barlow_loss": barlow_loss,
            "latent": post_feats[:, -1],
        }

    def encode_obs(self, obs: jax.Array) -> jax.Array:
        """Encodes one observation into one latent feature vector: [B, obs_dim] -> [B, feat_dim]."""
        return self.get_feat(self.encode_state(obs))

    def _rollout_prediction_loss(
        self,
        post_deters: jax.Array,
        post_stochs: jax.Array,
        post_feats: jax.Array,
        action: jax.Array,
        done: jax.Array,
    ) -> jax.Array:
        rollout_steps = min(self.prediction_loss_rolloutstep, action.shape[1])
        if rollout_steps <= 0:
            return jnp.array(0.0, dtype=post_feats.dtype)
        num_starts = action.shape[1] - rollout_steps + 1
        if num_starts <= 0:
            return jnp.array(0.0, dtype=post_feats.dtype)

        total_loss = jnp.array(0.0, dtype=post_feats.dtype)
        total_weight = jnp.array(0.0, dtype=post_feats.dtype)
        for start in range(num_starts):
            state = self.RSSMState(
                deter=post_deters[:, start],
                stoch=post_stochs[:, start],
                logit=post_stochs[:, start],
            )
            valid_chain = jnp.ones((post_feats.shape[0],), dtype=post_feats.dtype)
            for step in range(rollout_steps):
                state = self.imagine_step(state, action[:, start + step], rng=None)
                valid_chain = valid_chain * (1.0 - done[:, start + step].astype(post_feats.dtype))
                target = jax.lax.stop_gradient(post_feats[:, start + step + 1])
                step_loss = jnp.mean(jnp.square(self.get_feat(state) - target), axis=-1)
                total_loss = total_loss + jnp.sum(step_loss * valid_chain)
                total_weight = total_weight + jnp.sum(valid_chain)
        return total_loss / (total_weight + 1e-8)

    def predict_next_latent(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        """One-step prior rollout from an observation-encoded state."""
        assert action.shape == (obs.shape[0], self.action_dim), action.shape
        state_t = self.encode_state(obs)
        return self.get_feat(self.imagine_step(state_t, action, rng=None))

    def rollout_latent(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        """One-step prior rollout from an existing latent feature tensor."""
        flat_stoch = latent[..., self.deter_dim :]
        rssm_state = self.RSSMState(
            deter=latent[..., : self.deter_dim],
            stoch=flat_stoch.reshape(*flat_stoch.shape[:-1], self.stoch_dim, self.discrete_dim),
            logit=flat_stoch.reshape(*flat_stoch.shape[:-1], self.stoch_dim, self.discrete_dim),
        )
        return self.get_feat(self.imagine_step(rssm_state, action, rng=None))

    def loss(self, obs: jax.Array, action: jax.Array, next_obs: jax.Array) -> jax.Array:
        done = jnp.zeros((obs.shape[0], 1), dtype=bool)
        loss_info = self.rollout_observed_sequence(
            jnp.concatenate([obs[:, None], next_obs[:, None]], axis=1),
            action[:, None],
            done,
            rng=None,
        )
        return loss_info["loss"]


class DeterministicJepaWorldModel(nnx.Module):
    def __init__(
        self,
        *,
        obs_dim: int,
        action_dim: int,
        config: ModelConfig,
        rngs: nnx.Rngs,
    ):
        self.wm_type = config.wm_type
        self.action_dim = action_dim
        self.action_chunk_size = config.action_chunk_size
        self.deter_dim = config.channel_dim
        self.flat_stoch_dim = 0
        self.embed_dim = config.channel_dim
        self.prediction_loss_rolloutstep = config.prediction_loss_rolloutstep
        self.prediction_loss_scale = config.prediction_loss_scale
        self.reg_weight = config.jepa_reg_weight
        self.sigreg_knots = config.jepa_sigreg_knots
        self.sigreg_num_proj = config.jepa_sigreg_num_proj
        self.vc_std_coeff = config.jepa_vc_std_coeff
        self.vc_cov_coeff = config.jepa_vc_cov_coeff
        self.vc_sim_coeff_t = config.jepa_vc_sim_coeff_t

        self.encoder_wm = nnx.Sequential(
            nnx.Linear(obs_dim, config.jepa_obs_hidden_dim, rngs=rngs),
            nnx.RMSNorm(config.jepa_obs_hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.jepa_obs_hidden_dim, self.embed_dim, rngs=rngs),
        )
        self.projector = nnx.Sequential(
            nnx.Linear(self.embed_dim, config.jepa_predictor_hidden_dim, rngs=rngs),
            nnx.RMSNorm(config.jepa_predictor_hidden_dim, rngs=rngs),
            nnx.swish,
            nnx.Linear(config.jepa_predictor_hidden_dim, self.embed_dim, rngs=rngs),
        )
        self.latent_norm = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        self.action_encoder = JepaMLP(action_dim, config.jepa_action_hidden_dim, self.embed_dim, rngs=rngs)
        self.predictor = JepaPredictor(
            self.embed_dim,
            self.embed_dim,
            config.jepa_predictor_hidden_dim,
            rngs=rngs,
        )

    def encode_vector(self, obs: jax.Array) -> jax.Array:
        return self.latent_norm(self.projector(self.encoder_wm(obs)))

    def encode_obs(self, obs: jax.Array) -> jax.Array:
        return self.encode_vector(obs)

    def rollout_latent(self, latent: jax.Array, action: jax.Array) -> jax.Array:
        action_embed = self.action_encoder(action)
        return self.predictor(latent, action_embed)

    def predict_next_latent(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        return self.rollout_latent(self.encode_obs(obs), action)

    def _prediction_loss(
        self,
        obs: jax.Array,
        action: jax.Array,
        done: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        embeds = self.encode_obs(obs.reshape((-1, obs.shape[-1]))).reshape((obs.shape[0], obs.shape[1], self.embed_dim))
        rollout_steps = min(self.prediction_loss_rolloutstep, action.shape[1])
        if rollout_steps <= 0:
            return jnp.array(0.0, dtype=embeds.dtype), embeds, embeds[:, :-1]
        num_starts = action.shape[1] - rollout_steps + 1
        if num_starts <= 0:
            return jnp.array(0.0, dtype=embeds.dtype), embeds, embeds[:, :-1]

        total_loss = jnp.array(0.0, dtype=embeds.dtype)
        total_weight = jnp.array(0.0, dtype=embeds.dtype)
        final_pred = embeds[:, :-1]
        for start in range(num_starts):
            pred_latent = embeds[:, start]
            valid_chain = jnp.ones((embeds.shape[0],), dtype=embeds.dtype)
            for step in range(rollout_steps):
                pred_latent = self.rollout_latent(pred_latent, action[:, start + step])
                valid_chain = valid_chain * (1.0 - done[:, start + step].astype(embeds.dtype))
                target = jax.lax.stop_gradient(embeds[:, start + step + 1])
                step_loss = jnp.mean(jnp.square(pred_latent - target), axis=-1)
                total_loss = total_loss + jnp.sum(step_loss * valid_chain)
                total_weight = total_weight + jnp.sum(valid_chain)
                if start == num_starts - 1:
                    final_pred = final_pred.at[:, start + step].set(pred_latent)
        pred_loss = total_loss / (total_weight + 1e-8)
        return pred_loss, embeds, final_pred

    def rollout_observed_sequence(
        self,
        obs: jax.Array,
        action: jax.Array,
        done: jax.Array,
        rng: jax.Array | None = None,
    ) -> dict[str, jax.Array]:
        del rng
        pred_loss, embeds, pred = self._prediction_loss(obs, action, done)
        if self.wm_type == "lewm":
            reg_loss = _jepa_sigreg_loss(embeds, self.sigreg_knots, self.sigreg_num_proj, None)
        elif self.wm_type == "ebjepa":
            reg_loss, _ = _jepa_vc_loss(
                embeds,
                std_coeff=self.vc_std_coeff,
                cov_coeff=self.vc_cov_coeff,
                sim_coeff_t=self.vc_sim_coeff_t,
            )
        else:
            raise ValueError(f"Unsupported deterministic wm_type: {self.wm_type}")
        total_loss = self.prediction_loss_scale * pred_loss + self.reg_weight * reg_loss
        return {
            "loss": total_loss,
            "pred_loss": pred_loss,
            "reg_loss": reg_loss,
            "latent": pred[:, -1],
        }

    def loss(self, obs: jax.Array, action: jax.Array, next_obs: jax.Array) -> jax.Array:
        done = jnp.zeros((obs.shape[0], 1), dtype=bool)
        loss_info = self.rollout_observed_sequence(
            jnp.concatenate([obs[:, None], next_obs[:, None]], axis=1),
            action[:, None],
            done,
            rng=None,
        )
        return loss_info["loss"]


WorldModel: TypeAlias = LatentWorldModel | DeterministicJepaWorldModel


def build_world_model(*, obs_dim: int, action_dim: int, config: ModelConfig, rngs: nnx.Rngs) -> WorldModel:
    if config.wm_type == "rssm":
        return LatentWorldModel(obs_dim=obs_dim, action_dim=action_dim, config=config, rngs=rngs)
    if config.wm_type in ("lewm", "ebjepa"):
        return DeterministicJepaWorldModel(obs_dim=obs_dim, action_dim=action_dim, config=config, rngs=rngs)
    raise ValueError(f"Unknown wm_type: {config.wm_type}")
