import collections
import dataclasses
import functools
import math
import pathlib
import pickle
from typing import Literal, Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import pandas as pd
import tyro

import model as _model
import train_expert


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class AdaptiveChunkConfig:
    dicrepancy_measure: str = "action"
    dicrepancy_metric: str = "cos"  # or "l2norm"
    threshold: float = 0.97


@dataclasses.dataclass(frozen=True)
class DreamChunkConfig:
    n_samples: int = 16
    latent_similarity_metric: Literal["cos", "l2norm"] = "cos"
    temporal_constraint: Literal["none", "futureonly", "currentonly"] = "none"
    prediction_type: Literal["t0", "perstep"] = "t0"
    noise_injection: Literal["none", "obs", "action"] = "none"
    random_switch: bool = False


@dataclasses.dataclass(frozen=True)
class AdaptiveDreamChunkConfig:
    n_samples: int = 16
    latent_similarity_metric: Literal["cos", "l2norm"] = "cos"
    temporal_constraint: Literal["none", "futureonly", "currentonly"] = "futureonly"
    prediction_type: Literal["t0", "perstep"] = "t0"
    noise_injection: Literal["none", "obs", "action"] = "none"
    threshold: float = 0.9
    sinkhorn_epsilon: float = 0.1
    sinkhorn_iters: int = 20


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5
    action_noise: float = 0.1

    inference_delay: int = 0
    execute_horizon: int = 1
    method: (
        NaiveMethodConfig
        | RealtimeMethodConfig
        | BIDMethodConfig
        | AdaptiveChunkConfig
        | DreamChunkConfig
        | AdaptiveDreamChunkConfig
    ) = NaiveMethodConfig()

    model: _model.ModelConfig = _model.ModelConfig()


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy,
    world_model: _model.WorldModel | None,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | None = None,
):
    if isinstance(config.method, AdaptiveChunkConfig) and config.execute_horizon > 1:
        return {}, None, None
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(
            wrappers.AutoReplayWrapper(
                train_expert.NoisyActionWrapper(env, action_noise=config.action_noise)
            )
        ),
        config.num_evals,
    )
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"
    dream_method_types = (DreamChunkConfig, AdaptiveDreamChunkConfig)
    if isinstance(config.method, dream_method_types) and world_model is None:
        raise ValueError("World model checkpoint is required for DreamChunk.")
    if (
        isinstance(config.method, dream_method_types)
        and config.method.prediction_type == "perstep"
        and config.method.temporal_constraint != "currentonly"
    ):
        raise ValueError('DreamChunk prediction_type="perstep" requires temporal_constraint="currentonly".')

    def compute_action_metrics(action_a: jax.Array, action_b: jax.Array):
        a_norm = jnp.linalg.norm(action_a, axis=-1)
        b_norm = jnp.linalg.norm(action_b, axis=-1)
        cos_sim = jnp.sum(action_a * action_b, axis=-1) / (a_norm * b_norm + 1e-8)
        l2 = jnp.linalg.norm(action_a - action_b, axis=-1)
        return cos_sim, l2

    def use_old_chunk(metric_cos: jax.Array, metric_l2: jax.Array):
        if config.method.dicrepancy_metric == "cos":
            return metric_cos >= config.method.threshold
        if config.method.dicrepancy_metric == "l2norm":
            return metric_l2 <= config.method.threshold
        raise ValueError(f"Unknown dicrepancy_metric: {config.method.dicrepancy_metric}")

    def wasserstein_similarity(action_samples_a: jax.Array, action_samples_b: jax.Array):
        assert isinstance(config.method, AdaptiveDreamChunkConfig)
        cost = jnp.linalg.norm(action_samples_a[:, :, None, :] - action_samples_b[:, None, :, :], axis=-1)
        kernel = jnp.exp(-cost / config.method.sinkhorn_epsilon)
        n_samples = action_samples_a.shape[1]
        target = jnp.full((action_samples_a.shape[0], n_samples), 1.0 / n_samples)
        u = target
        v = target
        for _ in range(config.method.sinkhorn_iters):
            u = target / (jnp.einsum("bij,bj->bi", kernel, v) + 1e-8)
            v = target / (jnp.einsum("bij,bi->bj", kernel, u) + 1e-8)
        transport = u[:, :, None] * kernel * v[:, None, :]
        wasserstein = jnp.sum(transport * cost, axis=(1, 2))
        return jnp.exp(-wasserstein), wasserstein

    def step_env(carry, action):
        rng, obs, env_state = carry
        rng, key = jax.random.split(rng)
        next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
        return (rng, next_obs, next_env_state), (done, env_state, info)

    def execute_chunk_standard(carry, _):
        rng, obs, env_state, action_chunk, n, effective_horizon = carry
        rng, key = jax.random.split(rng)
        batch_size = obs.shape[0]
        effective_horizon_out = jnp.zeros((batch_size,), dtype=jnp.float32)

        if isinstance(config.method, AdaptiveChunkConfig):
            proposed_chunk = policy.action(key, obs, config.num_flow_steps)
            if config.method.dicrepancy_measure == "action":
                action_to_execute = action_chunk[:, : config.execute_horizon]
                if config.inference_delay == 0:
                    cos_sim, l2 = compute_action_metrics(action_chunk[:, 0], proposed_chunk[:, 0])
                    keep_old = use_old_chunk(cos_sim, l2)
                    action_to_execute = action_to_execute.at[:, 0].set(
                        jnp.where(keep_old[:, None], action_chunk[:, 0], proposed_chunk[:, 0])
                    )
                    effective_horizon_out = jnp.where(keep_old, effective_horizon + 1.0, 1.0)
                elif config.inference_delay == 1:
                    cos_sim, l2 = compute_action_metrics(action_chunk[:, 1], proposed_chunk[:, 1])
                    keep_old = use_old_chunk(cos_sim, l2)
                    if config.execute_horizon > 1:
                        action_to_execute = action_to_execute.at[:, 1].set(
                            jnp.where(keep_old[:, None], action_chunk[:, 1], proposed_chunk[:, 1])
                        )
                    effective_horizon_out = jnp.where(keep_old, effective_horizon + 1.0, 1.0)
                else:
                    raise ValueError(f"Unsupported {config.inference_delay=} for AdaptiveChunk")
            else:
                raise ValueError(f"Unknown dicrepancy_measure: {config.method.dicrepancy_measure}")

            action_chunk_to_execute = action_to_execute
            (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
                step_env, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
            )
            next_queue_old = jnp.concatenate([action_chunk[:, 1:], proposed_chunk[:, -1:]], axis=1)
            next_queue_new = jnp.concatenate([proposed_chunk[:, 1:], proposed_chunk[:, -1:]], axis=1)
            next_action_chunk = jnp.where(keep_old[:, None, None], next_queue_old, next_queue_new)
            next_n = jnp.concatenate([n[1:], jnp.zeros(1, dtype=jnp.int32)])
            effective_horizon_scan = jnp.repeat(effective_horizon_out[None, :], config.execute_horizon, axis=0)
            return (
                rng,
                next_obs,
                next_env_state,
                next_action_chunk,
                next_n,
                effective_horizon_out,
            ), (dones, env_states, infos, effective_horizon_scan)

        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        else:
            raise ValueError(f"Unknown method: {config.method}")

        action_chunk_to_execute = jnp.concatenate(
            [
                action_chunk[:, : config.inference_delay],
                next_action_chunk[:, config.inference_delay : config.execute_horizon],
            ],
            axis=1,
        )
        next_action_chunk = jnp.concatenate(
            [
                next_action_chunk[:, config.execute_horizon :],
                jnp.zeros((obs.shape[0], config.execute_horizon, policy.action_dim)),
            ],
            axis=1,
        )
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step_env, (rng, obs, env_state), action_chunk_to_execute.transpose(1, 0, 2)
        )
        effective_horizon_scan = jnp.repeat(effective_horizon_out[None, :], config.execute_horizon, axis=0)
        return (
            rng,
            next_obs,
            next_env_state,
            next_action_chunk,
            next_n,
            effective_horizon,
        ), (dones, env_states, infos, effective_horizon_scan)

    def build_dream_pool(pool_key: jax.Array, obs_in: jax.Array):
        assert isinstance(config.method, dream_method_types)
        assert world_model is not None
        batch_size = obs_in.shape[0]
        horizon = policy.action_chunk_size
        n_samples = config.method.n_samples
        latent_dim = world_model.deter_dim + world_model.flat_stoch_dim
        obs_repeated = jnp.repeat(obs_in[None, ...], n_samples, axis=0).reshape(n_samples * batch_size, -1)
        if config.method.noise_injection == "obs":
            pool_key, obs_noise_key = jax.random.split(pool_key)
            obs_repeated = obs_repeated + 0.1 * jax.random.normal(obs_noise_key, obs_repeated.shape)
        elif config.method.noise_injection not in ("none", "action"):
            raise ValueError(f"Unknown noise_injection: {config.method.noise_injection}")
        # Preserve key parity with the naive baseline when no extra noise is injected.
        if config.method.noise_injection == "none":
            action_key = pool_key
        else:
            pool_key, action_key = jax.random.split(pool_key)
        sampled_chunks = policy.action(action_key, obs_repeated, config.num_flow_steps).reshape(
            n_samples, batch_size, horizon, policy.action_dim
        )  # [n, b, h, d]
        if config.method.noise_injection == "action":
            pool_key, action_noise_key = jax.random.split(pool_key)
            sampled_chunks = sampled_chunks + 0.1 * jax.random.normal(action_noise_key, sampled_chunks.shape)
        z_t = world_model.encode_obs(obs_in)  # [b, c]
        z_t = jnp.repeat(z_t[None, ...], n_samples, axis=0)  # [n, b, c]
        rollout_actions_t = sampled_chunks.transpose(2, 0, 1, 3)  # [h, n, b, d]

        def rollout_step(z_prev: jax.Array, action_t: jax.Array):
            assert world_model is not None
            z_next = world_model.rollout_latent(z_prev, action_t)
            return z_next, z_next

        _, z_rollout = jax.lax.scan(rollout_step, z_t, rollout_actions_t)  # [h, n, b, c]
        z_t_h = z_t[:, :, None, :]  # [n, b, 1, c]
        z_prev_for_actions = jnp.concatenate([z_t_h, z_rollout[:-1].transpose(1, 2, 0, 3)], axis=2)  # [n, b, h, c]
        latent_pool = z_prev_for_actions.transpose(1, 0, 2, 3).reshape(batch_size, n_samples * horizon, latent_dim)
        action_pool = sampled_chunks.transpose(1, 0, 2, 3).reshape(
            batch_size, n_samples * horizon, policy.action_dim
        )
        base_chunk = sampled_chunks[0]  # [b, h, d], used for direct a_t initialization
        return latent_pool, action_pool, base_chunk

    def build_perstep_dream_pool(pool_key: jax.Array, obs_in: jax.Array):
        assert isinstance(config.method, dream_method_types)
        assert world_model is not None
        batch_size = obs_in.shape[0]
        horizon = policy.action_chunk_size
        n_samples = config.method.n_samples
        latent_dim = world_model.deter_dim + world_model.flat_stoch_dim
        obs_repeated = jnp.repeat(obs_in[None, ...], n_samples, axis=0).reshape(n_samples * batch_size, -1)
        if config.method.noise_injection == "obs":
            pool_key, obs_noise_key = jax.random.split(pool_key)
            obs_repeated = obs_repeated + 0.1 * jax.random.normal(obs_noise_key, obs_repeated.shape)
        elif config.method.noise_injection not in ("none", "action"):
            raise ValueError(f"Unknown noise_injection: {config.method.noise_injection}")
        if config.method.noise_injection == "none":
            action_key = pool_key
        else:
            pool_key, action_key = jax.random.split(pool_key)
        sampled_chunks = policy.action(action_key, obs_repeated, config.num_flow_steps).reshape(
            n_samples, batch_size, horizon, policy.action_dim
        )  # [n, b, h, d]
        if config.method.noise_injection == "action":
            pool_key, action_noise_key = jax.random.split(pool_key)
            sampled_chunks = sampled_chunks + 0.1 * jax.random.normal(action_noise_key, sampled_chunks.shape)

        z_t = world_model.encode_obs(obs_in)  # [b, c]
        z_t = jnp.repeat(z_t[None, ...], n_samples, axis=0)  # [n, b, c]
        z_t_h = jnp.repeat(z_t[:, :, None, :], horizon, axis=2)  # [n, b, h, c]
        z_one_step = world_model.rollout_latent(z_t_h, sampled_chunks)  # [n, b, h, c]
        z_for_actions = jnp.concatenate([z_t_h[:, :, :1, :], z_one_step[:, :, :-1, :]], axis=2)
        latent_pool = z_for_actions.transpose(1, 0, 2, 3).reshape(batch_size, n_samples * horizon, latent_dim)
        action_pool = sampled_chunks.transpose(1, 0, 2, 3).reshape(
            batch_size, n_samples * horizon, policy.action_dim
        )
        base_chunk = sampled_chunks[0]  # [b, h, d], used for direct a_t initialization
        return latent_pool, action_pool, base_chunk

    def select_dream_action_from_pool(
        rng: jax.Array,
        obs_in: jax.Array,
        pool_step: jax.Array,
        latent_pool: jax.Array,
        action_pool: jax.Array,
    ):
        assert isinstance(config.method, dream_method_types)
        assert world_model is not None
        if config.method.prediction_type == "perstep" and config.method.temporal_constraint != "currentonly":
            raise ValueError('DreamChunk prediction_type="perstep" requires temporal_constraint="currentonly".')
        horizon = policy.action_chunk_size
        n_samples = config.method.n_samples
        pool_time_idx = jnp.tile(jnp.arange(horizon, dtype=jnp.int32), n_samples)
        max_pool_step = jnp.maximum(horizon - 1, 0)
        clamped_step = jnp.minimum(pool_step, max_pool_step)
        z_cur = world_model.encode_obs(obs_in)  # [b, c]
        if config.method.latent_similarity_metric == "cos":
            z_norm = jnp.linalg.norm(z_cur, axis=-1, keepdims=True)  # [b, 1]
            pool_norm = jnp.linalg.norm(latent_pool, axis=-1)  # [b, n*h]
            score = jnp.sum(latent_pool * z_cur[:, None, :], axis=-1) / (pool_norm * z_norm + 1e-8)
            invalid_fill = -jnp.inf
            best_fn = jnp.argmax
        elif config.method.latent_similarity_metric == "l2norm":
            score = jnp.linalg.norm(latent_pool - z_cur[:, None, :], axis=-1)
            invalid_fill = jnp.inf
            best_fn = jnp.argmin
        else:
            raise ValueError(f"Unknown latent_similarity_metric: {config.method.latent_similarity_metric}")
        valid_mask = jnp.ones_like(score, dtype=jnp.bool_)
        if config.method.temporal_constraint == "futureonly":
            valid_mask = pool_time_idx[None, :] >= clamped_step[:, None]
            score = jnp.where(valid_mask, score, invalid_fill)
        elif config.method.temporal_constraint == "currentonly":
            valid_mask = pool_time_idx[None, :] == clamped_step[:, None]
            score = jnp.where(valid_mask, score, invalid_fill)
        elif config.method.temporal_constraint != "none":
            raise ValueError(f"Unknown temporal_constraint: {config.method.temporal_constraint}")
        if isinstance(config.method, DreamChunkConfig) and config.method.random_switch:
            random_score = jax.random.uniform(rng, score.shape)
            best_idx = jnp.argmax(jnp.where(valid_mask, random_score, -jnp.inf), axis=-1)[:, None, None]
        else:
            best_idx = best_fn(score, axis=-1)[:, None, None]  # [b, 1, 1]
        best_action = jnp.take_along_axis(action_pool, best_idx, axis=1).squeeze(1)  # [b, d]
        best_score = jnp.take_along_axis(score, best_idx.squeeze(-1), axis=1).squeeze(1)  # [b]
        return best_action, best_score

    def pool_by_time(pool: jax.Array):
        batch_size = pool.shape[0]
        return pool.reshape(batch_size, config.method.n_samples, policy.action_chunk_size, pool.shape[-1])

    def flatten_time_pool(pool: jax.Array):
        return pool.reshape(pool.shape[0], config.method.n_samples * policy.action_chunk_size, pool.shape[-1])

    def shift_append_pool(old_pool: jax.Array, append_pool: jax.Array, shift_steps: int):
        old_by_time = pool_by_time(old_pool)
        append_by_time = pool_by_time(append_pool)
        if shift_steps == 0:
            return old_pool
        if shift_steps >= policy.action_chunk_size:
            shifted = append_by_time[:, :, -policy.action_chunk_size :, :]
        else:
            shifted = jnp.concatenate(
                [old_by_time[:, :, shift_steps:, :], append_by_time[:, :, -shift_steps:, :]],
                axis=2,
            )
        return flatten_time_pool(shifted)

    def execute_chunk_dream(carry, _):
        assert isinstance(config.method, DreamChunkConfig)
        rng, obs, env_state, n, prev_latent_pool, prev_action_pool, prev_base_chunk, prev_pool_step = carry
        batch_size = obs.shape[0]
        max_pool_step = jnp.maximum(policy.action_chunk_size - 1, 0)
        rng, key = jax.random.split(rng)
        if config.method.prediction_type == "perstep":
            new_latent_pool, new_action_pool, new_base_chunk = build_perstep_dream_pool(key, obs)
        elif config.method.prediction_type == "t0":
            new_latent_pool, new_action_pool, new_base_chunk = build_dream_pool(key, obs)
        else:
            raise ValueError(f"Unknown prediction_type: {config.method.prediction_type}")
        delay_steps = config.inference_delay
        dream_steps = config.execute_horizon - delay_steps

        def make_t0_dream_step(latent_pool: jax.Array, action_pool: jax.Array):
            def dream_step(carry, _):
                rng, obs, env_state, action, pool_step = carry
                rng, key = jax.random.split(rng)
                next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
                next_pool_step = jnp.minimum(pool_step + 1, max_pool_step)
                rng, select_key = jax.random.split(rng)
                next_action, best_pool_score = select_dream_action_from_pool(
                    select_key,
                    next_obs, next_pool_step, latent_pool, action_pool
                )
                return (rng, next_obs, next_env_state, next_action, next_pool_step), (
                    done,
                    env_state,
                    info,
                    best_pool_score,
                )

            return dream_step

        def perstep_dream_step(carry, _):
            rng, obs, env_state, action, pool_step, latent_pool, action_pool = carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            next_pool_step = jnp.minimum(pool_step + 1, max_pool_step)
            rng, select_key = jax.random.split(rng)
            next_action, best_pool_score = select_dream_action_from_pool(
                select_key,
                next_obs, next_pool_step, latent_pool, action_pool
            )
            rng, pool_key = jax.random.split(rng)
            next_latent_pool, next_action_pool, _ = build_perstep_dream_pool(pool_key, next_obs)
            return (
                rng,
                next_obs,
                next_env_state,
                next_action,
                next_pool_step,
                next_latent_pool,
                next_action_pool,
            ), (
                done,
                env_state,
                info,
                best_pool_score,
            )
       
        delayed_outputs = None
        delayed_rng, delayed_obs, delayed_env_state = rng, obs, env_state
        delayed_latent_pool, delayed_action_pool = prev_latent_pool, prev_action_pool
        if delay_steps > 0:
            rng, select_key = jax.random.split(rng)
            init_delayed_action, _ = select_dream_action_from_pool(
                select_key,
                obs, prev_pool_step, prev_latent_pool, prev_action_pool
            )
            if config.method.prediction_type == "perstep":
                (
                    delayed_rng,
                    delayed_obs,
                    delayed_env_state,
                    _,
                    _,
                    delayed_latent_pool,
                    delayed_action_pool,
                ), delayed_outputs = jax.lax.scan(
                    perstep_dream_step,
                    (rng, obs, env_state, init_delayed_action, prev_pool_step, prev_latent_pool, prev_action_pool),
                    None,
                    length=delay_steps,
                )
            else:
                delayed_step = make_t0_dream_step(prev_latent_pool, prev_action_pool)
                (delayed_rng, delayed_obs, delayed_env_state, _, _), delayed_outputs = jax.lax.scan(
                    delayed_step,
                    (rng, obs, env_state, init_delayed_action, prev_pool_step),
                    None,
                    length=delay_steps,
                )

        if dream_steps > 0:
            new_pool_step_init = jnp.full((batch_size,), delay_steps, dtype=jnp.int32)
            if delay_steps == 0:
                init_new_action = new_base_chunk[:, 0, :]
            else:
                delayed_rng, select_key = jax.random.split(delayed_rng)
                init_new_action, _ = select_dream_action_from_pool(
                    select_key,
                    delayed_obs, new_pool_step_init, new_latent_pool, new_action_pool
                )
            if config.method.prediction_type == "perstep":
                (
                    rng,
                    next_obs,
                    next_env_state,
                    _,
                    next_prev_pool_step,
                    next_latent_pool,
                    next_action_pool,
                ), dream_outputs = jax.lax.scan(
                    perstep_dream_step,
                    (
                        delayed_rng,
                        delayed_obs,
                        delayed_env_state,
                        init_new_action,
                        new_pool_step_init,
                        new_latent_pool,
                        new_action_pool,
                    ),
                    None,
                    length=dream_steps,
                )
            else:
                new_step = make_t0_dream_step(new_latent_pool, new_action_pool)
                (rng, next_obs, next_env_state, _, next_prev_pool_step), dream_outputs = jax.lax.scan(
                    new_step,
                    (delayed_rng, delayed_obs, delayed_env_state, init_new_action, new_pool_step_init),
                    None,
                    length=dream_steps,
                )
                next_latent_pool, next_action_pool = new_latent_pool, new_action_pool
            if delayed_outputs is not None:
                scan_outputs = jax.tree.map(
                    lambda x, y: jnp.concatenate([x, y], axis=0), delayed_outputs, dream_outputs
                )
            else:
                scan_outputs = dream_outputs
        else:
            rng, next_obs, next_env_state = delayed_rng, delayed_obs, delayed_env_state
            assert delayed_outputs is not None, "DreamChunk requires some executed actions."
            scan_outputs = delayed_outputs
            next_prev_pool_step = jnp.full((batch_size,), delay_steps, dtype=jnp.int32)
            if config.method.prediction_type == "perstep":
                next_latent_pool, next_action_pool = delayed_latent_pool, delayed_action_pool
            else:
                next_latent_pool, next_action_pool = new_latent_pool, new_action_pool

        dones, env_states, infos, best_pool_score = scan_outputs
        next_n = jnp.concatenate([n[config.execute_horizon :], jnp.zeros(config.execute_horizon, dtype=jnp.int32)])
        return (
            rng,
            next_obs,
            next_env_state,
            next_n,
            next_latent_pool,
            next_action_pool,
            new_base_chunk,
            next_prev_pool_step,
        ), (dones, env_states, infos, best_pool_score)

    def execute_chunk_adaptive_dream(carry, _):
        assert isinstance(config.method, AdaptiveDreamChunkConfig)
        rng, obs, env_state, prev_latent_pool, prev_action_pool, prev_base_chunk, effective_horizon = carry
        batch_size = obs.shape[0]
        max_pool_step = jnp.maximum(policy.action_chunk_size - 1, 0)
        rng, key = jax.random.split(rng)
        if config.method.prediction_type == "perstep":
            new_latent_pool, new_action_pool, new_base_chunk = build_perstep_dream_pool(key, obs)
        elif config.method.prediction_type == "t0":
            new_latent_pool, new_action_pool, new_base_chunk = build_dream_pool(key, obs)
        else:
            raise ValueError(f"Unknown prediction_type: {config.method.prediction_type}")

        compare_step = min(config.inference_delay, policy.action_chunk_size - 1)
        prev_action_samples = pool_by_time(prev_action_pool)[:, :, compare_step, :]
        new_action_samples = pool_by_time(new_action_pool)[:, :, compare_step, :]
        adaptive_similarity, _ = wasserstein_similarity(prev_action_samples, new_action_samples)
        keep_old = adaptive_similarity >= config.method.threshold
        active_latent_pool = jnp.where(keep_old[:, None, None], prev_latent_pool, new_latent_pool)
        active_action_pool = jnp.where(keep_old[:, None, None], prev_action_pool, new_action_pool)
        effective_horizon_out = jnp.where(keep_old, effective_horizon + 1.0, 1.0)

        init_pool_step = jnp.zeros((batch_size,), dtype=jnp.int32)
        use_prev_init = init_pool_step < config.inference_delay
        init_latent_pool = jnp.where(use_prev_init[:, None, None], prev_latent_pool, active_latent_pool)
        init_action_pool = jnp.where(use_prev_init[:, None, None], prev_action_pool, active_action_pool)
        rng, select_key = jax.random.split(rng)
        init_action, _ = select_dream_action_from_pool(
            select_key, obs, init_pool_step, init_latent_pool, init_action_pool
        )

        def adaptive_dream_step(step_carry, _):
            rng, obs, env_state, action, pool_step = step_carry
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            next_pool_step = jnp.minimum(pool_step + 1, max_pool_step)
            use_prev_pool = next_pool_step < config.inference_delay
            select_latent_pool = jnp.where(use_prev_pool[:, None, None], prev_latent_pool, active_latent_pool)
            select_action_pool = jnp.where(use_prev_pool[:, None, None], prev_action_pool, active_action_pool)
            rng, select_key = jax.random.split(rng)
            next_action, best_pool_score = select_dream_action_from_pool(
                select_key,
                next_obs, next_pool_step, select_latent_pool, select_action_pool
            )
            return (rng, next_obs, next_env_state, next_action, next_pool_step), (
                done,
                env_state,
                info,
                best_pool_score,
                effective_horizon_out,
            )

        (rng, next_obs, next_env_state, _, _), scan_outputs = jax.lax.scan(
            adaptive_dream_step,
            (rng, obs, env_state, init_action, init_pool_step),
            None,
            length=config.execute_horizon,
        )
        kept_latent_pool = shift_append_pool(prev_latent_pool, new_latent_pool, config.execute_horizon)
        kept_action_pool = shift_append_pool(prev_action_pool, new_action_pool, config.execute_horizon)
        switched_latent_pool = shift_append_pool(new_latent_pool, new_latent_pool, config.execute_horizon)
        switched_action_pool = shift_append_pool(new_action_pool, new_action_pool, config.execute_horizon)
        next_latent_pool = jnp.where(keep_old[:, None, None], kept_latent_pool, switched_latent_pool)
        next_action_pool = jnp.where(keep_old[:, None, None], kept_action_pool, switched_action_pool)
        next_base_chunk = jnp.where(keep_old[:, None, None], prev_base_chunk, new_base_chunk)
        return (
            rng,
            next_obs,
            next_env_state,
            next_latent_pool,
            next_action_pool,
            next_base_chunk,
            effective_horizon_out,
        ), scan_outputs

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    if isinstance(config.method, AdaptiveDreamChunkConfig):
        assert world_model is not None
        assert policy.action_chunk_size > 1, "AdaptiveDreamChunk requires action_chunk_size > 1"
        assert config.execute_horizon <= policy.action_chunk_size, (
            f"AdaptiveDreamChunk requires {config.execute_horizon=} <= {policy.action_chunk_size=}"
        )
        rng, key = jax.random.split(rng)
        if config.method.prediction_type == "perstep":
            init_prev_latent_pool, init_prev_action_pool, init_prev_base_chunk = build_perstep_dream_pool(key, obs)
        elif config.method.prediction_type == "t0":
            init_prev_latent_pool, init_prev_action_pool, init_prev_base_chunk = build_dream_pool(key, obs)
        else:
            raise ValueError(f"Unknown prediction_type: {config.method.prediction_type}")
        init_effective_horizon = jnp.ones((obs.shape[0],), dtype=jnp.float32)
        _, (dones, env_states, infos, best_pool_score, effective_horizon_values) = jax.lax.scan(
            execute_chunk_adaptive_dream,
            (
                rng,
                obs,
                env_state,
                init_prev_latent_pool,
                init_prev_action_pool,
                init_prev_base_chunk,
                init_effective_horizon,
            ),
            None,
            length=scan_length,
        )
        dones, env_states, infos, best_pool_score, effective_horizon_values = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]),
            (dones, env_states, infos, best_pool_score, effective_horizon_values),
        )
    elif isinstance(config.method, DreamChunkConfig):
        assert world_model is not None
        assert policy.action_chunk_size > 1, "DreamChunk requires action_chunk_size > 1"
        assert config.execute_horizon <= policy.action_chunk_size, (
            f"DreamChunk requires {config.execute_horizon=} <= {policy.action_chunk_size=}"
        )
        n = jnp.ones(policy.action_chunk_size, dtype=jnp.int32)
        rng, key = jax.random.split(rng)
        if config.method.prediction_type == "perstep":
            init_prev_latent_pool, init_prev_action_pool, init_prev_base_chunk = build_perstep_dream_pool(key, obs)
        elif config.method.prediction_type == "t0":
            init_prev_latent_pool, init_prev_action_pool, init_prev_base_chunk = build_dream_pool(key, obs)
        else:
            raise ValueError(f"Unknown prediction_type: {config.method.prediction_type}")
        init_prev_pool_step = jnp.zeros((obs.shape[0],), dtype=jnp.int32)
        _, (dones, env_states, infos, best_pool_score) = jax.lax.scan(
            execute_chunk_dream,
            (
                rng,
                obs,
                env_state,
                n,
                init_prev_latent_pool,
                init_prev_action_pool,
                init_prev_base_chunk,
                init_prev_pool_step,
            ),
            None,
            length=scan_length,
        )
        dones, env_states, infos, best_pool_score = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos, best_pool_score)
        )
        effective_horizon_values = None
    else:
        rng, key = jax.random.split(rng)
        action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
        n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
        init_effective_horizon = jnp.ones((obs.shape[0],), dtype=jnp.float32)
        _, (dones, env_states, infos, effective_horizon_values) = jax.lax.scan(
            execute_chunk_standard,
            (rng, obs, env_state, action_chunk, n, init_effective_horizon),
            None,
            length=scan_length,
        )
        dones, env_states, infos, effective_horizon_values = jax.tree.map(
            lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos, effective_horizon_values)
        )
        best_pool_score = None
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        # only consider the first episode of each rollout
        first_done_idx = jnp.argmax(dones, axis=0)
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    sim_coverage_metrics = None
    done_any = jnp.any(dones, axis=0)
    first_done_idx = jnp.where(done_any, jnp.argmax(dones, axis=0), dones.shape[0])
    step_idx = jnp.arange(dones.shape[0])[:, None]
    valid_mask = step_idx < first_done_idx[None, :]
    if isinstance(config.method, (AdaptiveChunkConfig, AdaptiveDreamChunkConfig)):
        valid_effective_horizon = jnp.where(valid_mask, effective_horizon_values, jnp.nan)
        return_info["effective_horizon_mean"] = jnp.nanmean(valid_effective_horizon)
        return_info["effective_horizon_std"] = jnp.nanstd(valid_effective_horizon)
        return_info["effective_horizon_min"] = jnp.nanmin(valid_effective_horizon)
        return_info["effective_horizon_max"] = jnp.nanmax(valid_effective_horizon)
    if isinstance(config.method, dream_method_types):
        valid_best_pool_score = jnp.where(valid_mask, best_pool_score, jnp.nan)
        q1 = jnp.nanquantile(valid_best_pool_score, 0.25)
        med = jnp.nanquantile(valid_best_pool_score, 0.5)
        q3 = jnp.nanquantile(valid_best_pool_score, 0.75)
        mean = jnp.nanmean(valid_best_pool_score)
        std = jnp.nanstd(valid_best_pool_score)
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr
        whisker_mask = valid_mask & (best_pool_score >= lower_fence) & (best_pool_score <= upper_fence)
        whislo = jnp.nanmin(jnp.where(whisker_mask, best_pool_score, jnp.nan))
        whishi = jnp.nanmax(jnp.where(whisker_mask, best_pool_score, jnp.nan))
        sim_coverage_metrics = {
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": whislo,
            "whishi": whishi,
            "mean": mean,
            "std": std,
        }
    return return_info, video, None, sim_coverage_metrics


def main(
    flow_run_path: str,
    wm_run_path: str | None = None,
    config: EvalConfig = EvalConfig(),
    wm_type: Literal["rssm", "lewm", "ebjepa", "policyencoder"] | None = None,
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
    ),
    seed: int = 0,
    output_dir: str | None = "eval_output",
):
    if wm_type is not None:
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, wm_type=wm_type))
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # load flow policies and optionally world models from matching checkpoint trees
    state_dicts = []
    world_model_state_dicts = []
    weak_state_dicts = []
    flow_log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(flow_run_path).iterdir()))
    flow_log_dirs = sorted(flow_log_dirs, key=lambda p: int(p.name))
    if not flow_log_dirs:
        raise ValueError(f"No flow checkpoints found in {flow_run_path}")
    policy_step_dir = flow_log_dirs[config.step]
    weak_step_dir = flow_log_dirs[config.weak_step] if config.weak_step is not None else None

    has_world_model = wm_run_path is not None
    if has_world_model:
        wm_log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(wm_run_path).iterdir()))
        wm_log_dirs = sorted(wm_log_dirs, key=lambda p: int(p.name))
        if not wm_log_dirs:
            raise ValueError(f"No world model checkpoints found in {wm_run_path}")
        world_model_step_dir = wm_log_dirs[config.step]
        if not (world_model_step_dir / "world_models").is_dir():
            raise ValueError(f"Expected world model checkpoints in {world_model_step_dir / 'world_models'}")
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        # load policy
        with (policy_step_dir / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if has_world_model:
            with (world_model_step_dir / "world_models" / f"{level_name}.pkl").open("rb") as f:
                world_model_state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (weak_step_dir / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if has_world_model:
        world_model_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *world_model_state_dicts))
    else:
        world_model_state_dicts = None
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    if has_world_model:
        @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
        @functools.partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=(None, pspec, pspec, pspec, pspec, pspec),
            out_specs=pspec,
        )
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0))
        def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, world_model_state_dict, weak_state_dict):
            policy = _model.FlowPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config.model,
                rngs=nnx.Rngs(rng),
            )
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)
            world_model = _model.build_world_model(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config.model,
                rngs=nnx.Rngs(rng),
            )
            graphdef, state = nnx.split(world_model)
            state.replace_by_pure_dict(world_model_state_dict)
            world_model = nnx.merge(graphdef, state)
            if weak_state_dict is not None:
                graphdef, state = nnx.split(policy)
                state.replace_by_pure_dict(weak_state_dict)
                weak_policy = nnx.merge(graphdef, state)
            else:
                weak_policy = None
            eval_info, _, _, sim_coverage_metrics = eval(
                config,
                env,
                rng,
                level,
                policy,
                world_model,
                env_params,
                static_env_params,
                weak_policy,
            )
            return eval_info, sim_coverage_metrics
    else:
        @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
        @functools.partial(
            shard_map.shard_map,
            mesh=mesh,
            in_specs=(None, pspec, pspec, pspec, pspec),
            out_specs=pspec,
        )
        @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0))
        def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict):
            policy = _model.FlowPolicy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                config=config.model,
                rngs=nnx.Rngs(rng),
            )
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(state_dict)
            policy = nnx.merge(graphdef, state)
            if weak_state_dict is not None:
                graphdef, state = nnx.split(policy)
                state.replace_by_pure_dict(weak_state_dict)
                weak_policy = nnx.merge(graphdef, state)
            else:
                weak_policy = None
            eval_info, _, _, sim_coverage_metrics = eval(
                config,
                env,
                rng,
                level,
                policy,
                None,
                env_params,
                static_env_params,
                weak_policy,
            )
            return eval_info, sim_coverage_metrics

    rngs = jax.random.split(jax.random.key(seed), len(level_paths))
    results = collections.defaultdict(list)
    sim_coverage_results = collections.defaultdict(list)
    effective_horizon_metric_keys = [
        "effective_horizon_mean",
        "effective_horizon_std",
        "effective_horizon_min",
        "effective_horizon_max",
    ]

    def results_frame(data: collections.defaultdict):
        max_len = max((len(values) for values in data.values()), default=0)
        rectangular = {
            key: values + [float("nan")] * (max_len - len(values))
            for key, values in data.items()
        }
        return pd.DataFrame(rectangular)

    output_path = pathlib.Path(output_dir) if output_dir is not None else None
    active_wm_type = config.model.wm_type if has_world_model else "none"
    if output_path is not None:
        results_csv = output_path / "results.csv"
        if results_csv.exists():
            existing_df = pd.read_csv(results_csv)
            if "wm_type" not in existing_df:
                existing_df["wm_type"] = active_wm_type
            for metric_key in effective_horizon_metric_keys:
                if metric_key not in existing_df:
                    existing_df[metric_key] = float("nan")
            existing_results = existing_df.to_dict(orient="list")
            for key, values in existing_results.items():
                results[key].extend(values)
        sim_coverage_csv = output_path / "sim_coverage.csv"
        if sim_coverage_csv.exists():
            existing_sim_df = pd.read_csv(sim_coverage_csv)
            if "wm_type" not in existing_sim_df:
                existing_sim_df["wm_type"] = active_wm_type
            if "prediction_type" not in existing_sim_df:
                existing_sim_df["prediction_type"] = "t0"
            existing_sim_results = existing_sim_df.to_dict(orient="list")
            for key, values in existing_sim_results.items():
                sim_coverage_results[key].extend(values)
    for action_noise in [config.action_noise]:
        for inference_delay in [1]:
            for execute_horizon in range(max(1, inference_delay), 8 - inference_delay + 1, 2):
                print(f"{inference_delay=} {execute_horizon=}{action_noise=}")
                def run_method(method_config, method_name):
                    if isinstance(method_config, AdaptiveChunkConfig) and execute_horizon > 1:
                        print(f"skip {method_name} for {execute_horizon=}")
                        return
                    c = dataclasses.replace(
                        config,
                        inference_delay=inference_delay,
                        execute_horizon=execute_horizon,
                        action_noise=action_noise,
                        method=method_config,
                    )
                    if has_world_model:
                        out, sim_coverage_out = jax.device_get(
                            _eval(c, rngs, levels, state_dicts, world_model_state_dicts, weak_state_dicts)
                        )
                    else:
                        out, sim_coverage_out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                    if not out:
                        print(f"no metrics for {method_name} at {execute_horizon=}")
                        return
                    for i in range(len(level_paths)):
                        for k, v in out.items():
                            results[k].append(v[i])
                        for metric_key in effective_horizon_metric_keys:
                            if metric_key not in out:
                                results[metric_key].append(float("nan"))
                        results["delay"].append(inference_delay)
                        results["method"].append(method_name)
                        results["wm_type"].append(active_wm_type)
                        results["level"].append(level_paths[i])
                        results["execute_horizon"].append(execute_horizon)
                        results["action_noise"].append(action_noise)
                    if isinstance(method_config, (DreamChunkConfig, AdaptiveDreamChunkConfig)) and sim_coverage_out is not None:
                        for i in range(len(level_paths)):
                            for k, v in sim_coverage_out.items():
                                sim_coverage_results[k].append(v[i])
                            sim_coverage_results["delay"].append(inference_delay)
                            sim_coverage_results["method"].append(method_name)
                            sim_coverage_results["wm_type"].append(active_wm_type)
                            sim_coverage_results["level"].append(level_paths[i])
                            sim_coverage_results["execute_horizon"].append(execute_horizon)
                            sim_coverage_results["action_noise"].append(action_noise)
                            sim_coverage_results["dreamchunk_nsample"].append(method_config.n_samples)
                            sim_coverage_results["sim_metric"].append(method_config.latent_similarity_metric)
                            sim_coverage_results["prediction_type"].append(method_config.prediction_type)
                            sim_coverage_results["fliers"].append("[]")


                # for ni in []:
                # for ni in [20]:
                #     print(f"Dream Chunk sample n {ni}")
                #     run_method(DreamChunkConfig(n_samples=ni,temporal_constraint="currentonly",latent_similarity_metric="l2norm"), f"DreamChunk_current_l2_n{ni}")
                for ni in [1, 10, 20]:
                    print(f"Random Dream Chunk sample n {ni}")
                    run_method(
                        DreamChunkConfig(
                            n_samples=ni,
                            temporal_constraint="currentonly",
                            latent_similarity_metric="l2norm",
                            random_switch=True,
                        ),
                        f"random_chunk_current_n{ni}",
                    )
   
                for ni in [1,10, 20, 30]:
                    print(f"Dream Chunk sample n {ni}")
                    run_method(DreamChunkConfig(n_samples=ni,temporal_constraint="currentonly",latent_similarity_metric="l2norm",prediction_type="t0"), f"DreamChunk_perstepprediction_l2_n{ni}")

                
                run_method(RealtimeMethodConfig(), "realtime")
                run_method(BIDMethodConfig(), "bid")
                # 
                # run_method(RealtimeMethodConfig(prefix_attention_schedule="zeros"), "hard_masking")
                run_method(AdaptiveChunkConfig(dicrepancy_measure="action"), "AdapChunk")
                run_method(NaiveMethodConfig(), "naive")

            if output_path is not None:
                output_path.mkdir(parents=True, exist_ok=True)
                df = results_frame(results)
                # df.to_csv(output_path / "results.csv", index=False)
                df.to_csv(
                    output_path / "results.csv",
                    mode="a",
                    header=not (output_path / "results.csv").exists(),
                    index=False,
                )
                sim_df = results_frame(sim_coverage_results)
                sim_df.to_csv(output_path / "sim_coverage.csv", index=False)


if __name__ == "__main__":
    tyro.cli(main)
