#!/usr/bin/env bash
set -euo pipefail

levels=(
  "worlds/l/hard_lunar_lander.json"
  "worlds/l/mjc_half_cheetah.json"
  "worlds/l/mjc_swimmer.json"
  "worlds/l/mjc_walker.json"
  "worlds/l/h17_unicycle.json"
  "worlds/l/chain_lander.json"
  "worlds/l/catcher_v3.json"
  "worlds/l/trampoline.json"
  "worlds/l/car_launch.json"
  "worlds/l/grasp_easy.json"
  "worlds/l/catapult.json"
  "worlds/l/cartpole_thrust.json"
)

action_noises=(
  # "0.3"
  "0.2"
  "0.1"
)

# Iterate levels and run: train expert -> generate data -> train flow -> train wm -> eval
for action_noise in "${action_noises[@]}"; do
  noise_tag="${action_noise/./}"
  for level in "${levels[@]}"; do
    name="anoise${noise_tag}_$(basename "$level" .json)"
    flow_run="${name}"
    wm_run="rssm_${name}"
    echo "\n=== Level: $level | action_noise=$action_noise | run_name=$name ==="

    # Step 1: Train expert policies.
    echo "[1/5] Training expert..."
    uv run src/train_expert.py \
      --config.level-paths "$level" \
      --config.run_name "$name" \
      --config.action_noise "$action_noise" \
      --config.seed 77  \
      --config.num_seeds  8

    # Step 2: Generate trajectories from best expert checkpoints
    echo "[2/5] Generating data..."
    uv run src/generate_data.py \
      --config.run-path "./logs-expert/${name}" \
      --config.level-paths "$level" \
      --config.action_noise "$action_noise" \
      --config.num_steps 2_000_000

    # Step 3: Train imitation flow policy
    echo "[3/5] Training flow..."
    uv run src/train_flow.py \
      --config.run-path "./logs-expert/${name}" \
      --config.level-paths "$level" \
      --config.wandb_name "${flow_run}"

    # Step 4: Train latent dynamics rssm from the recorded trajectory
    echo "[4/5] Training RSSM world model..."
    uv run src/train_wm.py \
      --config.run-path "./logs-expert/${name}" \
      --config.level-paths "$level" \
      --config.wandb_name "${wm_run}"

    # # Step 5: Evaluate the final flow+wm policy
    echo "[5/5] Evaluating flow..."
    uv run src/eval_flow_wm.py \
      --flow-run-path "./logs-bc/${flow_run}" \
      --wm-run-path "./logs-wm/${wm_run}" \
      --output-dir "./logs-eval-expert-anoise${noise_tag}_testanoise03/${name}" \
      --level-paths "$level" \
      --config.action_noise 0.3
      # --config.action_noise "$action_noise" 

    echo "=== Completed pipeline for ${name} ===\n"
  done
done
