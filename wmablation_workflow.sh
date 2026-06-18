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

action_noise="0.2"
noise_tag="${action_noise/./}"

wm_types=("rssm" "lewm" "ebjepa")
prediction_loss_rolloutsteps=(1)

for rolloutstep in "${prediction_loss_rolloutsteps[@]}"; do

  for level in "${levels[@]}"; do
    name="anoise${noise_tag}_$(basename "$level" .json)"
    flow_run="${name}"
    expert_run="${name}"

    echo
    echo "=== Level: $level | action_noise=$action_noise | run_name=$name ==="

    if [[ ! -d "./logs-expert/${expert_run}" ]]; then
      echo "Missing expert data run: ./logs-expert/${expert_run}"
      exit 1
    fi

    if [[ ! -d "./logs-bc/${flow_run}" ]]; then
      echo "Missing BC policy run: ./logs-bc/${flow_run}"
      exit 1
    fi

    echo "[2/5] Generating data..."
    uv run src/generate_data.py \
      --config.run-path "./logs-expert/${expert_run}" \
      --config.level-paths "$level" \
      --config.action_noise "${action_noise}" \
      --config.num_steps 2000000

    for wm_type in "${wm_types[@]}"; do
      
      rollout_tag="predroll${rolloutstep}"

      if [[ "${wm_type}" == "rssm" ]]; then
        train_script="src/train_wm.py"
        wm_run="rssm_${rollout_tag}_${name}"
        wm_log_dir="./logs-wm/${wm_run}"
        eval_root="./logs-eval-rssm-${rollout_tag}-anoise${noise_tag}"
      elif [[ "${wm_type}" == "lewm" ]]; then
        train_script="src/train_lewm.py"
        wm_run="lewm_${rollout_tag}_${name}"
        wm_log_dir="./logs-lewm/${wm_run}"
        eval_root="./logs-eval-lewm-${rollout_tag}-anoise${noise_tag}"
      elif [[ "${wm_type}" == "ebjepa" ]]; then
        train_script="src/train_ebjepa.py"
        wm_run="ebjepa_${rollout_tag}_${name}"
        wm_log_dir="./logs-ebjepa/${wm_run}"
        eval_root="./logs-eval-ebjepa-${rollout_tag}-anoise${noise_tag}"
      else
        echo "Unsupported wm_type: ${wm_type}"
        exit 1
      fi

      echo "[4/5] Training ${wm_type} world model with prediction_loss_rolloutstep=${rolloutstep}..."
      train_args=(
        --config.run-path "./logs-expert/${expert_run}"
        --config.level-paths "$level"
        --config.wandb-name "${wm_run}"
        --config.prediction-loss-rolloutstep "${rolloutstep}"
      )

      uv run "${train_script}" "${train_args[@]}"

      echo "[5/5] Evaluating flow with ${wm_type} and prediction_loss_rolloutstep=${rolloutstep}..."
      uv run src/eval_flow_wm.py \
        --flow-run-path "./logs-bc/${flow_run}" \
        --wm-run-path "${wm_log_dir}" \
        --wm-type "${wm_type}" \
        --output-dir "${eval_root}/${name}" \
        --level-paths "$level" \
        --config.action_noise "${action_noise}"
      
    done

    echo "=== Completed WM tests for ${name} ==="
  done
done
