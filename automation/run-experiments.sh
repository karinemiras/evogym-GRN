#!/usr/bin/env bash
# Run from repo root:
#   ./automation/run-experiments.sh path/to/PARAMS.sh
set -euo pipefail

params_file=${1:-automation/setups/locomotion.sh}
source "$params_file"

# Defaults
: "${evogym_steps:=500}"
: "${evogym_num_workers:=0}"
: "${evogym_headless:=1}"
: "${evogym_render_mode:=screen}"
: "${RUN_ANALYSIS:=1}"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${out_path}/${study_name}" "${out_path}/${study_name}/analysis"

IFS=',' read -r -a EXP_LIST <<< "${experiments}"
IFS=',' read -r -a COND_LIST <<< "${env_conditions}"
IFS=',' read -r -a RUN_LIST <<< "${runs}"

if [[ ${#EXP_LIST[@]} -ne ${#COND_LIST[@]} ]]; then
  echo "Error: experiments and env_conditions must have same length."
  exit 1
fi

echo ">> Parallelism policy: single run at a time; intra-run CPU parallelism via --evogym_num_workers"

for idx in "${!EXP_LIST[@]}"; do
  exp="${EXP_LIST[$idx]}"
  cond="${COND_LIST[$idx]}"

  for run in "${RUN_LIST[@]}"; do
    logfile="${out_path}/${study_name}/${exp}_${run}.log"
    echo ">> Running experiment=${exp} run=${run}  (log: ${logfile})"

    cmd=(
      python3 -u "${REPO_ROOT}/experimental_setups/${algorithm}.py"
      --out_path "${out_path}"
      --experiment_name "${exp}"
      --env_conditions "${cond}"
      --run "${run}"
      --study_name "${study_name}"
      --algorithm "${algorithm}"
      --fitness_metric "${fitness_metric}"
      --num_generations "${num_generations}"
      --population_size "${population_size}"
      --offspring_size "${offspring_size}"
      --evogym_steps "${evogym_steps}"
      --evogym_num_workers "${evogym_num_workers}"
      --evogym_headless "${evogym_headless}"
      --evogym_render_mode "${evogym_render_mode}"
      --plastic "${plastic}"
      --crossover_prob "${crossover_prob}"
      --mutation_prob "${mutation_prob}"
      --max_voxels "${max_voxels}"
      --cube_face_size "${cube_face_size}"
      --run_simulation "${run_simulation}"
    )

    if [[ -n "${evogym_freeze_first_frame_seconds+x}" ]]; then
      cmd+=(--evogym_freeze_first_frame_seconds "${evogym_freeze_first_frame_seconds}")
    fi
    if [[ -n "${evogym_add_walls+x}" ]]; then
      cmd+=(--evogym_add_walls "${evogym_add_walls}")
    fi
    if [[ -n "${evogym_add_ceiling+x}" ]]; then
      cmd+=(--evogym_add_ceiling "${evogym_add_ceiling}")
    fi
    if [[ -n "${evogym_env_width+x}" ]]; then
      cmd+=(--evogym_env_width "${evogym_env_width}")
    fi
    if [[ -n "${evogym_env_height+x}" ]]; then
      cmd+=(--evogym_env_height "${evogym_env_height}")
    fi
    if [[ -n "${ppo_timesteps+x}" ]]; then
      cmd+=(--ppo_timesteps "${ppo_timesteps}")
    fi
    if [[ -n "${ppo_n_steps+x}" ]]; then
      cmd+=(--ppo_n_steps "${ppo_n_steps}")
    fi
    if [[ -n "${ppo_batch_size+x}" ]]; then
      cmd+=(--ppo_batch_size "${ppo_batch_size}")
    fi

    mkdir -p "${out_path}/${study_name}"
    "${cmd[@]}" >>"${logfile}" 2>&1
  done
done

if [[ "${RUN_ANALYSIS}" -eq 1 ]]; then
  echo ">> All runs finished. Starting analysis..."
  "${REPO_ROOT}/automation/run-analysis.sh" "$params_file"
fi
