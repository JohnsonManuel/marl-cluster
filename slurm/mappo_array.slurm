#!/bin/bash
#SBATCH --job-name=mappo-marl
#SBATCH --partition=batch
#SBATCH --array=1-9
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=32000M
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/users/%u/marl-cluster/logs/%x-%A_%a.out
#SBATCH --error=/projects/users/%u/marl-cluster/logs/%x-%A_%a.err
set -euo pipefail

# ==============================================================================
# MAPPO runs: 3 scenarios x 3 arms, one array task each.
#
# Each task reads its own line from configs/mappo_runs.txt, so all nine
# configurations are described in one place and only the array range needs to
# change when runs are added or removed.
#
#   --array=1-9      one task per configuration line
#   --cpus-per-task  MAPPO collects from 10 environments in parallel
#                    (batch_size_run=10), so it needs cores
#   --mem            each StarCraft II instance is roughly 600 MB; the cluster
#                    guide caps requests at 60000M (which is NOT 60G)
#   --time           24h fits 10M steps with headroom. Raise it for 40M.
#
# Submit only after 01_setup_env.slurm and 02_smoke_test.slurm have both
# succeeded.
# ==============================================================================

project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"
config="${project}/configs/mappo_runs.txt"

export SC2PATH="${project}/StarCraftII"

# One SC2 instance per core; without this, each of the 10 workers spawns its own
# thread pool and they fight each other for the same cores.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# --- pick this task's configuration line --------------------------------------
line="$(grep -vE '^\s*(#|$)' "${config}" | sed -n "${SLURM_ARRAY_TASK_ID}p")"
if [ -z "${line}" ]; then
    echo "[error] no configuration for array index ${SLURM_ARRAY_TASK_ID} in ${config}" >&2
    exit 1
fi
read -r ARM MAP OBS_SLICE STEPS SEED <<< "${line}"

echo "[run] array task ${SLURM_ARRAY_TASK_ID}"
echo "[run] arm=${ARM} map=${MAP} slice=${OBS_SLICE} steps=${STEPS} seed=${SEED}"
echo "[run] host=$(hostname)"

# --- build the mechanism flags ------------------------------------------------
# The baseline gets no sm_* flags at all, so it is the unmodified algorithm.
if [ "${ARM}" = "baseline" ]; then
    MECH_ARGS=()
else
    MECH_ARGS=(
        sm_active_selection=True
        sm_count_mode=per_agent
        sm_combine=minimum
        "sm_obs_slice=${OBS_SLICE}"
        sm_select_beta=0.1
        sm_hash_dim=16
        "sm_beta_anneal_time=${STEPS}"
    )
fi

results="${project}/results/${MAP}_${ARM}_s${SEED}"
mkdir -p "${results}"
cd "${epymarl}"

srun apptainer exec --nv \
    --env SC2PATH="${SC2PATH}" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" "${epymarl}/src/main.py" \
        --config=mappo --env-config=sc2 \
        with "env_args.map_name=${MAP}" \
             "t_max=${STEPS}" \
             test_nepisode=32 \
             test_interval=20000 \
             save_model=True \
             save_model_interval=1000000 \
             "local_results_path=${results}" \
             "seed=${SEED}" \
             "${MECH_ARGS[@]}"

echo "[run] finished arm=${ARM} map=${MAP} seed=${SEED}"
