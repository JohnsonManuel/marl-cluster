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
# Two corrections applied after the 2026-11-19 runs, which were invalid:
#
#   standardise_returns=True   Value normalisation, on EVERY arm including the
#                              baseline. EPyMARL ships it disabled; MAPPO's own
#                              paper names it one of five critical factors.
#                              Without it the critic diverged across seven orders
#                              of magnitude (critic_loss 4.3 -> 7.8e7) and every
#                              arm, baselines included, produced 0.0 on both hard
#                              scenarios.
#
#   sm_bonus_scaling=relative  Mechanism arms only. The absolute bonus decayed to
#                              ~0.004% of the value scale, so the mechanism was
#                              numerically inert. The bonus is now a fraction of
#                              the spread of the quantity the selector adds it to.
#
# Submit only after 01_setup_env.slurm and 02_smoke_test.slurm have both
# succeeded.
# ==============================================================================

project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"
# RUNS_CONFIG selects the budget set; pass it with
#   sbatch --export=ALL,RUNS_CONFIG=configs/mappo_runs_40m.txt ...
config="${project}/${RUNS_CONFIG:-configs/mappo_runs.txt}"

export SC2PATH="${project}/StarCraftII"

# --- optional Trackio monitoring ----------------------------------------------
# krull sits on a private university IP and cannot be reached from home, so there
# is no way to PULL logs remotely. Trackio pushes metrics to a Hugging Face Space
# instead, readable from anywhere including a phone. It is wandb-API-compatible
# but needs an HF token, not a wandb account.
#
# Credentials live in .hf_env, which is NOT tracked (this repo is public).
# Create it once on the cluster:
#
#   cat > /projects/users/$USER/marl-cluster/.hf_env <<'EOF'
#   export HF_TOKEN=<write-scoped token from huggingface.co/settings/tokens>
#   export TRACKIO_SPACE_ID=<your-hf-username>/marl-shared-memory
#   export TRACKIO_PROJECT=marl-shared-memory
#   EOF
#   chmod 600 /projects/users/$USER/marl-cluster/.hf_env
#
# The Space is created automatically on the first run. Without the file the runs
# proceed exactly as before, just unmonitored.
TRACKIO_ARGS=()
if [ -f "${project}/.hf_env" ]; then
    # shellcheck disable=SC1090
    . "${project}/.hf_env"
    if [ -n "${HF_TOKEN:-}" ]; then
        export HF_TOKEN
        TRACKIO_ARGS=(
            use_trackio=True
            "trackio_project=${TRACKIO_PROJECT:-marl-shared-memory}"
        )
        if [ -n "${TRACKIO_SPACE_ID:-}" ]; then
            TRACKIO_ARGS+=("trackio_space_id=${TRACKIO_SPACE_ID}")
            echo "[run] trackio -> ${TRACKIO_SPACE_ID}"
        else
            echo "[run] trackio enabled, local only (no TRACKIO_SPACE_ID)"
        fi
    else
        echo "[run] .hf_env present but HF_TOKEN empty; trackio disabled" >&2
    fi
else
    echo "[run] no .hf_env; trackio disabled"
fi

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
read -r ARM MAP OBS_SLICE STEPS SEED TEST_INTERVAL EPOCHS CLIP <<< "${line}"
: "${TEST_INTERVAL:=20000}"
: "${EPOCHS:=15}"
: "${CLIP:=0.2}"

echo "[run] array task ${SLURM_ARRAY_TASK_ID}"
echo "[run] arm=${ARM} map=${MAP} slice=${OBS_SLICE} steps=${STEPS} seed=${SEED} test_interval=${TEST_INTERVAL} epochs=${EPOCHS} clip=${CLIP}"
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
        sm_bonus_scaling=relative
        "sm_beta_anneal_time=${STEPS}"
    )
fi

results="${project}/results/${MAP}_${ARM}_s${SEED}_${STEPS}"
mkdir -p "${results}"
cd "${epymarl}"

srun apptainer exec --nv \
    --env SC2PATH="${SC2PATH}" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" "${epymarl}/src/main.py" \
        --config=mappo --env-config=sc2 \
        with "env_args.map_name=${MAP}" \
             "t_max=${STEPS}" \
             test_nepisode=32 \
             "test_interval=${TEST_INTERVAL}" \
             standardise_returns=True \
             lr=0.0005 \
             "epochs=${EPOCHS}" \
             "eps_clip=${CLIP}" \
             save_model=True \
             save_model_interval=1000000 \
             "local_results_path=${results}" \
             "seed=${SEED}" \
             "${MECH_ARGS[@]}" \
             "${TRACKIO_ARGS[@]}"

echo "[run] finished arm=${ARM} map=${MAP} seed=${SEED}"
