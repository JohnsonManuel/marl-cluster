#!/bin/bash
#SBATCH --job-name=marl-smoke
#SBATCH --partition=batch
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=12
#SBATCH --mem=24000M
#SBATCH --gres=gpu:1
#SBATCH --output=/projects/users/%u/marl-cluster/logs/%x-%j.out
#SBATCH --error=/projects/users/%u/marl-cluster/logs/%x-%j.err
set -euo pipefail

# ==============================================================================
# A short run that proves the whole chain works before any long job is queued:
# CUDA is visible, StarCraft II launches, SMAC loads a map, EPyMARL trains, and
# the shared exploration memory switches on.
#
# Expect this to finish in a few minutes. Do not submit the array job until the
# log shows the ENABLED line and a non-zero t_env.
# ==============================================================================

project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"

export SC2PATH="${project}/StarCraftII"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "${epymarl}"

srun apptainer exec --nv \
    --env SC2PATH="${SC2PATH}" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" -c \
    "import torch, smac, os; \
print('cuda_available =', torch.cuda.is_available()); \
print('gpu =', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'); \
print('torch =', torch.__version__); \
print('SC2PATH =', os.environ.get('SC2PATH')); \
print('smac ok')"

srun apptainer exec --nv \
    --env SC2PATH="${SC2PATH}" \
    --env OMP_NUM_THREADS=1 \
    --env MKL_NUM_THREADS=1 \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" "${epymarl}/src/main.py" \
        --config=mappo --env-config=sc2 \
        with env_args.map_name=3m \
             t_max=30000 \
             test_nepisode=8 test_interval=10000 \
             log_interval=10000 runner_log_interval=10000 \
             learner_log_interval=10000 \
             sm_active_selection=True \
             sm_count_mode=per_agent \
             sm_combine=minimum \
             sm_obs_slice=4:19 \
             sm_select_beta=0.1 \
             sm_hash_dim=16 \
             sm_beta_anneal_time=30000 \
             sm_bonus_scaling=relative \
             standardise_returns=True \
             seed=1

echo "[smoke] finished"
