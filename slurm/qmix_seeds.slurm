#!/bin/bash
#SBATCH --job-name=qmix-seeds
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=60000M
#SBATCH --time=4-00:00:00
#SBATCH --output=/projects/users/%u/marl-cluster/logs/%x-%j.out
#SBATCH --error=/projects/users/%u/marl-cluster/logs/%x-%j.err
set -uo pipefail

# ==============================================================================
# QMIX, five seeds on the two discriminating scenarios: 30 runs total.
#
#   2 maps (5m_vs_6m, 3s_vs_5z) x 3 arms x 5 seeds
#
# WHY ONE JOB SPANNING BOTH NODES
#
# The QOS allows one job per user (MaxJobsPU=1), so 30 separate jobs would run
# strictly sequentially -- over ten days. Instead this is a single job holding
# both nodes, with 15 concurrent runs per node. Each node has 24 CPUs and 62 GB,
# so that is ~1.6 cores and ~2.5 GB per run, matching what ran comfortably
# locally (9 runs on 16 cores, 23 GB peak).
#
# Each of the 2 tasks (one per node) launches its own half of the run list in
# the background and waits, which is the same pattern as run_qmix_local.sh.
#
# NO GPU IS REQUESTED. QMIX measured 213 steps/s on CPU against 161 on GPU: the
# networks are small, every step must return to the CPU for the StarCraft engine
# anyway, and avoiding --gres leaves the GPUs free and schedules sooner.
#
# --time=4-00:00:00 against a 5-day partition limit. Locally the 10.05M runs took
# 24-27 h at 1.78 cores each; here it is 1.6 cores each, so slightly slower. A
# TIMEOUT would kill all 30 runs at once, and asking for more wall time costs
# nothing, so the request is generous rather than tight.
# ==============================================================================

project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"
config="${project}/configs/qmix_seeds.txt"
logs="${project}/logs/qmix_seeds_${SLURM_JOB_ID}"

export SC2PATH="${project}/StarCraftII"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

mkdir -p "${logs}"

PER_NODE=15

# --- the per-node worker -------------------------------------------------------
# Written to a file so srun can run it on each node.
cat > "${logs}/worker.sh" <<'WORKER'
#!/bin/bash
set -uo pipefail
project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"
config="${project}/configs/qmix_seeds.txt"
logs="$1"
per_node="$2"

export SC2PATH="${project}/StarCraftII"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

offset=$(( SLURM_PROCID * per_node ))
echo "[worker] node=$(hostname) procid=${SLURM_PROCID} handling runs $((offset+1))..$((offset+per_node))"

cd "${epymarl}" || exit 1

for i in $(seq 1 "${per_node}"); do
    idx=$(( offset + i ))
    line="$(grep -vE '^\s*(#|$)' "${config}" | sed -n "${idx}p")"
    [ -z "${line}" ] && continue
    read -r ARM MAP OBS_SLICE STEPS SEED TEST_INTERVAL <<< "${line}"
    tag="${MAP}_${ARM}_s${SEED}"

    MECH=()
    if [ "${ARM}" != "baseline" ]; then
        MECH=(sm_active_selection=True sm_count_mode=per_agent sm_combine=minimum
              "sm_obs_slice=${OBS_SLICE}" sm_select_beta=0.1 sm_hash_dim=16
              "sm_beta_anneal_time=${STEPS}" sm_bonus_scaling=relative)
    fi

    echo "[worker] launching ${tag}"
    apptainer exec \
        --env SC2PATH="${SC2PATH}" \
        --env OMP_NUM_THREADS=1 \
        --env MKL_NUM_THREADS=1 \
        --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
        "${image}" \
        "${venv}/bin/python" "${epymarl}/src/main.py" \
            --config=qmix --env-config=sc2 \
            with "env_args.map_name=${MAP}" \
                 "t_max=${STEPS}" \
                 test_nepisode=32 \
                 "test_interval=${TEST_INTERVAL}" \
                 use_cuda=False \
                 use_rnn=True \
                 save_model=True \
                 save_model_interval=2000000 \
                 "local_results_path=${project}/results/qmix_seeds/${tag}" \
                 "seed=${SEED}" \
                 "${MECH[@]}" > "${logs}/${tag}.log" 2>&1 &
    sleep 10   # stagger SC2 launches so they do not race for ports
done

wait
echo "[worker] node=$(hostname) finished its ${per_node} runs"
WORKER
chmod +x "${logs}/worker.sh"

echo "[run] 30 QMIX runs, ${PER_NODE} per node across ${SLURM_NNODES} nodes"
echo "[run] logs: ${logs}"
grep -vE '^\s*(#|$)' "${config}" | nl | sed 's/^/[run]   /'

srun --ntasks=2 --ntasks-per-node=1 "${logs}/worker.sh" "${logs}" "${PER_NODE}"

echo "[run] all QMIX seed runs finished"
