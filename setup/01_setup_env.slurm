#!/bin/bash
#SBATCH --job-name=marl-setup
#SBATCH --partition=batch
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --output=/projects/users/%u/marl-cluster/logs/%x-%j.out
#SBATCH --error=/projects/users/%u/marl-cluster/logs/%x-%j.err
set -euo pipefail

# ==============================================================================
# One-time setup. Run this ONCE before submitting any training job.
#
# It does four things:
#   1. clones upstream EPyMARL at the exact commit this work was built on
#   2. copies our mechanism files over it
#   3. creates a venv that inherits the container's torch/CUDA
#   4. installs SMAC and downloads the StarCraft II headless build
#
# Everything durable lives under /projects/users/$USER, never /scratch.
# No GPU is requested here: this job only downloads and installs.
# ==============================================================================

project="/projects/users/${USER}/marl-cluster"
image="/datasets/containers/ml-workspace.sif"
epymarl="${project}/epymarl"
venv="${project}/.venv-cluster"
sc2dir="${project}/StarCraftII"

# The commit of uoe-agents/epymarl that the mechanism was developed against.
EPYMARL_COMMIT="cbc38c09588064eab978501d0f12c2cf58fa7fc2"

mkdir -p "${project}/logs"
cd "${project}"

# --- 1. upstream EPyMARL, pinned -----------------------------------------------
if [ ! -d "${epymarl}/.git" ]; then
    echo "[setup] cloning upstream EPyMARL"
    git clone https://github.com/uoe-agents/epymarl.git "${epymarl}"
fi
cd "${epymarl}"
git fetch --all --quiet || true
git checkout --quiet "${EPYMARL_COMMIT}"
echo "[setup] EPyMARL pinned at $(git rev-parse --short HEAD)"

# --- 2. our mechanism on top ---------------------------------------------------
echo "[setup] copying mechanism files"
cp -v "${project}/mechanism/src/components/action_selectors.py"  "${epymarl}/src/components/"
cp -v "${project}/mechanism/src/components/shared_memory.py"     "${epymarl}/src/components/"
cp -v "${project}/mechanism/src/config/default.yaml"             "${epymarl}/src/config/"
cp -v "${project}/mechanism/src/controllers/basic_controller.py" "${epymarl}/src/controllers/"
cp -v "${project}/mechanism/src/run.py"                          "${epymarl}/src/"
cp -v "${project}/mechanism/src/runners/episode_runner.py"       "${epymarl}/src/runners/"
cp -v "${project}/mechanism/src/runners/parallel_runner.py"      "${epymarl}/src/runners/"

# --- 3. venv inheriting the container's torch ----------------------------------
# --system-site-packages means the container's torch/CUDA build is reused rather
# than a second copy being installed into the venv.
if [ ! -d "${venv}" ]; then
    echo "[setup] creating venv"
    apptainer exec \
        --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
        "${image}" \
        /usr/bin/python3 -m venv --system-site-packages "${venv}"
fi

echo "[setup] installing python dependencies"
apptainer exec \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" -m pip install --upgrade pip

apptainer exec \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" -m pip install \
        "sacred" "pyyaml" "pygame" "matplotlib" "scipy" "tensorboard-logger"

# --- 3b. SMAC from a clone, not straight from git ------------------------------
# "pip install git+...smac" installs the Python modules but silently drops the
# .SC2Map files: they are data, not modules, and SMAC does not declare them as
# package data. Cloning and installing editable keeps the map pack on disk where
# it can be copied into SC2PATH below.
smac_src="${project}/smac"
if [ ! -d "${smac_src}/.git" ]; then
    echo "[setup] cloning SMAC"
    git clone https://github.com/oxwhirl/smac.git "${smac_src}"
fi

apptainer exec \
    --bind /projects:/projects,/datasets:/datasets,/scratch:/scratch \
    "${image}" \
    "${venv}/bin/python" -m pip install -e "${smac_src}"

# --- 4. StarCraft II headless ---------------------------------------------------
# SMAC needs the game binary and the SMAC map pack. The managed ML container does
# not ship StarCraft II, so it is installed into the project directory here.
if [ ! -d "${sc2dir}" ]; then
    echo "[setup] downloading StarCraft II 4.10 (about 3.6 GB)"
    cd "${project}"
    wget -q --show-progress -O SC2.4.10.zip \
        "http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip"
    unzip -q -P iagreetotheeula SC2.4.10.zip -d "${project}"
    rm -f SC2.4.10.zip
fi

mkdir -p "${sc2dir}/Maps"
smac_maps="${smac_src}/smac/env/starcraft2/maps/SMAC_Maps"
if [ ! -d "${smac_maps}" ]; then
    echo "[error] SMAC map pack missing at ${smac_maps}" >&2
    exit 1
fi
cp -r "${smac_maps}" "${sc2dir}/Maps/"

n_maps="$(ls -1 "${sc2dir}/Maps/SMAC_Maps" | wc -l)"
echo "[setup] installed ${n_maps} SMAC maps"
if [ "${n_maps}" -lt 20 ]; then
    echo "[error] expected at least 20 maps, found ${n_maps}" >&2
    exit 1
fi
for m in 3m 5m_vs_6m 3s_vs_5z; do
    if [ ! -f "${sc2dir}/Maps/SMAC_Maps/${m}.SC2Map" ]; then
        echo "[error] required map ${m}.SC2Map not installed" >&2
        exit 1
    fi
done
echo "[setup] required maps present: 3m, 5m_vs_6m, 3s_vs_5z"

echo "[setup] done"
echo "[setup] SC2PATH will be ${sc2dir}"
echo "[setup] next: sbatch ${project}/setup/02_smoke_test.slurm"
