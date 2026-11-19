# marl-cluster

MAPPO runs for the shared exploration memory, packaged for the TikiLab cluster.

Three scenarios (`3m`, `5m_vs_6m`, `3s_vs_5z`) times three arms (baseline, all
features, chosen features), submitted as one Slurm array job.

---

## Run it: five commands

Everything below is run **on krull**, after `ssh YOUR_USERNAME@krull`.

### 1. Get the project onto krull

```bash
cd "/projects/users/$USER"
git clone https://github.com/JohnsonManuel/marl-cluster.git
cd marl-cluster
mkdir -p logs
```

### 2. One-time setup (no GPU, about 20-40 minutes)

Clones EPyMARL at the pinned commit, copies the mechanism over it, builds a
venv, clones and installs SMAC, downloads StarCraft II, and installs the SMAC
map pack.

Note: SMAC is cloned and installed **editable** rather than with
`pip install git+...`. The latter installs the Python modules but silently drops
the `.SC2Map` files, because they are data rather than modules and are not
declared as package data -- which leaves SMAC unable to load any map. The job
now verifies that at least 20 maps and all three scenarios used here are present
before it reports success.

```bash
setup_id=$(sbatch --parsable "/projects/users/$USER/marl-cluster/setup/01_setup_env.slurm")
echo "setup job $setup_id"
squeue --job "$setup_id"
```

Wait for it to leave `squeue`, then confirm it succeeded:

```bash
sacct --jobs "$setup_id" --format=JobID,JobName,State,ExitCode,Elapsed
tail -20 "/projects/users/$USER/marl-cluster/logs/marl-setup-$setup_id.out"
```

### 3. Smoke test (1 GPU, a few minutes)

**Do not skip this.** It proves CUDA, StarCraft II, SMAC, EPyMARL and the
mechanism all work together before 9 long jobs are queued.

```bash
smoke_id=$(sbatch --parsable "/projects/users/$USER/marl-cluster/setup/02_smoke_test.slurm")
echo "smoke job $smoke_id"
squeue --job "$smoke_id"
```

When it finishes, the log must contain `cuda_available = True` and a line
beginning `Shared exploration memory ENABLED`:

```bash
grep -E "cuda_available|Shared exploration memory ENABLED|t_env" \
    "/projects/users/$USER/marl-cluster/logs/marl-smoke-$smoke_id.out"
```

### 4. Submit the 9 runs

```bash
array_id=$(sbatch --parsable "/projects/users/$USER/marl-cluster/slurm/mappo_array.slurm")
echo "array job $array_id"
squeue --job "$array_id"
```

### 5. Check progress and collect results

```bash
squeue --me
sacct --jobs "$array_id" --format=JobID,JobName,State,ExitCode,Elapsed,AllocTRES
bash "/projects/users/$USER/marl-cluster/scripts/collect_results.sh"
```

---

## What gets run

| Array task | Arm | Map | Hashed features |
|---|---|---|---|
| 1 | baseline | `3m` | mechanism off |
| 2 | all features | `3m` | whole observation (30) |
| 3 | chosen features | `3m` | enemy block `4:19` (15) |
| 4 | baseline | `5m_vs_6m` | mechanism off |
| 5 | all features | `5m_vs_6m` | whole observation (55) |
| 6 | chosen features | `5m_vs_6m` | enemy block `4:34` (30) |
| 7 | baseline | `3s_vs_5z` | mechanism off |
| 8 | all features | `3s_vs_5z` | whole observation (48) |
| 9 | chosen features | `3s_vs_5z` | enemy block `4:34` (30) |

Every arm within a scenario is identical except for the mechanism: same
algorithm, network, schedule, budget, seed and evaluation protocol. The baseline
receives no `sm_*` flags at all.

Common settings: `sm_count_mode=per_agent`, `sm_combine=minimum`,
`sm_select_beta=0.1`, `sm_hash_dim=16`, `sm_bonus_scaling=relative`, annealing
over the full run, 32 evaluation episodes every 20,000 steps, dense reward,
seed 1. `standardise_returns=True` is applied to **every** arm, baseline
included.

## Corrections after the 2026-11-19 runs

The first full set (array 321) was invalid. Both faults are recorded here so the
configuration is not silently reverted.

**MAPPO diverged.** `standardise_returns` defaults to `False` in EPyMARL, so
value normalisation was off. The critic grew across seven orders of magnitude
(`critic_loss` 4.3 to 7.8e7, final `q_taken_mean` -45,574 where SMAC scales the
maximum return to about 20), and every arm including the untouched baselines
scored 0.0 on both hard scenarios. The `3m` runs never stabilised at any point,
oscillating between 0.00 and 0.91 for the whole 10M steps. Value normalisation
is one of the five implementation factors MAPPO's own paper identifies as
critical. Fixed by `standardise_returns=True` on every arm.

**The mechanism was inert.** With an absolute `beta`, the bonus decayed to about
0.004% of the value scale once counts reached the hundreds, and `beta=0.1` meant
a 9x different perturbation per scenario because `q_taken_mean` ranged from 0.87
to 8.05. `sm_redundancy_ratio` was identical across arms (0.7673 against
0.7714), so the diagnostic built to measure the thesis's own claim showed no
effect. Fixed by `sm_bonus_scaling=relative`, which measures the bonus against
the spread of whatever the action selector adds it to and rescales novelty
across the available actions. Measured after the change: the bonus is about 5%
of the value spread and grows with it, rather than vanishing.

## The training budget

The configuration uses **10,050,000 steps**, matching the MAPPO paper's "at most
10M training timesteps", at which it reports 100.0 on `3m`, 89.1 on `5m_vs_6m`
and 100.0 on `3s_vs_5z`.

EPyMARL's own protocol is stricter: it trains **on-policy** algorithms for
**40M** steps against **4M** for off-policy ones, stating that on-policy methods
need ten times the samples for a fair comparison. Earlier local runs gave MAPPO
only 1-2M, which is why they produced zeros on `5m_vs_6m`.

For the full 40M protocol, edit `configs/mappo_runs.txt` (change `10050000` to
`40050000`) and raise `--time` in `slurm/mappo_array.slurm` above `24:00:00`.

## Changing what runs

Add or remove lines in `configs/mappo_runs.txt`, then set `--array` in
`slurm/mappo_array.slurm` to match the number of non-comment lines. Adding a
second seed means duplicating the nine lines with a different final column and
setting `--array=1-18`.

## Layout

```
configs/mappo_runs.txt        one line per run
slurm/mappo_array.slurm       the array job
setup/01_setup_env.slurm      one-time: EPyMARL + mechanism + SMAC + SC2
setup/02_smoke_test.slurm     short GPU run that verifies the whole chain
scripts/collect_results.sh    summarise finished runs
mechanism/                    our modified EPyMARL files (copied over upstream)
```

`epymarl/`, `smac/`, `.venv-cluster/`, `StarCraftII/`, `logs/` and `results/` are
created on the cluster and are deliberately not tracked.

## The mechanism

Seven files on top of upstream EPyMARL at commit
`cbc38c09588064eab978501d0f12c2cf58fa7fc2`:

| File | Role |
|---|---|
| `src/components/shared_memory.py` | **new** - the memory, SimHash, per-agent tables, redundancy |
| `src/controllers/basic_controller.py` | reads the counts at action selection, writes the chosen action, applies the anneal |
| `src/components/action_selectors.py` | applies the bonus to Q-values (QMIX) or logits (MAPPO) |
| `src/config/default.yaml` | the `sm_*` flags, all defaulting to off |
| `src/run.py` | creates the memory and injects it into the runner and controller |
| `src/runners/episode_runner.py` | diagnostics on the QMIX path |
| `src/runners/parallel_runner.py` | diagnostics on the MAPPO path |

## Known risk

The managed container `/datasets/containers/ml-workspace.sif` provides torch and
CUDA but **not** StarCraft II, so the setup job installs the game under
`/projects/users/$USER/marl-cluster/StarCraftII` and installs SMAC into a venv
that inherits the container's torch.

If the setup or smoke job fails on a missing system library that StarCraft II
needs, that cannot be fixed from inside the project and the cluster
administrators should be asked for a project-specific Apptainer image. The smoke
test exists to surface this in minutes rather than after a day of queued jobs.
