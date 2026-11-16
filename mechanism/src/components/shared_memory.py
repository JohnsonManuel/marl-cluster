"""Centralized shared exploration memory for collaborative active exploration.

This is the core mechanism of the thesis: a team-wide memory of what the agents
have already explored, shared across all agents and all parallel workers.

The memory supports TWO ways of using the shared knowledge, matching the task
description ("agents ... maintain a shared exploration memory and actively use this
shared information to coordinate future exploration and reduce redundant
state-action sampling"):

  1. PASSIVE (reward bonus) -- ``visit(state)`` counts state visits and returns an
     intrinsic reward bonus (coef / sqrt(count)) that the runner adds to the reward.
     The exploration policy itself does NOT depend on the memory here; the influence
     is only indirect, through the learned value/policy. Toggled by ``use_shared_memory``.

  2. ACTIVE (directed action selection) -- ``action_bonuses(obs, n_actions, agent_id)``
     returns a per-action novelty bonus, and ``record_action(obs, action, agent_id)``
     records what was taken. The action selector adds this bonus to the Q-values /
     policy logits at DECISION time, so the exploration POLICY depends directly on the
     collectively accumulated counts (UCB / count-based directed exploration). Because
     the table is shared, once the team has sampled a (state, action) many times every
     agent is steered away from it -> this directly "reduces redundant state-action
     sampling". Toggled by ``sm_active_selection``.

HOW THE ACTIVE COUNTS ARE ORGANISED (``count_mode``)
----------------------------------------------------
``team`` (default) keeps ONE table for the whole team, so the novelty signal is
"has *anyone* done this?". It cannot express *who* did what, so it can only make
agents avoid the team's history -- it cannot divide labour.

``per_agent`` keeps one table PER AGENT, all of them readable by every agent. Agent
i's bonus is then a function of how novel its candidate (obs, action) is according to
*every* agent's own table, following the multi-agent intrinsic-reward family of
Iqbal & Sha (2019), "Coordinated Exploration via Intrinsic Rewards for MARL". With
f_j = beta / sqrt(N_j(obs, a) + 1) the novelty according to agent j, ``combine`` selects:

  independent     g_i = f_i                      -- own history only (ablation: no sharing)
  minimum         g_i = min_j f_j                -- bonus only where NO agent has been;
                                                    the literal "no redundant sampling" rule
  covering        g_i = f_i if f_i >  mean_j f_j -- explore what is novel to me but known to
                                                    others -> agents spread out
  burrowing       g_i = f_i if f_i <= mean_j f_j -- push deeper into my own region
                                                    -> depth-first specialisation
  leader_follower agent 0 burrows, the rest cover

Iqbal & Sha report that which operator wins is TASK-DEPENDENT (burrowing won most of
their gridworld tasks, minimum won a VizDoom task), so ``combine`` is a config knob to
be swept rather than a fixed design choice. Their two stated desiderata are respected:
coordinate-wise monotonicity (a bonus never grows because some agent knows a state
better) and inner-directedness (zero own-novelty -> zero bonus).

Both modes hash continuous vectors with SimHash (random-hyperplane locality-sensitive
hashing, Tang et al. 2017 "#Exploration"): a vector is projected through a fixed random
matrix and reduced to the sign pattern of the k projections, so similar vectors share a
code. ``hash_dim`` (k) is the granularity knob -- smaller k -> coarser buckets -> more
collisions -> counts accumulate and the novelty signal is meaningful. (A plain floor
discretization is kept as a fallback for ``hash_dim <= 0``; it does not scale to large
state spaces -- see the thesis diagnostic.)
"""
import numpy as np


COMBINE_MODES = (
    "independent",
    "minimum",
    "covering",
    "burrowing",
    "leader_follower",
)


class SharedExplorationMemory:
    """Team-wide count tables for passive (state) and active (state-action) novelty."""

    def __init__(self, bonus_coef=0.05, discretization=0.25, hash_dim=16,
                 hash_seed=0, select_beta=0.5, n_agents=1, count_mode="team",
                 combine="minimum", obs_slice="all"):
        # Scale of the PASSIVE intrinsic reward bonus.
        self.bonus_coef = float(bonus_coef)
        # Scale of the ACTIVE action-selection bonus (added to Q-values / logits).
        self.select_beta = float(select_beta)
        self.hash_dim = int(hash_dim)
        self.hash_seed = int(hash_seed)
        self.discretization = float(discretization)
        self.n_agents = int(n_agents)
        self.count_mode = str(count_mode)
        self.combine = str(combine)
        if self.count_mode not in ("team", "per_agent"):
            raise ValueError(
                f"sm_count_mode must be 'team' or 'per_agent', got {self.count_mode!r}"
            )
        if self.combine not in COMBINE_MODES:
            raise ValueError(
                f"sm_combine must be one of {COMBINE_MODES}, got {self.combine!r}"
            )
        # Restricted-space projection (Liu et al. 2021, CMAE): count over a chosen few
        # observation features rather than the whole vector. Their ablation drops 47.7%
        # -> 9.4% without it, making the CHOICE of features more consequential than the
        # hash granularity. ``obs_slice`` is "all" or an index spec like "4:19,29:30".
        self.obs_slice = str(obs_slice)
        self._slice_idx = self._parse_slice(self.obs_slice)
        # Fixed random projections, cached per input dimension so that state vectors
        # (passive) and per-agent observation vectors (active) -- which have different
        # lengths -- can both be hashed by the same object.
        self._projs = {}
        # Passive: state-bucket -> visit count.
        self.counts = {}
        # Active, count_mode="team": obs-bucket -> np.array of per-action counts.
        self.action_counts = {}
        # Active, count_mode="per_agent": one such table per agent, all readable by all.
        self.agent_action_counts = [{} for _ in range(self.n_agents)]
        # obs-bucket -> bitmask of which agents have sampled it. Used only to measure
        # redundancy; see ``stats()["redundancy_ratio"]``.
        self._owner_mask = {}
        self.shared_buckets = 0
        # Running totals, for logging/inspection.
        self.total_visits = 0
        self.bonus_sum = 0.0
        self.total_action_records = 0

    @property
    def per_agent(self):
        return self.count_mode == "per_agent"

    def _projection(self, dim):
        """Return (building if needed) the fixed SimHash projection for ``dim`` inputs."""
        proj = self._projs.get(dim)
        if proj is None:
            # Seeded per-dimension so hashes are stable and reproducible within a run.
            rng = np.random.RandomState(self.hash_seed + dim)
            proj = rng.randn(self.hash_dim, dim).astype(np.float32)
            self._projs[dim] = proj
        return proj

    @staticmethod
    def _parse_slice(spec):
        """Parse "all" / "4:19,29:30" / "0,3,7" into an index array (None = whole vector).

        Sacred coerces a bare comma list on the command line (sm_obs_slice=8,13,18)
        into a Python list, so accept list/tuple input as well as the string forms,
        and tolerate the bracket characters that str() of a list produces.
        """
        if isinstance(spec, (list, tuple)):
            idx = [int(x) for x in spec]
            return np.asarray(idx, dtype=np.int64) if idx else None
        spec = str(spec if spec is not None else "all").strip()
        spec = spec.strip("[]()").strip()
        if spec in ("", "all", "none"):
            return None
        idx = []
        for part in spec.split(","):
            part = part.strip()
            if not part:
                continue
            if ":" in part:
                a, b = part.split(":")
                idx.extend(range(int(a), int(b)))
            else:
                idx.append(int(part))
        if not idx:
            return None
        return np.asarray(idx, dtype=np.int64)

    def _key(self, vec, project=False):
        """Map a continuous vector to a hashable bucket key (SimHash, or floor fallback).

        ``project=True`` first restricts the vector to ``obs_slice`` -- used for the
        active (observation) path only; the passive path hashes the full global state.
        """
        v = np.asarray(vec, dtype=np.float32).ravel()
        if project and self._slice_idx is not None:
            if self._slice_idx.max() >= v.shape[0]:
                raise ValueError(
                    f"sm_obs_slice={self.obs_slice!r} indexes feature "
                    f"{int(self._slice_idx.max())} but the observation has only "
                    f"{v.shape[0]} features."
                )
            v = v[self._slice_idx]
        if self.hash_dim > 0:
            code = (self._projection(v.shape[0]) @ v) > 0.0
            return np.packbits(code).tobytes()
        bucket = np.floor(v / self.discretization).astype(np.int64)
        return bucket.tobytes()

    # ------------------------------------------------------------------ passive
    def visit(self, state):
        """PASSIVE: record one team visit to ``state``, return its reward bonus.

        Bonus uses the count *after* incrementing, so a bucket's first visit gets the
        largest bonus (coef / sqrt(1) = coef) and repeats decay it toward zero.
        """
        key = self._key(state)
        n = self.counts.get(key, 0) + 1
        self.counts[key] = n
        self.total_visits += 1
        bonus = self.bonus_coef / np.sqrt(n)
        self.bonus_sum += bonus
        return bonus

    def novelty(self, state):
        """PASSIVE: read-only reward bonus for ``state`` without recording a visit."""
        key = self._key(state)
        n = self.counts.get(key, 0)
        return self.bonus_coef / np.sqrt(n + 1)

    # ------------------------------------------------------------------- tables
    @staticmethod
    def _read_counts(table, key, n_actions):
        """Per-action counts for ``key`` in ``table``, zero-padded to ``n_actions``."""
        counts = np.zeros(n_actions, dtype=np.float64)
        stored = table.get(key)
        if stored is not None:
            k = min(n_actions, stored.shape[0])
            counts[:k] = stored[:k]
        return counts

    @staticmethod
    def _bump(table, key, action):
        """Increment ``table[key][action]``, growing the row lazily if needed."""
        a = int(action)
        counts = table.get(key)
        if counts is None:
            # Grow lazily; sized to the action index seen (SMAC n_actions is fixed).
            counts = np.zeros(a + 1, dtype=np.float64)
            table[key] = counts
        elif a >= counts.shape[0]:
            grown = np.zeros(a + 1, dtype=np.float64)
            grown[: counts.shape[0]] = counts
            table[key] = grown
            counts = grown
        counts[a] += 1.0

    # ------------------------------------------------------------------- active
    def action_bonuses(self, obs, n_actions, agent_id=0):
        """ACTIVE: per-action novelty bonus for ``obs`` as seen by ``agent_id`` (read-only).

        Returns an array of length ``n_actions``. In ``team`` mode this is
        beta / sqrt(N(obs, a) + 1) over the single shared table. In ``per_agent`` mode the
        per-agent novelties are combined by ``self.combine`` (see the module docstring),
        so the bonus depends on what *every* agent has sampled, not just the team total.
        """
        key = self._key(obs, project=True)
        if not self.per_agent:
            counts = self._read_counts(self.action_counts, key, n_actions)
            return self.select_beta / np.sqrt(counts + 1.0)

        # (n_agents, n_actions): how novel this (obs, action) is to each agent.
        nov = np.stack(
            [
                self.select_beta
                / np.sqrt(self._read_counts(table, key, n_actions) + 1.0)
                for table in self.agent_action_counts
            ]
        )
        own = nov[agent_id]
        mode = self.combine
        if mode == "independent":
            return own
        if mode == "minimum":
            return nov.min(axis=0)
        mean = nov.mean(axis=0)
        if mode == "covering":
            return np.where(own > mean, own, 0.0)
        if mode == "burrowing":
            return np.where(own <= mean, own, 0.0)
        # leader_follower: agent 0 burrows (goes deep), everyone else covers (spreads).
        if agent_id == 0:
            return np.where(own <= mean, own, 0.0)
        return np.where(own > mean, own, 0.0)

    def record_action(self, obs, action, agent_id=0):
        """ACTIVE: record that ``agent_id`` took ``action`` in ``obs``.

        In ``team`` mode this increments the single shared table. In ``per_agent`` mode it
        increments only that agent's own table -- which every other agent can still read,
        so the knowledge is shared while remaining attributable to its source.
        """
        key = self._key(obs, project=True)
        if self.per_agent:
            self._bump(self.agent_action_counts[agent_id], key, action)
            # Track how many obs-buckets more than one agent has sampled. A bucket
            # becomes "shared" the moment a second agent touches it: the mask goes from
            # exactly one bit set to two.
            mask = self._owner_mask.get(key, 0)
            bit = 1 << int(agent_id)
            if not mask & bit:
                if mask and not mask & (mask - 1):
                    self.shared_buckets += 1
                self._owner_mask[key] = mask | bit
        else:
            self._bump(self.action_counts, key, action)
        self.total_action_records += 1

    # -------------------------------------------------------------- diagnostics
    @property
    def n_buckets(self):
        """Distinct passive (state) buckets discovered so far."""
        return len(self.counts)

    def stats(self):
        """Diagnostic summary. ``unique_ratio`` = n_buckets / total_visits: near 1.0
        means almost every state is a new bucket (degenerate novelty -- hash too fine);
        well below 1.0 means buckets are revisited and the signal is meaningful.

        ``redundancy_ratio`` (per_agent mode only) is the fraction of visited obs-buckets
        that MORE THAN ONE agent has sampled -- a direct measure of the redundant
        state-action sampling the mechanism is supposed to reduce, independent of win rate.
        """
        tv = self.total_visits
        out = {
            "n_buckets": self.n_buckets,
            "total_visits": tv,
            "unique_ratio": (self.n_buckets / tv) if tv else 0.0,
            "mean_bonus": (self.bonus_sum / tv) if tv else 0.0,
            "action_records": self.total_action_records,
        }
        if self.per_agent:
            distinct = len(self._owner_mask)
            out["action_buckets"] = distinct
            out["shared_buckets"] = self.shared_buckets
            out["redundancy_ratio"] = (self.shared_buckets / distinct) if distinct else 0.0
            for i, table in enumerate(self.agent_action_counts):
                out[f"agent{i}_buckets"] = len(table)
        else:
            out["action_buckets"] = len(self.action_counts)
        return out
