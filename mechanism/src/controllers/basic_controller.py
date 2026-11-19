from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import numpy as np
import torch as th


# This multi-agent controller shares parameters between agents
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

        # Centralized shared exploration memory, injected by run.py when the
        # mechanism is enabled (None otherwise). Used here for ACTIVE directed
        # exploration: the memory enters the action decision, so the exploration
        # policy depends on the collectively accumulated shared knowledge.
        self.shared_memory = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)

        # ACTIVE shared-memory exploration: build a per-agent, per-action novelty
        # bonus from the shared state-action counts so the selector is steered
        # toward collectively under-sampled state-actions. Training only -- test
        # uses the pure learned policy.
        novelty_bonus = None
        obs_np = None
        active = (
            self.shared_memory is not None
            and getattr(self.args, "sm_active_selection", False)
            and not test_mode
        )
        # Exploration-influence anneal, adapted from CMAE (Liu et al. 2021, Alg. 1),
        # which selects actions with a mixture "alpha*mu + (1-alpha)*pi" where "alpha
        # decreases linearly to 0". We have no separate exploration policy -- our
        # novelty enters as an additive bonus -- so the faithful analogue is to scale
        # that bonus by the same alpha. At alpha=1 novelty has full influence; at
        # alpha=0 the action comes purely from the learned values. sm_beta_anneal_time=0
        # disables the anneal and reproduces the previous (constant-beta) behaviour.
        anneal_t = getattr(self.args, "sm_beta_anneal_time", 0)
        sm_scale = 1.0
        if active and anneal_t and anneal_t > 0:
            sm_scale = max(0.0, 1.0 - float(t_env) / float(anneal_t))
        self.sm_last_scale = sm_scale
        if active and sm_scale <= 0.0:
            # Bonus is identically zero: skip the hashing work for the read, but still
            # record below so the counts and the redundancy diagnostic stay correct.
            active_read = False
        else:
            active_read = active

        if active:
            # (bs, n_agents, obs_dim)
            obs_np = ep_batch["obs"][:, t_ep][bs].detach().cpu().numpy()
            n_actions = avail_actions.shape[-1]
            # SMAC returns an ALL-ZERO observation for a dead unit. Those must be
            # skipped: recording them would pile every corpse into a single bucket,
            # which both floods the counts and makes sm_redundancy_ratio meaningless
            # (every agent "shares" that one bucket). A dead agent's only available
            # action is no-op anyway, so skipping changes no behaviour.
            alive_np = np.abs(obs_np).sum(axis=-1) > 0.0
            bonus = np.zeros(
                (obs_np.shape[0], self.n_agents, n_actions), dtype=np.float32
            )
            if active_read:
                for i in range(obs_np.shape[0]):
                    for a in range(self.n_agents):
                        if alive_np[i, a]:
                            bonus[i, a] = sm_scale * self.shared_memory.action_bonuses(
                                obs_np[i, a], n_actions, a
                            )
            novelty_bonus = th.as_tensor(bonus, device=ep_batch.device)
            if active_read and getattr(
                self.args, "sm_bonus_scaling", "absolute"
            ) == "relative":
                novelty_bonus = self._scale_bonus_to_decision_spread(
                    novelty_bonus, agent_outputs[bs], avail_actions[bs], sm_scale
                )

        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs],
            avail_actions[bs],
            t_env,
            test_mode=test_mode,
            novelty_bonus=novelty_bonus,
        )

        # Record the taken (obs, action) pairs so future selections by ANY agent
        # see these state-actions as more sampled -> collaborative, non-redundant
        # exploration.
        if active:
            acts_np = chosen_actions.detach().cpu().numpy()  # (bs, n_agents)
            for i in range(acts_np.shape[0]):
                for a in range(self.n_agents):
                    if alive_np[i, a]:
                        self.shared_memory.record_action(
                            obs_np[i, a], int(acts_np[i, a]), a
                        )

        return chosen_actions

    def _scale_bonus_to_decision_spread(self, bonus, outputs, avail, sm_scale):
        """Express the novelty bonus as a fraction of the agent's own decision spread.

        An absolute beta is not comparable across scenarios. Measured on the runs of
        2026-11-19, ``q_taken_mean`` was 0.87 on ``5m_vs_6m`` against 8.05 on
        ``3s_vs_5z``, so one beta meant a 9x different perturbation per map. Worse, the
        ``1/sqrt(N+1)`` term shrank the raw bonus to roughly 0.004% of the Q-scale once
        counts reached the hundreds, leaving the mechanism numerically inert.

        Two changes fix both problems. The bonus is measured against the spread of the
        quantity the action selector actually adds it to -- Q-values for epsilon-greedy,
        log-probabilities for the policy-gradient selector, which multiplies by
        ``exp(bonus)`` and so adds to logits. And novelty is rescaled across the
        available actions, so the most novel one receives the full ``beta * spread``
        and the least novel receives zero, regardless of how large the counts have
        grown. What survives is the *ordering* of novelty, which is the part that
        should steer the choice; its absolute magnitude, which only ever reflected how
        long training had run, does not.

        The influence on the argmax is then invariant to reward scale and stays
        meaningful for the whole run, while remaining bounded by beta.
        """
        beta = float(getattr(self.args, "sm_select_beta", 0.1))
        avail_b = avail == 0
        big = 1e20

        # The selector adds the bonus to Q-values directly, but multiplies
        # probabilities by exp(bonus), which is an addition in log space.
        if getattr(self.args, "agent_output_type", "q") == "pi_logits":
            decision = th.log(outputs.clamp(min=1e-8))
        else:
            decision = outputs

        d_hi = decision.masked_fill(avail_b, -big).max(dim=-1).values
        d_lo = decision.masked_fill(avail_b, big).min(dim=-1).values
        spread = (d_hi - d_lo).clamp(min=0.0).unsqueeze(-1)       # (bs, n_agents, 1)

        b_hi = bonus.masked_fill(avail_b, -big).max(dim=-1, keepdim=True).values
        b_lo = bonus.masked_fill(avail_b, big).min(dim=-1, keepdim=True).values
        b_norm = ((bonus - b_lo) / (b_hi - b_lo).clamp(min=1e-12)).clamp(0.0, 1.0)
        b_norm = b_norm * avail.float()

        scaled = sm_scale * beta * spread * b_norm

        # Diagnostics: these make inertness visible within minutes instead of after a
        # full run. Averaged only over agents that actually have a choice, and smoothed.
        # A dead or pinned agent has a single available action, so its spread is zero by
        # definition; the runner reads these at episode end, which is exactly when most
        # agents are in that state, so an unfiltered last-value reading always reports 0.
        with th.no_grad():
            choosing = avail.sum(dim=-1) > 1
            if bool(choosing.any()):
                sp = float(spread.squeeze(-1)[choosing].mean().item())
                bm = float(scaled.max(dim=-1).values[choosing].mean().item())
                a = 0.01
                prev_sp = getattr(self, "sm_last_decision_spread", None)
                prev_bm = getattr(self, "sm_last_bonus_max", None)
                self.sm_last_decision_spread = sp if prev_sp is None else (1 - a) * prev_sp + a * sp
                self.sm_last_bonus_max = bm if prev_bm is None else (1 - a) * prev_bm + a * bm
        return scaled

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents

        return input_shape
