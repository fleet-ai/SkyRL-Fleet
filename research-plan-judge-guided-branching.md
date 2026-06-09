# Research Plan: LM Judge-Guided Branching for Credit Assignment in RL Training

> **Note:** One research direction was left incomplete during initial scoping — this plan marks it as TBD and should be revisited.

---

## Problem Statement

Standard RL training for LLMs (GRPO, PPO) smears a terminal reward across an entire trace, which gives a noisy learning signal over long horizons. Existing fixes either learn a separate process reward model (expensive, requires labeled data) or apply MCTS-style uniform branching (compute-intensive, ignores semantic structure).

This work proposes using an LM judge to identify *interesting* states in anchor traces, then rolling out from those states to estimate local value functions. The judge provides a cheap, semantically-grounded signal for *where* to branch — replacing visit counts or uniform step coverage with language understanding.

---

## Core Mechanism

1. **Anchor trace generation**: Roll out 1–2 complete traces per training example using the current policy.
2. **Judge scoring**: Pass the anchor trace to an LM judge. The judge returns a saliency score (or ranked set of states) indicating which intermediate states are high-leverage — i.e., where the outcome could meaningfully diverge.
3. **Branching rollouts**: From each flagged state, sample K continuations to the end. Observe terminal rewards.
4. **Value estimation**: The empirical mean reward across rollouts from state s is the value estimate V(s). Reward variance across rollouts is a signal of causal importance.
5. **Training**: Use value estimates for better advantage computation, or train directly on contrastive pairs at branch points (step-level DPO).

---

## Research Questions

### RQ1: What is the right notion of "interesting" for prefix selection?

This is the central question. The judge's interestingness criterion determines everything downstream — which states get value estimates, which get training signal, and whether the whole thing is more efficient than uniform branching.

Candidate framings:

- **Uncertainty**: States where the judge believes the model is likely to diverge. Operationalized as perplexity of the continuation, or judge confidence scores.
- **Causal influence**: States that appear to control whether the final answer is correct. Operationalized by comparing reward variance with/without conditioning on the state.
- **Decision-point detection**: States where the trace makes a semantically distinct choice (branches in reasoning, tool selection, etc.). Can be elicited directly via judge prompting.
- **Surprise / low prior probability**: Steps that are unexpected given the prefix. High negative log-likelihood under a reference model.

Key empirical question: do these criteria correlate with each other, and which best predicts reward variance in downstream rollouts? The ground truth is: a good interestingness score should rank states by how much the reward distribution changes when you branch from them.

**Experiment 1a**: For a held-out set of anchor traces with ground-truth reward variance at each step (computed by uniform branching), evaluate how well each judge criterion predicts that variance. Measure rank correlation.

**Experiment 1b**: Train with branching guided by each criterion and compare learning efficiency (reward improvement per rollout).

### RQ2: How does anchor trace quality affect the value estimates?

The anchor trace defines the prefix distribution. Branching from a bad prefix (off-policy, or from a trace that was already wrong early) gives value estimates that may not generalize to the on-policy distribution.

Sub-questions:
- Should anchors be sampled from the current policy, a fixed reference policy, or a mixture?
- Should anchors be filtered by quality (e.g., only branch from traces that pass some intermediate check)?
- Does branching from "good" traces vs. "bad" traces teach different things? (Recovery vs. exploitation)

**Experiment 2**: Compare value estimate accuracy and downstream training performance across anchor selection strategies: random policy rollout, best-of-N selection, reward-stratified sampling (mix of good and bad anchors).

### RQ3: How should partial traces be used for training?

Given a branch point with K rollouts and observed rewards, several training strategies are possible:

- **Advantage correction only**: Use rollout rewards to compute better per-step advantages for the anchor trace; don't train on branch rollouts directly. Cleanest, no distribution shift.
- **Step-level DPO**: At each branch point, form contrastive pairs (best rollout vs. worst rollout) conditioned on the shared prefix. Train with DPO loss from the branch point forward.
- **Weighted behavioral cloning**: Train on the best rollout with weight proportional to reward gap.
- **Value-function auxiliary loss**: Train a value head using the MC estimates as supervision.

**Experiment 3**: Ablate training strategy on a fixed branching schedule. Measure sample efficiency and final performance.

### RQ4: How does compute allocation interact with branch point selection?

Given a fixed rollout budget, is it better to branch from many states shallowly (few rollouts each) or few states deeply (many rollouts each)? And how does this interact with judge confidence — should high-confidence interesting states get more rollouts?

This connects to the analysis in Snell et al. on optimal compute allocation for test-time search.

**Experiment 4**: Sweep branch budget (states × rollouts per state = constant) and measure value estimate quality as a function of allocation.

### RQ5 (TBD)

*[Left incomplete during initial scoping — to be filled in.]*

---

## Baselines

- **Math-Shepherd**: Uniform MC rollouts at every step, no judge.
- **OmegaPRM**: MCTS with binary search for branch point selection.
- **Standard GRPO/PPO**: No branching, terminal reward only.
- **Oracle branching**: Branch at steps with highest ground-truth reward variance (upper bound on judge quality).

---

## Evaluation

Primary: learning efficiency (final reward as a function of total rollouts generated).

Secondary:
- Value estimate accuracy (MSE vs. ground-truth MC values computed with large K).
- Branching efficiency: fraction of rollout budget spent on high-variance states.
- Generalization: does training on partial traces hurt performance on full trace distributions?

Domains: start with math reasoning (verifiable reward, well-studied baselines), then extend to agentic/tool-use tasks where long-horizon credit assignment matters most.

---

## Implementation Notes

- **Environment**: Internal computer use environment (TBD).
- **Judge**: Start with the training model itself (self-evaluation) to avoid external dependencies. Compare against a stronger judge model.
- **Branching implementation**: Can be layered on top of existing SkyRL rollout infrastructure — anchor traces are just standard rollouts, branching is re-running from a saved KV cache state.
- **KV cache reuse**: Branching from a prefix is only efficient if the prefix KV cache is reused across rollouts. Ensure vLLM prefix caching is enabled for anchor traces.

---

## Rough Timeline

| Phase | Focus | Duration |
|---|---|---|
| 0 | Literature review (Math-Shepherd, OmegaPRM, Snell et al.) | 1–2 days |
| 1 | Prototype: uniform branching + judge scoring, measure Exp 1a correlation | 1 week |
| 2 | RQ1 ablations (interestingness criteria) | 1–2 weeks |
| 3 | RQ3 ablations (training strategy) | 1 week |
| 4 | Full comparison vs. baselines, compute scaling analysis | 1–2 weeks |
| 5 | Write-up | 1 week |

---

## Open Questions / Risks

- **Judge cost**: If the judge is a large model, its inference cost may negate the savings from targeted branching. Need to establish that a small/fast judge is sufficient for RQ1.
- **Distribution shift from partial traces**: Training on suffix-only traces could degrade coherence. Monitor full-trace evaluation separately from branch-point evaluation.
- **Variance of value estimates**: With small K, MC estimates are noisy. Need to establish minimum K for reliable signal before using estimates for training.
- **Interaction with FSDP/vLLM colocated training**: KV cache reuse for prefix branching may conflict with vLLM's sleep/wake cycle in colocated setup. Verify feasibility before committing to this optimization.
