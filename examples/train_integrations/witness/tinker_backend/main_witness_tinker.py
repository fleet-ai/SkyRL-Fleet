"""Witness GRPO RL on cloud-tinker — ported into Deniz's Fleet×Tinker loop.

REUSES `integrations/fleet/entrypoints/main_fleet_tinker.py` verbatim for everything that
is env-agnostic — the GRPO advantage math, the Datum construction (DAPO overlong filter,
target-shift, logprob/shape guards — the team's recent fixes), the metrics, the
forward_backward/optim_step cycle. The ONLY witness-specific piece is rollout collection:
we swap `FleetTaskEnv` for the witness `AgentRolloutWrapper` bridge.

Key difference from FleetTaskEnv (and why per-call datums): the witness ORAI loop is
SINGLE-TURN-PER-CALL — each call is a fresh (system,user) prompt, NOT an accumulating
conversation. So the policy sampled each response under its own single-turn context; to keep
the training context == the sampling context (correct importance ratios), each ORAI call
becomes its OWN datum (prompt = that single-turn chat), and the trajectory-level GRPO
advantage is broadcast across all of a trajectory's calls (outcome-supervised GRPO).

Runs LOCALLY; tinker's cloud does GPU forward_backward/sample. Reached via
`scripts/2026-06-23_witness_rl_tinker_launch.sh` (which the BACKEND=tinker switch dispatches to).

  TINKER_API_KEY=… python -m tinker_backend.main_witness_tinker \
      --model-name Qwen/Qwen3.5-9B --load-checkpoint-path tinker://<sft-lora> \
      --game-ids tw07,tw03,tw06,tw02,tw05,tw08,tw11,tw13 \
      --val-game-ids tw01,tw09,tw10,tw12 --inject-mode off
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tinker
import wandb
from tinker import types
from transformers import AutoTokenizer
from pydantic import BaseModel

# --- env-agnostic GRPO machinery, VENDORED verbatim from Deniz's loop (see _grpo_utils.py
#     for why we vendor rather than import: his module's top-level deps — omegaconf/skyrl_gym/
#     skyrl.train — aren't in the lean agent venv and are only used by the OpenEnv collector). ---
from tinker_backend._grpo_utils import (  # noqa: E402
    RolloutOutput,
    prepare_training_data,
    normalize_advantages,
    compute_pass_at_n,
    compute_per_env_metrics,
    tokenize_chat,
    set_seed,
)

# --- witness harness + agent on path (mirrors agent_wrapper.py) ---
_WITNESS_DIR = Path(__file__).resolve().parent.parent          # .../train_integrations/witness
sys.path.insert(0, str(_WITNESS_DIR))
from harness.agent_wrapper import AgentRolloutWrapper           # noqa: E402

_EXECUTOR = ThreadPoolExecutor(max_workers=64)                 # bridge calls block → run off the event loop


async def _blocking(fn, *args):
    return await asyncio.get_event_loop().run_in_executor(_EXECUTOR, lambda: fn(*args))


def _load_agent_config(path: Optional[str]) -> dict:
    """Load config.yaml + apply the SAME RL overrides env_agent.py does, so only the bridged
    ORAI call hits the policy: ground-truth perception ON, MLLM enhancement OFF (no out-of-band
    LLM calls to a nonexistent endpoint), provider=vllm. Mirrors env_agent.py:79-82 + the RL recipe.
    """
    import yaml
    cand = path or os.environ.get("WITNESS_AGENT_CONFIG") or str(
        Path(os.environ.get("ARC_WITNESS_AGENT",
             Path.home() / "Documents/obsidian/research-repos/arc-witness-agent")) / "configs" / "witness_canonical.yaml")
    with open(cand) as f:
        cfg = yaml.safe_load(f) or {}
    ascii_cfg = cfg.setdefault("semantic_ascii", {})
    ascii_cfg["enabled"] = True
    ascii_cfg["mllm_enabled"] = False                                      # RL: no MLLM enhancement
    ascii_cfg.setdefault("env_ground_truth", {})["enabled"] = \
        os.environ.get("USE_ENV_GROUND_TRUTH", "1") == "1"                 # GT perception (no vision LLM)
    cfg.setdefault("llm", {})["provider"] = "vllm"
    # Fix-3 (efficiency): disable the IncrementalPlanner's out-of-band LLM call. It has no
    # endpoint in the tinker rollout → fails + self-disables after 3 tries anyway; a high
    # min_rules makes can_plan() never fire → zero wasted planner calls per rollout.
    cfg.setdefault("stages", {}).setdefault("planning", {}).setdefault("rule_planner", {})["min_rules"] = 100000
    return cfg


_WRONG_POOL = ["tw03", "tw06", "tw02", "tw05", "tw08", "tw11", "tw13", "tw07"]


def _rule_card_lines(game_id: str, mode: str) -> list[str]:
    from oversight import WitnessOracle
    card_game = game_id if mode == "rules" else \
        ([g for g in _WRONG_POOL if g != game_id] or _WRONG_POOL)[hash(game_id) % max(1, len(_WRONG_POOL) - 1)]
    card = WitnessOracle().rule_card(card_game)
    return [ln for ln in card.splitlines() if ln.strip() and not ln.startswith("#")]


def _arc_agi3_format_score(game_id: str, gm, max_levels: int) -> float:
    """Official ARC-AGI-3 RHAE per-game score (0–115) for one rollout, from the agent's per-level
    metrics. Reuses the OFFICIAL scorer + baselines (bench.scoring / bench.catalog, on PYTHONPATH).
    Per level: min(115, (baseline/actions)²·100), 0 if unsolved; game = Σ(score·weight)/Σweight
    (weight=1-based level), capped at the completion fraction. score_cap = max_levels (=5)."""
    try:
        from bench.catalog import load_game_info
        from bench.scoring import WitnessScoreCalculator
        gi = load_game_info(game_id)
        n = min(gi.scoreable_levels, max_levels, len(gi.baseline_actions))
        by_idx: Dict[int, Any] = {}
        for lm in (getattr(gm, "level_metrics", None) or []):
            if lm.level_index not in by_idx or lm.completed:   # prefer the completed entry per level
                by_idx[lm.level_index] = lm
        calc = WitnessScoreCalculator(game_id)
        for L in range(n):
            lm = by_idx.get(L)
            calc.add_level(level_index=L + 1,
                           completed=bool(lm.completed) if lm else False,
                           actions_taken=int(lm.actions_taken) if lm else 0,
                           baseline_actions=int(gi.baseline_actions[L]))
        return float(calc.to_score().score)
    except Exception as e:  # noqa: BLE001
        print(f"[arc_score] {game_id}: {e}", flush=True)
        return 0.0


def _solved_level_steps(game_id: str, gm, max_levels: int) -> List[int]:
    """Per-SOLVED-level action counts (steps spent on the levels the rollout actually WON) → feeds the
    'avg steps per solved level' eval metric. Same per-level dedup (prefer completed) + level cap as the
    arc scorer, so it counts only genuine solves and excludes steps wasted on unsolved levels."""
    try:
        from bench.catalog import load_game_info
        gi = load_game_info(game_id)
        n = min(gi.scoreable_levels, max_levels, len(gi.baseline_actions))
        by_idx: Dict[int, Any] = {}
        for lm in (getattr(gm, "level_metrics", None) or []):
            if lm.level_index not in by_idx or lm.completed:
                by_idx[lm.level_index] = lm
        out = []
        for L in range(n):
            lm = by_idx.get(L)
            if lm and lm.completed and int(lm.actions_taken) > 0:
                out.append(int(lm.actions_taken))
        return out
    except Exception:  # noqa: BLE001
        return []


def _level_detail(game_id: str, gm, max_levels: int) -> List[List[int]]:
    """Per-level [level_index(0-based), completed(0/1), actions_taken] using the SAME dedup + level cap
    as the arc scorer — dumped into the eval JSON so evals are RE-SCORABLE offline under any baseline
    convention (baseline = k×optimal etc.) without re-running rollouts. Join with bench.catalog baselines."""
    try:
        from bench.catalog import load_game_info
        gi = load_game_info(game_id)
        n = min(gi.scoreable_levels, max_levels, len(gi.baseline_actions))
        by_idx: Dict[int, Any] = {}
        for lm in (getattr(gm, "level_metrics", None) or []):
            if lm.level_index not in by_idx or lm.completed:
                by_idx[lm.level_index] = lm
        out = []
        for L in range(n):
            lm = by_idx.get(L)
            out.append([L, int(bool(lm.completed)) if lm else 0, int(lm.actions_taken) if lm else 0])
        return out
    except Exception:  # noqa: BLE001
        return []


class WitnessTrajectory(BaseModel):
    """One witness game rollout = a list of per-ORAI-call (prompt, response, logprobs) + a return."""
    calls: List[Dict[str, Any]]      # each: {prompt_ids, response_ids, logprobs}
    reward: float                    # trajectory return (sum of per-call event rewards)
    task_key: str                    # game_id (GRPO group key)
    turns: int
    wins: int = 0                    # # ORAI calls that completed a level (ev_r>0) — outcome-signal diagnostic
    reward_breakdown: Dict[str, float] = {}   # per-component reward sums (outcome/step_penalty/...)
    game_steps: int = 0              # agent game-actions this rollout (max event.step) — for steps/level
    completed_levels: int = 0        # genuine levels won (_levels_genuine; skip-path excluded)
    arc_score: float = 0.0           # ARC-AGI-3 RHAE per-game format score (0–115), official scorer
    solved_level_steps: List[int] = []  # actions_taken per SOLVED level → 'avg steps per solved level'
    level_detail: List[List[int]] = []  # [lvl_idx, completed, actions] per level (scorer's view) → re-scorable evals
    seeded: bool = False             # rollout started from a Go-Explore frontier snapshot
    stop_reason: str = "stop"
    duration: float = 0.0
    total_gen_time: float = 0.0
    total_tokens: int = 0
    error: Optional[str] = None


async def collect_witness_rollout(
    game_id: str, seed: int, agent_config: dict,
    sampling_client: tinker.SamplingClient, tokenizer: AutoTokenizer,
    max_levels: int = 5, max_generate_length: int = 2048, max_input_length: int = 16384,
    temperature: float = 1.0, top_p: float = 1.0, stop_sequences: List[str] = None,
    max_orai_steps: int = 30, inject_mode: str = "off", inject_p: float = 0.0,
    frontier=None, start_snapshot=None,
) -> WitnessTrajectory:
    """Drive the witness ORAI bridge; sample each call via tinker. Per-call records.

    Go-Explore (goal-3): start_snapshot (if given) restarts the rollout from a restored
    mid-game state (on-policy continuation); frontier (if given) harvests progress states
    for future seeding. Both default off → identical to the plain T1 path."""
    t0 = time.time()
    wrapper = AgentRolloutWrapper(game_id=game_id, seed=seed, agent_config=agent_config,
                                  vllm_base_url="http://unused-bridged", max_levels=max_levels, mode="bridged")
    # T1 curriculum injection (training-only; tainted 'oracle_injected')
    if inject_mode != "off" and inject_p > 0 and random.random() < inject_p:
        from agent.oversight import inject_rules
        inject_rules(wrapper.agent, _rule_card_lines(game_id, inject_mode))

    calls: List[Dict[str, Any]] = []
    total_reward, gen_time, total_tokens, stop_reason = 0.0, 0.0, 0, "stop"
    wins = 0
    rbd: Dict[str, float] = {}                        # per-component reward sums (for the outcome metric)
    last_step = 0                                     # max event.step = agent game-actions (for steps/level)
    try:
        prompt = await _blocking(wrapper.start_bridged_rollout, start_snapshot)   # restored mid-game state if Go-Explore
        while prompt is not None and len(calls) < max_orai_steps:  # cap ORAI calls/rollout (Deniz's max_turns)
            system_prompt, messages = prompt
            if not messages and not system_prompt:
                break
            chat = [{"role": "system", "content": system_prompt}] + list(messages)
            input_ids = tokenize_chat(tokenizer, chat, add_generation_prompt=True)
            if len(input_ids) > max_input_length:
                stop_reason = "length"
                break
            sp = {"max_tokens": max_generate_length, "temperature": temperature, "top_p": top_p}
            if stop_sequences:
                sp["stop"] = stop_sequences
            g0 = time.time()
            result = await sampling_client.sample_async(
                prompt=types.ModelInput.from_ints(tokens=input_ids), num_samples=1,
                sampling_params=types.SamplingParams(**sp))
            gen_time += time.time() - g0
            if not result.sequences:
                break
            seq = result.sequences[0]
            out_ids = list(seq.tokens)
            out_lp = list(seq.logprobs) if seq.logprobs else [0.0] * len(out_ids)
            if len(out_lp) != len(out_ids):   # tinker shape guard (mirror Deniz)
                out_lp = out_lp[:len(out_ids)] + [0.0] * max(0, len(out_ids) - len(out_lp))
            total_tokens += len(out_ids)
            completion = tokenizer.decode(out_ids, skip_special_tokens=True)
            calls.append({"prompt_ids": input_ids, "response_ids": out_ids, "logprobs": out_lp})
            next_prompt, event = await _blocking(wrapper.feed_completion_and_get_next, completion)
            ev_r = float(getattr(event, "reward", 0.0)); total_reward += ev_r
            for _k, _v in (getattr(event, "reward_breakdown", None) or {}).items():
                rbd[_k] = rbd.get(_k, 0.0) + float(_v)       # accumulate per-component reward (→ outcome metric)
            _st = int(getattr(event, "step", 0) or 0)
            if _st > last_step: last_step = _st              # agent game-action count (for steps/level)
            if ev_r > 0:                                     # ev_r>0 ⟺ a level completed this call (outcome +1 − 0.005 step)
                wins += 1
                if frontier is not None:                     # Go-Explore: harvest the next-level-start state (engine forks, not tinker)
                    try:
                        from oversight import snapshot as _snap
                        await _blocking(frontier.harvest, game_id, await _blocking(_snap, wrapper.game))
                    except Exception:
                        pass
            prompt = next_prompt
        if prompt is not None:                 # exited via the ORAI-step cap, not natural end → truncated
            stop_reason = "length"
    except Exception as e:  # noqa: BLE001
        import traceback; traceback.print_exc()
        return WitnessTrajectory(calls=calls, reward=total_reward, task_key=game_id,
                                 turns=len(calls), stop_reason="error", error=str(e),
                                 duration=time.time() - t0, wins=wins, seeded=start_snapshot is not None,
                                 reward_breakdown=rbd, game_steps=last_step, completed_levels=wins,
                                 arc_score=0.0)
    finally:
        try:
            await _blocking(wrapper.join_bridged_rollout)
        except Exception:  # noqa: BLE001
            pass
    completed = int(getattr(wrapper.agent, "_levels_genuine", wins))   # genuine levels won (skip-excluded)
    _gm = getattr(wrapper, "metrics", None); _ml = getattr(wrapper, "max_levels", 5)
    arc = _arc_agi3_format_score(game_id, _gm, _ml)
    sls = _solved_level_steps(game_id, _gm, _ml)
    return WitnessTrajectory(calls=calls, reward=total_reward, task_key=game_id, turns=len(calls),
                             stop_reason=stop_reason, duration=time.time() - t0,
                             total_gen_time=gen_time, total_tokens=total_tokens,
                             wins=wins, seeded=start_snapshot is not None,
                             reward_breakdown=rbd, game_steps=last_step, completed_levels=completed,
                             arc_score=arc, solved_level_steps=sls,
                             level_detail=_level_detail(game_id, _gm, _ml))


async def collect_batch(game_ids, agent_config, sampling_client, tokenizer, n_samples_per_prompt,
                        seed_base, max_concurrent=8, inject_mode="off", inject_p=0.0,
                        frontier=None, seed_frac=0.0, **kw) -> List[WitnessTrajectory]:
    sem = asyncio.Semaphore(max_concurrent)
    seeded_k = round(seed_frac * n_samples_per_prompt)   # Go-Explore: # of N rollouts that start from a frontier snapshot

    async def one(game_id, idx, start_snap):
        async with sem:
            t = await collect_witness_rollout(
                game_id=game_id, seed=seed_base + idx, agent_config=agent_config,
                sampling_client=sampling_client, tokenizer=tokenizer,
                inject_mode=inject_mode, inject_p=inject_p,
                frontier=frontier, start_snapshot=start_snap, **kw)
            print(f"  [rollout {game_id} #{idx}{'*GE' if start_snap is not None else ''}] calls={len(t.calls)} "
                  f"reward={t.reward:.3f} stop={t.stop_reason} dur={t.duration:.0f}s gen={t.total_gen_time:.0f}s "
                  f"agent={(t.duration - t.total_gen_time):.0f}s" + (f" ERR={t.error}" if t.error else ""), flush=True)
            return t
    tasks, idx = [], 0
    for g in game_ids:                       # n_samples consecutive rollouts of the SAME game (GRPO group order)
        for j in range(n_samples_per_prompt):
            # seed the first `seeded_k` of each group from the frontier (empty early → None → fresh)
            snap = frontier.sample(g) if (frontier is not None and j < seeded_k) else None
            tasks.append(one(g, idx, snap)); idx += 1
    return list(await asyncio.gather(*tasks))


def _traj_to_percall_rollouts(traj: WitnessTrajectory, advantage: float):
    """Expand one trajectory into per-ORAI-call RolloutOutputs (+ broadcast advantage)."""
    outs, advs = [], []
    for c in traj.calls:
        outs.append(RolloutOutput(
            prompt_ids=c["prompt_ids"], response_ids=c["response_ids"], logprobs=c["logprobs"],
            loss_mask=[1] * len(c["response_ids"]), reward=traj.reward,
            task_key=traj.task_key, env_key=traj.task_key, turns=traj.turns,
            tool_calls=0, tool_errors=0, stop_reason=traj.stop_reason, duration=traj.duration))
        advs.append(advantage)
    return outs, advs


def compute_grpo_advantages_and_signal(valid, normalize: bool = True):
    """Group trajectories by task_key (GRPO group = same game) → center rewards within group →
    optional global normalize. Returns (advantages, metrics).

    Why group by task_key and not the vendored fixed-`group_size` slicing: `valid` drops
    errored/empty rollouts, so a fixed N-chunk straddles two games whenever a drop happens →
    advantages centered against the wrong group (silent, wrong gradients). Grouping by task_key
    is correct regardless of drops; identical to the old path when nothing drops.

    The metrics diagnose the removed-process-reward (near-sparse) signal and whether Go-Explore
    seeding restores it. Two variances, deliberately: with reward = +1/level − 0.005/ORAI-call,
    non-solving rollouts STILL differ in reward via step count → `reward` variance is ~always
    nonzero (a length gradient, not task signal). The variance GRPO needs to learn the TASK is in
    solve count (`wins`). We log both so the contrast is explicit; `groups/frac_solve_var` is the
    headline (off vs GE-on)."""
    from collections import defaultdict
    idx_by_game: Dict[str, List[int]] = defaultdict(list)
    for i, t in enumerate(valid):
        idx_by_game[t.task_key].append(i)
    advantages = [0.0] * len(valid)
    n_reward_var = n_solve_var = n_any_solve = seed_rescued = 0
    solve_var_sum = 0.0
    for idxs in idx_by_game.values():
        rs = np.array([valid[i].reward for i in idxs], dtype=float)
        ws = np.array([float(valid[i].wins) for i in idxs], dtype=float)
        centered = rs - rs.mean()
        for k, i in enumerate(idxs):
            advantages[i] = float(centered[k])
        if float(rs.var()) > 1e-12:
            n_reward_var += 1
        sv = float(ws.var())
        solve_var_sum += sv
        if ws.max() > 0:
            n_any_solve += 1
        if sv > 1e-12:
            n_solve_var += 1
            unseeded = np.array([float(valid[i].wins) for i in idxs if not valid[i].seeded], dtype=float)
            if unseeded.size <= 1 or float(unseeded.var()) <= 1e-12:
                seed_rescued += 1   # solve-variance exists ONLY because a seeded rollout solved differently
    if normalize:
        advantages = normalize_advantages(advantages)
    n_groups = max(1, len(idx_by_game))
    adv = np.asarray(advantages, dtype=float) if advantages else np.zeros(1)
    metrics = {
        "groups/n": len(idx_by_game),
        "groups/frac_any_solve": n_any_solve / n_groups,
        "groups/frac_solve_var": n_solve_var / n_groups,     # headline: groups with learnable OUTCOME signal
        "groups/frac_reward_var": n_reward_var / n_groups,   # ~1 (step-length noise) — contrast vs solve_var
        "groups/mean_solve_var": solve_var_sum / n_groups,
        "groups/seed_rescued": seed_rescued,
        "rollouts/seeded": sum(1 for t in valid if t.seeded),
        "rollouts/seeded_solved": sum(1 for t in valid if t.seeded and t.wins > 0),
        "advantage/l2": float(np.sqrt(np.mean(adv ** 2))),
        "advantage/abs_mean": float(np.mean(np.abs(adv))),
    }
    return advantages, metrics


def _compute_eval_metrics(ev, egroups):
    """Held-out eval metrics from eval trajectories `ev` (already filtered to calls & not error).
    Returns (metrics_dict, eval_str). Shared by the in-loop periodic eval AND the post-loop FINAL
    eval (step==max_steps) so the two never drift."""
    metrics = {}
    er = [t.reward for t in ev]
    metrics["eval/all/avg_reward"] = float(np.mean(er))
    metrics["eval/all/pass_at_1"] = compute_pass_at_n([t.model_dump() for t in ev], 1)
    metrics["eval/num_samples"] = len(ev)
    ev_lv = [t.completed_levels for t in ev]
    metrics["eval/avg_completed_levels"] = float(np.mean(ev_lv))
    metrics["eval/avg_outcome_reward"]   = float(np.mean([t.reward_breakdown.get("outcome", 0.0) for t in ev]))
    metrics["eval/avg_steps_per_level"]  = float(sum(t.game_steps for t in ev) / max(1, sum(ev_lv)))  # total steps (incl. failed level) / solved levels
    # avg # steps per SOLVED level: mean actions_taken over levels actually won (excludes steps wasted on
    # unsolved levels) — pure solving efficiency, vs avg_steps_per_level which charges failed-level steps too.
    def _steps_per_solved(tl):
        ss = [s for t in tl for s in getattr(t, "solved_level_steps", [])]
        return float(np.mean(ss)) if ss else 0.0
    metrics["eval/avg_steps_per_solved_level"] = _steps_per_solved(ev)
    by_g = {}
    for t in ev:
        by_g.setdefault(getattr(t, "task_key", "?"), []).append(t)
    # RHAE per game: mean-of-N (smooth training signal) AND best-of-N (= official ARC-AGI-3 per-game
    # aggregation, arc_agi.scorecard.EnvironmentScoreList.score = max over runs). Report BOTH — ~free.
    game_arc_mean, game_arc_best, game_lv_best, arcs_by_game = {}, {}, {}, {}
    for g, ts in by_g.items():
        lv = [t.completed_levels for t in ts]
        arc = [t.arc_score for t in ts]
        arcs_by_game[g] = arc
        game_arc_mean[g] = float(np.mean(arc)); game_arc_best[g] = float(np.max(arc)); game_lv_best[g] = float(np.max(lv))
        metrics[f"eval/{g}/avg_reward"]           = float(np.mean([t.reward for t in ts]))
        metrics[f"eval/{g}/avg_completed_levels"] = float(np.mean(lv))            # mean-of-N seeds
        metrics[f"eval/{g}/avg_completed_levels_bestN"] = game_lv_best[g]         # best-of-N (max seed)
        metrics[f"eval/{g}/avg_outcome_reward"]   = float(np.mean([t.reward_breakdown.get("outcome", 0.0) for t in ts]))
        metrics[f"eval/{g}/avg_steps_per_level"]  = float(sum(t.game_steps for t in ts) / max(1, sum(lv)))
        metrics[f"eval/{g}/avg_steps_per_solved_level"] = _steps_per_solved(ts)
        metrics[f"eval/{g}/arc_meanN"] = game_arc_mean[g]
        metrics[f"eval/{g}/arc_bestN"] = game_arc_best[g]
        metrics[f"eval/{g}/arc_std"] = float(np.std(arc, ddof=1)) if len(arc) > 1 else 0.0  # seed-to-seed std of this game's RHAE
    # official aggregation = per-game score THEN mean across games (arc_agi: avg_score = mean over envs)
    metrics["eval/arc_meanN"] = float(np.mean(list(game_arc_mean.values()))) if game_arc_mean else 0.0
    metrics["eval/arc_bestN"] = float(np.mean(list(game_arc_best.values()))) if game_arc_best else 0.0  # OFFICIAL
    metrics["eval/avg_completed_levels_bestN"] = float(np.mean(list(game_lv_best.values()))) if game_lv_best else 0.0
    metrics["eval/avg_arc_agi_3_format_score"] = metrics["eval/arc_meanN"]  # legacy alias (= mean-of-N)
    # seed-to-seed std of the GROUP RHAE (per-seed group score = mean over the group's games of that seed's
    # arc; std over seeds) → error bars on the reported means. SEM = arc_std / sqrt(n_seed). ALWAYS report std.
    def _group_arc_std(gnames):
        cols = [arcs_by_game[g] for g in gnames if g in arcs_by_game]
        nmin = min((len(c) for c in cols), default=0)
        if not cols or nmin < 2:
            return 0.0
        per_seed = [float(np.mean([c[s] for c in cols])) for s in range(nmin)]
        return float(np.std(per_seed, ddof=1))
    metrics["eval/arc_std"] = _group_arc_std(list(arcs_by_game))
    # respective aggregates for named held-out subgroups (composites vs variants, etc.)
    for _lab, _set in egroups.items():
        gts = [t for t in ev if getattr(t, "task_key", "?") in _set]
        if not gts: continue
        glv = [t.completed_levels for t in gts]
        gn = [g for g in game_arc_best if g in _set]
        metrics[f"eval/{_lab}/avg_reward"]            = float(np.mean([t.reward for t in gts]))
        metrics[f"eval/{_lab}/avg_completed_levels"]  = float(np.mean(glv))
        metrics[f"eval/{_lab}/avg_outcome_reward"]    = float(np.mean([t.reward_breakdown.get("outcome", 0.0) for t in gts]))
        metrics[f"eval/{_lab}/avg_steps_per_level"]   = float(sum(t.game_steps for t in gts) / max(1, sum(glv)))
        metrics[f"eval/{_lab}/avg_steps_per_solved_level"] = _steps_per_solved(gts)
        metrics[f"eval/{_lab}/arc_meanN"] = float(np.mean([game_arc_mean[g] for g in gn])) if gn else 0.0
        metrics[f"eval/{_lab}/arc_bestN"] = float(np.mean([game_arc_best[g] for g in gn])) if gn else 0.0
        metrics[f"eval/{_lab}/arc_std"] = _group_arc_std(gn)
        metrics[f"eval/{_lab}/avg_completed_levels_bestN"] = float(np.mean([game_lv_best[g] for g in gn])) if gn else 0.0
        metrics[f"eval/{_lab}/avg_arc_agi_3_format_score"] = metrics[f"eval/{_lab}/arc_meanN"]  # legacy alias
    eval_str = (f" eval/lvls(mean/best)={metrics['eval/avg_completed_levels']:.2f}/{metrics['eval/avg_completed_levels_bestN']:.2f}"
                f" arc(mean/best/std)={metrics['eval/arc_meanN']:.1f}/{metrics['eval/arc_bestN']:.1f}/{metrics['eval/arc_std']:.1f}"
                f" steps/lvl={metrics['eval/avg_steps_per_level']:.0f}"
                f" steps/solved={metrics['eval/avg_steps_per_solved_level']:.0f}(n={len(ev)})")
    for _lab in egroups:
        _k = f"eval/{_lab}/avg_completed_levels"
        if _k in metrics: eval_str += f" {_lab}/lvls={metrics[_k]:.2f}"
    return metrics, eval_str


async def main(model_name, game_ids, val_game_ids, load_checkpoint_path, batch_size, learning_rate,
               lora_rank, max_steps, max_levels, max_generate_length, max_input_length,
               max_sequence_length, n_samples_per_prompt, eval_every, seed, wandb_project, wandb_name,
               temperature, top_p, stop_sequences, loss_fn, inject_mode, agent_config_path,
               max_orai_steps, seed_mode="off", seed_frac=0.0, max_concurrent=8, save_every=5,
               eval_groups="", wandb_run_id="", start_step=0, eval_n_samples=5, eval_seed_base=7_000_000,
               eval_greedy=True, eval_max_orai_steps=250):
    set_seed(seed)
    games = [g.strip() for g in game_ids.split(",") if g.strip()]
    vals = [g.strip() for g in val_game_ids.split(",") if g.strip()]
    # named held-out subgroups → respective eval aggregates, e.g. "composites=tw08,tw11;variants=tw09,tw10"
    egroups = {}
    for _part in (eval_groups or "").split(";"):
        _part = _part.strip()
        if not _part or "=" not in _part: continue
        _lab, _gs = _part.split("=", 1)
        _set = {g.strip() for g in _gs.split(",") if g.strip()}
        if _lab.strip() and _set: egroups[_lab.strip()] = _set
    agent_config = _load_agent_config(agent_config_path)
    frontier = None
    if seed_mode in ("goexplore", "preseed"):         # Go-Explore dense-signal seeding (goal-3)
        from oversight import WitnessOracle
        from tinker_backend.frontier import FrontierBuffer
        frontier = FrontierBuffer(WitnessOracle())
        if seed_mode == "preseed":                    # option D: oracle-solution pre-seed (the random
            n = frontier.preseed_from_oracle(games, seed=seed)   # harvest filter is blind to witness — 2026-06-24)
            print(f"[preseed] frontier from oracle solutions: added={n} stats={frontier.stats()}", flush=True)
    # wandb_run_id set → RESUME that run (continue its curves); else a fresh run. start_step offsets
    # every logged/saved/eval step so a resumed run's points land AFTER the previous run's last step.
    # Resume the SAME wandb run only when a run-id is given; for a FRESH run pass NEITHER id nor resume
    # (wandb 0.28 raises "Run ID cannot be empty" if id=None/resume=None are passed explicitly).
    _wb_resume = {"id": wandb_run_id, "resume": "allow"} if wandb_run_id else {}
    wandb.init(project=wandb_project, name=wandb_name, **_wb_resume, config={
        "model_name": model_name, "n_games": len(games), "lora_rank": lora_rank,
        "n_samples_per_prompt": n_samples_per_prompt, "loss_fn": loss_fn, "inject_mode": inject_mode,
        "learning_rate": learning_rate, "backend": "tinker"})

    service_client = tinker.ServiceClient()                         # reads TINKER_API_KEY / TINKER_API_URL
    # async client creation (we're inside async main; sync variants warn "deadlock from async ctx")
    if load_checkpoint_path:
        training_client = await service_client.create_training_client_from_state_async(load_checkpoint_path)
    else:
        training_client = await service_client.create_lora_training_client_async(base_model=model_name, rank=lora_rank)
    adam = types.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    common = dict(max_levels=max_levels, max_generate_length=max_generate_length,
                  max_input_length=max_input_length, temperature=temperature, top_p=top_p,
                  stop_sequences=stop_sequences or [], max_orai_steps=max_orai_steps)

    # iterate over GLOBAL step numbers so a resumed run continues (start_step..start_step+max_steps-1);
    # start_step=0 → range(0, max_steps), the original behavior. Every step usage below (wandb.log,
    # save_state name, eval cadence, seeds, prints) is thus the continued/global step, for free.
    for step in range(start_step, start_step + max_steps):
        t0 = time.time()
        sampling_path = training_client.save_weights_for_sampler(name=f"step_{step:06d}").result().path
        sampling_client = service_client.create_sampling_client(model_path=sampling_path)
        # inject anneal uses LOCAL progress (step-start_step) so a resume doesn't start past the schedule
        inject_p = max(0.0, 1.0 - (step - start_step) / (0.4 * max_steps)) if inject_mode != "off" else 0.0

        trajs = await collect_batch(games, agent_config, sampling_client, tokenizer,
                                    n_samples_per_prompt, seed_base=1000 * (step + 1),
                                    max_concurrent=max_concurrent,
                                    inject_mode=inject_mode, inject_p=inject_p,
                                    frontier=frontier, seed_frac=seed_frac, **common)
        valid = [t for t in trajs if t.calls and not t.error]
        if not valid:
            print(f"[step {step}] no valid trajectories, skipping"); continue

        rewards = [t.reward for t in valid]
        advantages, signal_metrics = compute_grpo_advantages_and_signal(valid, normalize=True)
        t_roll = time.time()                       # rollout (sampling) done

        percall_rollouts, percall_advs = [], []
        for traj, adv in zip(valid, advantages):
            o, a = _traj_to_percall_rollouts(traj, adv)
            percall_rollouts += o; percall_advs += a
        datums, truncated = prepare_training_data(percall_rollouts, percall_advs, tokenizer, max_sequence_length)
        if not datums:
            print(f"[step {step}] no datums, skipping"); continue

        fb_res = training_client.forward_backward(datums, loss_fn=loss_fn).result()
        training_client.optim_step(adam).result()
        t_train = time.time()                      # train (fwd/bwd + optim) done
        # save_state periodically → resumable across interruptions + an evaluable tinker:// ckpt
        # for the held-out official arc_score (blueprint T1 primary endpoint).
        if save_every > 0 and (step % save_every == 0 or step == max_steps - 1):
            try:
                sp = training_client.save_state(name=f"{wandb_name}_s{step:03d}").result()
                path = getattr(sp, "path", None) or getattr(sp, "state_path", sp)
                print(f"[step {step}] saved state -> {path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[step {step}] save_state failed: {e}", flush=True)

        fb_metrics = getattr(fb_res, "metrics", None) or {}
        metrics = {"step": step,
                   "time/total": time.time() - t0, "time/rollout": t_roll - t0, "time/train": t_train - t_roll,
                   "rollouts/truncated": truncated, "rollouts/valid": len(valid),
                   "rollouts/trunc_frac": truncated / max(1, len(datums)),
                   "reward/avg_raw_reward": float(np.mean(rewards)), "reward/std": float(np.std(rewards)),
                   "reward/min": float(np.min(rewards)), "reward/max": float(np.max(rewards)),
                   "reward/avg_pass_at_n": compute_pass_at_n([t.model_dump() for t in valid], n_samples_per_prompt),
                   "advantage/mean": float(np.mean(advantages)), "advantage/std": float(np.std(advantages)),
                   "advantage/min": float(np.min(advantages)), "advantage/max": float(np.max(advantages)),
                   "datums/count": len(datums), "learning_rate": learning_rate, "inject/p": inject_p}
        # tinker forward_backward metrics (loss + whatever the loss_fn exposes) -> train/*  (SkyRL logs policy loss)
        for k, v in fb_metrics.items():
            try: metrics[f"train/{k}"] = float(v)
            except (TypeError, ValueError): pass
        metrics.update(compute_per_env_metrics([t.model_dump() for t in valid], n_samples_per_prompt))
        metrics.update(signal_metrics)
        # outcome-only reward signal (geometric off): mean per-rollout outcome reward over the train batch
        metrics["reward/avg_outcome_reward"] = float(np.mean([t.reward_breakdown.get("outcome", 0.0) for t in valid]))
        metrics["reward/avg_arc_agi_3_format_score"] = float(np.mean([t.arc_score for t in valid]))  # official RHAE, 0–115
        if frontier is not None:
            metrics.update({f"frontier/{k}": v for k, v in frontier.stats().items()})

        # Held-out eval — MERGE into the SAME wandb.log as the training metrics (one commit/step).
        # A separate wandb.log(step=step) AFTER the training commit is silently dropped by wandb
        # ("step must be monotonically increasing"), losing the decisive held-out metric (2026-06-24).
        # Also echoed into the [step] line so it lands in the node logs (log-reconstruction fallback).
        # Uses sampling_client = this step's pre-update weights (eval@entry-to-step convention).
        # eval_n_samples rollouts/game (default 5) averaged in _compute_eval_metrics → smoother curves;
        # seed_base is CONSTANT (not +step) so the eval set is FIXED across steps/resumes/ablation arms
        # (a proper validation set → clearer ablation signal; matches the frontier-model eval's 5 seeds).
        eval_str = ""
        if eval_every > 0 and vals and step % eval_every == 0:
            ev = await collect_batch(vals, agent_config, sampling_client, tokenizer, eval_n_samples,
                                     seed_base=eval_seed_base, max_concurrent=max_concurrent,
                                     inject_mode="off", inject_p=0.0, **{**common, "max_orai_steps": eval_max_orai_steps})
            ev = [t for t in ev if t.calls and not t.error]
            if ev:
                em, eval_str = _compute_eval_metrics(ev, egroups)
                metrics.update(em)
            # GREEDY eval: 1 deterministic rollout/game (temperature 0). Near-reproducible (seed is a
            # no-op for witness games; only continuous-batching FP wobbles), so it's a clean low-variance
            # tracking signal — complements the noisy 5-seed sampled mean/Bo5. Logged under eval/greedy/*.
            if eval_greedy:
                evg = await collect_batch(vals, agent_config, sampling_client, tokenizer, 1,
                                          seed_base=eval_seed_base, max_concurrent=max_concurrent,
                                          inject_mode="off", inject_p=0.0, **{**common, "temperature": 0.0, "max_orai_steps": eval_max_orai_steps})
                evg = [t for t in evg if t.calls and not t.error]
                if evg:
                    emg, _gs = _compute_eval_metrics(evg, egroups)
                    metrics.update({k.replace("eval/", "eval/greedy/", 1): v for k, v in emg.items()})
                    eval_str += f" | greedy_arc={emg.get('eval/arc_meanN', 0.0):.1f}"

        wandb.log(metrics, step=step, commit=True)
        print(f"[step {step}] reward={metrics['reward/avg_raw_reward']:.3f} "
              f"pass@{n_samples_per_prompt}={metrics['reward/avg_pass_at_n']:.3f} "
              f"solve_var={metrics['groups/frac_solve_var']:.2f} datums={len(datums)}{eval_str}")

    # FINAL eval at step == max_steps. The periodic eval above uses each step's PRE-update
    # weights (eval@entry-to-step), so the fully-trained policy — after the last optim_step —
    # is otherwise never scored (the loop ends at max_steps-1). Evaluate the final weights and
    # log at step=max_steps so every run ends on a held-out number for the final policy.
    final_step = start_step + max_steps
    if eval_every > 0 and vals:
        final_path = training_client.save_weights_for_sampler(name=f"step_{final_step:06d}").result().path
        final_sampling = service_client.create_sampling_client(model_path=final_path)
        ev = await collect_batch(vals, agent_config, final_sampling, tokenizer, eval_n_samples,
                                 seed_base=eval_seed_base, max_concurrent=max_concurrent,
                                 inject_mode="off", inject_p=0.0, **{**common, "max_orai_steps": eval_max_orai_steps})
        ev = [t for t in ev if t.calls and not t.error]
        if ev:
            em, eval_str = _compute_eval_metrics(ev, egroups)
            evg = []
            if eval_greedy:      # greedy (temp 0, 1 rollout/game) → eval/greedy/* in the dumped JSON too
                evg = await collect_batch(vals, agent_config, final_sampling, tokenizer, 1,
                                          seed_base=eval_seed_base, max_concurrent=max_concurrent,
                                          inject_mode="off", inject_p=0.0, **{**common, "temperature": 0.0, "max_orai_steps": eval_max_orai_steps})
                evg = [t for t in evg if t.calls and not t.error]
                if evg:
                    emg, _gs = _compute_eval_metrics(evg, egroups)
                    em.update({k.replace("eval/", "eval/greedy/", 1): v for k, v in emg.items()})
            em["step"] = final_step
            wandb.log(em, step=final_step, commit=True)
            # dump the final-eval metrics to a JSON so eval-only runs (MAX_STEPS=0) are trivially
            # collectable without the wandb API — one self-contained file per run.
            try:
                import json as _json
                # "detail"/"greedy_detail": per-rollout per-level [lvl_idx, completed, actions] (scorer's
                # deduped view) → the eval is RE-SCORABLE offline under any baseline convention (k×optimal
                # sensitivity, human recalibration) without re-running rollouts.
                _detail = [{"game": t.task_key, "arc": t.arc_score, "levels": t.level_detail} for t in ev]
                _gdetail = [{"game": t.task_key, "arc": t.arc_score, "levels": t.level_detail} for t in evg]
                with open(f"eval_metrics_{wandb_name}.json", "w") as _f:
                    _json.dump({"run": wandb_name, "model": model_name, "ckpt": load_checkpoint_path,
                                "step": final_step, "eval_n_samples": eval_n_samples,
                                "eval_seed_base": eval_seed_base, "metrics": em,
                                "detail": _detail, "greedy_detail": _gdetail}, _f, indent=2)
                print(f"[step {final_step}] eval metrics -> eval_metrics_{wandb_name}.json", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[eval-dump] failed: {_e}", flush=True)
            print(f"[step {final_step}] FINAL eval{eval_str}", flush=True)
    wandb.finish()
    print("training complete")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Witness GRPO RL on cloud-tinker (ports Deniz's loop)")
    p.add_argument("--model-name", default="Qwen/Qwen3.5-9B")
    p.add_argument("--load-checkpoint-path", default=None, help="tinker:// SFT-LoRA warm-start")
    p.add_argument("--game-ids", default="tw07,tw03,tw06,tw02,tw05,tw08,tw11,tw13")
    p.add_argument("--val-game-ids", default="tw01,tw09,tw10,tw12")
    p.add_argument("--eval-groups", default="",
                   help='named held-out subgroups for respective eval/<label>/* aggregates, e.g. '
                        '"composites=tw08,tw11,tw12,tw13;variants=tw09,tw10"')
    p.add_argument("--batch-size", type=int, default=8)         # = #games per step (all, here)
    p.add_argument("--learning-rate", type=float, default=2.0e-6)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-levels", type=int, default=5)
    p.add_argument("--max-orai-steps", type=int, default=30, help="ORAI calls/rollout cap (SkyRL MAX_ORAI_STEPS=30)")
    p.add_argument("--max-generate-length", type=int, default=2048)
    p.add_argument("--max-input-length", type=int, default=16384)
    p.add_argument("--max-sequence-length", type=int, default=18432)
    p.add_argument("--n-samples-per-prompt", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--eval-n-samples", type=int, default=5,
                   help="rollouts per held-out game at eval, averaged for smoother curves (default 5; "
                        "fixed seed set across steps → comparable, consistent with the frontier-model eval)")
    p.add_argument("--eval-seed-base", type=int, default=7_000_000,
                   help="base seed for held-out eval instances (seed = base + game/sample idx). Default "
                        "7_000_000 (constant, comparable across steps). Set to 7_000_000+step to replicate "
                        "the OLD in-RL eval seed convention for a like-for-like comparison.")
    p.add_argument("--eval-max-orai-steps", type=int, default=250,
                   help="ORAI-call cap for EVAL rollouts (in-loop, final, greedy) — decoupled from the "
                        "training rollout cap (--max-orai-steps, recipe 150). 250 = the unified eval "
                        "budget, matched by evaluate.py --max-orai-calls for API models (2026-07-02).")
    p.add_argument("--eval-greedy", type=int, default=1,
                   help="also run a greedy (temperature 0) 1-rollout/game eval each cadence + final, logged "
                        "under eval/greedy/* — a near-deterministic low-variance tracking signal alongside "
                        "the sampled mean/Bo5. 1=on (default), 0=off.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb-project", default="arc-agi-3")
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--stop-sequences", default="[]", help='JSON list, e.g. ["</tool_call>"]')
    p.add_argument("--loss-fn", default="importance_sampling", help="importance_sampling | ppo")
    p.add_argument("--inject-mode", default="off", help="off | rules (A1-train) | shuf (A1-shuf)")
    p.add_argument("--agent-config-path", default=None)
    p.add_argument("--seed-mode", default="off",
                   help="off | goexplore (online random-harvest seeding; barely fires — 2026-06-24 probe) | "
                        "preseed (option D: oracle-solution pre-seeded frontier + online harvest)")
    p.add_argument("--seed-frac", type=float, default=0.25, help="fraction of each GRPO group seeded from the frontier")
    p.add_argument("--max-concurrent", type=int, default=8, help="concurrent rollouts (lower when running many arms on one node to stay under tinker's rate limit)")
    p.add_argument("--save-every", type=int, default=5, help="save_state every N steps (resumable + evaluable tinker:// ckpt)")
    p.add_argument("--wandb-run-id", default="", help="resume this wandb run id (continue its curves) instead of a new run")
    p.add_argument("--start-step", type=int, default=0,
                   help="global step of the first iteration; for a continued run set it to the previous "
                        "run's last step so curves/ckpts/evals continue (e.g. resume from s020 -> --start-step 21)")
    a = p.parse_args()
    import json as _json
    nm = a.wandb_name or f"witness_grpo_tinker_{a.inject_mode}_{datetime.now().strftime('%m%d_%H%M')}"
    asyncio.run(main(
        model_name=a.model_name, game_ids=a.game_ids, val_game_ids=a.val_game_ids,
        load_checkpoint_path=a.load_checkpoint_path, batch_size=a.batch_size,
        learning_rate=a.learning_rate, lora_rank=a.lora_rank, max_steps=a.max_steps,
        max_levels=a.max_levels, max_generate_length=a.max_generate_length,
        max_input_length=a.max_input_length, max_sequence_length=a.max_sequence_length,
        n_samples_per_prompt=a.n_samples_per_prompt, eval_every=a.eval_every, seed=a.seed,
        wandb_project=a.wandb_project, wandb_name=nm, temperature=a.temperature, top_p=a.top_p,
        stop_sequences=_json.loads(a.stop_sequences), loss_fn=a.loss_fn, inject_mode=a.inject_mode,
        agent_config_path=a.agent_config_path, max_orai_steps=a.max_orai_steps,
        seed_mode=a.seed_mode, seed_frac=a.seed_frac, max_concurrent=a.max_concurrent,
        save_every=a.save_every, eval_groups=a.eval_groups,
        wandb_run_id=a.wandb_run_id, start_step=a.start_step, eval_n_samples=a.eval_n_samples,
        eval_seed_base=a.eval_seed_base, eval_greedy=bool(a.eval_greedy),
        eval_max_orai_steps=a.eval_max_orai_steps))
