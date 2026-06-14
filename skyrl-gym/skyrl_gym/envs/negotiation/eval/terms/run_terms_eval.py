#!/usr/bin/env python3
"""Offline TERMS-Bench evaluation harness for negotiation checkpoints.

Faithful reconstruction of the bilateral price-negotiation instantiation of TERMS-Bench
(Zhang et al., 2026, arXiv:2605.13909). The evaluated policy is served through an
OpenAI-compatible chat endpoint (local vLLM, OpenRouter, etc.) and negotiates, over many
seeded episodes, against the *fixed stochastic simulator counterpart* implemented in
``kernel.py``. The environment is the verifier: every metric is computed from the
simulator state and the agent's self-reported belief -- there is NO LLM grader. The
counterpart's natural-language "voice" is cosmetic (templated by default; ``--voice-model``
optionally renders it but never changes the economic outcome).

Cost: evaluating a locally-served checkpoint is ~free in API terms (counterpart = simulator,
policy served locally). See README.md / SPEC.md.

Examples:
  # quick snapshot (144 episodes balanced across regime x family x role x opener)
  python3 run_terms_eval.py --model <ckpt> --base-url http://localhost:8000/v1 --n 144 --no-think

  # full 1,800-episode suite
  python3 run_terms_eval.py --model <ckpt> --base-url http://localhost:8000/v1 --full-suite

  # offline self-test (scripted agent, no API calls)
  python3 run_terms_eval.py --dry-run --n 72
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
# Put THIS dir first so the local config/kernel/scenarios/prompts/metrics shadow any
# similarly-named modules in the parent negotiation package (e.g. scenarios.py/prompts.py).
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import kernel  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import prompts  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402
from config import (  # noqa: E402
    AgentAction,
    CounterpartMove,
    EpisodeResult,
    RoundLog,
    Scenario,
    TermsConfig,
)

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - only needed for live runs
    AsyncOpenAI = None


NO_THINK_TOKEN = "/no_think"
NO_THINK_BODY = {"reasoning": {"enabled": False}}


# ----------------------------------------------------------------------------------
# Endpoint plumbing (mirrors run_tom_eval.py).
# ----------------------------------------------------------------------------------
def make_client(base_url: str, timeout: float = 90.0, max_retries: int = 2):
    if AsyncOpenAI is None:
        raise RuntimeError("`openai` package required for endpoint calls (pip install openai).")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


async def chat(client, model, messages, temperature, max_tokens=512, retries=4, extra_body=None):
    """Robust chat call that adapts to provider parameter quirks."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if temperature is not None and temperature >= 0:
        kwargs["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            if not getattr(resp, "choices", None):
                return ""
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            if "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
            elif "max_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif ("reasoning" in msg or "extra_body" in msg) and "extra_body" in kwargs:
                kwargs.pop("extra_body")
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return ""


def _maybe_no_think(text: str, no_think: bool) -> str:
    return text + "\n\n" + NO_THINK_TOKEN if no_think else text


def sanitize_model_name(model: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model).strip("_") or "model"


def redact_base_url(url: str) -> str:
    try:
        parts = urlsplit(url)
        if not parts.scheme or not parts.netloc:
            return url
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, parts.query, parts.fragment))
    except Exception:  # noqa: BLE001
        return url


# ----------------------------------------------------------------------------------
# Agent backends: a backend maps a list of chat messages -> raw assistant text.
# ----------------------------------------------------------------------------------
class LLMAgent:
    def __init__(self, client, model, temperature, max_tokens, no_think):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.no_think = no_think
        self.body = NO_THINK_BODY if no_think else None

    async def act(self, messages: list[dict]) -> str:
        return await chat(
            self.client, self.model, messages, self.temperature,
            max_tokens=self.max_tokens, extra_body=self.body,
        )


class ScriptedAgent:
    """Deterministic heuristic agent for offline self-test (no API).

    Concedes gradually toward its reservation and accepts a counterpart offer that is
    individually rational and not improving much. Emits a valid JSON action (so it flows
    through the same parser/violation path as a real model)."""

    async def act(self, messages: list[dict]) -> str:
        user = json.loads(messages[-1]["content"])
        role = user["private_context"]["role"]
        r_a = float(user["private_context"]["reservation_price"])
        p_min, p_max = user["constraints"]["price_bounds"]
        cp_offer = user["observation"]["counterpart_offer"]
        own_prev = user["protocol_state"]["own_previous_offer"]
        k = user["protocol_state"]["round_number"]
        K = user["protocol_state"]["max_rounds"]
        is_buyer = role == "buyer"

        # Accept if the offer on the table is individually rational and we're late or it's good.
        if cp_offer is not None:
            ir_ok = (cp_offer <= r_a) if is_buyer else (cp_offer >= r_a)
            late = k >= max(2, K - 2)
            margin = (r_a - cp_offer) if is_buyer else (cp_offer - r_a)
            if ir_ok and (late or margin >= 0.10 * (p_max - p_min)):
                return json.dumps({
                    "decision": "Accept", "price": None, "message": "Agreed.",
                    "belief": _scripted_belief(r_a, p_min, p_max, is_buyer),
                })

        # Otherwise make a conceding offer toward r_a.
        if own_prev is None:
            # Opening: anchor away from reservation by ~30% of range.
            offer = (r_a - 0.30 * (p_max - p_min)) if is_buyer else (r_a + 0.30 * (p_max - p_min))
        else:
            step = 0.10 * (p_max - p_min)
            offer = (own_prev + step) if is_buyer else (own_prev - step)
            # Never cross own reservation (IR) and keep monotone.
            offer = min(offer, r_a) if is_buyer else max(offer, r_a)
        offer = max(p_min, min(p_max, offer))
        # Round to the individually-rational side so a 2-dp price never crosses r_a
        # (a buyer floors, a seller ceils) -- otherwise rounding can trip the strict-IR check.
        price = math.floor(offer * 100) / 100 if is_buyer else math.ceil(offer * 100) / 100
        return json.dumps({
            "decision": "Offer", "price": price, "message": "Here is my offer.",
            "belief": _scripted_belief(r_a, p_min, p_max, is_buyer),
        })


def _scripted_belief(r_a, p_min, p_max, is_buyer) -> dict:
    # Guess the counterpart reservation sits on the other side of our own.
    r_hat = (p_max + r_a) / 2 if is_buyer else (p_min + r_a) / 2
    return {
        "r_hat": round(r_hat, 2), "kappa_hat": 0.5,
        "stance_probs": {"conciliatory": 0.34, "neutral": 0.33, "aggressive": 0.33},
    }


class VoiceLLM:
    """Optional cosmetic voice layer; renders a counterpart message but never changes the
    economic action. Falls back to the deterministic template on any failure."""

    def __init__(self, client, model):
        self.client = client
        self.model = model

    async def render(self, move: CounterpartMove, scenario: Scenario, is_opening: bool) -> str:
        fallback = prompts.template_voice(move, scenario, is_opening)
        if self.client is None or not self.model:
            return fallback
        sys_prompt = (
            "You are the natural-language voice of a simulated negotiation counterpart. "
            "The economic decision and price are ALREADY FIXED; only write a short (1-2 "
            "sentence) message consistent with them. Never change the action or price, never "
            "introduce new numbers, never reveal hidden info."
        )
        price_str = f"{move.price:.2f}" if move.price is not None else "(none)"
        user = (
            f"role={scenario.counterpart_role}; opening={is_opening}; "
            f"action={move.decision}; price={price_str}; "
            f"sentiment={move.sentiment}; strategy_cue={move.strategy_cue}. "
            "If the action is an Offer, include the exact price. Output only the message text."
        )
        try:
            txt = await chat(
                self.client, self.model,
                [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}],
                temperature=0.7, max_tokens=80,
            )
            txt = (txt or "").strip()
            if move.decision == "Offer" and move.price is not None and price_str not in txt:
                return fallback  # voice dropped/altered the price -> use safe template
            return txt or fallback
        except Exception:  # noqa: BLE001
            return fallback


# ----------------------------------------------------------------------------------
# Single-episode rollout.
# ----------------------------------------------------------------------------------
async def play_episode(scenario: Scenario, cfg: TermsConfig, agent, voice, no_think: bool) -> EpisodeResult:
    rng = random.Random(scenario.seed)
    cp = kernel.Counterpart(scenario, cfg, rng)

    system = _maybe_no_think(prompts.build_system_prompt(scenario, cfg), no_think)
    messages: list[dict] = [{"role": "system", "content": system}]

    rounds: list[RoundLog] = []
    history: list[dict] = []
    belief_samples: list[dict] = []
    violation_tags: list[str] = []
    critical = False

    agent_offers: list[float] = []
    counterpart_offers: list[float] = []
    current_cp_offer: float | None = None
    current_cp_message: str | None = None
    own_previous_offer: float | None = None
    agent_opening_price: float | None = None

    def _record_critical(tags: list[str]):
        nonlocal critical
        for t in tags:
            violation_tags.append(t)
            if t in ("price_bound", "ir", "invalid_action"):
                critical = True

    # Counterpart opens (if applicable).
    if scenario.opener == "CounterpartOpens":
        opening = cp.opening_offer()
        current_cp_offer = opening.price
        counterpart_offers.append(opening.price)
        msg = await voice.render(opening, scenario, is_opening=True)
        current_cp_message = msg
        rounds.append(RoundLog(0, "counterpart", "Offer", opening.price, msg,
                               opening.sentiment, opening.strategy_cue))
        history.append({"round": 0, "actor": "counterpart", "decision": "Offer",
                        "price": opening.price, "message": msg})

    agreed = False
    terminal_price: float | None = None
    termination = "Timeout"

    try:
        for k in range(1, cfg.K + 1):
            user_msg = prompts.build_user_message(
                scenario, k,
                counterpart_offer=current_cp_offer,
                counterpart_message=current_cp_message,
                own_previous_offer=own_previous_offer,
                history=history,
                cfg=cfg,
            )
            messages.append({"role": "user", "content": user_msg})
            raw = await agent.act(messages)
            messages.append({"role": "assistant", "content": raw})

            action = prompts.parse_agent_action(raw, scenario, own_previous_offer, cfg)
            crit, sec = prompts.detect_violations(action, scenario, own_previous_offer, cfg)
            _record_critical(crit)

            # Record belief (only if the agent reported something parseable).
            b = action.belief
            if b.r_hat is not None or b.kappa_hat is not None or b.stance_probs is not None:
                belief_samples.append({
                    "r_hat": b.r_hat, "kappa_hat": b.kappa_hat, "stance_probs": b.stance_probs,
                })

            rounds.append(RoundLog(k, "agent", action.decision, action.price, action.message,
                                   belief=b, violations=crit + sec))
            history.append({"round": k, "actor": "agent", "decision": action.decision,
                            "price": action.price, "message": action.message})

            # Unparseable / illegal -> invalid action terminates with no deal.
            if action.parse_error is not None:
                termination = "Error"
                break

            if action.decision == "Accept":
                if current_cp_offer is None:
                    _record_critical(["invalid_action"])
                    termination = "Error"
                    break
                agreed = True
                terminal_price = current_cp_offer
                termination = "Agreement"
                break

            if action.decision == "Reject":
                termination = "AgentReject"
                break

            # Offer.
            price = action.price
            if price is None:
                _record_critical(["invalid_action"])
                termination = "Error"
                break
            # Clamp into bounds for continuation (the violation is already recorded).
            price = max(scenario.p_min, min(scenario.p_max, price))
            agent_offers.append(price)
            if agent_opening_price is None:
                agent_opening_price = price
            own_previous_offer = price

            move = cp.respond(price, k, list(agent_offers), list(counterpart_offers))

            if move.decision == "Accept":
                agreed = True
                terminal_price = price
                termination = "Agreement"
                cp_msg = await voice.render(move, scenario, is_opening=False)
                rounds.append(RoundLog(k, "counterpart", "Accept", None, cp_msg,
                                       move.sentiment, move.strategy_cue))
                history.append({"round": k, "actor": "counterpart", "decision": "Accept",
                                "price": None, "message": cp_msg})
                break

            if move.decision == "Reject":
                termination = "CounterpartWalkAway"
                cp_msg = await voice.render(move, scenario, is_opening=False)
                rounds.append(RoundLog(k, "counterpart", "Reject", None, cp_msg,
                                       move.sentiment, move.strategy_cue))
                history.append({"round": k, "actor": "counterpart", "decision": "Reject",
                                "price": None, "message": cp_msg})
                break

            # Counter-offer.
            is_open = not counterpart_offers
            current_cp_offer = move.price
            counterpart_offers.append(move.price)
            cp_msg = await voice.render(move, scenario, is_opening=is_open)
            current_cp_message = cp_msg
            rounds.append(RoundLog(k, "counterpart", "Offer", move.price, cp_msg,
                                   move.sentiment, move.strategy_cue))
            history.append({"round": k, "actor": "counterpart", "decision": "Offer",
                            "price": move.price, "message": cp_msg})

        util = metrics_mod.episode_utility(scenario, agreed, terminal_price)
        return EpisodeResult(
            scenario=scenario, rounds=rounds, agreed=agreed, terminal_price=terminal_price,
            termination=termination, agent_utility=util, critical_violation=critical,
            violation_tags=violation_tags, agent_opening_price=agent_opening_price,
            belief_samples=belief_samples,
        )
    except Exception as e:  # noqa: BLE001
        return EpisodeResult(
            scenario=scenario, rounds=rounds, agreed=False, terminal_price=None,
            termination="Error", agent_utility=0.0, critical_violation=critical,
            violation_tags=violation_tags, agent_opening_price=agent_opening_price,
            belief_samples=belief_samples, error=str(e),
        )


# ----------------------------------------------------------------------------------
# Serialization helpers.
# ----------------------------------------------------------------------------------
def _round_to_dict(r: RoundLog) -> dict:
    d = {"k": r.k, "actor": r.actor, "decision": r.decision, "price": r.price,
         "message": r.message, "sentiment": r.sentiment, "strategy_cue": r.strategy_cue,
         "violations": r.violations}
    if r.belief is not None and (r.belief.r_hat is not None or r.belief.stance_probs is not None):
        d["belief"] = {"r_hat": r.belief.r_hat, "kappa_hat": r.belief.kappa_hat,
                       "stance_probs": r.belief.stance_probs}
    return d


def _episode_to_dict(res: EpisodeResult, include_rounds: bool) -> dict:
    sc = res.scenario
    d = {
        "episode_id": sc.episode_id,
        "regime": sc.regime, "family": sc.family,
        "agent_role": sc.agent_role, "opener": sc.opener,
        "r_agent": sc.r_agent, "r_counterpart": sc.r_counterpart,
        "kappa_B": sc.kappa_B, "eta_B": sc.eta_B, "delta": sc.delta, "feasible": sc.feasible,
        "agreed": res.agreed, "terminal_price": res.terminal_price,
        "termination": res.termination, "agent_utility": res.agent_utility,
        "critical_violation": res.critical_violation, "violation_tags": res.violation_tags,
        "agent_opening_price": res.agent_opening_price,
        "n_rounds": len(res.rounds), "error": res.error,
    }
    if include_rounds:
        d["rounds"] = [_round_to_dict(r) for r in res.rounds]
    return d


# ----------------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------------
async def main_async(args):
    cfg = TermsConfig(K=args.max_rounds) if args.max_rounds else config.DEFAULT_CONFIG
    scs = scenarios_mod.generate_scenarios(
        n=args.n, base_seed=args.seed, cfg=cfg, full_suite=args.full_suite,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        agent = ScriptedAgent()
        voice = VoiceLLM(None, None)
        model_label = "scripted-dry-run"
        base_url_label = "(none)"
    else:
        client = make_client(args.base_url)
        agent = LLMAgent(client, args.model, args.temperature, args.max_tokens, args.no_think)
        voice = VoiceLLM(client if args.voice_model else None, args.voice_model)
        model_label = args.model
        base_url_label = redact_base_url(args.base_url)

    sem = asyncio.Semaphore(args.concurrency)
    done = {"k": 0}
    total = len(scs)

    async def run_one(sc: Scenario) -> EpisodeResult:
        async with sem:
            res = await play_episode(sc, cfg, agent, voice, args.no_think)
            done["k"] += 1
            if done["k"] % 25 == 0 or done["k"] == total:
                print(f"  [terms] {done['k']}/{total} episodes", flush=True)
            return res

    t0 = time.time()
    print(f"running {total} TERMS-Bench episodes "
          f"({'full suite' if args.full_suite else f'n={args.n}'})...", flush=True)
    results = await asyncio.gather(*[run_one(sc) for sc in scs])
    elapsed = time.time() - t0

    agg = metrics_mod.compute_metrics(results, cfg)
    cfg_block = {
        "model": model_label, "base_url": base_url_label,
        "n": total, "full_suite": args.full_suite, "seed": args.seed,
        "K": cfg.K, "temperature": args.temperature, "max_tokens": args.max_tokens,
        "no_think": args.no_think, "voice_model": args.voice_model,
        "concurrency": args.concurrency, "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "Reconstructed from arXiv:2605.13909 (official code not public).",
    }
    payload = {
        "config": cfg_block,
        "metrics": agg,
        "episodes": [_episode_to_dict(r, include_rounds=args.save_transcripts) for r in results],
    }
    safe = sanitize_model_name(model_label)
    tag = "full" if args.full_suite else f"n{total}"
    out_path = out_dir / f"{safe}_terms_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print_report(cfg_block, agg)
    print(f"\nwrote {out_path}")
    return agg


def _fmt(x):
    return "  n/a" if x is None else f"{x:.4f}"


def print_report(cfg_block: dict, agg: dict) -> None:
    o = agg["overall"]
    print("\n" + "=" * 64)
    print("TERMS-Bench (reconstruction) — " + cfg_block["model"])
    print("=" * 64)
    print(f"  episodes      : {o['n']}  (feasible {o['n_feasible']} / infeasible {o['n_infeasible']})")
    print(f"  K             : {cfg_block['K']}   seed: {cfg_block['seed']}   elapsed_s: {cfg_block['elapsed_s']}")
    print("-" * 64)
    print(f"  SE+  (surplus efficiency)     : {_fmt(o['SE_plus'])}")
    print(f"     = AGR+ x CSE+              : {_fmt(o['AGR_plus'])} x {_fmt(o['CSE_plus'])}")
    print(f"  FAGR- (false agree, infeas.)  : {_fmt(o['FAGR_minus'])}   safe-term: {_fmt(o['safe_term_minus'])}")
    print(f"  BE_type (belief error)        : {_fmt(o['BE_type'])}   "
          f"(r {_fmt(o['BE_r'])} / kappa {_fmt(o['BE_kappa'])} / brier {_fmt(o['Brier_eta'])})")
    print(f"  CritViol%                     : {_fmt(o['CritViol_pct'])}")
    print(f"  mean utility (secondary)      : {_fmt(o['mean_utility'])}")
    print("-" * 64)
    print("  SE+ by family:")
    for fam, blk in agg["by_family"].items():
        print(f"    - {fam:<12}: SE+ {_fmt(blk['SE_plus'])}  AGR+ {_fmt(blk['AGR_plus'])}  "
              f"FAGR- {_fmt(blk['FAGR_minus'])}  BE_type {_fmt(blk['BE_type'])}")
    print("  SE+ by regime:")
    for reg, blk in agg["by_regime"].items():
        print(f"    - {reg:<14}: SE+ {_fmt(blk['SE_plus'])}  AGR+ {_fmt(blk['AGR_plus'])}  "
              f"FAGR- {_fmt(blk['FAGR_minus'])}")
    print("=" * 64)


def parse_args():
    p = argparse.ArgumentParser(description="TERMS-Bench reconstruction eval harness")
    p.add_argument("--model", default="scripted", help="Model name served at the endpoint")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--n", type=int, default=144, help="Total episodes (balanced across cells)")
    p.add_argument("--full-suite", action="store_true", help="Run the full 6x3x100=1,800 suite")
    p.add_argument("--seed", type=int, default=0, help="Base seed for the scenario set")
    p.add_argument("--max-rounds", type=int, default=0, help="Override horizon K (0 = config default 10)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max-tokens", type=int, default=512)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--no-think", action="store_true")
    p.add_argument("--voice-model", default=None,
                   help="Optional model slug for the cosmetic counterpart voice (default: templates).")
    p.add_argument("--save-transcripts", action="store_true",
                   help="Include full per-round transcripts in the results JSON.")
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--dry-run", action="store_true",
                   help="Offline self-test: scripted agent, no API calls.")
    return p.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
