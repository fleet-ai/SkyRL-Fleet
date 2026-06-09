#!/usr/bin/env python3
"""Baseline eval for the negotiation task via LLM self-play.

Runs each scenario as a 2-player negotiation (same model on both sides by
default), parses each side's final `<deal>`, and computes the *verifiable*
metrics we plan to use as RLVR rewards:

  - outcome reward  : normalized self-score (score / max possible)  [no-deal = 0]
  - pareto-optimal? : was the agreed split on the Pareto frontier?
  - joint efficiency: achieved joint score / best achievable joint

No GPU required -- uses an OpenAI-compatible endpoint (OpenRouter by default).

Example:
  OPENROUTER_API_KEY=... python3 run_eval.py \
      --model openai/gpt-4o-mini --dataset dnd --split val --n 40 --concurrency 8
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

# Import the dependency-free core game logic from the parent package dir.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import game  # noqa: E402
import prompts  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:
    print("ERROR: `openai` package required (pip install openai).", file=sys.stderr)
    raise


def make_client(base_url: str, timeout: float = 90.0, max_retries: int = 2) -> "AsyncOpenAI":
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY (or OPENAI_API_KEY).")
    # Bound each request: without a timeout a single stalled provider response
    # (seen with some OpenRouter-hosted models) blocks the whole asyncio.gather.
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


async def chat(client, model, messages, temperature, max_tokens=500, retries=4, extra_body=None):
    """Robust chat call that adapts to reasoning-model parameter quirks.

    Some frontier/reasoning models reject `temperature` or require
    `max_completion_tokens` instead of `max_tokens`; we strip/swap on error.
    """
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if temperature is not None and temperature >= 0:
        kwargs["temperature"] = temperature
    for attempt in range(retries):
        try:
            resp = await client.chat.completions.create(**kwargs)
            # Some providers (via OpenRouter) occasionally return 200 with empty
            # `choices` or null content; treat as an empty turn rather than crashing.
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
                kwargs.pop("extra_body")  # provider rejects the reasoning toggle; drop it
            if attempt == retries - 1:
                raise
            await asyncio.sleep(1.5 * (attempt + 1))
    return ""


# Qwen3 hybrid-reasoning soft switch: appending this token disables the thinking
# block for that turn. We also pass OpenRouter's reasoning toggle as a belt-and-braces.
NO_THINK_TOKEN = "/no_think"
NO_THINK_BODY = {"reasoning": {"enabled": False}}


def _maybe_no_think(system_prompt: str, no_think: bool) -> str:
    return system_prompt + "\n\n" + NO_THINK_TOKEN if no_think else system_prompt


async def play_game(client, sc, model_a, model_b, max_turns, temperature, max_tokens=500,
                    protocol="single", no_think=False):
    """Self-play one scenario under the chosen protocol.

    Agent A holds you_values, agent B holds them_values.
    """
    if protocol == "dual":
        return await _play_dual(client, sc, model_a, model_b, max_turns, temperature, max_tokens, no_think)
    return await _play_single(client, sc, model_a, model_b, max_turns, temperature, max_tokens, no_think)


def _scenario_dict(sc):
    return {
        "item_names": list(sc.item_names),
        "counts": list(sc.counts),
        "you_values": list(sc.you_values),
        "them_values": list(sc.them_values),
    }


async def _play_dual(client, sc, model_a, model_b, max_turns, temperature, max_tokens, no_think=False):
    """DUAL-TAG protocol: both agents must each emit a `<deal>` and the two
    claims must exactly partition the pool, else both score 0."""
    items = list(sc.item_names)
    body = NO_THINK_BODY if no_think else None
    sys_a = _maybe_no_think(prompts.build_system_prompt(items, list(sc.counts), list(sc.you_values), max_turns, protocol="dual"), no_think)
    sys_b = _maybe_no_think(prompts.build_system_prompt(items, list(sc.counts), list(sc.them_values), max_turns, protocol="dual"), no_think)

    hist_a = [{"role": "system", "content": sys_a},
              {"role": "user", "content": prompts.OPENING_USER_MSG}]
    hist_b = [{"role": "system", "content": sys_b}]

    last = {"a": None, "b": None}
    count = {"a": 0, "b": 0}
    transcript = []
    speaker = "a"

    while True:
        if speaker == "a":
            text = await chat(client, model_a, hist_a, temperature, max_tokens, extra_body=body)
            hist_a.append({"role": "assistant", "content": text})
            hist_b.append({"role": "user", "content": text})
            last["a"] = game.parse_deal(text, items)
            count["a"] += 1
        else:
            text = await chat(client, model_b, hist_b, temperature, max_tokens, extra_body=body)
            hist_b.append({"role": "assistant", "content": text})
            hist_a.append({"role": "user", "content": text})
            last["b"] = game.parse_deal(text, items)
            count["b"] += 1
        transcript.append({"speaker": "you" if speaker == "a" else "them", "text": text})

        if last["a"] is not None and last["b"] is not None:
            break
        if count["a"] >= max_turns and count["b"] >= max_turns:
            break
        speaker = "b" if speaker == "a" else "a"

    outcome = game.evaluate(list(sc.counts), list(sc.you_values), list(sc.them_values),
                            last["a"], last["b"])
    return {
        "scenario": _scenario_dict(sc),
        "num_turns": len(transcript),
        "transcript": transcript,
        "outcome": outcome.to_dict(),
        "protocol": "dual",
    }


async def _play_single(client, sc, model_a, model_b, max_turns, temperature, max_tokens, no_think=False):
    """SINGLE-PROPOSER protocol: one agent proposes a full split via `<propose>`
    (listing what THEY keep; partner gets the rest); the other agent finalizes it
    with `<accept>`. A single offer always partitions the pool, so the only
    failure mode is no_deal (nothing accepted within the budget)."""
    items = list(sc.item_names)
    counts = list(sc.counts)
    n = len(counts)
    body = NO_THINK_BODY if no_think else None
    sys_a = _maybe_no_think(prompts.build_system_prompt(items, counts, list(sc.you_values), max_turns, protocol="single"), no_think)
    sys_b = _maybe_no_think(prompts.build_system_prompt(items, counts, list(sc.them_values), max_turns, protocol="single"), no_think)

    hist_a = [{"role": "system", "content": sys_a},
              {"role": "user", "content": prompts.OPENING_USER_MSG}]
    hist_b = [{"role": "system", "content": sys_b}]

    count = {"a": 0, "b": 0}
    transcript = []
    speaker = "a"
    pending = None            # {"by": "a"/"b", "keep": [...]} most recent valid offer
    final_proposal = None     # last offer seen, for reporting on no-deal
    you_take = them_take = None

    while True:
        model = model_a if speaker == "a" else model_b
        hist = hist_a if speaker == "a" else hist_b
        text = await chat(client, model, hist, temperature, max_tokens, extra_body=body)
        (hist_a if speaker == "a" else hist_b).append({"role": "assistant", "content": text})
        (hist_b if speaker == "a" else hist_a).append({"role": "user", "content": text})
        count[speaker] += 1
        transcript.append({"speaker": "you" if speaker == "a" else "them", "text": text})

        # An <accept> finalizes the OTHER agent's pending offer (accept wins over a co-occurring propose).
        if game.has_accept(text) and pending and pending["by"] != speaker:
            keep = pending["keep"]
            other_take = [counts[i] - keep[i] for i in range(n)]
            if pending["by"] == "a":
                you_take, them_take = keep, other_take
            else:
                them_take, you_take = keep, other_take
            break

        prop = game.parse_proposal(text, items)
        if prop is not None:
            keep = [min(counts[i], max(0, prop[i])) for i in range(n)]
            pending = {"by": speaker, "keep": keep}
            final_proposal = {"by": "you" if speaker == "a" else "them", "keep": keep}

        if count["a"] >= max_turns and count["b"] >= max_turns:
            break
        speaker = "b" if speaker == "a" else "a"

    outcome = game.evaluate(counts, list(sc.you_values), list(sc.them_values), you_take, them_take)
    return {
        "scenario": _scenario_dict(sc),
        "num_turns": len(transcript),
        "transcript": transcript,
        "outcome": outcome.to_dict(),
        "final_proposal": final_proposal,
        "protocol": "single",
    }


def aggregate(results):
    n = len(results)
    if n == 0:
        return {}
    outs = [r["outcome"] for r in results]
    agreed = [o for o in outs if o["agreed"]]
    na = len(agreed)

    def rate(pred):
        return round(sum(1 for o in outs if pred(o)) / n, 4)

    self_norms = []
    for o in outs:
        # No-deal contributes 0 reward to both sides (terminal outcome reward).
        self_norms.append(o["you_norm"] if o["agreed"] else 0.0)
        self_norms.append(o["them_norm"] if o["agreed"] else 0.0)

    return {
        "n": n,
        "agreement_rate": rate(lambda o: o["agreed"]),
        "no_deal_rate": rate(lambda o: o["reason"] == "no_deal"),
        "conflict_rate": rate(lambda o: o["reason"] == "conflict"),
        "incomplete_rate": rate(lambda o: o["reason"] == "incomplete"),
        "avg_outcome_reward": round(sum(self_norms) / len(self_norms), 4),
        "avg_points_per_agent": round(
            sum(o["you_score"] + o["them_score"] for o in agreed) / (2 * na), 3) if na else 0.0,
        "pareto_rate_of_agreements": round(sum(o["pareto_optimal"] for o in agreed) / na, 4) if na else 0.0,
        "pareto_rate_overall": round(sum(o["pareto_optimal"] for o in agreed) / n, 4),
        "avg_joint_efficiency_of_agreements": round(
            sum(o["joint_efficiency"] for o in agreed) / na, 4) if na else 0.0,
        "avg_turns": round(sum(r["num_turns"] for r in results) / n, 2),
    }


def print_report(cfg, agg):
    print("\n" + "=" * 60)
    print("NEGOTIATION SELF-PLAY EVAL")
    print("=" * 60)
    print(f"  model         : {cfg['model']}")
    print(f"  protocol      : {cfg.get('protocol', 'single')}")
    print(f"  dataset/split : {cfg['dataset']}/{cfg['split']}")
    print(f"  scenarios     : {agg['n']}   max_turns/agent: {cfg['max_turns']}   temp: {cfg['temperature']}")
    print("-" * 60)
    print(f"  agreement rate          : {agg['agreement_rate']:.1%}")
    print(f"    no-deal / conflict / incomplete: "
          f"{agg['no_deal_rate']:.1%} / {agg['conflict_rate']:.1%} / {agg['incomplete_rate']:.1%}")
    print(f"  avg OUTCOME reward      : {agg['avg_outcome_reward']:.3f}   (normalized self-score, no-deal=0)")
    print(f"  avg points / agent      : {agg['avg_points_per_agent']:.2f}   (raw, on agreements)")
    print(f"  PARETO-optimal rate     : {agg['pareto_rate_of_agreements']:.1%} of agreements "
          f"({agg['pareto_rate_overall']:.1%} overall)")
    print(f"  joint efficiency        : {agg['avg_joint_efficiency_of_agreements']:.1%}   (achieved / best-possible joint)")
    print(f"  avg turns               : {agg['avg_turns']}")
    print("=" * 60 + "\n")


async def evaluate_model(
    model: str,
    dataset: str = "dnd",
    split: str = "val",
    n: int = 40,
    max_turns: int = 8,
    temperature: float = 0.7,
    concurrency: int = 8,
    seed: int = 1,
    base_url: str = "https://openrouter.ai/api/v1",
    partner_model: str = None,
    max_tokens: int = 500,
    out_dir: str = None,
    client=None,
    quiet: bool = False,
    label: str = "",
    protocol: str = "single",
    no_think: bool = False,
):
    """Run self-play eval for one (model, dataset, split). Returns a dict with
    keys: config, aggregate, results. Optionally writes a per-run JSON.
    """
    rng = random.Random(seed)
    scs = scenarios_mod.load_scenarios(dataset, split)
    rng.shuffle(scs)
    scs = scs[:n]

    client = client or make_client(base_url)
    partner = partner_model or model
    sem = asyncio.Semaphore(concurrency)
    done = {"k": 0}
    tag = label or model

    async def run_one(sc):
        async with sem:
            try:
                r = await play_game(client, sc, model, partner, max_turns, temperature, max_tokens, protocol, no_think)
            except Exception as e:  # noqa: BLE001
                r = {"error": str(e), "scenario": {"counts": list(sc.counts)}, "num_turns": 0,
                     "outcome": game.evaluate(list(sc.counts), list(sc.you_values),
                                              list(sc.them_values), None, None).to_dict()}
            done["k"] += 1
            if not quiet and (done["k"] % 10 == 0 or done["k"] == len(scs)):
                print(f"  [{tag} / {dataset}] {done['k']}/{len(scs)} games", flush=True)
            return r

    t0 = time.time()
    results = await asyncio.gather(*[run_one(sc) for sc in scs])
    elapsed = time.time() - t0

    agg = aggregate(results)
    n_err = sum(1 for r in results if "error" in r)
    cfg = {"model": model, "partner_model": partner, "dataset": dataset, "split": split,
           "n": len(scs), "max_turns": max_turns, "temperature": temperature,
           "max_tokens": max_tokens, "seed": seed, "elapsed_s": round(elapsed, 1),
           "n_errors": n_err, "label": label, "protocol": protocol, "no_think": no_think}
    payload = {"config": cfg, "aggregate": agg, "results": results}

    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        safe = model.replace("/", "_")
        proto_tok = f"{protocol}-nothink" if no_think else protocol
        (od / f"{safe}_{dataset}_{split}_{proto_tok}_n{len(scs)}.json").write_text(json.dumps(payload, indent=2))
    return payload


async def main_async(args):
    payload = await evaluate_model(
        model=args.model, dataset=args.dataset, split=args.split, n=args.n,
        max_turns=args.max_turns, temperature=args.temperature, concurrency=args.concurrency,
        seed=args.seed, base_url=args.base_url, partner_model=args.partner_model,
        max_tokens=args.max_tokens, out_dir=args.out_dir, protocol=args.protocol,
        no_think=args.no_think,
    )
    print_report(payload["config"], payload["aggregate"])
    print(f"(completed in {payload['config']['elapsed_s']}s, errors={payload['config']['n_errors']})")
    return payload["aggregate"]


def parse_args():
    p = argparse.ArgumentParser(description="Negotiation self-play eval")
    p.add_argument("--model", default="openai/gpt-4o-mini")
    p.add_argument("--partner-model", default=None, help="defaults to --model (self-play)")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--dataset", default="dnd", choices=["dnd", "casino"])
    p.add_argument("--protocol", default="single", choices=["single", "dual"],
                   help="single = one proposer + accept; dual = both emit <deal> tags")
    p.add_argument("--no-think", action="store_true",
                   help="disable hybrid-reasoning thinking (Qwen3 /no_think + OpenRouter reasoning off)")
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--max-turns", type=int, default=8, help="max messages per agent")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=500)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default=str(Path(__file__).resolve().parent / "results"))
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(parse_args()))
