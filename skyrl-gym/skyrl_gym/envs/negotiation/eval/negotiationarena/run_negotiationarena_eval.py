#!/usr/bin/env python3
"""Offline NegotiationArena evaluation harness for negotiation checkpoints.

Faithful reconstruction of NegotiationArena (Bianchi et al., ICML 2024,
arXiv:2402.05863; code github.com/vinid/NegotiationArena). The evaluated policy is
served through an OpenAI-compatible chat endpoint (local vLLM, OpenRouter, ...) and
occupies one **seat** of an LLM-vs-LLM negotiation. The counterpart is either another
LLM (a frontier opponent over OpenRouter, like cross-play) or a deterministic
**scripted** agent (offline `--dry-run`, no API). The *environment is the verifier*:
trade legality, payoffs, and win are computed by ``games.py`` from the structured
trades -- there is NO LLM grader.

Examples:
  # quick snapshot vs a frontier opponent (balanced across game x focal_seat cells)
  python3 run_negotiationarena_eval.py --model <ckpt> --base-url http://localhost:8000/v1 \
      --opponent-model openai/gpt-5.5 --n 36 --no-think

  # full suite
  python3 run_negotiationarena_eval.py --model <ckpt> --base-url http://localhost:8000/v1 \
      --opponent-model openai/gpt-5.5 --full-suite

  # free run vs the scripted opponent (policy still served locally; no opponent API)
  python3 run_negotiationarena_eval.py --model <ckpt> --base-url http://localhost:8000/v1 \
      --scripted-opponent --n 36 --no-think

  # offline self-test: BOTH seats scripted, zero API calls
  python3 run_negotiationarena_eval.py --dry-run --n 12
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

HERE = Path(__file__).resolve().parent
# Put THIS dir first so the local config/games/scenarios/prompts/opponents/metrics
# shadow any similarly-named modules in the parent negotiation package.
sys.path.insert(0, str(HERE))

import config  # noqa: E402
import games  # noqa: E402
import metrics as metrics_mod  # noqa: E402
import prompts  # noqa: E402
import scenarios as scenarios_mod  # noqa: E402
from config import (  # noqa: E402
    GAMES_ORDER,
    SOCIAL_BEHAVIOURS,
    AgentAction,
    GameResult,
    NegArenaConfig,
    Scenario,
    Trade,
    TurnLog,
)
from opponents import ScriptedOpponent  # noqa: E402

try:
    from openai import AsyncOpenAI
except ImportError:  # pragma: no cover - only needed for live runs
    AsyncOpenAI = None


NO_THINK_TOKEN = "/no_think"
NO_THINK_BODY = {"reasoning": {"enabled": False}}
DEFAULT_OPPONENT_BASE_URL = "https://openrouter.ai/api/v1"


# ----------------------------------------------------------------------------------
# Endpoint plumbing (mirrors run_terms_eval.py / run_tom_eval.py).
# ----------------------------------------------------------------------------------
def make_client(base_url: str, timeout: float = 120.0, max_retries: int = 2):
    if AsyncOpenAI is None:
        raise RuntimeError("`openai` package required for endpoint calls (pip install openai).")
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
    return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=timeout, max_retries=max_retries)


async def chat(client, model, messages, temperature, max_tokens=400, retries=4, extra_body=None):
    """Robust chat call that adapts to provider parameter quirks."""
    kwargs = {"model": model, "messages": messages, "max_tokens": max_tokens}
    if extra_body:
        kwargs["extra_body"] = extra_body
    if temperature is not None and temperature >= 0:
        kwargs["temperature"] = temperature
    for attempt in range(retries):
        try:
            try:
                from nemo_relay_runtime import orchestrated_openai_chat_call_async
            except ImportError:
                resp = await client.chat.completions.create(**kwargs)
            else:
                resp = await orchestrated_openai_chat_call_async(
                    request=kwargs,
                    invoke=lambda effective_request: client.chat.completions.create(
                        **dict(effective_request)
                    ),
                    call_site="skyrl_gym.negotiation.eval.negotiationarena",
                    metadata={
                        "producer_session_id": os.environ.get("SKYRL_ATOF_PRODUCER_SESSION_ID"),
                    },
                )
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
# Seat backends: each seat is driven by either an LLM endpoint (own chat history) or
# the deterministic scripted opponent.
# ----------------------------------------------------------------------------------
class LLMSeat:
    """An OpenAI-compatible chat-endpoint seat with its own conversation history."""

    def __init__(self, client, model, temperature, max_tokens, no_think, system_prompt):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.no_think = no_think
        self.body = NO_THINK_BODY if no_think else None
        self.messages: list[dict] = [{"role": "system", "content": _maybe_no_think(system_prompt, no_think)}]

    async def act(self, scenario, seat, state, user_msg) -> str:
        self.messages.append({"role": "user", "content": user_msg})
        raw = await chat(
            self.client,
            self.model,
            self.messages,
            self.temperature,
            max_tokens=self.max_tokens,
            extra_body=self.body,
        )
        self.messages.append({"role": "assistant", "content": raw})
        return raw


class ScriptedSeat:
    """Wraps the deterministic ScriptedOpponent behind the same async act() API."""

    def __init__(self):
        self.opp = ScriptedOpponent()

    async def act(self, scenario, seat, state, user_msg) -> str:
        return self.opp.act(scenario, seat, state)


# ----------------------------------------------------------------------------------
# Single game rollout.
# ----------------------------------------------------------------------------------
def _trade_price(trade: Trade) -> float | None:
    """ZUP amount in a sell_buy trade (the sale price), regardless of which side gave it."""
    for s in (0, 1):
        z = trade.give(s).get("ZUP")
        if z:
            return float(z)
    return None


def _turn_log(turn: int, seat: int, action: AgentAction, violations: list[str]) -> TurnLog:
    return TurnLog(
        turn=turn,
        seat=seat,
        answer=action.answer,
        proposed_trade=action.proposed_trade.to_dict() if action.proposed_trade else None,
        message=action.message,
        has_reasoning=bool(action.reasoning),
        violations=list(violations),
    )


async def play_game(scenario: Scenario, seat_agents: dict, save_transcripts: bool) -> GameResult:
    """Run one full game. ``seat_agents`` maps seat index -> backend (LLMSeat/ScriptedSeat)."""
    focal = scenario.focal_seat
    other_of = {0: 1, 1: 0}
    initial = {0: games.initial_resources(scenario, 0), 1: games.initial_resources(scenario, 1)}
    current_resources = initial  # resources only change on acceptance (legality uses initial)

    last_proposal_by: dict[int, Trade | None] = {0: None, 1: None}
    proposals_made = {0: 0, 1: 0}
    focal_opening_price: float | None = None

    turns: list[TurnLog] = []
    n_moves = 0
    violation_tags: list[str] = []
    format_violation = False
    accepted_trade: Trade | None = None
    deal = False
    termination = "Timeout"
    error: str | None = None

    # First mover gets the kickoff prompt; the other seat's first user message is the
    # first mover's filtered public reply.
    pending_user = {scenario.first_mover: prompts.build_opening_user_message(scenario, scenario.first_mover)}
    current = scenario.first_mover

    try:
        for t in range(1, scenario.max_turns + 1):
            seat = current
            opp = other_of[seat]
            user_msg = pending_user.get(seat) or "It is your turn. Respond using the required format."

            state = {
                "turn": t,
                "max_turns": scenario.max_turns,
                "standing_offer": last_proposal_by[opp],
                "own_resources": dict(current_resources[seat]),
                "proposals_made": proposals_made[seat],
            }
            raw = await seat_agents[seat].act(scenario, seat, state, user_msg)
            n_moves += 1
            action = prompts.parse_agent_action(raw, scenario, seat)

            # Structural + economic violations.
            viol = prompts.detect_violations(action, scenario, seat, proposals_made[seat])
            if action.has_proposal and action.proposed_trade is not None:
                legal, econ_tags = games.is_legal_trade(scenario, action.proposed_trade, current_resources)
                if not legal:
                    viol = viol + econ_tags

            if seat == focal and viol:
                format_violation = True
                violation_tags.extend(viol)

            turns.append(_turn_log(t, seat, action, viol))

            # Unparseable response -> the strict NegotiationArena parser interrupts the game.
            if action.parse_error is not None:
                termination = "FormatError"
                break

            # ACCEPT: ends the game; accepts the most recent proposal by the other seat.
            if action.is_accept:
                standing = last_proposal_by[opp]
                if standing is None:
                    viol.append("accept_without_offer")
                    if seat == focal:
                        format_violation = True
                        violation_tags.append("accept_without_offer")
                    termination = "FormatError"
                    break
                accepted_trade = standing
                deal = True
                termination = "Agreement"
                break

            # A proposal that is illegal or past the per-seat limit interrupts the game.
            if action.has_proposal and action.proposed_trade is not None:
                blocking = [
                    v
                    for v in viol
                    if v
                    in (
                        "proposal_after_limit",
                        "non_integer",
                        "negative_amount",
                        "unknown_token",
                        "insufficient_resources",
                        "illegal_ultimatum_trade",
                        "illegal_sellbuy_trade",
                    )
                ]
                if blocking:
                    termination = "FormatError"
                    break
                last_proposal_by[seat] = action.proposed_trade
                proposals_made[seat] += 1
                if scenario.game == "sell_buy" and seat == focal and focal_opening_price is None:
                    focal_opening_price = _trade_price(action.proposed_trade)

            # Forward the filtered public surface to the other seat for its next turn.
            pending_user[opp] = prompts.filter_public(raw)
            current = opp

    except Exception as e:  # noqa: BLE001
        error = str(e)
        termination = "Error"

    payoffs = games.compute_payoffs(scenario, accepted_trade if deal else None)
    decisive, winner = games.decisive_winner(payoffs)
    focal_win = (winner == focal) if decisive else None

    if not save_transcripts:
        turns = []  # keep only the per-turn count via n_turns below if not saving

    return GameResult(
        scenario=scenario,
        turns=turns,
        deal=deal,
        accepted_trade=accepted_trade,
        termination=termination,
        payoffs=payoffs,
        focal_payoff=payoffs[focal],
        opp_payoff=payoffs[other_of[focal]],
        decisive=decisive,
        focal_win=focal_win,
        sale_price=games.extract_price(scenario, accepted_trade if deal else None),
        proposer_give=games.extract_proposer_give(scenario, accepted_trade if deal else None),
        focal_opening_price=focal_opening_price,
        format_violation=format_violation,
        violation_tags=violation_tags,
        n_turns=n_moves,
        error=error,
    )


# ----------------------------------------------------------------------------------
# Seat construction.
# ----------------------------------------------------------------------------------
def _apply_opponent_persona(scenario: Scenario, persona: str | None) -> None:
    """Attach a social-behaviour persona to the OPPONENT seat only (Appendix F.2)."""
    if not persona:
        return
    text = SOCIAL_BEHAVIOURS[persona][scenario.game]
    sb = list(scenario.social_behaviour)
    sb[scenario.opponent_seat] = text
    scenario.social_behaviour = (sb[0], sb[1])


def build_seat_agents(scenario, args, policy_client, opponent_client) -> dict:
    """Return {seat: backend}. focal seat = the evaluated policy; the other = opponent."""
    focal = scenario.focal_seat
    opp = scenario.opponent_seat
    agents: dict[int, object] = {}

    policy_scripted = args.dry_run
    opponent_scripted = args.dry_run or args.scripted_opponent

    # Focal (policy) seat.
    if policy_scripted:
        agents[focal] = ScriptedSeat()
    else:
        sp = prompts.build_system_prompt(scenario, focal)
        agents[focal] = LLMSeat(
            policy_client,
            args.model,
            args.temperature,
            args.max_tokens,
            args.no_think,
            sp,
        )

    # Opponent seat.
    if opponent_scripted:
        agents[opp] = ScriptedSeat()
    else:
        sp = prompts.build_system_prompt(scenario, opp)
        agents[opp] = LLMSeat(
            opponent_client,
            args.opponent_model,
            args.opponent_temperature,
            args.max_tokens,
            args.no_think,
            sp,
        )
    return agents


# ----------------------------------------------------------------------------------
# Serialization.
# ----------------------------------------------------------------------------------
def _result_to_dict(res: GameResult, include_turns: bool) -> dict:
    sc = res.scenario
    d = {
        "episode_id": sc.episode_id,
        "game": sc.game,
        "focal_seat": sc.focal_seat,
        "focal_role": sc.focal_role,
        "seller_cost": sc.seller_cost,
        "buyer_willingness": sc.buyer_willingness,
        "amount_to_split": sc.amount_to_split,
        "deal": res.deal,
        "termination": res.termination,
        "accepted_trade": res.accepted_trade.to_dict() if res.accepted_trade else None,
        "focal_payoff": res.focal_payoff,
        "opp_payoff": res.opp_payoff,
        "decisive": res.decisive,
        "focal_win": res.focal_win,
        "sale_price": res.sale_price,
        "proposer_give": res.proposer_give,
        "focal_opening_price": res.focal_opening_price,
        "format_violation": res.format_violation,
        "violation_tags": res.violation_tags,
        "n_turns": res.n_turns,
        "error": res.error,
    }
    if include_turns:
        d["turns"] = [
            {
                "turn": tl.turn,
                "seat": tl.seat,
                "answer": tl.answer,
                "proposed_trade": tl.proposed_trade,
                "message": tl.message,
                "has_reasoning": tl.has_reasoning,
                "violations": tl.violations,
            }
            for tl in res.turns
        ]
    return d


# ----------------------------------------------------------------------------------
# Driver.
# ----------------------------------------------------------------------------------
async def main_async(args):
    cfg = NegArenaConfig(
        ultimatum_amount=args.ultimatum_amount,
        buyer_budget=args.buyer_budget,
        sell_buy_cost=args.sell_buy_cost,
        sell_buy_willingness=args.sell_buy_willingness,
        vary_sell_buy=not args.no_vary_sell_buy,
        vary_amount=args.vary_amount,
    )
    games_subset = tuple(g for g in GAMES_ORDER if g in set(args.games)) if args.games else GAMES_ORDER
    scs = scenarios_mod.generate_scenarios(
        n=args.n,
        base_seed=args.seed,
        cfg=cfg,
        games=games_subset,
        full_suite=args.full_suite,
    )
    for sc in scs:
        _apply_opponent_persona(sc, args.opponent_persona)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy_client = None
    opponent_client = None
    if not args.dry_run:
        policy_client = make_client(args.base_url)
        if not args.scripted_opponent:
            opponent_client = make_client(args.opponent_base_url)

    model_label = "scripted-dry-run" if args.dry_run else args.model
    opp_label = "scripted" if (args.dry_run or args.scripted_opponent) else args.opponent_model

    sem = asyncio.Semaphore(args.concurrency)
    done = {"k": 0}
    total = len(scs)

    async def run_one(sc: Scenario) -> GameResult:
        async with sem:
            agents = build_seat_agents(sc, args, policy_client, opponent_client)
            res = await play_game(sc, agents, args.save_transcripts)
            done["k"] += 1
            if done["k"] % 10 == 0 or done["k"] == total:
                print(f"  [neg-arena] {done['k']}/{total} games", flush=True)
            return res

    t0 = time.time()
    print(
        f"running {total} NegotiationArena games "
        f"({'full suite' if args.full_suite else f'n={args.n}'}); "
        f"policy={model_label} vs opponent={opp_label}...",
        flush=True,
    )
    results = await asyncio.gather(*[run_one(sc) for sc in scs])
    elapsed = time.time() - t0

    agg = metrics_mod.compute_metrics(results, cfg)
    cfg_block = {
        "model": model_label,
        "base_url": redact_base_url(args.base_url),
        "opponent_model": opp_label,
        "opponent_base_url": (
            None if (args.dry_run or args.scripted_opponent) else redact_base_url(args.opponent_base_url)
        ),
        "opponent_persona": args.opponent_persona,
        "n": total,
        "full_suite": args.full_suite,
        "games": list(games_subset),
        "seed": args.seed,
        "temperature": args.temperature,
        "opponent_temperature": args.opponent_temperature,
        "max_tokens": args.max_tokens,
        "no_think": args.no_think,
        "vary_sell_buy": cfg.vary_sell_buy,
        "vary_amount": cfg.vary_amount,
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 2),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provenance": "Reconstructed from arXiv:2402.05863 + the public NegotiationArena repo.",
    }
    payload = {
        "config": cfg_block,
        "metrics": agg,
        "games": [_result_to_dict(r, include_turns=args.save_transcripts) for r in results],
    }
    safe = sanitize_model_name(model_label)
    tag = "full" if args.full_suite else f"n{total}"
    out_path = out_dir / f"{safe}_negotiationarena_{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))

    print_report(cfg_block, agg)
    print(f"\nwrote {out_path}")
    return agg


def _fmt(x):
    return "  n/a" if x is None else f"{x:.4f}"


def print_report(cfg_block: dict, agg: dict) -> None:
    o = agg["overall"]
    print("\n" + "=" * 68)
    print("NegotiationArena (reconstruction) — " + cfg_block["model"])
    print(
        f"  vs opponent: {cfg_block['opponent_model']}"
        + (f"  persona={cfg_block['opponent_persona']}" if cfg_block["opponent_persona"] else "")
    )
    print("=" * 68)
    print(f"  games         : {o['n']}  ({o['n_deals']} deals)   elapsed_s: {cfg_block['elapsed_s']}")
    print("-" * 68)
    print(f"  deal_rate              : {_fmt(o['deal_rate'])}")
    print(f"  mean_focal_payoff      : {_fmt(o['mean_focal_payoff'])}   (opp {_fmt(o['mean_opp_payoff'])})")
    print(f"  focal_win_rate         : {_fmt(o['focal_win_rate'])}   (decisive {o['n_decisive']})")
    print(f"  format_violation_rate  : {_fmt(o['format_violation_rate'])}   errors: {o['n_errors']}")
    print(f"  mean_n_turns           : {_fmt(o['mean_n_turns'])}")
    print("-" * 68)
    print("  by game:")
    for game, blk in agg["by_game"].items():
        line = (
            f"    - {game:<18}: deal {_fmt(blk['deal_rate'])}  "
            f"focal_payoff {_fmt(blk['mean_focal_payoff'])}  "
            f"win {_fmt(blk['focal_win_rate'])}"
        )
        if blk.get("mean_sale_price") is not None:
            line += f"  sale_price {_fmt(blk['mean_sale_price'])}  anchor_rho {_fmt(blk['anchoring_spearman'])}"
        if blk.get("mean_proposer_share") is not None:
            line += f"  proposer_share {_fmt(blk['mean_proposer_share'])}"
        print(line)
    print("  by focal role:")
    for role, blk in agg["by_focal_role"].items():
        print(
            f"    - {role:<12}: deal {_fmt(blk['deal_rate'])}  "
            f"focal_payoff {_fmt(blk['mean_focal_payoff'])}  win {_fmt(blk['focal_win_rate'])}"
        )
    print("=" * 68)


def parse_args():
    p = argparse.ArgumentParser(description="NegotiationArena reconstruction eval harness")
    p.add_argument("--model", default="scripted", help="Policy model name served at --base-url")
    p.add_argument("--base-url", default="http://localhost:8000/v1", help="Policy endpoint")
    p.add_argument("--opponent-model", default="openai/gpt-5.5", help="Frontier opponent model (OpenRouter slug)")
    p.add_argument(
        "--opponent-base-url", default=DEFAULT_OPPONENT_BASE_URL, help="Opponent endpoint (default OpenRouter)"
    )
    p.add_argument(
        "--opponent-persona",
        choices=sorted(SOCIAL_BEHAVIOURS.keys()),
        default=None,
        help="Prime the OPPONENT seat with a social-behaviour persona (Appendix F.2)",
    )
    p.add_argument(
        "--scripted-opponent",
        action="store_true",
        help="Use the deterministic scripted opponent (no opponent API; policy still served)",
    )
    p.add_argument("--n", type=int, default=36, help="Total games, balanced across game x focal_seat cells")
    p.add_argument("--full-suite", action="store_true", help="Run N_PER_CELL_FULL_SUITE per cell instead of --n")
    p.add_argument(
        "--games",
        nargs="*",
        default=None,
        help="Subset of games to run (default: all). e.g. --games sell_buy ultimatum",
    )
    p.add_argument("--seed", type=int, default=0, help="Base seed for the scenario set")
    p.add_argument("--temperature", type=float, default=0.0, help="Policy temperature (eval default 0)")
    p.add_argument("--opponent-temperature", type=float, default=0.7, help="Opponent temperature (paper default 0.7)")
    p.add_argument("--max-tokens", type=int, default=400, help="Per-turn token budget (paper default 400)")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--no-think", action="store_true")
    p.add_argument(
        "--save-transcripts", action="store_true", help="Include full per-turn transcripts in the results JSON"
    )
    p.add_argument("--out-dir", default=str(HERE / "results"))
    p.add_argument("--dry-run", action="store_true", help="Offline self-test: BOTH seats scripted, no API calls")
    # Scenario-config overrides.
    p.add_argument("--ultimatum-amount", type=int, default=config.ULTIMATUM_DEFAULT_AMOUNT)
    p.add_argument("--buyer-budget", type=int, default=config.SELL_BUY_BUYER_BUDGET)
    p.add_argument("--sell-buy-cost", type=int, default=config.SELL_BUY_DEFAULT_COST)
    p.add_argument("--sell-buy-willingness", type=int, default=config.SELL_BUY_DEFAULT_WILLINGNESS)
    p.add_argument(
        "--no-vary-sell-buy",
        action="store_true",
        help="Use fixed (cost, willingness) instead of the U{20,40}/U{60,80} draws",
    )
    p.add_argument("--vary-amount", action="store_true", help="Draw the ultimatum pot from the numerosity pool (§5.2)")
    return p.parse_args()


def main():
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
