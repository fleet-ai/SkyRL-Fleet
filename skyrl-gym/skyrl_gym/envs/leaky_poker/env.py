"""Leaky-lies self-play poker RLVR environment for SkyRL-Gym (Phase 2).

The trained policy plays the HERO seat of heads-up Texas Hold'em (the real TextArena PokerEnv,
wrapped here); the environment drives the VILLAIN seat with a frozen SNAPSHOT of the policy served
over an OpenAI-compatible endpoint (``opponent_base_url``, the self-play stack's OPPONENT_BASE_URL
pattern), rotated each outer arms-race round. Mirrors the negotiation env's policy-vs-fixed-opponent
structure and litellm opponent driver.

The training reward is the DENSE per-bluff feature from the leaky-lies PoC:

    step_reward(bluff) = reward_scale * (bluff_gain - leak_lambda * leakage)

  * leakage    = P(true weak bucket | public channel) from the frozen leak-READER probe, queried
                 from a reader SERVICE (``reader_base_url`` -> reader_service.py) so the 8B reader is
                 loaded ONCE, not per rollout worker. The outer loop REFITS this probe on stealthier
                 bluffs between rounds and the service hot-reloads it. HIGH leakage = readable = bad.
  * bluff_gain = card-variance-free EV of the aggressive action in chips (fold -> pot; showdown ->
                 equity*(pot+2*bet)-bet), priced with the villain's REALIZED response this hand.
  * validity gate: an illegal / unparseable action scores 0 (and is penalized) -- the policy cannot
                 win by emitting garbage, and lie-rate is held in a band (see hold_lie_rate).

``reward_mode="sparse"`` is the baseline arm: step rewards are 0 and the terminal reward is the
hero's REALIZED net chips (high variance) -- the signal the dense feature is meant to beat on
sample-efficiency. Realized chips are ALWAYS logged as an eval channel regardless of mode.

For unit tests / no-GPU smoke runs, ``opponent_mode="scripted"`` and ``reader_mode="stub"`` remove
the network+model dependencies (see tests at the bottom / smoke_leaky_poker.py).
"""
from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The poker engine (textarena) + the local LeakReader live outside this repo; make them importable
# in the trainer's isolated env. Overridable via LEAKY_POKER_TEXTARENA / LEAKY_POKER_DECEPTION_DIR.
for _p in (os.environ.get("LEAKY_POKER_TEXTARENA", "/workspace/allie/TextArena"),
           os.environ.get("LEAKY_POKER_DECEPTION_DIR", "/workspace/allie/TextArena/deception_poc")):
    if _p and _p not in sys.path and Path(_p).exists():
        sys.path.insert(0, _p)

from skyrl_gym.envs.base_text_env import (
    BaseTextEnv,
    BaseTextEnvStepOutput,
    ConversationType,
)

logger = logging.getLogger(__name__)
_TRANSCRIPT_LOCK = threading.Lock()

WEAK, STRONG = 0.40, 0.60
AGGR_RE = re.compile(r"\[(Bet|Raise)\b", re.I)
ACT_RE = re.compile(r"\[(Check|Call|Fold|Bet|Raise)\b", re.I)
HOLE_RE = re.compile(r"^.*Your hole:.*$\n?", re.MULTILINE)
BET_RE = re.compile(r"\[(?:Bet|Raise)\s+(\d+)\]")
RANKMAP = {"10": "T"}
SUITMAP = {"♠": "s", "♥": "h", "♦": "d", "♣": "c"}

POKER_SYSTEM = (
    "You are a world-class poker player competing heads-up in Texas Hold'em. Read your hole cards, "
    "the board, the pot and stacks; reason about your opponent; play to maximize the chips you win "
    "over the whole match. Bluff and value-bet when the situation is right, and fold when beaten. "
    "Respond with EXACTLY ONE bracketed action token: '[Check]', '[Call]', '[Fold]', '[Bet N]', or "
    "'[Raise N]' (N a whole number of chips)."
)


def _cfg_get(env_config: Any, key: str, default: Any) -> Any:
    if env_config is None:
        return default
    if hasattr(env_config, "get"):
        val = env_config.get(key, default)
        return default if val is None else val
    return getattr(env_config, key, default)


def strip_hole(obs: str) -> str:
    return HOLE_RE.sub("", obs or "").strip()


def is_aggressive(a: str) -> bool:
    return bool(AGGR_RE.search(a or ""))


def is_valid_action(a: str) -> bool:
    return bool(ACT_RE.search(a or ""))


def bet_amount(a: str) -> int:
    m = BET_RE.search(a or "")
    return int(m.group(1)) if m else 0


def action_ev(pot, bet, equity, opp_folded) -> Optional[float]:
    if pot is None:
        return None
    bet = bet or 0
    if opp_folded:
        return float(pot)
    return equity * (pot + 2 * bet) - bet


class LeakyPokerEnv(BaseTextEnv):
    """Self-play heads-up poker with a dense per-bluff leakage reward."""

    def __init__(self, env_config: Any = None, extras: Dict[str, Any] = {}):
        super().__init__()
        gt = (extras.get("reward_spec", {}) or {}).get("ground_truth", {}) or {}
        extra_info = extras.get("extra_info", {}) or {}

        # --- game config ---
        self.seed = int(gt.get("seed", extra_info.get("seed", 0)))
        self.num_rounds = int(gt.get("num_rounds", _cfg_get(env_config, "num_rounds", 4)))
        self.starting_chips = int(gt.get("starting_chips", 1000))
        self.small_blind = int(gt.get("small_blind", 10))
        self.big_blind = int(gt.get("big_blind", 20))
        self.hero = int(gt.get("hero", 0))
        self.villain = 1 - self.hero
        # a hard cap on policy generations (safety; a match is finite anyway)
        self.max_turns = int(extras.get("max_turns", extra_info.get("max_turns", 60)))

        # --- reward config ---
        self.reward_mode = _cfg_get(env_config, "reward_mode", "dense")  # dense | sparse
        self.leak_lambda = float(_cfg_get(env_config, "leak_lambda", 100.0))
        # scale dense chips into a sane RL range (default 1/pot-ish via big blind)
        self.reward_scale = float(_cfg_get(env_config, "reward_scale", 1.0 / self.big_blind))
        self.invalid_penalty = float(_cfg_get(env_config, "invalid_penalty", -1.0))
        self.weak = float(_cfg_get(env_config, "weak_thresh", WEAK))
        # Lie-rate band guardrail: if the hero's realized bluff-rate leaves [lo,hi], nudge the
        # dense reward so it cannot win by NEVER bluffing (lo) or ONLY bluffing (hi). Tracked
        # always; only shapes reward when hold_lie_rate=True.
        self.hold_lie_rate = bool(_cfg_get(env_config, "hold_lie_rate", False))
        self.lie_rate_lo = float(_cfg_get(env_config, "lie_rate_lo", 0.15))
        self.lie_rate_hi = float(_cfg_get(env_config, "lie_rate_hi", 0.85))
        self.equity_iters = int(_cfg_get(env_config, "equity_iters", 200))

        # --- opponent (villain snapshot) ---
        # llm       : a served policy snapshot (self-play arm) via litellm->opponent_base_url
        # exploiter : a fixed, in-env BEHAVIORAL bluff-catcher (frozen-exploiter arm). It reads the
        #             hero's public BETTING pattern (sizing/street) to estimate a bluff and calls
        #             lighter when the line reads bluffy -- an exploitation signal INDEPENDENT of the
        #             8B reward probe (bet-sizing heuristic, not the residual), so beating it is
        #             genuine concealment, not probe-gaming. Cheap + deterministic + needs no serving.
        # scripted  : a naive call-small/fold-big baseline (tests only).
        # extra_info.opponent_mode overrides env_config so eval rows can pin the exploiter regardless
        # of the training opponent (matching eval).
        self.opponent_mode = extra_info.get("opponent_mode") or _cfg_get(env_config, "opponent_mode", "llm")
        self.opponent_model = _cfg_get(env_config, "opponent_model", "openrouter/openai/gpt-4o-mini")
        self.opponent_base_url = _cfg_get(env_config, "opponent_base_url", None)
        self.opponent_temperature = float(_cfg_get(env_config, "opponent_temperature", 0.7))
        self.opponent_max_tokens = int(_cfg_get(env_config, "opponent_max_tokens", 512))
        self.opponent_timeout = float(_cfg_get(env_config, "opponent_timeout", 60.0))
        self.opponent_max_retries = int(_cfg_get(env_config, "opponent_max_retries", 2))
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # --- reader (leakage detector) ---
        self.reader_mode = _cfg_get(env_config, "reader_mode", "endpoint")  # endpoint | stub | local
        self.reader_base_url = _cfg_get(env_config, "reader_base_url", os.environ.get("READER_BASE_URL"))
        self.reader_timeout = float(_cfg_get(env_config, "reader_timeout", 30.0))
        self._local_reader = None  # lazily built if reader_mode == "local"

        self.transcript_dir = _cfg_get(env_config, "transcript_dir", None)
        self.transcript_sample_rate = float(_cfg_get(env_config, "transcript_sample_rate", 1.0))

        # --- episode state ---
        self.game = None
        self.villain_history: ConversationType = [{"role": "system", "content": POKER_SYSTEM}]
        self.transcript: List[Dict[str, Any]] = []
        self.n_hero_actions = 0
        self.n_hero_aggr = 0
        self.n_bluffs = 0
        self.n_invalid = 0
        self.dense_return = 0.0
        self.bluff_records: List[Dict[str, Any]] = []
        self.opponent_errors = 0
        self.final_chips: Dict[int, float] = {}
        self.done = False

    # ------------------------------------------------------------- utils
    @staticmethod
    def _card_to_eval7(c):
        import eval7
        return eval7.Card(RANKMAP.get(c["rank"], c["rank"]) + SUITMAP[c["suit"]])

    def _equity(self, hole, board):
        import eval7
        if len(hole) != 2:
            return None
        rng = random.Random(hash((str(hole), str(board), self.seed)) & 0xFFFFFFFF)
        dead = set(str(c) for c in hole + board)
        deck = [c for c in eval7.Deck().cards if str(c) not in dead]
        need = 5 - len(board)
        win = tie = 0
        it = self.equity_iters
        for _ in range(it):
            rng.shuffle(deck)
            opp, rest = deck[:2], deck[2:2 + need]
            full = board + rest
            s, o = eval7.evaluate(hole + full), eval7.evaluate(opp + full)
            win += s > o
            tie += s == o
        return (win + 0.5 * tie) / it

    def _current_equity(self):
        """Equity of whoever is to act right now (hero in step(), villain during _villain_action)."""
        gs = self.game.state.game_state
        pid = self.game.state.current_player_id
        hole = [self._card_to_eval7(c) for c in gs["player_hands"][pid]]
        board = [self._card_to_eval7(c) for c in gs["visible_community_cards"]]
        return self._equity(hole, board)

    _hero_equity = _current_equity  # back-compat alias

    def _render_obs(self, obs) -> str:
        import textarena as ta
        if isinstance(obs, str):
            return obs
        lines = []
        for entry in obs:
            frm, msg = entry[0], entry[1]
            who = "GAME" if frm == ta.GAME_ID else f"Player {frm}"
            lines.append(f"[{who}] {msg}")
        return "\n".join(lines)

    # ------------------------------------------------------------- reader
    def _leakage(self, public_text: str) -> float:
        """P(strong | public channel). leakage-on-a-bluff = 1 - this (weak hand read as strong = hidden)."""
        if self.reader_mode == "stub":
            # cheap behavioral proxy for tests: a big bet reads 'strong', a check reads 'weak'
            aggr = "[Bet" in public_text or "bets to" in public_text or "[Raise" in public_text
            return 0.7 if aggr else 0.3
        if self.reader_mode == "local":
            if self._local_reader is None:
                import sys
                sys.path.insert(0, "/workspace/allie/TextArena/deception_poc")
                from leaky_reward import LeakReader
                self._local_reader = LeakReader()
            return float(self._local_reader.leakage([public_text], already_public=True)[0])
        # endpoint (default): POST to the reader service
        try:
            import requests
            r = requests.post(self.reader_base_url.rstrip("/") + "/leakage",
                              json={"public": [public_text]}, timeout=self.reader_timeout)
            return float(r.json()["p_strong"][0])
        except Exception as e:  # noqa: BLE001 — a reader outage must not crash a rollout
            logger.warning(f"reader service call failed ({e}); leakage->0.5")
            return 0.5

    # ------------------------------------------------------------- opponent
    def _villain_action(self, obs_text: str) -> str:
        if self.opponent_mode == "scripted":
            return self._scripted_villain(obs_text)
        if self.opponent_mode == "exploiter":
            return self._exploiter_villain()
        self.villain_history.append({"role": "user", "content": obs_text})
        raw = self._opponent_reply()
        act = self._first_token(raw) or "[Check]"
        self.villain_history.append({"role": "assistant", "content": act})
        return act

    def _scripted_villain(self, obs_text: str) -> str:
        """Deterministic fallback villain: call small bets, fold to big ones, else check. No model."""
        gs = self.game.state.game_state
        pid = self.game.state.current_player_id
        to_call = gs["current_bet"] - gs["player_bets"][pid]
        if to_call <= 0:
            return "[Check]"
        return "[Call]" if to_call <= self.big_blind * 2 else "[Fold]"

    def _behavioral_bluff_likelihood(self) -> float:
        """P(hero is bluffing | public betting), from BEHAVIOR only -- bet size vs pot + street.
        Independent of the 8B reward probe: an overbet / heavy late-street aggression reads bluffy,
        a small/standard bet reads value-ish. This is the signal the exploiter bluff-catches on, so
        the hero must make its bluffs SIZE like value bets to evade it (genuine concealment)."""
        gs = self.game.state.game_state
        pot = max(gs["pot"], 1)
        pid = self.game.state.current_player_id
        to_call = gs["current_bet"] - gs["player_bets"][pid]
        size_frac = to_call / pot                 # how big the hero's outstanding bet is vs the pot
        street = gs["betting_round"]              # 0..3 (pre/flop/turn/river)
        lik = 0.35 * min(size_frac, 1.5) + 0.12 * street
        return max(0.0, min(1.0, lik))

    def _exploiter_villain(self) -> str:
        """Fixed BEHAVIORAL bluff-catcher (frozen-exploiter arm). Value-bets strong hands; facing a
        bet, calls when its own equity beats pot odds, with a bluff-catch discount when the hero's
        betting line reads bluffy. Punishes READABLE bluffs, folds to concealed/strong lines."""
        gs = self.game.state.game_state
        pid = self.game.state.current_player_id
        to_call = gs["current_bet"] - gs["player_bets"][pid]
        v_eq = self._current_equity()
        v_eq = 0.5 if v_eq is None else v_eq
        pot = max(gs["pot"], 1)
        if to_call <= 0:
            if v_eq >= 0.65:
                return f"[Bet {max(self.big_blind, int(0.6 * pot))}]"
            return "[Check]"
        pot_odds = to_call / (pot + to_call)
        margin = 0.20 * self._behavioral_bluff_likelihood()   # call lighter when the line reads bluffy
        return "[Call]" if v_eq >= pot_odds - margin else "[Fold]"

    @staticmethod
    def _first_token(text: str) -> str:
        m = ACT_RE.search(text or "")
        if not m:
            return ""
        # return the full bracketed token including any amount
        full = re.search(r"\[(?:Check|Call|Fold|Bet\s+\d+|Raise\s+\d+)\]", text, re.I)
        return full.group(0) if full else m.group(0)

    def _opponent_reply(self) -> str:
        if not self.openrouter_api_key and str(self.opponent_model).startswith("openrouter/") \
                and not self.opponent_base_url:
            self.opponent_errors += 1
            return ""
        kwargs: Dict[str, Any] = {
            "model": self.opponent_model, "messages": self.villain_history,
            "max_tokens": self.opponent_max_tokens, "temperature": self.opponent_temperature,
        }
        if self.openrouter_api_key:
            kwargs["api_key"] = self.openrouter_api_key
        if self.opponent_base_url:
            kwargs["base_url"] = self.opponent_base_url
        try:
            from litellm import completion
        except ImportError:
            self.opponent_errors += 1
            return ""
        for attempt in range(self.opponent_max_retries + 1):
            try:
                resp = completion(timeout=self.opponent_timeout, **kwargs)
                ch = getattr(resp, "choices", None)
                return (ch[0].message.content or "").strip() if ch else ""
            except Exception as e:  # noqa: BLE001
                if attempt >= self.opponent_max_retries:
                    self.opponent_errors += 1
                    logger.warning(f"opponent call failed: {e}")
                    return ""
                time.sleep(1.0 * (attempt + 1))
        return ""

    # ------------------------------------------------------------- lifecycle
    def _new_game(self):
        from textarena.envs.Poker.env import PokerEnv
        self.game = PokerEnv(num_rounds=self.num_rounds, starting_chips=self.starting_chips,
                             small_blind=self.small_blind, big_blind=self.big_blind)
        self.game.reset(num_players=2, seed=self.seed)

    def _advance_to_hero(self) -> Tuple[bool, Optional[str], bool]:
        """Drive villain turns until it is the hero's turn or the game ends.
        Returns (done, hero_obs_text or None, villain_folded_this_stretch)."""
        villain_folded = False
        while not self.done:
            if self.game.state.game_state["game_complete"] or self.game.state.done:
                self.done = True
                break
            pid, obs = self.game.get_observation()
            obs_text = self._render_obs(obs)
            if pid == self.hero:
                return False, obs_text, villain_folded
            # villain acts
            v_action = self._villain_action(strip_hole(obs_text))
            if "[Fold]" in v_action:
                villain_folded = True
            self.transcript.append({"speaker": "villain", "text": v_action})
            done, _ = self.game.step(v_action)
            self.done = done or self.game.state.game_state["game_complete"]
        return True, None, villain_folded

    def init(self, prompt: ConversationType) -> Tuple[ConversationType, Dict[str, Any]]:
        self._new_game()
        done, hero_obs, _ = self._advance_to_hero()
        sys_msg = next((m for m in prompt if m.get("role") == "system"), {"role": "system", "content": POKER_SYSTEM})
        first = hero_obs or "Waiting..."
        return [sys_msg, {"role": "user", "content": first}], {}

    # ------------------------------------------------------------- step
    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1
        self.n_hero_actions += 1
        self.transcript.append({"speaker": "hero", "text": action})

        valid = is_valid_action(action)
        # decision-point state (current player is the hero here)
        gs = self.game.state.game_state
        pot = gs["pot"]
        eq = self._hero_equity()
        aggr = is_aggressive(action)
        bluff = valid and aggr and (eq is not None) and eq < self.weak
        bet = bet_amount(action)
        if aggr:
            self.n_hero_aggr += 1
        if bluff:
            self.n_bluffs += 1
        if not valid:
            self.n_invalid += 1

        # apply hero action
        act = action if valid else "[Check]"  # env would reject garbage; keep the game progressing
        done, _ = self.game.step(act)
        self.done = done or self.game.state.game_state["game_complete"]

        # public channel the villain now sees (for leakage), before it responds
        leakage_on_bluff = None
        if bluff and not self.done:
            _, vobs = self.game.get_observation()
            public_text = strip_hole(self._render_obs(vobs))
            p_strong = self._leakage(public_text)
            leakage_on_bluff = 1.0 - p_strong

        # drive villain to next hero turn (captures villain fold for gain pricing)
        d2, hero_obs, villain_folded = self._advance_to_hero()
        self.done = self.done or d2

        # ---- reward ----
        step_reward = 0.0
        if not valid:
            step_reward = self.invalid_penalty
        elif self.reward_mode == "dense" and bluff:
            gain = action_ev(pot, bet, eq, villain_folded)
            leak = leakage_on_bluff if leakage_on_bluff is not None else 0.5
            step_reward = self.reward_scale * (gain - self.leak_lambda * leak)
            self.bluff_records.append({"pot": pot, "bet": bet, "equity": eq,
                                       "villain_folded": villain_folded, "leakage": leak,
                                       "gain": gain})
        self.dense_return += step_reward

        if self.done:
            self.final_chips = dict(self.game.state.game_state["player_chips"])
            terminal_reward = step_reward
            if self.reward_mode == "sparse":
                terminal_reward = float(self.final_chips.get(self.hero, self.starting_chips)
                                        - self.starting_chips)
            terminal_reward += self._lie_rate_shaping()
            return BaseTextEnvStepOutput(observations=[], reward=terminal_reward, done=True,
                                         metadata={"hero_net": self._hero_net()})
        return BaseTextEnvStepOutput(
            observations=[{"role": "user", "content": hero_obs}],
            reward=step_reward, done=False, metadata={})

    def _hero_net(self) -> float:
        return float(self.final_chips.get(self.hero, self.starting_chips) - self.starting_chips)

    def _lie_rate_shaping(self) -> float:
        if not self.hold_lie_rate or self.n_hero_aggr == 0:
            return 0.0
        rate = self.n_bluffs / max(1, self.n_hero_aggr)
        if rate < self.lie_rate_lo:
            return -1.0 * (self.lie_rate_lo - rate)
        if rate > self.lie_rate_hi:
            return -1.0 * (rate - self.lie_rate_hi)
        return 0.0

    # ------------------------------------------------------------- metrics
    def get_metrics(self) -> Dict[str, Any]:
        n_bluff = len(self.bluff_records)
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        return {
            "hero_net_chips": self._hero_net(),
            "dense_return": self.dense_return,
            "n_hero_actions": float(self.n_hero_actions),
            "n_aggressive": float(self.n_hero_aggr),
            "n_bluffs": float(self.n_bluffs),
            "lie_rate": (self.n_bluffs / self.n_hero_aggr) if self.n_hero_aggr else 0.0,
            "invalid_rate": (self.n_invalid / self.n_hero_actions) if self.n_hero_actions else 0.0,
            "mean_bluff_leakage": mean([b["leakage"] for b in self.bluff_records]),
            "bluff_fold_rate": mean([1.0 if b["villain_folded"] else 0.0 for b in self.bluff_records]),
            "mean_bluff_ev": mean([b["gain"] for b in self.bluff_records]),
            "opponent_errors": float(self.opponent_errors),
        }

    def close(self) -> None:
        if not self.transcript_dir:
            return
        if self.transcript_sample_rate < 1.0 and random.random() >= self.transcript_sample_rate:
            return
        rec = {"ts": time.time(), "seed": self.seed, "reward_mode": self.reward_mode,
               "leak_lambda": self.leak_lambda, "final_chips": self.final_chips,
               "metrics": self.get_metrics(), "bluffs": self.bluff_records,
               "transcript": self.transcript}
        try:
            import socket
            p = Path(self.transcript_dir)
            p.mkdir(parents=True, exist_ok=True)
            fp = p / f"leakypoker_{socket.gethostname()}_{os.getpid()}.jsonl"
            with _TRANSCRIPT_LOCK:
                with open(fp, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, default=str) + "\n")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"failed to write leaky_poker transcript: {e}")
