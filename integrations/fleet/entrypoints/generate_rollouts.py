"""Generate rollouts from Claude + Qwen on booking tasks, then run the trace judge.

No live Fleet environment needed — uses a stateful mock booking tool server.

Usage:
    # Set API keys first
    export ANTHROPIC_API_KEY=...
    export OPENROUTER_API_KEY=...

    python -m integrations.fleet.entrypoints.generate_rollouts \\
        --dataset ~/Work/data/fleet/v7/openenv/all_tool_use.json \\
        --n-tasks 12 \\
        --rollouts-per-model 2 \\
        --output-dir ~/Work/data/fleet/traces \\
        --judge-output ~/Work/data/fleet/analysis/judge_results.json

Outputs:
    {output-dir}/booking_rollouts.jsonl   — all traces (task_key, model, chat_history, reward)
    {judge-output}                        — judge analysis results
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from integrations.fleet.trace_judge import (
    calibrate_batch,
    divergence_judge,
    direct_judge,
    parse_steps,
)


# ---------------------------------------------------------------------------
# Mock booking tool environment
# ---------------------------------------------------------------------------

# Fixed hotel catalog — realistic enough for multi-step reasoning
_HOTELS = [
    {"id": 101, "name": "The Grand Amsterdam", "city": "Amsterdam", "country": "NL",
     "star_rating": 5, "average_rating": 9.1, "property_type": "Resort",
     "price_per_night": 420, "facilities": ["Gym", "Pool", "Spa", "Restaurant"],
     "nearby_attractions": [{"name": "Vondelpark", "type": "nature", "distance_km": 0.4}]},
    {"id": 102, "name": "Canal View Resort", "city": "Amsterdam", "country": "NL",
     "star_rating": 5, "average_rating": 8.8, "property_type": "Resort",
     "price_per_night": 310, "facilities": ["Gym", "Pool", "Bar"],
     "nearby_attractions": [{"name": "Artis Zoo", "type": "attraction", "distance_km": 0.7}]},
    {"id": 103, "name": "Amsterdam Bay Resort", "city": "Amsterdam", "country": "NL",
     "star_rating": 5, "average_rating": 8.5, "property_type": "Resort",
     "price_per_night": 275, "facilities": ["Pool", "Restaurant"],
     "nearby_attractions": [{"name": "IJ River Park", "type": "nature", "distance_km": 0.2}]},
    {"id": 201, "name": "Austin Skyline Hotel", "city": "Austin", "country": "US",
     "star_rating": 5, "average_rating": 9.3, "property_type": "Hotel",
     "price_per_night": 380, "facilities": ["Gym", "Pool", "Business Center"],
     "nearby_attractions": [{"name": "Barton Creek Greenbelt", "type": "nature", "distance_km": 2.1}]},
    {"id": 202, "name": "Hotel Congress Austin", "city": "Austin", "country": "US",
     "star_rating": 4, "average_rating": 8.6, "property_type": "Hotel",
     "price_per_night": 195, "facilities": ["Restaurant", "Bar"],
     "nearby_attractions": [{"name": "Lady Bird Lake", "type": "nature", "distance_km": 1.3}]},
    {"id": 301, "name": "Miami Beach Resort & Spa", "city": "Miami", "country": "US",
     "star_rating": 5, "average_rating": 9.0, "property_type": "Resort",
     "price_per_night": 450, "facilities": ["Gym", "Pool", "Spa", "Beach Access"],
     "nearby_attractions": [{"name": "South Beach", "type": "beach", "distance_km": 0.1}]},
    {"id": 401, "name": "The Ritz Lisbon", "city": "Lisbon", "country": "PT",
     "star_rating": 5, "average_rating": 9.4, "property_type": "Hotel",
     "price_per_night": 520, "facilities": ["Gym", "Pool", "Spa", "Michelin Restaurant"],
     "nearby_attractions": [{"name": "Eduardo VII Park", "type": "nature", "distance_km": 0.3}]},
]

_ROOMS = {
    101: [
        {"id": 1001, "hotel_id": 101, "type": "Standard", "price_per_night": 380, "max_guests": 2, "available": True},
        {"id": 1002, "hotel_id": 101, "type": "Deluxe", "price_per_night": 520, "max_guests": 2, "available": True},
        {"id": 1003, "hotel_id": 101, "type": "Suite", "price_per_night": 850, "max_guests": 4, "available": False},
    ],
    102: [
        {"id": 2001, "hotel_id": 102, "type": "Standard", "price_per_night": 280, "max_guests": 2, "available": True},
        {"id": 2002, "hotel_id": 102, "type": "Deluxe", "price_per_night": 360, "max_guests": 2, "available": True},
    ],
    103: [
        {"id": 3001, "hotel_id": 103, "type": "Standard", "price_per_night": 250, "max_guests": 2, "available": True},
        {"id": 3002, "hotel_id": 103, "type": "Deluxe", "price_per_night": 320, "max_guests": 2, "available": True},
    ],
    201: [
        {"id": 4001, "hotel_id": 201, "type": "Standard", "price_per_night": 340, "max_guests": 2, "available": True},
        {"id": 4002, "hotel_id": 201, "type": "Deluxe", "price_per_night": 430, "max_guests": 2, "available": True},
    ],
    202: [{"id": 5001, "hotel_id": 202, "type": "Standard", "price_per_night": 180, "max_guests": 2, "available": True}],
    301: [
        {"id": 6001, "hotel_id": 301, "type": "Ocean View", "price_per_night": 420, "max_guests": 2, "available": True},
        {"id": 6002, "hotel_id": 301, "type": "Deluxe Suite", "price_per_night": 680, "max_guests": 4, "available": True},
    ],
    401: [
        {"id": 7001, "hotel_id": 401, "type": "Classic", "price_per_night": 480, "max_guests": 2, "available": True},
        {"id": 7002, "hotel_id": 401, "type": "Deluxe", "price_per_night": 620, "max_guests": 2, "available": True},
    ],
}

_BOOKING_COUNTER = [25590]  # mutable counter
_PM_COUNTER = [8330]


class MockBookingEnv:
    """Stateful mock booking environment. One instance per rollout."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.wishlists: Dict[str, List[Dict]] = {}  # list_name -> [{hotel_id, notes}]
        self.payment_methods: List[Dict] = [
            {"id": 1, "last_four": "1234", "card_type": "Visa",
             "expiry_month": 12, "expiry_year": 2027, "cardholder_name": "Test User",
             "is_default": True}
        ]
        self.bookings: List[Dict] = []
        self.ops_log: List[str] = []  # track what the agent did

    def execute(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatch tool call and return result dict."""
        self.ops_log.append(f"{tool_name}({list(args.keys())})")
        handler = {
            "search_hotels": self._search_hotels,
            "get_price_distribution": self._get_price_distribution,
            "get_hotel_details": self._get_hotel_details,
            "get_room_availability": self._get_room_availability,
            "get_nearby_attractions": self._get_nearby_attractions,
            "get_wishlists": self._get_wishlists,
            "create_wishlist": self._create_wishlist,
            "add_to_wishlist": self._add_to_wishlist,
            "delete_wishlist": self._delete_wishlist,
            "get_payment_methods": self._get_payment_methods,
            "add_payment_method": self._add_payment_method,
            "book_hotel": self._book_hotel,
            "final_answer": self._final_answer,
            # aliases
            "search": self._search_hotels,
            "get_hotels": self._search_hotels,
            "check_wishlists": self._get_wishlists,
            "list_wishlists": self._get_wishlists,
            "check_payment_methods": self._get_payment_methods,
            "list_payment_methods": self._get_payment_methods,
            "create_booking": self._book_hotel,
            "make_booking": self._book_hotel,
        }.get(tool_name.lower())

        if handler is None:
            return {"error": f"Unknown tool: {tool_name}. Available: search_hotels, get_hotel_details, get_room_availability, get_nearby_attractions, get_wishlists, create_wishlist, add_to_wishlist, delete_wishlist, get_payment_methods, add_payment_method, book_hotel, final_answer"}

        try:
            return handler(args)
        except Exception as e:
            return {"error": f"Tool execution error: {e}"}

    def _search_hotels(self, args: Dict) -> Dict:
        city = args.get("city", "")
        stars = args.get("stars") or args.get("star_rating") or args.get("min_stars")
        prop_type = (args.get("property_type") or args.get("type") or "").lower()
        facilities = args.get("facilities") or args.get("required_facilities") or []
        if isinstance(facilities, str):
            facilities = [facilities]

        results = []
        for h in _HOTELS:
            if city and city.lower() not in h["city"].lower():
                continue
            if stars and h["star_rating"] < int(stars):
                continue
            if prop_type and prop_type not in h["property_type"].lower():
                continue
            if facilities:
                has_all = all(any(f.lower() in fac.lower() for fac in h["facilities"]) for f in facilities)
                if not has_all:
                    continue
            results.append({k: v for k, v in h.items() if k != "nearby_attractions"})

        return {"hotels": results, "total": len(results)}

    def _get_price_distribution(self, args: Dict) -> Dict:
        city = args.get("city", "")
        prop_type = (args.get("property_type") or "").lower()
        hotels = [h for h in _HOTELS
                  if (not city or city.lower() in h["city"].lower())
                  and (not prop_type or prop_type in h["property_type"].lower())]
        prices = [h["price_per_night"] for h in hotels] if hotels else [200, 300, 400]
        avg = sum(prices) / len(prices)
        return {
            "city": city,
            "property_type": prop_type,
            "avg_price_per_night": round(avg, 2),
            "min_price": min(prices),
            "max_price": max(prices),
            "count": len(prices),
            "currency": "USD",
        }

    def _get_hotel_details(self, args: Dict) -> Dict:
        hotel_id = int(args.get("hotel_id", 0))
        hotel = next((h for h in _HOTELS if h["id"] == hotel_id), None)
        if not hotel:
            return {"error": f"Hotel {hotel_id} not found"}
        return {"hotel": hotel, "rooms": _ROOMS.get(hotel_id, [])}

    def _get_room_availability(self, args: Dict) -> Dict:
        hotel_id = int(args.get("hotel_id", 0))
        room_type = (args.get("room_type") or "").lower()
        rooms = _ROOMS.get(hotel_id, [])
        if room_type:
            rooms = [r for r in rooms if room_type in r["type"].lower()]
        available = [r for r in rooms if r["available"]]
        return {"hotel_id": hotel_id, "available_rooms": available, "total_available": len(available)}

    def _get_nearby_attractions(self, args: Dict) -> Dict:
        hotel_id = int(args.get("hotel_id", 0))
        hotel = next((h for h in _HOTELS if h["id"] == hotel_id), None)
        if not hotel:
            return {"error": f"Hotel {hotel_id} not found"}
        return {"hotel_id": hotel_id, "nearby_attractions": hotel["nearby_attractions"]}

    def _get_wishlists(self, _: Dict) -> Dict:
        return {"wishlists": [
            {"name": k, "properties": v} for k, v in self.wishlists.items()
        ]}

    def _create_wishlist(self, args: Dict) -> Dict:
        name = args.get("name") or args.get("list_name") or ""
        if not name:
            return {"error": "list name required"}
        if name in self.wishlists:
            return {"status": "already_exists", "name": name}
        self.wishlists[name] = []
        return {"status": "created", "name": name}

    def _add_to_wishlist(self, args: Dict) -> Dict:
        name = args.get("list_name") or args.get("name") or ""
        hotel_id = args.get("hotel_id")
        notes = args.get("notes") or args.get("note") or ""
        if name not in self.wishlists:
            self.wishlists[name] = []
        self.wishlists[name].append({"hotel_id": hotel_id, "notes": notes})
        return {"status": "added", "list_name": name, "hotel_id": hotel_id}

    def _delete_wishlist(self, args: Dict) -> Dict:
        name = args.get("name") or args.get("list_name") or ""
        if name not in self.wishlists:
            return {"error": f"Wishlist '{name}' not found"}
        del self.wishlists[name]
        return {"status": "deleted", "name": name}

    def _get_payment_methods(self, _: Dict) -> Dict:
        return {"payment_methods": self.payment_methods}

    def _add_payment_method(self, args: Dict) -> Dict:
        _PM_COUNTER[0] += 1
        pm = {
            "id": _PM_COUNTER[0],
            "card_number": args.get("card_number", ""),
            "last_four": str(args.get("card_number", "0000"))[-4:],
            "card_type": args.get("card_type", "Visa"),
            "expiry_month": args.get("expiry_month") or args.get("exp_month"),
            "expiry_year": args.get("expiry_year") or args.get("exp_year"),
            "cvv": args.get("cvv") or args.get("security_code"),
            "cardholder_name": args.get("cardholder_name") or args.get("name"),
            "billing_address": args.get("billing_address", ""),
            "billing_postal_code": args.get("billing_postal_code") or args.get("zip_code"),
            "is_default": args.get("is_default", False),
        }
        self.payment_methods.append(pm)
        return {"status": "added", "payment_method": pm}

    def _book_hotel(self, args: Dict) -> Dict:
        _BOOKING_COUNTER[0] += 1
        booking = {
            "id": _BOOKING_COUNTER[0],
            "booking_reference": f"BK{_BOOKING_COUNTER[0]:06d}",
            "hotel_id": args.get("hotel_id"),
            "room_id": args.get("room_id"),
            "check_in_date": args.get("check_in") or args.get("check_in_date"),
            "check_out_date": args.get("check_out") or args.get("check_out_date"),
            "adults": args.get("adults", 1),
            "guest_first_name": args.get("guest_first_name") or args.get("first_name"),
            "guest_last_name": args.get("guest_last_name") or args.get("last_name"),
            "guest_email": args.get("guest_email") or args.get("email"),
            "guest_phone_number": args.get("guest_phone_number") or args.get("phone"),
            "payment_method_id": args.get("payment_method_id"),
            "status": "confirmed",
            "payment_status": "paid",
            "agreed_to_terms": True,
        }
        self.bookings.append(booking)
        return {"status": "confirmed", "booking": booking}

    def _final_answer(self, args: Dict) -> Dict:
        return {"status": "recorded", "answer": args.get("answer", "")}

    def compute_reward(self, task_prompt: str) -> float:
        """Heuristic reward: fraction of key booking operations attempted."""
        prompt_lower = task_prompt.lower()
        ops = " ".join(self.ops_log).lower()
        score = 0.0
        checks = 0

        def check(condition: bool):
            nonlocal score, checks
            checks += 1
            if condition:
                score += 1

        # Was a hotel searched?
        check("search_hotels" in ops or "search" in ops)
        # Was a booking made?
        check(len(self.bookings) > 0)
        # Was a payment method checked or added?
        check("payment" in ops)
        # If wishlist was mentioned, was one created?
        if "wishlist" in prompt_lower:
            check(len(self.wishlists) > 0)
        else:
            check(True)
        # Did agent complete?
        check("final_answer" in ops or "done" in ops)

        return round(score / checks, 2) if checks > 0 else 0.0


# ---------------------------------------------------------------------------
# Tool schema for system prompt
# ---------------------------------------------------------------------------

_BOOKING_TOOLS = [
    {
        "name": "search_hotels",
        "description": "Search for hotels by city, star rating, property type, and required facilities.",
        "parameters": {
            "city": "string - city name",
            "stars": "integer - minimum star rating (1-5)",
            "property_type": "string - e.g. 'Hotel', 'Resort'",
            "facilities": "list[string] - required facilities e.g. ['Gym', 'Pool']",
            "check_in": "string - check-in date YYYY-MM-DD",
            "check_out": "string - check-out date YYYY-MM-DD",
            "guests": "integer - number of guests",
        }
    },
    {
        "name": "get_price_distribution",
        "description": "Get price statistics for hotels in a city filtered by property type.",
        "parameters": {"city": "string", "property_type": "string"}
    },
    {
        "name": "get_hotel_details",
        "description": "Get full details for a specific hotel including rooms and facilities.",
        "parameters": {"hotel_id": "integer"}
    },
    {
        "name": "get_room_availability",
        "description": "Check room availability for a hotel, optionally filtered by room type.",
        "parameters": {"hotel_id": "integer", "room_type": "string (optional)", "check_in": "string", "check_out": "string"}
    },
    {
        "name": "get_nearby_attractions",
        "description": "Get nearby attractions for a hotel.",
        "parameters": {"hotel_id": "integer"}
    },
    {
        "name": "get_wishlists",
        "description": "Get all wishlists for the current user.",
        "parameters": {}
    },
    {
        "name": "create_wishlist",
        "description": "Create a new named wishlist.",
        "parameters": {"name": "string"}
    },
    {
        "name": "add_to_wishlist",
        "description": "Add a hotel to a wishlist with an optional note.",
        "parameters": {"list_name": "string", "hotel_id": "integer", "notes": "string (optional)"}
    },
    {
        "name": "delete_wishlist",
        "description": "Delete a wishlist by name.",
        "parameters": {"name": "string"}
    },
    {
        "name": "get_payment_methods",
        "description": "Get all saved payment methods for the current user.",
        "parameters": {}
    },
    {
        "name": "add_payment_method",
        "description": "Save a new credit/debit card.",
        "parameters": {
            "card_number": "string", "cvv": "string",
            "expiry_month": "integer", "expiry_year": "integer",
            "cardholder_name": "string", "billing_address": "string",
            "billing_postal_code": "string", "card_type": "string"
        }
    },
    {
        "name": "book_hotel",
        "description": "Book a hotel room.",
        "parameters": {
            "hotel_id": "integer", "room_id": "integer",
            "check_in": "string (YYYY-MM-DD)", "check_out": "string (YYYY-MM-DD)",
            "adults": "integer", "children": "integer",
            "guest_first_name": "string", "guest_last_name": "string",
            "guest_email": "string", "guest_phone_number": "string",
            "payment_method_id": "integer", "agreed_to_terms": "boolean"
        }
    },
    {
        "name": "final_answer",
        "description": "Submit your final answer and mark the task complete.",
        "parameters": {"answer": "string"}
    },
]

_SYSTEM_PROMPT = """\
You are a helpful travel booking assistant. Complete the task by calling tools.

## Tool Call Format
Format each call as:
<tool_call>{{"name": "<tool_name>", "arguments": {{...}}}}</tool_call>

## Available Tools
{tools_json}

## Rules
- After each tool call you will receive the result. Use it to decide your next action.
- When the task is fully complete, call final_answer with a summary, then say <done>.
- Do NOT repeat the same tool call with the same arguments if it already returned a result.
- Every response MUST end with either a tool call OR <done>.
"""


# ---------------------------------------------------------------------------
# Model clients
# ---------------------------------------------------------------------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def _parse_tool_call(content: str) -> Optional[Tuple[str, Dict]]:
    m = _TOOL_CALL_RE.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group(1))
        return obj.get("name", ""), obj.get("arguments", {})
    except json.JSONDecodeError:
        return None


def call_claude(
    messages: List[Dict],
    system: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 1024,
) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=messages,
    )
    return resp.content[0].text


def call_openrouter(
    messages: List[Dict],
    system: str,
    model: str = "qwen/qwen-2.5-7b-instruct",
    max_tokens: int = 1024,
) -> str:
    import requests
    headers = {
        "Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/fleet-ai/SkyRL",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system}] + messages,
    }
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Rollout loop
# ---------------------------------------------------------------------------

def run_rollout(
    task_key: str,
    task_prompt: str,
    env_variables: Dict,
    model_name: str,
    call_fn,  # callable(messages, system) -> str
    max_turns: int = 20,
    rng_seed: int = 0,
    temperature_hint: str = "",
) -> Tuple[List[Dict], float]:
    """Run one full rollout for a task. Returns (chat_history, reward)."""

    env = MockBookingEnv(seed=rng_seed)
    tools_json = json.dumps(_BOOKING_TOOLS, indent=2)

    # Inject env context into task prompt
    env_ctx = ""
    if env_variables:
        lines = []
        for k, v in env_variables.items():
            if k == "LOGGED_IN_USER":
                lines.append(f"Logged in as: {v}")
            elif k == "CURRENT_DATE":
                lines.append(f"Current date: {v}")
        if lines:
            env_ctx = "\n\nContext: " + "; ".join(lines)

    system = _SYSTEM_PROMPT.format(tools_json=tools_json)
    full_prompt = task_prompt + env_ctx

    messages: List[Dict] = [{"role": "user", "content": full_prompt}]
    chat_history: List[Dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": full_prompt},
    ]

    for turn in range(max_turns):
        try:
            reply = call_fn(messages, system)
        except Exception as e:
            logger.warning(f"  [{model_name}] API error at turn {turn}: {e}")
            time.sleep(2)
            break

        chat_history.append({"role": "assistant", "content": reply})
        messages.append({"role": "assistant", "content": reply})

        # Check done
        is_done = "<done>" in reply.lower() or "[done]" in reply.lower()
        tc = _parse_tool_call(reply)

        if tc:
            tool_name, tool_args = tc
            result = env.execute(tool_name, tool_args)
            obs = f"Tool result:\n{json.dumps(result, indent=2)}"
            chat_history.append({"role": "user", "content": obs})
            messages.append({"role": "user", "content": obs})

        if is_done or (not tc and turn > 0):
            break

    reward = env.compute_reward(task_prompt)
    return chat_history, reward


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_booking_tasks(path: str, n: int, seed: int = 42) -> List[Dict]:
    """Load N random booking tasks from the dataset."""
    expanded = os.path.expanduser(path)
    with open(expanded) as f:
        data = json.load(f)
    tasks = data["tasks"] if isinstance(data, dict) else data
    booking = [t for t in tasks if t.get("env_key") == "booking"]
    rng = random.Random(seed)
    sample = rng.sample(booking, min(n, len(booking)))
    logger.info(f"Loaded {len(sample)} booking tasks from {expanded}")
    return sample


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="~/Work/data/fleet/v7/openenv/all_tool_use.json")
    parser.add_argument("--n-tasks", type=int, default=12)
    parser.add_argument("--rollouts-per-model", type=int, default=2,
                        help="Rollouts per model per task (different rng seeds)")
    parser.add_argument("--output-dir", default="~/Work/data/fleet/traces")
    parser.add_argument("--judge-output", default="~/Work/data/fleet/analysis/judge_results.json")
    parser.add_argument("--claude-model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--qwen-model", default="qwen/qwen-2.5-7b-instruct")
    parser.add_argument("--max-turns", type=int, default=18)
    parser.add_argument("--judge-method", choices=["divergence", "direct", "both"], default="both")
    parser.add_argument("--judge-model", default="claude-haiku-4-5-20251001",
                        help="Model for direct judge (defaults to claude-haiku)")
    parser.add_argument("--skip-qwen", action="store_true", help="Only run Claude (faster)")
    args = parser.parse_args()

    # Validate API keys
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set")
    if not args.skip_qwen and not os.environ.get("OPENROUTER_API_KEY"):
        logger.warning("OPENROUTER_API_KEY not set — skipping Qwen rollouts")
        args.skip_qwen = True

    # Load tasks
    tasks = load_booking_tasks(args.dataset, args.n_tasks)

    # Output paths
    out_dir = os.path.expanduser(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    traces_path = os.path.join(out_dir, "booking_rollouts.jsonl")

    # Models to run
    models = [("claude", args.claude_model, lambda m, s: call_claude(m, s, model=args.claude_model))]
    if not args.skip_qwen:
        models.append(("qwen", args.qwen_model, lambda m, s: call_openrouter(m, s, model=args.qwen_model)))

    all_records = []
    total = len(tasks) * len(models) * args.rollouts_per_model
    done = 0

    with open(traces_path, "w") as out_f:
        for task in tasks:
            task_key = task["task_key"]
            prompt = task["prompt"]
            env_vars = task.get("env_variables", {})

            for model_short, model_id, call_fn in models:
                for roll_idx in range(args.rollouts_per_model):
                    done += 1
                    logger.info(f"[{done}/{total}] task={task_key[:30]} model={model_short} roll={roll_idx}")
                    try:
                        history, reward = run_rollout(
                            task_key=task_key,
                            task_prompt=prompt,
                            env_variables=env_vars,
                            model_name=model_id,
                            call_fn=call_fn,
                            max_turns=args.max_turns,
                            rng_seed=roll_idx * 17 + 3,
                        )
                    except Exception as e:
                        logger.error(f"  Rollout failed: {e}")
                        continue

                    record = {
                        "task_key": f"{task_key}__{model_short}_{roll_idx}",
                        "base_task_key": task_key,
                        "env_key": "booking",
                        "model": model_id,
                        "model_short": model_short,
                        "roll_idx": roll_idx,
                        "chat_history": history,
                        "reward": reward,
                        "n_turns": sum(1 for m in history if m["role"] == "assistant"),
                    }
                    all_records.append(record)
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    logger.info(f"  → {record['n_turns']} turns, reward={reward:.2f}")
                    time.sleep(0.5)  # gentle rate limiting

    logger.info(f"\nGenerated {len(all_records)} traces → {traces_path}")

    # ---------------------------------------------------------------------------
    # Run trace judge
    # ---------------------------------------------------------------------------
    logger.info("\n=== Running trace judge ===")

    # Group by base_task_key for divergence (all rollouts of same task)
    from collections import defaultdict
    by_task: Dict[str, List] = defaultdict(list)
    for rec in all_records:
        by_task[rec["base_task_key"]].append(rec)

    task_to_rewards = {tk: [r["reward"] for r in recs] for tk, recs in by_task.items()}

    div_scores = {}
    if args.judge_method in ("divergence", "both"):
        for task_key, recs in by_task.items():
            if len(recs) < 2:
                continue
            all_steps = [parse_steps(r["chat_history"]) for r in recs]
            div_scores[task_key] = divergence_judge(all_steps)

        div_cal = calibrate_batch(div_scores, task_to_rewards)
        print("\n=== Divergence Judge ===")
        print(f"Tasks scored: {div_cal['n_tasks']}")
        print(f"Mean max interestingness: {div_cal['mean_max_interestingness']}")
        print(f"Mean reward variance:     {div_cal['mean_reward_variance']}")
        print(f"Spearman(score, reward_var): {div_cal['spearman_max_score_vs_reward_var']}")
        print("\nTop findings per task:")
        for task_key, scores in sorted(div_scores.items(),
                                       key=lambda kv: max((s.score for s in kv[1]), default=0), reverse=True)[:5]:
            rewards = task_to_rewards.get(task_key, [])
            top = sorted(scores, key=lambda s: s.score, reverse=True)[:2]
            print(f"  {task_key[:40]}  rewards={[round(r,2) for r in rewards]}")
            for s in top:
                print(f"    turn {s.turn_idx:>2d}  score={s.score:.3f}  {s.rationale[:70]}")

    direct_scores = {}
    if args.judge_method in ("direct", "both"):
        try:
            import anthropic as _ant
            _judge_client_obj = _ant.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

            class _AnthropicAsOpenAI:
                """Thin shim: direct_judge() expects openai-style .chat.completions.create()."""
                def __init__(self, client, model):
                    self._c = client
                    self._m = model
                    self.chat = self
                    self.completions = self

                def create(self, model, messages, temperature=0, max_tokens=200):
                    sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
                    usr_msgs = [m for m in messages if m["role"] != "system"]
                    resp = self._c.messages.create(
                        model=self._m, max_tokens=max_tokens, system=sys_msg, messages=usr_msgs
                    )
                    class _Choice:
                        class _Msg:
                            content = resp.content[0].text
                        message = _Msg()
                    class _Resp:
                        choices = [_Choice()]
                    return _Resp()

            judge_client = _AnthropicAsOpenAI(_judge_client_obj, args.judge_model)

            print("\n=== Direct Judge (sampling first rollout per task) ===")
            for task_key, recs in list(by_task.items())[:8]:  # cap API calls
                rec = recs[0]
                steps = parse_steps(rec["chat_history"])
                task_prompt = next(
                    (m["content"] for m in rec["chat_history"] if m["role"] == "user"),
                    ""
                )
                if isinstance(task_prompt, list):
                    task_prompt = " ".join(b.get("text", "") for b in task_prompt if isinstance(b, dict))
                scores = direct_judge(steps, task_prompt[:400], client=judge_client, model=args.judge_model)
                direct_scores[task_key] = scores
                top = sorted(scores, key=lambda s: s.score, reverse=True)[:2]
                print(f"  {task_key[:40]}")
                for s in top:
                    print(f"    turn {s.turn_idx:>2d}  score={s.score:.3f}  {s.rationale[:80]}")

        except Exception as e:
            logger.error(f"Direct judge failed: {e}")

    # Write judge results
    judge_out_path = os.path.expanduser(args.judge_output)
    os.makedirs(os.path.dirname(judge_out_path) or ".", exist_ok=True)

    def _scores_to_dict(sm):
        return {tk: [{"turn_idx": s.turn_idx, "score": s.score, "rationale": s.rationale} for s in sc]
                for tk, sc in sm.items()}

    with open(judge_out_path, "w") as f:
        json.dump({
            "divergence": _scores_to_dict(div_scores),
            "direct_judge": _scores_to_dict(direct_scores),
            "calibration": {
                "divergence": calibrate_batch(div_scores, task_to_rewards) if div_scores else {},
            },
            "task_rewards": task_to_rewards,
        }, f, indent=2)

    logger.info(f"\nJudge results → {judge_out_path}")
    logger.info(f"Traces saved → {traces_path}")
    print(f"\nDone. {len(all_records)} traces, judge results at {judge_out_path}")


if __name__ == "__main__":
    main()
