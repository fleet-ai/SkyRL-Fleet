# Are agents using their value function, or vibe-negotiating?

_Analysis of `eval/results` traces · gpt-4o-mini & qwen3.5-35b-a3b · single-proposer vs dual-tag._
_Reproduce: `python3 eval/analyze_value_alignment.py` (metrics dumped to `value_alignment_metrics.json`)._

## TL;DR

- **Both models use their value function, but only when the protocol forces them to claim.** Under **dual-tag** (each side states its own keep) allocations are strongly value-aligned. Under **single-proposer** the *accepter* frequently rubber-stamps a split that ignores its own values — i.e. vibe-accepts.
- **Your hypothesis holds.** Dual aligns better than single for both models, on the *same* dnd scenarios:
  - qwen3.5-35b-a3b: **55% → 82%** value-capture (+28 pts)
  - gpt-4o-mini: **68% → 79%** value-capture (+11 pts)
- The gap is almost entirely the **single-proposer accepter**: in single, the proposer designs a value-greedy split for itself (qwen proposer 68%) while the accepter takes whatever it's handed (qwen accepter **40%**, near random). Dual removes the passive role, so both sides chase value.

## Metric

**Value-capture (conditional on quantity).** For one side of an agreed deal that ends up with `T` units: compare its achieved points to the *best* and *worst* it could have done with *that many* units (grab the `T` highest- vs lowest-value units in the pool).

```
value_capture = (achieved - worst_for_T) / (best_for_T - worst_for_T)
```

This controls for *how much* a side won and isolates the real question: **of the units it took, were they the high-value ones?** A value-blind negotiator who ends with a random `T` units scores ≈0.5; one that actually steers toward its high-value items scores ≈1.0. (Skipped when `T=0`, `T=all`, or all values equal.)

## Results

`VC prop` = proposer/"you" side, `VC acc` = accepter/"them" side. `0-conc` = of value-0 items, fraction correctly given away (dnd only). `top` = secured all of its single highest-value item. `1stoff` = value-capture of the opener's first concrete offer. `talk` = fraction of negotiations whose prose explicitly reasons about item value/priority (you/them).

| model | data | proto | deals | VC both | VC prop | VC acc | 0-conc | top | 1stoff | talk y/t |
|---|---|---|---|---|---|---|---|---|---|---|
| gpt-4o-mini | dnd | single | 19 | 67% | 67% | 67% | 63% | 58% | 86% | 50% / 85% |
| gpt-4o-mini | dnd | **dual** | 24 | **74%** | 75% | 73% | 78% | 55% | 75% | 27% / 40% |
| gpt-4o-mini | casino | single | 20 | 54% | 51% | 57% | – | 20% | 50% | 10% / 40% |
| gpt-4o-mini | casino | dual | 27 | 57% | 54% | 60% | – | 17% | 54% | 33% / 57% |
| qwen3.5-35b-a3b | dnd | single | 20 | 55% | **68%** | **40%** | 62% | 34% | 79% | 75% / 65% |
| qwen3.5-35b-a3b | dnd | **dual** | 10 | **81%** | 81% | 81% | 83% | 79% | 81% | 75% / 83% |
| qwen3.5-35b-a3b | casino | single | 20 | 55% | 57% | 54% | – | 8% | 57% | 35% / 60% |

**Paired (same dnd scenarios, agreed under *both* protocols):**

| model | # shared | single | dual | Δ |
|---|---|---|---|---|
| gpt-4o-mini | 14 | 68% | 79% | **+11%** |
| qwen3.5-35b-a3b | 10 | 55% | 82% | **+28%** |

**"Someone got burned" rate** (a side's value-capture < 0.34, i.e. closer to the *worst* placement of its units than the middle):

| | single | dual |
|---|---|---|
| gpt-4o-mini (dnd) | 26% | 15% |
| qwen3.5-35b-a3b (dnd) | 32% | **11%** |

## Why single-proposer hides value-awareness (the smoking gun)

dnd scenario 0 — pool `1 book / 1 hat / 3 balls`; "you" values **book = 9**, hat = 1, **ball = 0**. The single most valuable item is the book.

**Dual (value-capture 1.0):** "you" claims the book + hat, gives away the worthless balls. Clean, value-grounded, both `<deal>` tags consistent.

**Single (value-capture 0.10 — "you" loses its 9-point book):**

```
[you]  "I value the book at 9 points and the other items at 0 ... I propose keeping the book."
       <propose>{"book": 1, "hat": 0, "ball": 0}</propose>          ← perfect value reasoning

[them] "...How about you keep the book and 1 ball."                 ← prose says YOU keep the book
       <propose>{"book": 1, "hat": 0, "ball": 1}</propose>          ← but the TAG claims the book for THEM

[you]  "Your counter-offer gives me the book and one ball ... I accept."   ← read the prose, not the tag
       <accept>
```

"you" opens with textbook value reasoning, then **accepts based on the natural-language story** ("gives me the book") **without checking the formal tag against its own values**. The tag actually handed its 9-point book to the partner. Result: `you_take = [0 book, 1 hat, 2 balls]` → **1 of a possible 10 points**.

This is the canonical vibe-negotiating failure, and it is *structural* to single-proposer: the accepter's only lever is accept/reject, and here it didn't verify the offer against its value vector. It shows up as the depressed `VC acc` column (qwen 40%) and the higher single "burned" rate.

## Dataset note: casino is a weak probe

On CaSiNo, value-capture sits near 0.5 for everyone (54–60%) and `top`-item capture is very low — but this is largely **metric insensitivity, not pure vibing**: CaSiNo values are a compressed permutation of {5,4,3} with no zero items, so a value-blind even split and a value-greedy split score almost the same and there is little to gain from concentrating on the top item. dnd (values 0–10, summing to 10, with genuine zeros) is the discriminating probe — and there the protocol effect is clear.

## Takeaways for RLVR

- **Agents *can* read and use their values** — the proposer side and every opener (`1stoff` 75–86% on dnd) are strongly value-aligned. The failure is **passive acceptance**, not comprehension.
- **The outcome reward already punishes vibe-accepting** (the burned accepter scores near 0), so RL on outcome reward should directly push accepters to verify offers against their value function before `<accept>`. This is exactly the headroom the report flags for the 7B train target.
- **Single-proposer trades a coordination tax for an acceptance tax:** it removes spurious `conflict`/`no_deal`, but introduces value-blind acceptances that don't exist in dual (where both sides must actively claim). Worth weighting/curriculum-mixing dual scenarios if the training goal is *value-aware* negotiation rather than just *closing* deals.
