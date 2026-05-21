# Ticketmaster Qualitative Results Analysis

This note summarizes candidate Ticketmaster eval trajectories where the verifier-only GRPO run failed but the taste-shaped GRPO run received `reward >= 0.5`, which corresponds to verifier success under the reward construction:

```text
reward = 0.5 * verifier_binary
       + (redundancy + consistency + recovery + visual_grounding) / 40
```

Important caveat: these examples are from the same fixed 100-task Ticketmaster subset used for the controlled overnight ablation. They are useful for qualitative comparison, but not a clean held-out generalization claim.

## Summary

| Row | Verifier-only Score | Taste-shaped Reward | Qualitative Usefulness |
|---:|---:|---:|---|
| 13 | 0.00 | 0.600 | Mixed. Taste run reaches event/ticket page but appears repetitive afterward. |
| 40 | 0.00 | 0.650 | Moderate. Taste run explores event listings more productively than verifier-only. |
| 63 | 0.00 | 0.675 | Weak. Taste run is marked successful by reward threshold, but visual trace looks repetitive. |
| 70 | 0.00 | 0.700 | Strongest. Taste run reaches ticket selection/checkout; verifier-only loops on search. |

Recommended slide example: **row 70**. It has the clearest qualitative contrast.

## Row 70: Strongest Example

**Task.** Buy concert tickets for music events in Los Angeles. The task includes verifier feedback requiring that a purchase be made for `robertbrown@msn.com`.

**Verifier-only GRPO behavior.**

- Starts correctly by accepting cookies and entering `Los Angeles`.
- Then repeatedly cycles through search-field clicks, `Taylor Swift` searches, search button clicks, and scrolls.
- Does not reach ticket selection or checkout.
- Top repeated actions:
  - `scroll down 500`: 21 times
  - `click [742, 222]`: 18 times
  - `click [597, 227]`: 18 times
  - `type 'Taylor Swift'`: 16 times

**Taste-shaped GRPO behavior.**

- Accepts cookies, searches Los Angeles, identifies an event, clicks `Get Tickets`.
- Proceeds through ticket terms, ticket selection, checkout, saved payment method, and `Complete Purchase`.
- The later part of the trajectory waits on a loading/session state, but it gets much farther through the purchase funnel.
- This is the best example for the story that taste-shaped training pushed behavior toward purposeful task completion.

**Screenshots.**

Verifier-only:

![Row 70 verifier-only contact sheet](analysis_assets/ticketmaster_qual/row_0070_verifier_contact.jpg)

Taste-shaped:

![Row 70 taste-shaped contact sheet](analysis_assets/ticketmaster_qual/row_0070_taste_contact.jpg)

## Row 40: Moderate Example

**Task.** Buy tickets for a live concert happening soon, preferably under `$300` per ticket, and provide order confirmation.

**Verifier-only GRPO behavior.**

- Dismisses cookies and searches for `Taylor Swift`.
- Then gets stuck repeatedly clicking around essentially the same region.
- Top repeated action:
  - `click [836, 362]`: 75 times

**Taste-shaped GRPO behavior.**

- Performs broader search attempts and reaches visible event listings.
- The trajectory shows more exploration of available concerts rather than immediate single-coordinate thrashing.
- Still not as clean as row 70, so use this as a secondary example only if needed.

**Screenshots.**

Verifier-only:

![Row 40 verifier-only contact sheet](analysis_assets/ticketmaster_qual/row_0040_verifier_contact.jpg)

Taste-shaped:

![Row 40 taste-shaped contact sheet](analysis_assets/ticketmaster_qual/row_0040_taste_contact.jpg)

## Row 13: Mixed Example

**Task.** Buy 2 tickets for a live music concert after December 8, 2025, using the saved payment method and provide order confirmation.

**Verifier-only GRPO behavior.**

- Searches for `music concert` / `concert`.
- Repeatedly alternates between search-button clicks and scrolling.
- Fails with repeated tool-call errors and does not make meaningful progress.
- Top repeated actions:
  - `click [742, 226]`: 41 times
  - `scroll down 500`: 36 times

**Taste-shaped GRPO behavior.**

- Dismisses cookie popup, searches, and reaches an event/ticket page.
- However, it then appears to repeat clicks around the ticket terms / selection area.
- This is a weaker slide example because the trajectory is still visibly repetitive despite crossing the success threshold.

**Screenshots.**

Verifier-only:

![Row 13 verifier-only contact sheet](analysis_assets/ticketmaster_qual/row_0013_verifier_contact.jpg)

Taste-shaped:

![Row 13 taste-shaped contact sheet](analysis_assets/ticketmaster_qual/row_0013_taste_contact.jpg)

## Row 63: Weak Example

**Task.** Buy 2 tickets for an upcoming music/concert event in a major city, including total cost breakdown and confirmation details.

**Verifier-only GRPO behavior.**

- Tries several artist searches.
- Mostly repeats search-button clicks.
- Top repeated action:
  - `click [742, 223]`: 69 times

**Taste-shaped GRPO behavior.**

- Accepts cookies and enters `New York`.
- The screenshot trace appears repetitive around the search/location fields.
- I would not use this one in the deck unless a more detailed manual review finds a later verifier-relevant state not obvious from the contact sheet.

**Screenshots.**

Verifier-only:

![Row 63 verifier-only contact sheet](analysis_assets/ticketmaster_qual/row_0063_verifier_contact.jpg)

Taste-shaped:

![Row 63 taste-shaped contact sheet](analysis_assets/ticketmaster_qual/row_0063_taste_contact.jpg)

## Suggested Slide Framing

Use row 70 as the qualitative example:

```text
Verifier-only GRPO gets stuck in search loops.
Taste-shaped GRPO reaches the ticket purchase flow and checkout.
```

Tie it back to the rubric:

- **Redundancy:** fewer repeated search attempts before progressing.
- **Recovery:** moves from search results into event selection instead of retrying the same search loop.
- **Visual grounding:** acts on visible `Get Tickets`, ticket terms, seat selection, and checkout controls.
- **Consistency:** stated intent matches the next purchase-flow action more often.

## Source Files

- Verifier-only eval JSONL: `local_runs/tm-grpo-verifier-s42/global_step_4/ticketmaster.jsonl`
- Taste-shaped eval JSONL: `local_runs/tm-grpo-taste-s42-v2/global_step_4/ticketmaster.jsonl`
- Contact sheets: `analysis_assets/ticketmaster_qual/`

