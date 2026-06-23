# TERMS-Bench reimplementation spec (faithful to arXiv:2605.13909)

Reconstructed from the paper appendices because the official repo
(`github.com/zou-group/terms-bench`) is **not public**. All constants live in
`config.py` (`TermsConfig`, `FAMILIES`). This file is the equation reference.

Notation: price bounds `[p_min, p_max]`, range `R = p_max - p_min`. Horizon `K`.
Round index `k = 1..K`. Counterpart type `t_B = (r_B, kappa_B, eta_B)`:
reservation value, urgency in `[0,1]`, stance in {conciliatory, neutral, aggressive}.
`sigma(x) = 1/(1+exp(-x))`. Deadline clock `Dtil_k = sqrt(k/K)`, `Dbar_k = 1 - Dtil_k`.

The counterpart may be **buyer or seller** (opposite of the agent). Define seller
direction `s_B = +1` if counterpart is seller else `-1`. Favorable direction for the
counterpart is `s_B*(p - r_B) > 0`.

---

## 1. Regime generator (`scenarios.py`; §3.1)

Sample once per `(family, agent_role, opener, episode_index)` cell using the seeding
scheme below. Midpoint `m` is sampled within bounds so reservations stay in `[p_min,p_max]`.

- **overlap**: `z ~ U[zopa_min, zopa_max]`; `r_buyer = m + z/2`, `r_seller = m - z/2`.
  `kappa_B ~ Beta(urgency_alpha, urgency_beta)` rescaled to `[0,1]`.
- **urgency_shift**: same reservation geometry as overlap, but
  `kappa_B ~ Beta(urgency_shift_alpha, urgency_shift_beta)` (counterpart-more-urgent).
- **no_deal**: `g ~ U[gap_min, gap_max]`; `r_buyer = m - g/2`, `r_seller = m + g/2`
  (so `delta = r_buyer - r_seller = -g < 0`). Baseline urgency Beta.

`eta_B` is drawn from the **family** stance prior (`FamilyPreset.stance_prior`;
uniform for core families, `(0.05,0.15,0.80)` for adversarial). `d0e ~ U[d0_min,d0_max]`.

### Seeding (H.1.3)
Given `base_seed b`, family index `f`, agent_role index `r`, opener index `o`,
episode index `e`: `cell = b*1e7 + f*1e5 + r*1e4 + o*1e3 + e*10`. Use disjoint
streams `cell + i` for the latent draws. A separate shared stream draws a geometry
percentile `u_e ~ U(0,1)` mapped to `z_e = zopa_min + u_e*(zopa_max - zopa_min)` and
`q_e = gap_min + u_e*(gap_max - gap_min)` so overlap/no-deal siblings differ only by sign.
The point: identical scenario set across evaluated models. Use Python `random.Random(seed)`
per stream; document that this is the chosen RNG.

### Episode grid (H.1.2)
Full suite per family = `3 regimes x 2 agent_roles x 2 openers x n_per_cell`.
`n_per_cell=25` reproduces the paper's 100 episodes/(regime,family) and 1,800 total.
The harness exposes `n` as the **total target** episodes and balances across cells.

---

## 2. Counterpart kernel `pi_B` (`kernel.py`; §3.2, Appendix C)

History features over the agent's own offers (C.3). Let `s_A = +1` if the **agent** is
a buyer (concession = raising price) else `-1`. With `J_k = {j : max(2,k-3) <= j <= k-1}`
indexing rounds where two consecutive agent offers exist:

- `ConcedeMagnitude_k = mean_{j in J_k} max(0, s_A*(p^A_j - p^A_{j-1})) / R`  (0 if `|J_k|<1`)
- `ConcedeSpeed_k    = mean_{j in J_k} (s_A*(p^A_j - p^A_{j-1})) / R`         (0 if `|J_k|<1`)
- `Rigidity_k = 1` if `max(0, s_A*(p^A_{k-1} - p^A_{k-2}))/R < tau_rigid` else 0
  (needs last two agent offers; else 0)

If fewer than two agent offers exist, all three are 0.

### 2a. Acceptance (eqs. 5-6)
Role-normalized favorability of the agent's current offer `p^A_k`:
`Delta_k = (p^A_k - r_B)/R` if counterpart is **seller**, else `(r_B - p^A_k)/R`.
(So `Delta_k >= 0` iff individually rational for the counterpart.)

```
g = alpha*Delta_k + beta*kappa_B - gamma*Dbar_k
      + rho_F(eta_B)*ConcedeSpeed_k + xi_F(eta_B)*Rigidity_k
a_k = 1{Delta_k >= 0} * sigma(g)
```
`rho_F, xi_F` come from the family preset indexed by stance.

### 2b. Walk-away (eqs. 7/21), only if not accepted
```
tauW_k = clip((k - k_walk)/(K - k_walk), 0, 1)
omega_k = 1{k >= k_walk} * 1{Delta_k < 0} * sigma(phi0 + phi_delta*max(0,-Delta_k) + phi_T*tauW_k)
```
Walk-away => counterpart decision `Reject` (terminal, no deal).

### 2c. Counter-offer (Table 5), if neither accept nor walk-away and `k < K`
Latent concession score:
```
lam = lambda0 + lambda1*kappa_B - lambda2_F(eta_B)*ConcedeMagnitude_k
        - lambda3*1{eta_B==aggressive} + lambda4*1{eta_B==conciliatory}
lam = clip(lam, 0, 1)
```
Price update moves weakly toward `r_B` (so `(p - r_B)` shrinks by factor `(1-lam)`):
```
p_target = p^B_{k-1} + lam*(r_B - p^B_{k-1}) + eps,   eps ~ N(0, sigma_p^2), sigma_p = price_noise*R
```
Project onto monotone feasible interval `M_B(k)`:
seller counterpart `[r_B, p^B_{k-1}]`, buyer counterpart `[p^B_{k-1}, r_B]`.
If noise would reverse the concession direction, **hold** at `p^B_{k-1}`.

### 2d. Opening offer (eqs. 15-16), used whenever the counterpart makes its FIRST offer
```
S_open = (p_max - r_B) if counterpart seller else (r_B - p_min)
phi = clip(1 - omega_kappa*kappa_B + omega_eta*1{aggr} - omega_eta_prime*1{conc}, phi_min, phi_max)
p_init = project_to([r_B,p_max] if seller else [p_min,r_B],
                    r_B + s_B*d0e*phi*S_open + eps0),  eps0 ~ N(0, sigma0^2), sigma0 = sigma0_bar*R
```

### 2e. Cues (C.5) — drive the (optional) message; logged for belief scoring
Counterpart concession magnitude (for cue logits):
`C^B_k = min(1, |p^B_k - p^B_{k-1}| / (|p^B_{k-1} - r_B| + eps_c))` if Offer with prev offer, else 0.

Strategic cue base:
- Accept -> `Concede`; Reject -> `Pressure`.
- Offer -> logits over (Concede, Hold, Pressure):
  - bias `b(eta)`: conciliatory `(b_C, 0, -b_C)`, neutral `(0, b_H, 0)`, aggressive `(-b_P, 0, b_P)`
  - `l(Concede) = b[0] + alpha_C*(C^B_k - tau_conc)`
  - `l(Hold)    = b[1]`
  - `l(Pressure)= b[2] + alpha_P*(Dtil_k - tau_dead) - beta_C*C^B_k`
  - sample `Categorical(softmax(l))`.

Sentiment cue base: `z = mu(eta) + N(0, sigma_s^2)` with `mu = +mu_s/0/-mu_s` for
conc/neutral/aggr; `positive` if `z>tau_s`, `negative` if `z<-tau_s`, else `neutral`.

Family cue overrides (C.5.3): base for candid/expressive; **uninformative**
(`neutral`/`Hold`) for taciturn/strategic; **noisy** for stochastic (sentiment with
`sigma_s_stoch`, strategy `softmax(l / T_stoch)`); **pressuring** (`negative`/`Pressure`)
for adversarial.

### Opener protocol
- `CounterpartOpens`: counterpart emits opening offer (2d) at round 1.
- `AgentOpens`: agent offers first; `Accept` unavailable round 1. Counterpart then runs
  the response model; if it neither accepts nor walks away and must produce its first
  offer, it uses (2d).

---

## 3. Agent interface (`prompts.py`; Appendix K)

System prompt (buyer variant verbatim in the paper; seller is symmetric with
`u(p)=p-reservation`, IR `counterpart_offer >= reservation`, monotonically non-increasing
seller offers). Each round the agent gets a single JSON user message with keys:
`private_context` (role, reservation_price), `protocol_state` (round_number, max_rounds,
rounds_remaining, counterpart_offer present?, legal decision set, own_previous_offer),
`constraints` (price_bounds, monotone-concession rule), `observation`
(counterpart_offer/price, counterpart_message, accept-utility u_A when an offer is on the
table), `history` (last W=6 rounds).

Agent output JSON schema (must parse):
```
{ "decision": "Offer"|"Accept"|"Reject",
  "price": <float>|null,
  "message": <string>,
  "belief": { "r_hat": <float>, "kappa_hat": <float>,
              "stance_probs": {"conciliatory":<f>,"neutral":<f>,"aggressive":<f>} } }
```
Parser must be tolerant (strip code fences/prose, extract first JSON object) and record a
`parse_error` (counts as an invalid-action critical violation) when it cannot.

Voice layer is **cosmetic** (§H.1.4, Appendix K Fig 22). Default = deterministic templates
keyed on (decision, cue); optional `--voice-model` may render messages but must never
change the economic action/price.

---

## 4. Metrics (`metrics.py`; Table 1, Appendix F)

Per episode: `delta_i = r_buyer - r_seller`; feasible set `I+ = {delta_i>0}`,
infeasible `I- = {delta_i<0}`. Agent utility `u_A(f) = r_A - p` (buyer) or `p - r_A`
(seller) at deal price `p`; `0` if no deal.

- `AGR+ = mean over I+ of 1[agreement]`
- `CSE+ = mean over agreed feasible of u_A(f_i)/delta_i`  (undefined if none agreed)
- `SE+  = AGR+ * CSE+`
- `FAGR- = mean over I- of 1[agreement]`  (lower better; safe-term = 1 - FAGR-)
- Belief error (from the agent's self-reported belief, averaged over rounds it reported):
  `BE_r = mean |r_hat - r_B| / (p_max - p_min)`,
  `BE_kappa = mean |kappa_hat - kappa_B|`,
  `Brier_eta = mean sum_c (stance_probs[c] - 1[c == eta_B])^2`,
  `BE_type = (BE_r + BE_kappa + Brier_eta)/3`.
- `CritViol% = mean over all episodes of 1[any critical violation]`. Critical violations:
  price-bound (offer outside `[p_min,p_max]`), individual-rationality (Accept/Offer a price
  strictly worse than `r_A`), invalid action (unparseable/illegal decision). Monotonicity
  and turn-budget are **secondary** (track separately, not in CritViol%).

Report aggregates overall and broken down by regime and by family. Conditional metrics are
`null` when their denominator is empty (never impute 0).

---

## 5. Determinism
Everything stochastic in the kernel/scenarios uses a per-episode `random.Random` seeded
from the scenario seed, so a fixed `base_seed` yields the same scenario set and the same
counterpart draws **given identical agent actions** (the agent can still steer trajectories).
