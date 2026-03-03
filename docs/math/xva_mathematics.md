# XVA Engine — Mathematical Reference
## End-to-End Walkthroughs: Zero Coupon Bond & Interest Rate Swap

This document traces two trades — a Zero Coupon Bond (ZCB) and a payer Interest Rate Swap (IRS) — through the complete XVA engine pipeline. Every formula maps 1-to-1 to code in `src/xva_engine/`. Each section gives the **intuition first**, then the **mathematics**.

---

# Part 1 — Zero Coupon Bond (ZCB)

**In plain English:** You lend $1,000,000 to a counterparty today. They promise to pay it back in full, two years from now, with no coupons in between. You carry the credit risk the entire time — if they default before maturity, you lose money.

---

## 1. Trade Definition

**Intuition:** A ZCB has exactly one cash flow: the counterparty pays you the notional $N$ at maturity $T$. Between now and $T$, the bond has a positive market value to you (you are owed money), which creates one-sided credit exposure throughout the trade's life.

$$\boxed{V_{ZCB}(T, \omega) = N \quad \forall \omega}$$

**Example trade:**
| Field        | Value              |
| ------------ | ------------------ |
| trade_id     | ZCB-1              |
| notional $N$ | 1,000,000 USD      |
| maturity $T$ | 730 days (2 years) |

---

## 2. Market Data

### 2.1 Discount Curve

**Intuition:** A dollar paid two years from now is worth less than a dollar today. The discount curve tells us *how much less*. $P(0, t)$ is the price today of receiving $1 at time $t$ — it is always below 1 for $t > 0$ in a positive rate environment.

**Log-linear interpolation** between pillars (`market/curve.py`):

$$\ln P(0, t) = \ln P(0, t_i) + \frac{t - t_i}{t_{i+1} - t_i} \left[ \ln P(0, t_{i+1}) - \ln P(0, t_i) \right]$$

*Why log-linear?* Interpolating in log space assumes a **piecewise-constant forward rate** between pillars — a standard market convention. For dates beyond the last pillar the last forward rate is held flat.

The continuously-compounded zero rate:

$$r(0, t) = -\frac{\ln P(0, t)}{t / 365}$$

**Example curve:**
| Tenor (days) | $P(0, t)$ | Zero rate |
| ------------ | --------- | --------- |
| 0            | 1.0000    | 0.00%     |
| 182          | 0.9880    | 2.42%     |
| 365          | 0.9750    | 2.53%     |
| 730          | 0.9500    | 2.56%     |
| 1825         | 0.8800    | 2.48%     |

So the ZCB with $N = 1{,}000{,}000$ has an initial present value of $0.950 \times 1{,}000{,}000 = 950{,}000$ USD.

### 2.2 Credit Curves

**Intuition:** The survival probability $S(t)$ is the chance the counterparty is still alive (has not defaulted) at time $t$. It starts at 1.0 today and decays over time. The faster it decays, the riskier the counterparty.

$$S(t) = \exp\!\left(-\int_0^t h(u)\,du\right)$$

$h(t)$ is the **hazard rate** — the instantaneous rate of default. Log-linear interpolation of $S(t)$ is equivalent to a **piecewise-constant hazard rate** between pillars.

**Marginal default probability** (unconditional) in bucket $[t_{i-1}, t_i]$ (`market/credit.py`):

$$\Delta PD(t_{i-1}, t_i) = S(t_{i-1}) - S(t_i)$$

*What this means:* The probability the counterparty defaults specifically in this time bucket — not before, not after. This is what we weight each exposure bucket by in CVA.

The conditional probability (given survival to $t_{i-1}$):

$$PD^{\text{cond}}(t_{i-1}, t_i) = 1 - \frac{S(t_i)}{S(t_{i-1})}$$

`CreditCurveModel.marginal_pd()` implements the unconditional definition.

---

## 3. Hull-White 1-Factor Model

**Intuition:** Interest rates are not fixed — they move randomly over time. The Hull-White model describes *how* rates move. It says: rates have a tendency to drift back toward a long-run level (mean reversion), but are also pushed around by random shocks. We need this model to figure out what the ZCB is worth at each future date under each random scenario.

### 3.1 Short Rate Process

The model decomposes the short rate into a deterministic part $\varphi(t)$ (which exactly fits today's curve) and a random part $x_t$:

$$r_t = \varphi(t) + x_t$$

The random factor $x_t$ follows an **Ornstein-Uhlenbeck** process — pulled back to zero at speed $a$, while $\sigma$ controls how violently it is pushed around:

$$dx_t = -a\,x_t\,dt + \sigma\,dW_t, \quad x_0 = 0$$

| Symbol   | Name                 | Typical value | Intuition                               |
| -------- | -------------------- | ------------- | --------------------------------------- |
| $a$      | Mean reversion speed | 0.05          | Higher = rates snap back to mean faster |
| $\sigma$ | Volatility           | 0.01          | Higher = rates jump around more         |

### 3.2 Exact Simulation Step

For each time step $\Delta t$, the next value of $x_t$ is drawn exactly (no approximation error):

$$x_{t+\Delta t} = x_t \cdot \underbrace{e^{-a\,\Delta t}}_{\text{decay toward zero}} + \underbrace{\sigma\sqrt{\frac{1 - e^{-2a\,\Delta t}}{2a}}\cdot Z}_{\text{random shock}}, \quad Z \sim \mathcal{N}(0,1)$$

### 3.3 Bond Pricing Under the Model

**Intuition:** On each Monte Carlo path, $x_t$ tells us where rates are at time $t$. High $x_t$ (high rates) → bond price falls. Low $x_t$ → bond price rises.

$$\boxed{P(t, T \mid x_t) = \frac{P(0, T)}{P(0, t)}\exp\!\left(-B(t,T)\,x_t - \tfrac{1}{2}V(t,T)\right)}$$

$$B(t, T) = \frac{1 - e^{-a(T-t)}}{a} \qquad \text{(sensitivity to } x_t\text{, like duration)}$$

$$V(t, T) = \frac{\sigma^2}{2a}\,B(t,T)^2\,\left(1 - e^{-2at}\right) \qquad \text{(Jensen convexity correction)}$$

- $B(t,T)$: how much the bond price reacts to $x_t$. At maturity $B = 0$ — no sensitivity left.
- $V(t,T)$: because $\exp$ is convex, the average bond price is slightly *above* the price at the average rate.
- At maturity $t = T$: $B = 0$, $V = 0 \Rightarrow P(T,T) = 1$ ✓

**ZCB mark-to-market on path $j$ at time $t_i$:**

$$V_{ZCB}(t_i, j) = N \cdot P\!\left(t_i/365,\; T/365 \mid x_{t_i}^{(j)}\right)$$

---

## 4. Simulation Time Grid

**Intuition:** We only need to know the portfolio value at a finite set of dates. The engine builds this grid automatically from all relevant cash flow dates.

Grid construction (`sim/timegrid.py`):
1. Always include $t = 0$
2. Include the ZCB maturity $T$
3. Include all IRS payment dates
4. Optionally add dense monitoring points every $\delta$ days

$$\mathcal{T} = \{0, 182, 365, 548, 730\} \text{ days (ZCB-only example)}$$

---

## 5. Monte Carlo Engine

**Intuition:** We cannot know exactly where rates will be in the future. Instead, we simulate thousands of possible futures (paths), compute the portfolio value on each path at each date, and average the results.

### 5.1 Path Generation (`sim/batching.py`)

1. Fix a random seed for reproducibility
2. Draw $Z_{i,j} \sim \mathcal{N}(0,1)$ for each grid step $i$ and path $j$
3. Propagate $x_{t_i}^{(j)}$ via the exact OU step
4. Output: matrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{T}| \times M}$

### 5.2 Pathwise Portfolio Valuation

At each grid point $t_i$ and path $j$, value every trade and sum them:

$$V_{NS}(t_i, j) = \sum_{k \in \text{trades}} V_k\!\left(t_i,\, x_{t_i}^{(j)}\right)$$

**For ZCB-1:**

$$V_{ZCB}(t_i, j) = \begin{cases} N \cdot P\!\left(t_i/365,\; T/365 \mid x_{t_i}^{(j)}\right) & t_i < T \\ 0 & t_i \ge T \end{cases}$$

Since the ZCB is always worth $N \ge 0$ at maturity and model prices are always positive, **every path** has $V_{ZCB}(t_i, j) > 0$ for $t_i < T$.

---

## 6. Collateral (`exposure/collateral.py`)

**Intuition:** If there is a collateral agreement (CSA), the counterparty posts cash equal to the mark-to-market value. This reduces the loss if they default — you already hold their collateral. `perfect_vm` means full daily margining (typical for cleared trades).

| CSA mode     | Collateral $C(t, \omega)$                    | Effect                             |
| ------------ | -------------------------------------------- | ---------------------------------- |
| `none`       | $0$                                          | Full credit exposure               |
| `perfect_vm` | $V_{NS}(t, \omega)$                          | Exposure drops to zero             |
| `threshold`  | $\max(V_{NS} - H, 0) - \max(-V_{NS} - H, 0)$ | Partial protection above threshold |

Sign convention: $C > 0$ = bank holds collateral (received); $C < 0$ = bank has posted collateral.

---

## 7. Exposure (`exposure/exposure.py`)

**Intuition:** We only suffer a loss if two things happen simultaneously: the counterparty defaults *and* we are owed money. The exposure is the net amount we'd lose in that scenario.

### 7.1 Pathwise Exposures

$$E(t_i, j) = \max\!\left(V_{NS}(t_i, j) - C(t_i, j),\; 0\right) \quad \text{(counterparty owes us)}$$

$$NE(t_i, j) = \max\!\left(C(t_i, j) - V_{NS}(t_i, j),\; 0\right) \quad \text{(we owe counterparty)}$$

### 7.2 Expected Exposures

Average over all paths:

$$EE(t_i) = \frac{1}{M}\sum_{j=1}^{M} E(t_i, j)$$

$$ENE(t_i) = \frac{1}{M}\sum_{j=1}^{M} NE(t_i, j)$$

**For ZCB with no CSA:** $V_{ZCB}(t) > 0$ on all paths $\Rightarrow$ $EE(t_i) \approx \mathbb{E}[V_{ZCB}(t_i)]$ and $ENE(t_i) \approx 0$.

---

## 8. XVA Calculations

### 8.1 CVA — Credit Valuation Adjustment (`xva/cva.py`)

**Intuition:** CVA is the expected cost of counterparty default. For each future time bucket, ask: "What is the chance they default in this period? If they do, how much do I lose?" Multiply and sum across all buckets.

$$\boxed{CVA = (1 - R_c)\sum_{i=1}^{n} P(0, t_i)\cdot EE(t_i)\cdot \Delta PD_c(t_{i-1}, t_i)}$$

| Symbol        | Meaning                                                  |
| ------------- | -------------------------------------------------------- |
| $R_c$         | Recovery rate — fraction you get back even after default |
| $(1-R_c)$     | Loss given default                                       |
| $P(0, t_i)$   | Discount factor — future losses are worth less today     |
| $EE(t_i)$     | Average amount you'd lose if default occurs now          |
| $\Delta PD_c$ | Probability of default specifically in this bucket       |

**For ZCB:** CVA > 0 always — you are owed money the entire time.

### 8.2 DVA — Debt Valuation Adjustment (`xva/dva.py`)

**Intuition:** DVA is the mirror of CVA — the benefit to us from our *own* potential default. If we default and we owe money, they lose — so from our accounting perspective, this is a benefit.

$$\boxed{DVA = (1 - R_b)\sum_{i=1}^{n} P(0, t_i)\cdot ENE(t_i)\cdot \Delta PD_b(t_{i-1}, t_i)}$$

**For ZCB held long:** $ENE \approx 0$ $\Rightarrow$ DVA $\approx 0$.

### 8.3 FVA — Funding Valuation Adjustment (`xva/fva.py`)

**Intuition:** If the trade is uncollateralised, you effectively funded the counterparty's position. You borrow in the market at a spread above risk-free to do so. FVA is the cost of that funding.

$$\boxed{FVA = \sum_{i=1}^{n} P(0, t_i)\cdot s_f(t_i)\cdot EE_{net}(t_i)\cdot \Delta t_i}$$

$s_f$ = funding spread, $\Delta t_i = (t_i - t_{i-1}) / 365$.

### 8.4 MVA — Margin Valuation Adjustment (`xva/mva.py`)

**Intuition:** For cleared trades you must post initial margin — cash locked up at the CCP. Funding that cash has a cost.

$$\boxed{MVA = \sum_{i=1}^{n} P(0, t_i)\cdot s_f(t_i)\cdot \mathbb{E}[IM(t_i)]\cdot \Delta t_i}$$

### 8.5 KVA — Capital Valuation Adjustment (`xva/kva.py`)

**Intuition:** Regulators require capital to be held against trading book risk. Equity investors expect a return on that capital — KVA is the cost of providing it.

$$\boxed{KVA = \sum_{i=1}^{n} P(0, t_i)\cdot c_{cap}(t_i)\cdot \mathbb{E}[K(t_i)]\cdot \Delta t_i}$$

### 8.6 Net XVA

$$\text{Net XVA} = CVA - DVA + FVA + MVA + KVA$$

A positive Net XVA is a **cost** to the dealer — it reduces the mark-to-market profit on the trade.

---

## 9. Numerical Example — ZCB-1

**Inputs:** $N = 1{,}000{,}000$ USD, $T = 730$ days, $R_c = 0.40$, $S_c(365) = 0.98$, $S_c(730) = 0.96$, no CSA.

| Bucket | $t_i$ | $P(0,t_i)$ | $EE(t_i)$ ≈ | $\Delta PD_c$ | CVA bucket                                                      |
| ------ | ----- | ---------- | ----------- | ------------- | --------------------------------------------------------------- |
| 1      | 1 yr  | 0.975      | ~975,000    | 0.020         | $(1-0.4) \times 0.975 \times 975{,}000 \times 0.020 = 11{,}408$ |
| 2      | 2 yr  | 0.950      | ~950,000    | 0.020         | $(1-0.4) \times 0.950 \times 950{,}000 \times 0.020 = 10{,}830$ |

**Total CVA ≈ 22,238 USD** (≈ 2.3% of the initial PV of 950,000)

- DVA ≈ 0 (we never owe the counterparty anything)
- FVA > 0 if uncollateralised (spread applies to the always-positive EE profile)
- Net XVA = CVA + FVA must be charged when pricing the trade

---

---

# Part 2 — Interest Rate Swap (IRS)

**In plain English:** An Interest Rate Swap is an agreement to exchange fixed-rate payments for floating-rate payments on a notional. One party pays a fixed rate $K$ every 6 months; the other pays whatever the market floating rate turns out to be. Nobody exchanges the notional — only the rate difference matters. Because rates can move up *or* down, the swap can be an asset *or* a liability — creating exposure in **both** directions.

---

## IRS-1. Trade Definition

**Payer IRS:** We **pay fixed** rate $K$ and **receive floating** on notional $N$.

- **Fixed leg:** We pay $N \cdot \alpha_i \cdot K$ on each payment date $T_i$ ($\alpha_i$ = year fraction of period $i$).
- **Float leg:** We receive $N \cdot \alpha_i \cdot L(T_{i-1}, T_i)$ where $L$ is the floating rate set at the start of each period.

**Intuition for value:** If rates *rise* after inception, the floating receipts you receive go up but your fixed payments stay the same → swap becomes valuable to you (positive MtM). If rates *fall*, the swap loses value (negative MtM). This two-sided behaviour is the key difference from a ZCB.

**Example trade:**
| Field             | Value               |
| ----------------- | ------------------- |
| trade_id          | IRS-1               |
| notional $N$      | 1,000,000 USD       |
| maturity $T$      | 1825 days (5 years) |
| fixed rate $K$    | 2.5%                |
| payment frequency | 6 months            |
| direction         | payer (pay fixed)   |

---

## IRS-2. IRS Valuation

### At Inception ($t = 0$)

**Fixed leg PV** — known schedule, just discount it:

$$V_{\text{fixed}}(0) = N \cdot K \sum_{i=1}^{n} \alpha_i \cdot P(0, T_i)$$

**Float leg PV — par approximation:**

$$V_{\text{float}}(0) = N \cdot \left(1 - P(0, T)\right)$$

*Why?* A floating-rate bond always trades at par ($N$). Strip off the notional repayment and what remains is the float leg: $N - N \cdot P(0,T)$.

**Payer IRS value at $t = 0$:**

$$\boxed{V_{IRS}(0) = N\!\left(1 - P(0,T)\right) - N\cdot K\sum_{i=1}^{n} \alpha_i P(0, T_i)}$$

The **at-market (fair) fixed rate** $K^*$ sets $V_{IRS}(0) = 0$:

$$K^* = \frac{1 - P(0, T)}{\displaystyle\sum_{i=1}^{n} \alpha_i P(0, T_i)}$$

### Pathwise Valuation Under Hull-White

**Intuition:** On each simulation path at future date $t$, we re-price the remaining cashflows using model-consistent bond prices $P(t, T_i \mid x_t)$.

**Fixed leg at $(t, x_t)$:**

$$V_{\text{fixed}}(t, x_t) = N \cdot K \sum_{T_i > t} \alpha_i \cdot P(t, T_i \mid x_t)$$

**Float leg at $(t, x_t)$:**

$$V_{\text{float}}(t, x_t) = N \cdot \left(1 - P(t, T \mid x_t)\right)$$

**Payer IRS MtM on path $j$ at time $t_i$:**

$$\boxed{V_{IRS}(t_i, j) = N\!\left(1 - P(t_i, T \mid x_{t_i}^{(j)})\right) - N\cdot K\sum_{T_k > t_i} \alpha_k \cdot P(t_i, T_k \mid x_{t_i}^{(j)})}$$

Implemented in `products/irs.py` → `IrsPricer.pv_pathwise()`.

---

## IRS-3. Simulation Time Grid

**Intuition:** The IRS value has kinks at every coupon date — the grid must include all payment dates to capture the exposure profile accurately.

Grid for IRS-1 (5-year, semi-annual):

$$\mathcal{T} = \{0, 182, 365, 547, 730, 912, 1095, 1277, 1460, 1642, 1825\} \text{ days}$$

All payment dates are added automatically by `build_irs_schedule()`.

---

## IRS-4. Monte Carlo Engine

Same path generation as Part 1. At each grid point the netting-set value is the IRS value:

$$V_{NS}(t_i, j) = V_{IRS}(t_i, j)$$

**Key difference from ZCB:** $V_{IRS}$ can be **positive or negative** on each path:
- Paths where $x_{t_i}^{(j)} > 0$ (rates above initial curve) → floating receipts high → $V_{IRS} > 0$
- Paths where $x_{t_i}^{(j)} < 0$ (rates below initial curve) → $V_{IRS} < 0$

This means **both EE and ENE are non-zero** for an IRS — unlike the ZCB.

---

## IRS-5. Exposure

**EE and ENE formulas are identical to Part 1** — only the inputs differ.

**Exposure profile shape for payer IRS (no CSA):**
- $EE(t)$ rises initially as rate volatility accumulates, **peaks mid-life** (hump shape), then declines to zero as remaining cashflows approach zero.
- $ENE(t)$ mirrors $EE(t)$ approximately (symmetric for at-market IRS).
- Compare to ZCB: ZCB EE declines monotonically from a high level; IRS EE is much lower but two-sided.

| $t$ (yr) | ZCB EE (approx) | IRS EE (approx) | IRS ENE (approx) |
| -------- | --------------- | --------------- | ---------------- |
| 0.5      | ~980,000        | ~8,000          | ~8,000           |
| 1.0      | ~975,000        | ~14,000         | ~14,000          |
| 2.0      | ~960,000        | ~20,000         | ~20,000          |
| 3.0      | —               | ~18,000         | ~18,000          |
| 4.0      | —               | ~12,000         | ~12,000          |
| 5.0      | —               | ~0              | ~0               |

---

## IRS-6. XVA Calculations

All XVA formulas are identical to Part 1. The key differences are in the exposure inputs:

| XVA | ZCB                                   | IRS (payer, at-market)                   |
| --- | ------------------------------------- | ---------------------------------------- |
| CVA | Large (EE ≈ full notional)            | Moderate hump-shaped EE                  |
| DVA | ≈ 0 (ENE ≈ 0)                         | Non-trivial (ENE ≈ EE for at-market)     |
| FVA | Large positive (always funding asset) | Smaller; partially offsets with ENE side |
| MVA | Bilateral — not applicable            | Cleared IRS: significant initial margin  |

**CVA for IRS:**
$$CVA_{IRS} = (1 - R_c)\sum_{i=1}^{n} P(0, t_i)\cdot EE_{IRS}(t_i)\cdot \Delta PD_c(t_{i-1}, t_i)$$

**DVA for IRS:**
$$DVA_{IRS} = (1 - R_b)\sum_{i=1}^{n} P(0, t_i)\cdot ENE_{IRS}(t_i)\cdot \Delta PD_b(t_{i-1}, t_i)$$

For an at-market IRS with similar bank/counterparty credit: **CVA ≈ DVA**, so the bilateral XVA is dominated by FVA, MVA, and KVA.

---

## IRS-7. Numerical Example — IRS-1

**Inputs:** $N = 1{,}000{,}000$, $T = 1825$ days (5 yr), $K = 2.5\%$, semi-annual payer, $R_c = R_b = 0.40$, $S_c(365) = 0.98$, $S_c(1825) = 0.90$, no CSA.

**Two-bucket CVA illustration (years 0–1, 1–2):**

| Bucket | $t_i$ | $P(0,t_i)$ | $EE(t_i)$ | $\Delta PD_c$ | CVA bucket                                                     |
| ------ | ----- | ---------- | --------- | ------------- | -------------------------------------------------------------- |
| 1      | 1 yr  | 0.975      | ~14,000   | 0.020         | $(1-0.4)\times 0.975 \times 14{,}000 \times 0.020 \approx 164$ |
| 2      | 2 yr  | 0.950      | ~20,000   | 0.020         | $(1-0.4)\times 0.950 \times 20{,}000 \times 0.020 \approx 228$ |

**Rough total CVA over 5yr ≈ 600–900 USD** on $1M notional (≈ 0.07% of notional).

Compare to ZCB CVA ≈ 22,238 USD — the IRS CVA is roughly **25× smaller** per unit notional because the hump-shaped EE is far smaller than a bond's full-notional exposure.

**DVA ≈ CVA** for an at-market IRS, so bilateral Net XVA for an IRS is primarily:

$$\text{Net XVA}_{IRS} \approx \underbrace{CVA - DVA}_{\approx 0 \text{ for at-market}} + FVA + MVA + KVA$$

---

# Appendix: Complete Code → Math Mapping

| Math symbol           | Code location            | Python symbol                                      |
| --------------------- | ------------------------ | -------------------------------------------------- |
| $P(0,t)$              | `market/curve.py`        | `DiscountCurve.df(t)`                              |
| $r(0,t)$              | `market/curve.py`        | `DiscountCurve.zero_rate(t)`                       |
| $S(t)$                | `market/credit.py`       | `CreditCurveModel.survival_prob(t)`                |
| $\Delta PD(t_1, t_2)$ | `market/credit.py`       | `CreditCurveModel.marginal_pd(t1, t2)`             |
| $B(t,T)$              | `models/hw1f.py`         | `B_func(t, T, a)`                                  |
| $V(t,T)$              | `models/hw1f.py`         | `V_func(t, T, a, sigma)`                           |
| $P(t,T \mid x_t)$     | `models/hw1f.py`         | `hw1f_bond_price(t, T, x_t, P_0t, P_0T, a, sigma)` |
| $x_{t+\Delta t}$      | `models/hw1f.py`         | `simulate_xt_step(x_t, dt, a, sigma, Z)`           |
| $\mathcal{T}$         | `sim/timegrid.py`        | `build_simulation_grid(netting_set)`               |
| $\mathbf{X}$          | `sim/batching.py`        | `MonteCarloEngine.simulate_paths(grid_days)`       |
| $V_{ZCB}(t, x_t)$     | `products/zcb.py`        | `ZcbPricer.pv_pathwise(t, x_t, curve, a, sigma)`   |
| $V_{IRS}(t, x_t)$     | `products/irs.py`        | `IrsPricer.pv_pathwise(t, x_t, curve, a, sigma)`   |
| $\alpha_i$            | `products/schedule.py`   | `year_fraction_act365f(start, end)`                |
| $C(t,\omega)$         | `exposure/collateral.py` | `calculate_collateral(V_ns, csa)`                  |
| $EE(t)$               | `exposure/exposure.py`   | `calculate_exposures(V_ns, C)["EE"]`               |
| $ENE(t)$              | `exposure/exposure.py`   | `calculate_exposures(V_ns, C)["ENE"]`              |
| $CVA$                 | `xva/cva.py`             | `compute_cva(grid_days, EE, curve, cpty_credit)`   |
| $DVA$                 | `xva/dva.py`             | `compute_dva(grid_days, ENE, curve, bank_credit)`  |
| $FVA$                 | `xva/fva.py`             | `compute_fva(grid_days, EE_net, C, curve, s_f)`    |
| $MVA$                 | `xva/mva.py`             | `compute_mva(grid_days, V_paths, curve, s_f)`      |
| $KVA$                 | `xva/kva.py`             | `compute_kva(grid_days, V_paths, curve, c_cap)`    |
