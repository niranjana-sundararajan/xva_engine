# XVA Engine — Mathematical Reference
# End-to-End Walkthrough: Zero Coupon Bond (ZCB)

This document traces a single Zero Coupon Bond through the complete engine pipeline — from trade definition through to final CVA and DVA numbers. All symbols and formulas correspond 1-to-1 with the code in `src/xva_engine/`.

---

## 1. Trade Definition

A **Zero Coupon Bond** pays a fixed notional $N$ at maturity $T$:

$$V_{ZCB}(T, \omega) = N \quad \forall \omega$$

**Example trade (inputs/trades/portfolio.csv):**
| Field        | Value               |
| ------------ | ------------------- |
| trade_id     | ZCB-1               |
| notional $N$ | 1,000,000           |
| maturity $T$ | 730 days from $t=0$ |

---

## 2. Market Data

### 2.1 Discount Curve

The initial discount curve $P(0, t)$ is supplied as a set of pillar points $(t_i, D_i)$ where $D_i = P(0, t_i)$.

**Log-linear interpolation** between pillars (implemented in `market/curve.py`):

$$\ln P(0, t) = \ln P(0, t_i) + \frac{t - t_i}{t_{i+1} - t_i} \left[ \ln P(0, t_{i+1}) - \ln P(0, t_i) \right]$$

For times $t$ beyond the last pillar, the curve assumes a **flat forward rate** (log-linear extrapolation). For $t=0$, $P(0,0)=1$ is assumed if not provided.

The continuously-compounded zero rate is:

$$r(0, t) = -\frac{\ln P(0, t)}{t / 365}$$

**Example (inputs/market/discount_curve.json):**
| Tenor (days) | $P(0, t)$ | Zero rate |
| ------------ | --------- | --------- |
| 0            | 1.0000    | 0.00%     |
| 182          | 0.9880    | 2.42%     |
| 365          | 0.9750    | 2.53%     |
| 730          | 0.9500    | 2.56%     |
| 1825         | 0.8800    | 2.48%     |

### 2.2 Credit Curves

Survival probability $S(t)$ for each entity (counterparty, bank) is also log-linearly interpolated:

$$S(t) = \exp\!\left(-\int_0^t h(u)\,du\right)$$

where $h(t)$ is the instantaneous hazard rate. Log-linear interpolation of $S(t)$ corresponds to a **piecewise-constant hazard rate** between pillars. For extrapolation beyond the last pillar, a **flat hazard rate** ($S(t)$ decays log-linearly) is assumed.

**Marginal probability of default** (unconditional) between $t_{i-1}$ and $t_i$ (`market/credit.py`):

$$\Delta PD(t_{i-1}, t_i) = S(t_{i-1}) - S(t_i)$$

The corresponding **conditional** probability of default given survival to $t_{i-1}$ is:

$$PD^{\text{cond}}(t_{i-1}, t_i) = 1 - \frac{S(t_i)}{S(t_{i-1})}$$

`CreditCurveModel.marginal_pd()` implements the unconditional definition.

---

## 3. Hull-White 1-Factor Model

### 3.1 Short Rate Process

The Hull-White 1F model specifies the short rate as:

$$r_t = \varphi(t) + x_t$$

where $\varphi(t)$ is the **deterministic drift** (fitted to the initial curve) and $x_t$ is the **stochastic factor** satisfying the Ornstein-Uhlenbeck (OU) process:

$$dx_t = -a\,x_t\,dt + \sigma\,dW_t, \quad x_0 = 0$$

**Parameters (inputs/config/model_config.json):**
| Symbol   | Name                  | Typical value |
| -------- | --------------------- | ------------- |
| $a$      | Mean reversion speed  | 0.05          |
| $\sigma$ | Short rate volatility | 0.01          |

### 3.2 Exact OU Simulation

For a time step $\Delta t$ (`models/hw1f.py`, `simulate_xt_step`):

$$x_{t+\Delta t} = x_t \cdot e^{-a\,\Delta t} + \sigma\sqrt{\frac{1 - e^{-2a\,\Delta t}}{2a}}\cdot Z, \quad Z \sim \mathcal{N}(0,1)$$

This is the **exact** transition — no Euler discretisation error.

### 3.3 Bond Pricing Formula

Given $x_t$ (the factor value on a Monte Carlo path at time $t$), the model-consistent price of a zero coupon bond maturing at $T$ is (`models/hw1f.py`, `hw1f_bond_price`):

$$\boxed{P(t, T \mid x_t) = \frac{P(0, T)}{P(0, t)}\exp\!\left(-B(t,T)\,x_t - \tfrac{1}{2}V(t,T)\right)}$$

**where:**

$$B(t, T) = \frac{1 - e^{-a(T-t)}}{a}$$

$$V(t, T) = \frac{\sigma^2}{2a}\,B(t,T)^2\,\left(1 - e^{-2at}\right)$$

**Intuition:**
- $B(t,T)$ is the **duration-like sensitivity** of the bond to the factor $x_t$. At $t = T$, $B = 0$ (no sensitivity at maturity).
- $V(t,T)$ is a **convexity/Jensen correction** — it grows with variance accumulated up to $t$.
- The ratio $P(0,T)/P(0,t)$ is the **initial forward bond price**.

**At maturity** $t = T$: $B = 0$, $V = 0$, so $P(T,T) = 1$ regardless of $x_T$ ✓

**ZCB mark-to-market at time $t$:**

$$V_{ZCB}(t, \omega) = N \cdot P\!\left(t, T \mid x_t(\omega)\right)$$

---

## 4. Simulation Time Grid

The engine builds an event-driven grid (`sim/timegrid.py`):

1. Always include $t = 0$
2. Include ZCB maturity date $T_{ZCB}$
3. Include all IRS payment dates from `build_irs_schedule()`
4. Optionally add **dense monitoring points** every $\delta$ days (e.g. monthly)

**Example for ZCB-1 + IRS-1 portfolio:**

$$\mathcal{T} = \{0, 182, 365, 548, 730, 913, \ldots, 1825\} \text{ days}$$

---

## 5. Monte Carlo Engine

### 5.1 Path Generation (`sim/batching.py`)

For $M$ paths and $|\mathcal{T}|$ grid steps:

1. Seed the RNG: `np.random.default_rng(seed)`
2. Draw $Z_{i,j} \sim \mathcal{N}(0,1)$ for each step $i$ and path $j$
3. Propagate $x_{t_i}^{(j)}$ via the exact OU step above
4. Return matrix $\mathbf{X} \in \mathbb{R}^{|\mathcal{T}| \times M}$

### 5.2 Pathwise Netting-Set Valuation

For each grid point $t_i$ and path $j$:

$$V_{NS}(t_i, j) = \sum_{k \in \text{trades}} V_k\!\left(t_i,\, x_{t_i}^{(j)}\right)$$

For our ZCB:

$$V_{ZCB}(t_i, j) = \begin{cases} N \cdot P\!\left(\tfrac{t_i}{365},\, \tfrac{T}{365} \mid x_{t_i}^{(j)}\right) & t_i < T \\ 0 & t_i \ge T \end{cases}$$

**Output:** matrix $\mathbf{V}_{NS} \in \mathbb{R}^{|\mathcal{T}| \times M}$

*Note on IRS Float Leg Valuation:* Float leg pricing within the Hull-White framework is performed using the model-consistent forward rates (or exactly through the equivalent par-bond approximation $N(1 - P(t,T))$ where applicable), ensuring consistent valuation under the $x_t$ paths.

---

## 6. Collateral (`exposure/collateral.py`)

Given a CSA, the collateral held $C(t, \omega)$ reduces effective exposure:

| CSA mode     | Collateral $C(t, \omega)$                                        |
| ------------ | ---------------------------------------------------------------- |
| `none`       | $0$                                                              |
| `perfect_vm` | $V_{NS}(t, \omega)$ (full variation margin)                      |
| `threshold`  | $\max(V_{NS}(t,\omega) - H, 0) - \max(-V_{NS}(t,\omega) - H, 0)$ |

**Sign Convention**: $V_{NS}(t, \omega) > 0$ means the netting-set is an asset to the bank. $C(t, \omega) > 0$ means collateral is **held** by the bank (received from the counterparty), thereby reducing the bank's credit exposure. $C(t, \omega) < 0$ means collateral is posted by the bank.

---

## 7. Exposure (`exposure/exposure.py`)

### 7.1 Pathwise Exposures

$$E(t_i, j) = \max\!\left(V_{NS}(t_i, j) - C(t_i, j),\; 0\right)$$

$$NE(t_i, j) = \max\!\left(C(t_i, j) - V_{NS}(t_i, j),\; 0\right)$$

$E$ is the **credit exposure to the counterparty** (we lose if they default); $NE$ is our **exposure to our own default** (counterparty loses if we default).

### 7.2 Expected Exposures

Averaging over the $M$ Monte Carlo paths:

$$EE(t_i) = \frac{1}{M}\sum_{j=1}^{M} E(t_i, j) = \mathbb{E}\!\left[\max(V_{NS}(t_i) - C(t_i), 0)\right]$$

$$ENE(t_i) = \frac{1}{M}\sum_{j=1}^{M} NE(t_i, j) = \mathbb{E}\!\left[\max(C(t_i) - V_{NS}(t_i), 0)\right]$$

**For a ZCB with no CSA:** Since $V_{ZCB}(t) > 0$ for $t < T$ in all paths (bond has positive value to holder), $EE(t_i) \approx \mathbb{E}[V_{ZCB}(t_i)]$ and $ENE(t_i) \approx 0$.

---

## 8. XVA Calculations

### 8.1 CVA — Credit Valuation Adjustment (`xva/cva.py`)

The unilateral CVA captures the expected loss from **counterparty default**:

$$\boxed{CVA = (1 - R_c)\sum_{i=1}^{n} P(0, t_i)\cdot EE(t_i)\cdot \Delta PD_c(t_{i-1}, t_i)}$$

| Symbol        | Meaning                                                              |
| ------------- | -------------------------------------------------------------------- |
| $R_c$         | Counterparty recovery rate (e.g. 0.40)                               |
| $P(0, t_i)$   | Risk-free discount factor to bucket end                              |
| $EE(t_i)$     | **Undiscounted** expected credit exposure at $t_i$                   |
| $\Delta PD_c$ | Counterparty marginal default probability in bucket $[t_{i-1}, t_i]$ |

**For our ZCB:** CVA is positive since $EE > 0$ (we are always owed money if they default).

### 8.2 DVA — Debt Valuation Adjustment (`xva/dva.py`)

The bilateral DVA captures the benefit from **our own potential default**:

$$\boxed{DVA = (1 - R_b)\sum_{i=1}^{n} P(0, t_i)\cdot ENE(t_i)\cdot \Delta PD_b(t_{i-1}, t_i)}$$

Where $ENE(t_i)$ is our **undiscounted** expected negative exposure to the counterparty.

**For a ZCB held long:** ENE ≈ 0 because we always owe positive value, so DVA ≈ 0.

### 8.3 FVA — Funding Valuation Adjustment (`xva/fva.py`)

The funding cost of carrying uncollateralised positive exposure:

$$\boxed{FVA = \sum_{i=1}^{n} P(0, t_i)\cdot s_f(t_i)\cdot EE_{net}(t_i)\cdot \Delta t_i}$$

where $s_f$ is the funding spread above risk-free and $\Delta t_i = (t_i - t_{i-1}) / 365$.

### 8.4 MVA — Margin Valuation Adjustment (`xva/mva.py`)

The funding cost associated with posting **initial margin** $IM(t)$:

$$\boxed{MVA = \sum_{i=1}^{n} P(0, t_i)\cdot s_f(t_i)\cdot \mathbb{E}[IM(t_i)]\cdot \Delta t_i}$$

### 8.5 KVA — Capital Valuation Adjustment (`xva/kva.py`)

The cost of holding regulatory capital $K(t)$ against the trade:

$$\boxed{KVA = \sum_{i=1}^{n} P(0, t_i)\cdot c_{cap}(t_i)\cdot \mathbb{E}[K(t_i)]\cdot \Delta t_i}$$

### 8.6 Net XVA

$$\text{Net XVA} = CVA - DVA + FVA + MVA + KVA$$

A positive Net XVA is a **cost** to the dealer (reduces mark-to-market profit).

---

## 9. Numerical Example (ZCB-1)

**Inputs:**
- $N = 1{,}000{,}000$ USD, $T = 730$ days (2 years)
- $P(0, 730) = 0.950$, so initial ZCB PV $= 950{,}000$
- $a = 0.05$, $\sigma = 0.01$
- Counterparty: $R_c = 0.40$, $S_c(365) = 0.98$, $S_c(730) = 0.96$
- No CSA ($C = 0$)

**Step-by-step CVA (simplified, 2-bucket):**

| Bucket | $t_i$ (yrs) | $P(0,t_i)$ | $EE(t_i)$ ≈ | $\Delta PD_c$ | CVA contribution                                            |
| ------ | ----------- | ---------- | ----------- | ------------- | ----------------------------------------------------------- |
| 1      | 1.0         | 0.975      | ~975,000    | 0.020         | ≈ $(1-0.4) \cdot 0.975 \cdot 975000 \cdot 0.020 = 11{,}408$ |
| 2      | 2.0         | 0.950      | ~950,000    | 0.020         | ≈ $(1-0.4) \cdot 0.950 \cdot 950000 \cdot 0.020 = 10{,}830$ |

**Total CVA ≈ 22,238 USD** (≈ 2.3% of initial PV)

---

## 10. Code → Math Mapping

| Math symbol           | Code location            | Python symbol                                      |
| --------------------- | ------------------------ | -------------------------------------------------- |
| $P(0,t)$              | `market/curve.py`        | `DiscountCurve.df(t)`                              |
| $S(t)$                | `market/credit.py`       | `CreditCurveModel.survival_prob(t)`                |
| $\Delta PD(t_1, t_2)$ | `market/credit.py`       | `CreditCurveModel.marginal_pd(t1, t2)`             |
| $B(t,T)$              | `models/hw1f.py`         | `B_func(t, T, a)`                                  |
| $V(t,T)$              | `models/hw1f.py`         | `V_func(t, T, a, sigma)`                           |
| $P(t,T \mid x_t)$     | `models/hw1f.py`         | `hw1f_bond_price(t, T, x_t, P_0t, P_0T, a, sigma)` |
| $x_{t+\Delta t}$      | `models/hw1f.py`         | `simulate_xt_step(x_t, dt, a, sigma, Z)`           |
| $\mathcal{T}$         | `sim/timegrid.py`        | `build_simulation_grid(netting_set)`               |
| $\mathbf{X}$          | `sim/batching.py`        | `MonteCarloEngine.simulate_paths(grid_days)`       |
| $V_{ZCB}$             | `products/zcb.py`        | `ZcbPricer.pv_pathwise(t, x_t, curve, a, sigma)`   |
| $C(t,\omega)$         | `exposure/collateral.py` | `calculate_collateral(V_ns, csa)`                  |
| $EE(t)$               | `exposure/exposure.py`   | `calculate_exposures(V_ns, C)["EE"]`               |
| $ENE(t)$              | `exposure/exposure.py`   | `calculate_exposures(V_ns, C)["ENE"]`              |
| $CVA$                 | `xva/cva.py`             | `compute_cva(grid_days, EE, curve, cpty_credit)`   |
| $DVA$                 | `xva/dva.py`             | `compute_dva(grid_days, ENE, curve, bank_credit)`  |
| $FVA$                 | `xva/fva.py`             | `compute_fva(grid_days, EE_net, curve, s_f)`       |
