# XVA Engine (Hull-White 1F) — Project Plan

## Project summary (reflecting our conversation)

You want an XVA engine that is small enough to demo early (ZCB + IRS), but architected to scale to more products and methodologies. The engine should:

- Read **CSV/JSON** inputs:
  - Trades/portfolios (start: **Zero-Coupon Bond** and **plain-vanilla IRS**)
  - Market data (discount curve; later projection/funding curves)
  - Counterparty + own credit (survival/hazard, recovery)
  - CSA/collateral terms (VM/IM rules; start simple)
  - Model config (Hull–White parameters, Monte Carlo settings)

- Produce **mathematically auditable outputs**:
  - Deterministic PVs (per trade, per netting set)
  - Monte Carlo exposure profiles (EE/EPE/ENE, optional PFE)
  - XVA components: **CVA, DVA, FVA, MVA, KVA** (with explicit assumptions)
  - A complete **explainability bundle**: formulas + intermediate arrays + contribution tables so you can work through the mathematics step-by-step.

- Keep a **minimal demo UI** (not UI-driven design) so you can run scenarios and view key outputs. UI should come in Phase 3.

---

## Non-negotiables

1. **Single source of truth for “states”**
   - Every stage produces a named artifact (tables/arrays) that is saved and referenced later.
2. **Event-driven time grid**
   - Grid is built from trade cashflow/reset dates + margin dates (later) + optional dense points.
3. **Netting set is the exposure unit**
   - Aggregation, collateral, exposure, XVA all operate per netting set.
4. **Deterministic reproducibility**
   - Same inputs + config + seed => same outputs and the same `run_id`.
5. **Explainability-by-construction**
   - Every XVA number must have a time-bucket contribution breakdown.
6. **Batching**
   - Path simulation/valuation must be batched; avoid large cubes by default.

---

## Key components

### A) Market & curve layer
- Curve snapshot with interpolation/extrapolation rules
- Accessors: discount factors, zeros, forwards (as needed)

### B) Product layer
- Schedule generation + accrual fractions
- ZCB PV
- IRS PV (fixed leg + float leg), consistent with chosen curve/model approach

### C) Model layer (Hull–White 1F)
- Short-rate dynamics under Q:
\[
dr_t = a(\theta(t) - r_t)dt + \sigma dW_t
\]
- Curve fitting / representation so the model matches the initial discount curve

### D) Simulation & valuation layer
- Monte Carlo paths (batched)
- Pathwise PV of trades and netting sets at each time grid point

### E) Collateral & exposure layer
- Collateral model (start minimal; structure for realism)
- Exposure definitions and statistics:
\[
E(t,\omega)=\max(V(t,\omega)-C(t,\omega),0),\quad
NE(t,\omega)=\max(C(t,\omega)-V(t,\omega),0)
\]
\[
EE(t)=\mathbb{E}[E(t,\omega)],\quad ENE(t)=\mathbb{E}[NE(t,\omega)]
\]

### F) XVA layer
- CVA/DVA core formulas with time-bucket contributions:
\[
\text{CVA} \approx (1-R_c)\sum_i P(0,t_i)\,EE(t_i)\,\Delta PD_c(t_i)
\]
\[
\text{DVA} \approx (1-R_b)\sum_i P(0,t_i)\,ENE(t_i)\,\Delta PD_b(t_i)
\]
- FVA/MVA/KVA as explicitly-assumed plug-ins (start with “math-correct skeleton”)

### G) Explainability bundle
- `math_report.md` containing:
  - stage-by-stage formulas,
  - parameter values,
  - tables used in each calculation,
  - contribution breakdowns per XVA component.
- Persisted arrays/tables for audit and debugging.

---

# Three-phase plan

## Phase 1 — Rate model + product pricing (deterministic and model-consistent)

**Goal:** Get the core interest-rate mathematics correct and testable, before Monte Carlo.

### Deliverables
1. **Input schemas + validators**
   - `trades`, `netting_sets`, `csas`, `curves`, `credit_curves`, `config`.
2. **Time conventions & schedules**
   - Day count (at least ACT/365F; stub others)
   - Coupon schedule builder for IRS
3. **Curve object**
   - Log-linear DF interpolation (robust MVP)
   - `df(t)` working for any time grid point
4. **Deterministic pricing**
   - ZCB:
     \[
     PV_{ZCB}(0;T)=P(0,T)
     \]
   - IRS (single-curve MVP, clearly documented assumption):
     - Fixed leg:
       \[
       PV_{fix}=N\sum_j \alpha_jK\,P(0,T_j)
       \]
     - Float leg (par approximation for MVP):
       \[
       PV_{flt}\approx N(1-P(0,T_{end}))
       \]
     - Payer fixed: \(PV=PV_{flt}-PV_{fix}\)
5. **Hull–White 1F “math pieces”**
   - Implement:
     \[
     B(t,T)=\frac{1-e^{-a(T-t)}}{a}
     \]
   - Choose a representation (recommended for clarity): **Gaussian factor form**
     - \(r_t = \phi(t) + x_t\)
     - \(dx_t = -a x_t dt + \sigma dW_t\)
     - \(\phi(t)\) chosen so that model reproduces the initial curve.
   - Implement model-consistent bond pricing **under this representation**:
     \[
     P(t,T)=\frac{P(0,T)}{P(0,t)}\exp\left(-B(t,T)x_t - \tfrac{1}{2}V(t,T)\right)
     \]
     where \(V(t,T)\) is the HW variance term for the bond (derive and document in the math report).
     - (Exact form depends on representation; you will derive it once and then test it.)
6. **Phase-1 explainability**
   - Produce a `math_report.md` for:
     - curve interpolation,
     - schedules/accruals,
     - ZCB PV,
     - IRS PV,
     - HW bond formula derivation summary + parameterization.

### Phase 1 tests (edge cases + invariants)
- Curve:
  - DF > 0, DF(0)≈1
  - Handle negative rates safely
  - Extrapolation policy explicit (forbid or flat; tested)
- ZCB:
  - PV equals DF at maturity (exact)
- IRS:
  - Par swap test: compute par rate \(K^*\) and PV ≈ 0
  - Sign test: payer vs receiver swap PV sign flips
- Schedule:
  - Duplicate dates removed, proper ordering
  - Accrual fractions sum approximately to tenor

---

## Phase 2 — Monte Carlo Hull–White simulation + pathwise valuation + exposure

**Goal:** Build the full **simulation → valuation → exposure** pipeline with the correct HW math.

### Deliverables
1. **Time grid builder (event-driven)**
   - Union of all relevant trade event dates (payments/resets)
   - Optional dense points (monthly/weekly) *as config*
2. **Monte Carlo engine (batched, deterministic)**
   - Simulate \(x_t\) exactly on the grid:
     \[
     x_{t+\Delta}=x_t e^{-a\Delta}+\sigma\sqrt{\frac{1-e^{-2a\Delta}}{2a}}Z
     \]
   - Mapping from `(run_id, seed, batch_id)` → RNG stream (reproducible)
3. **Pathwise valuation**
   - ZCB at time t: \(V_{ZCB}(t,\omega)=P(t,T;\omega)\)
   - IRS at time t: value remaining fixed and float legs using model-consistent discounting along the path:
     \[
     PV_{fix}(t,\omega)=N\sum_{j:T_j>t}\alpha_jK\,P(t,T_j;\omega)
     \]
     - Float leg: choose one coherent approach and document it:
       - (a) model-consistent par-float using bond prices, or
       - (b) projection curve later (Phase 3+), but for now keep the simplest consistent choice.
4. **Netting set aggregation**
   - \(V_{NS}(t,\omega)=\sum_{trade\in NS}V_{trade}(t,\omega)\)
5. **Collateral model (minimal but structured)**
   - Two modes:
     - `none`: \(C(t,\omega)=0\)
     - `perfect_vm`: \(C(t,\omega)=V_{NS}(t,\omega)\) (validation)
   - Interface must be stateful (for future thresholds/MTA/MPOR)
6. **Exposure statistics**
   - Compute pathwise exposures and EE/ENE (and optional PFE)
   - Persist exposure profile tables + sample path slices for explainability
7. **Phase-2 explainability**
   - Save:
     - time grid table
     - sample paths (x_t)
     - PV paths for netting set
     - collateral series (per mode)
     - exposure distributions at selected times

### Phase 2 tests (edge cases + connectivity)
- Determinism:
  - Same inputs/config/seed => identical results
  - Batch size changes do not change results (same seed)
- Model sanity:
  - OU mean/variance checks for x_t
- Pipeline invariants:
  - Grid used in simulation == valuation == exposure
  - PV_ns equals sum of PV_trade (where trade-level computed)
- Collateral validation:
  - Perfect VM => EE≈0 across all times (tolerance)
- Numerical stability:
  - No NaN/Inf in PV/exposure arrays for long maturities or small dt

---

## Phase 3 — XVA components + explainability bundle + minimal demo UI

**Goal:** Add the full XVA layer with explicit assumptions, full contribution breakdowns, and a demo UI to run scenarios and visualize results.

### Deliverables
1. **Credit curve layer**
   - Support survival pillars \(S(t)\) or hazard \(h(t)\)
   - Interpolate \(\log S(t)\) linearly (robust)
   - Time-bucket default probabilities:
     \[
     \Delta PD(t_i)=S(t_{i-1})-S(t_i)
     \]
2. **CVA/DVA calculators with contribution tables**
   - CVA:
     \[
     \text{CVA} \approx (1-R_c)\sum_i P(0,t_i)\,EE(t_i)\,\Delta PD_c(t_i)
     \]
   - DVA:
     \[
     \text{DVA} \approx (1-R_b)\sum_i P(0,t_i)\,ENE(t_i)\,\Delta PD_b(t_i)
     \]
   - Write per-bucket tables:
     - `df * EE * dPD * (1-R)` and cumulative totals
3. **FVA (explicit assumption version)**
   - Define funding requirement \(F(t)\) (document it; e.g., positive exposure net of collateral)
   - Simple discrete approximation:
     \[
     \text{FVA} \approx \sum_i P(0,t_i)\,s_f(t_i)\,\mathbb{E}[F(t_i)]\,\Delta t_i
     \]
4. **MVA skeleton (IM plumbing)**
   - Implement `InitialMarginModel` interface returning \(IM(t)\)
   - For now:
     - `IM(t)=0` (wired), or a simple proxy (document clearly)
   - MVA:
     \[
     \text{MVA} \approx \sum_i P(0,t_i)\,s_f(t_i)\,\mathbb{E}[IM(t_i)]\,\Delta t_i
     \]
5. **KVA skeleton (capital plumbing)**
   - Implement `CapitalModel` interface returning \(K(t)\)
   - For now: constant or proxy (document clearly)
   - KVA:
     \[
     \text{KVA} \approx \sum_i P(0,t_i)\,c_{cap}(t_i)\,\mathbb{E}[K(t_i)]\,\Delta t_i
     \]
6. **Explainability bundle (complete)**
   - `inputs_snapshot/`
   - `run_config.json` (model params, seed, methodology versions)
   - `tables/`:
     - curve points, time grid, survival + dPD, exposure profile
     - CVA/DVA/FVA/MVA/KVA contribution tables
   - `arrays/` (npz or parquet): PV paths samples, exposures
   - `math_report.md`: all formulas + parameter values + tables used
7. **Minimal demo UI (Streamlit recommended)**
   - Purpose: run a configuration and view results; not “UI-first”
   - Features:
     - load sample dataset / upload dataset folder
     - select paths, seed, collateral mode, HW params
     - run engine and show:
       - PV summary
       - EE/ENE curves
       - CVA/DVA (and FVA/MVA/KVA if enabled)
       - contribution tables
       - link/download the explainability bundle artifacts

### Phase 3 tests (edge cases + end-to-end)
- Credit curve sanity:
  - S(0)=1, S(t) in [0,1], non-increasing
  - recovery in [0,1]
- XVA monotonic direction checks:
  - increase hazard => CVA increases
  - increase recovery => CVA decreases
  - perfect VM => CVA≈0
- Connectivity invariants:
  - vector lengths match: DF, EE, ΔPD
  - no NaNs/Infs in contribution tables
- Regression tests:
  - golden file comparisons for a fixed seed/sample dataset
- Convergence checks (light):
  - CVA stabilizes as paths increase

---

## Suggested minimal sample dataset (to support all phases)
- One netting set, one CSA, one counterparty, one “bank entity”
- Trades:
  - one 2Y ZCB
  - one ~5Y payer-fixed IRS close to par
- Market:
  - discount curve pillars: 0D, 6M, 1Y, 2Y, 5Y
- Credit:
  - counterparty survival curve
  - bank survival curve
- Config:
  - default paths (e.g., 5k), seed fixed, batch size set

This dataset ensures:
- deterministic PV tests pass
- MC exposure is non-trivial
- CVA/DVA contributions are visible and explainable
- perfect VM mode validates the entire exposure/XVA wiring

---

## Implementation shape (compact, scalable boundaries)
- `io/` schemas + validation
- `market/curve.py`
- `products/zcb.py`, `products/irs.py`, `products/schedule.py`
- `models/hw1f.py` (bond pricing pieces + OU simulation)
- `sim/timegrid.py`, `sim/batching.py`
- `exposure/collateral.py`, `exposure/exposure.py`
- `xva/cva.py`, `xva/dva.py`, `xva/fva.py`, `xva/mva.py`, `xva/kva.py`
- `explain/bundle.py` (math report + tables + arrays)
- `ui/streamlit_app.py` (minimal demo UI in Phase 3)

---

## Section 6 — Input Data Format

All sample inputs live under `inputs/` and can be uploaded from the Streamlit sidebar.

### Trades — `inputs/trades/portfolio.csv`
Flat CSV with one row per trade. Columns:
| Column              | ZCB             | IRS                    |
| ------------------- | --------------- | ---------------------- |
| `trade_id`          | ✓               | ✓                      |
| `trade_type`        | `ZCB`           | `IRS`                  |
| `netting_set_id`    | ✓               | ✓                      |
| `notional`          | ✓               | ✓                      |
| `maturity_date`     | days from today | days from today        |
| `start_date`        | _(blank)_       | days from today        |
| `receive_fixed`     | _(blank)_       | `True`/`False`         |
| `fixed_rate`        | _(blank)_       | decimal (e.g. `0.025`) |
| `payment_frequency` | _(blank)_       | months (e.g. `6`)      |

### Netting sets — `inputs/netting/netting_sets.csv`
Maps each netting set to its counterparty and optional CSA:
```
netting_set_id,counterparty_id,csa_id
NS-CPTY1,CPTY-1,CSA-1
NS-CPTY2,CPTY-2,
```

### Discount curve — `inputs/market/discount_curve.json`
```json
{ "currency": "USD", "points": [{"tenor": 0, "discount_factor": 1.0}, ...] }
```

### Credit curves — `inputs/market/credit_curves.json`
Array of entity objects. A `"BANK"` entry is required.
```json
[{"entity_id":"CPTY-1", "recovery_rate":0.40, "points":[{"tenor":0,"survival_prob":1.0},...]}]
```

### Model config — `inputs/config/model_config.json`
```json
{
  "hw": {"mean_reversion": 0.05, "volatility": 0.01},
  "mc": {"num_paths": 5000, "seed": 42, "batch_size": 1000},
  "funding_spread": 0.01, "cost_of_capital": 0.08, "collateral_mode": "none"
}
```

### Loaders — `src/xva_engine/io/loaders.py`
Four public functions that hydrate Pydantic schema objects:
- `load_portfolio_csv(trades_path, netting_path) → List[NettingSet]`
- `load_discount_curve_json(path) → CurveSnapshot`
- `load_credit_curves_json(path) → Dict[str, CreditCurve]`
- `load_model_config_json(path) → Tuple[ModelConfig, float, float, str]`

---

## Section 7 — Streamlit UI Design (6 Tabs)

**Entry point:** `uv run streamlit run src/xva_engine/ui/streamlit_app.py`

### Sidebar
- File uploaders: `portfolio.csv`, `netting_sets.csv`, `discount_curve.json`, `credit_curves.json` (all optional; fall back to built-in sample defaults)
- Editable model params: `hw_a`, `hw_σ`, `num_paths` (slider), `seed`, `s_f`, `c_cap`
- `▶ Run XVA Engine` primary button (runs all netting sets; stores results in `st.session_state`)
- `st.selectbox` to choose which netting set to display

### Tab 1 — Summary
- Six metric cards: CVA, DVA, FVA, MVA, KVA, Net XVA
- Plotly waterfall chart showing XVA sign/magnitude decomposition
- Plotly area chart: EE and ENE profiles over the time grid
- Download: `xva_summary.csv`

### Tab 2 — Market Data
- Dual-axis line chart: discount factors (left axis) + zero rates % (right axis)
- Survival probability curves for all entities in `credit_curves.json`
- LaTeX expanders: log-linear interpolation formula, marginal PD formula
- Downloads: `discount_curve.csv`, per-entity credit data

### Tab 3 — Simulation
- Horizontal strip scatter showing the time grid structure (coloured by step type)
- HW factor fan chart: 50 sample paths of `x_t` + mean path (supplied via `x_sample` in engine return dict)
- LaTeX expander: exact OU step formula

### Tab 4 — Exposure
- MtM path fan: 50 sample paths of `V_NS(t)` + mean (from `V_ns_sample`)
- Stacked area chart: EE (green) and ENE (red) profiles
- Histogram of `V_NS` distribution at a user-selected time step (slider)
- LaTeX expanders: EE and ENE formulas
- Download: `exposure_profile.csv`

### Tab 5 — XVA Breakdown
Five collapsible expanders (CVA expanded by default), each containing:
- Bucket bar chart (per time step contribution)
- Styled contribution table (pandas-backed Streamlit dataframe with max-value highlight)
- LaTeX formula for the metric
- Download: `{metric}_buckets.csv`

### Tab 6 — Sensitivity
- Sliders: hazard rate multiplier (×), funding spread bump (bps), HW σ scaling (%)
- `▶ Run Sensitivity` button: constructs a bumped `XVAEngine` via `__new__` (avoids double init)
- Grouped bar chart: base vs bumped CVA/DVA/FVA
- Delta table: Metric, Base, Bumped, Delta_USD, Delta_%
- Download: `sensitivity.csv`

### Engine patching for sensitivity
Bypasses `__init__` using `XVAEngine.__new__(XVAEngine)` and directly assigns attributes
(`netting_set`, `config`, `discount_curve`, `cpty_credit_model`, `bank_credit_model`, `pricers`,
`csa_schema`, `funding_spread`, `cost_of_capital`, `output_dir`, `run_id`) before calling `.run()`.

### Technology choices
- **Plotly** (`plotly.graph_objects`) for all charts — already in `pyproject.toml`
- **Pandas** for table display + CSV serialisation
- **`st.session_state`** for result persistence across reruns
- **`st.latex()`** for in-app formula rendering

---

## Section 8 — Mathematical Reference

`docs/math/xva_mathematics.md` provides a fully worked example of a 2Y ZCB through
every stage of the engine pipeline.

### Sections covered
1. **Trade Definition** — ZCB cashflow, notional, maturity
2. **Market Data** — discount curve log-linear interpolation; survival curve piecewise-constant hazard
3. **Hull-White 1F Model** — SDE, risk-neutral measure, exact bond pricing formula
4. **Exact OU Simulation** — conditional distribution, variance, Euler-free stepping
5. **ZCB Pricing Under HW** — `P(t,T;x_t)` formula, sigma_P, time-zero check
6. **Time Grid** — quarterly + maturity bucketing; `sim/timegrid.py` behaviour
7. **Monte Carlo EE/ENE** — netting set aggregation, path-by-path max(V−C,0), expectation
8. **Collateral** — VM formula, IM placeholder (zero for uncollateralised)
9. **XVA Formulas** — CVA, DVA, FVA, MVA, KVA with bucket-by-bucket summations
10. **Numerical Example** — concrete numbers for a 5,000-path 2Y ZCB run
11. **Code → Math Mapping Table** — maps every key function/variable to its mathematical symbol

