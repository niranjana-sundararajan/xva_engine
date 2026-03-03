# XVA Engine — Future Implementation Plan

> **Status:** Planning
> **Purpose:** Comprehensive roadmap for evolving the engine from its current Hull-White 1F / ZCB+IRS baseline into a full production-grade XVA platform. Each section states what to build, why, and — where known — what the prevailing industry practice is.

---

## Layer 1 — Market & Curve (`market/`)

### What to build
- Multi-curve framework: separate OIS discount curve from SOFR/IBOR projection curves per tenor (1M, 3M, 6M)
- Interpolation methods: monotone convex, natural cubic spline, Bessel/Hermite; selectable per curve
- Tenor basis spread curves
- Cross-currency (xccy) discount curves: FX-implied, basis swap adjusted
- Inflation curves: CPI index ratio, zero-coupon swap pricing
- Volatility surfaces: swaption vol cube (strike × expiry × tenor); cap/floor vol surface; Black/Bachelier
- Dividend curves: discrete schedule + continuous yield
- Repo / haircut curves for SFTs
- Curve bumping / scenario generation API: parallel shift, twist, fly

### Industry practice
- **OIS discounting** has been standard since the 2008–2012 transition (CSA discounting at OIS); all major dealers moved to OIS/CSA-consistent pricing by 2012.
- **SOFR/€STR/SONIA** have replaced IBOR tenors post-2021 (LIBOR cessation). Projection curves are built from OIS futures and basis swaps.
- **Monotone convex interpolation** (Hagan & West, 2006) is the industry default for discount curves — it guarantees positive forward rates. Bloomberg uses piecewise linear on log-discount; most risk systems use monotone convex.
- **Volatility surfaces** are stored as SABR parameter grids or as normal vol grids in basis points; Black vol is increasingly replaced by Bachelier (normal) vol for near-zero/negative rate environments.
- Major systems (Murex, Sophis/Calypso, OpenGamma Strata) expose a `MarketData` snapshot object per valuation date containing all curves, surfaces, and fixings.

---

## Layer 2 — Product (`products/`)

### Interest rate products to add

| Product                   | Notes                                           |
| ------------------------- | ----------------------------------------------- |
| OIS swap                  | Compounded overnight fixing (SOFR, €STR, SONIA) |
| Basis swap                | Float-float; two projection curves              |
| Cross-currency swap (CCS) | FX notional exchange; MTM reset variant         |
| FRA                       | Single-period forward rate agreement            |
| Cap / Floor               | Strip of caplets/floorlets; Black or Bachelier  |
| European swaption         | Analytical HW1F formula; Black vol benchmark    |
| Bermudan swaption         | LSM / regression for exercise boundary under HW |
| Callable / puttable bond  | Embedded optionality; HW tree or LSM            |
| Amortising IRS            | Variable notional schedule                      |
| CMS swap / spread         | Convexity adjustment under HW/LMM               |
| Inflation-linked swap     | Zero-coupon and year-on-year variants           |
| Asset swap                | Fixed rate bond + IRS; ASW spread               |
| Repo / reverse repo       | Collateral haircut model                        |

### FX products

| Product            | Notes                                          |
| ------------------ | ---------------------------------------------- |
| FX Forward         | Log-normal or normal FX under domestic measure |
| FX European Option | Garman-Kohlhagen + stochastic-rate correction  |
| FX Barrier Option  | Path-dependent payoff on MC paths              |
| FX Variance Swap   | Realised vol contract                          |

### Equity products

| Product                | Notes                                               |
| ---------------------- | --------------------------------------------------- |
| Equity Forward / TRS   | Continuous/discrete dividend model                  |
| European Equity Option | Black-Scholes + stochastic-rate via hybrid HW + GBM |
| Asian Option           | Arithmetic/geometric average; MC                    |
| Equity Variance Swap   | Realised variance payoff                            |
| Convertible Bond       | Equity + credit + rates hybrid; PDE or MC           |
| Equity Swap            | Periodic equity return vs SOFR leg                  |

### Credit products

| Product            | Notes                                          |
| ------------------ | ---------------------------------------------- |
| CDS (single-name)  | Premium + protection leg; hazard bootstrapping |
| CDS Index          | Spread/upfront; constituent credit curves      |
| CLN / TARN         | Credit-linked structured note                  |
| Drawn/undrawn loan | EAD model for KVA                              |

### Industry practice
- **FpML** is the ISDA-endorsed standard for trade representation. All major trade repositories (DTCC, REGIS-TR) accept FpML. Internal systems map FpML to pricing schemas.
- **CCP clearing** (LCH SwapClear, CME Clearing) is mandatory for vanilla IRS and CDS index under Dodd-Frank / EMIR. Cleared trade economics differ from bilateral only in collateral/IM terms.
- **SOFR compounded in arrears** (RFR) is the standard for USD OIS swaps post-July 2023. This changes the fixing calculation significantly versus forward-looking LIBOR.
- Bermudan swaptions are the most computationally demanding vanilla product; **Longstaff-Schwartz Monte Carlo** (2001) is industry standard for their valuation under a path-based model.

---

## Layer 3 — Model (`models/`)

### Short-rate model extensions

| Model                        | What it adds                                                             |
| ---------------------------- | ------------------------------------------------------------------------ |
| Hull-White 2F (G2++)         | Two correlated factors; richer yield curve dynamics; better swaption fit |
| CIR++                        | Non-negative rates (shifted); closed-form bond price                     |
| Black-Karasinski             | Log-normal rates; no negative rates; numerical only                      |
| Extended Nelson-Siegel (DNS) | Dynamic macro factor model; level/slope/curvature                        |

### Term-structure / market models

| Model                            | What it adds                                                    |
| -------------------------------- | --------------------------------------------------------------- |
| LIBOR Market Model (LMM)         | Forward rate dynamics; calibrated to cap + swaption vols        |
| Stochastic Alpha Beta Rho (SABR) | Smile-consistent vol; used as local vol per tenor pillar        |
| SABR-LMM                         | LMM with stochastic vol per forward; best-in-class market model |

### Hybrid models (for multi-asset XVA)

| Model                     | What it adds                                   |
| ------------------------- | ---------------------------------------------- |
| HW + GBM (rates + equity) | Correlated equity and rate paths               |
| HW + HW FX                | Domestic + foreign rates + FX; XCCY XVA        |
| HW + JCIR++               | Jump-to-default credit + rates; wrong-way risk |
| Heston + HW               | Stochastic equity vol + stochastic rates       |

### Calibration
- Automatic calibration of HW a/σ to ATM swaption vols (Levenberg-Marquardt or Nelder-Mead)
- G2++ calibration to co-terminal swaptions + caps simultaneously
- LMM calibration to full vol cube with cascade calibration

### Industry practice
- **G2++ (Hull-White 2 factor)** is the most widely used short-rate model for XVA engines at major banks (reference: Brigo & Mercurio, *Interest Rate Models*, 2nd ed.). It fits the swaption vol surface far better than HW1F while remaining analytically tractable.
- **SABR** (Hagan et al., 2002) is the market standard for smile interpolation for IR options. Every major dealer stores SABR parameters per expiry/tenor.
- **LMM / BGM** is used by the most sophisticated desks for exotic IR products (Bermudan, CMS, TARNs). It is calibrated through the cap/swaption vol surface.
- For **XVA specifically**, most tier-1 banks use G2++ or HW2F for rates with a correlated jump diffusion for credit (JCIR++) to capture wrong-way risk. Pure HW1F is acceptable for Phase 1–3 but insufficient for regulatory CVA capital purposes.

---

## Layer 4 — Simulation (`sim/`)

### What to build
- **Variance reduction:** antithetic variates, quasi-Monte Carlo (Sobol / Halton), stratified sampling, importance sampling for tail exposures
- **Multi-factor correlated simulation:** Cholesky decomposition of factor correlation matrix for G2++ / hybrid models
- **American Monte Carlo (Longstaff-Schwartz):** regression-based optimal stopping; Bermudan swaptions, callable bonds
- **Nested Monte Carlo:** inner simulations for MVA/KVA IM estimation (ISDA SIMM proxy)
- **PDE / ADI solver:** alternating direction implicit for HW 1F/2F as analytical benchmark
- **Jump process simulation:** compound Poisson jumps for credit/FX jump risk
- **Path storage:** compressed sparse; streaming valuation to avoid large in-memory path cubes
- **GPU acceleration:** CUDA/CuPy pathwise pricing; 10–100× speedup for large path counts

### Industry practice
- **10,000–50,000 paths** is the typical range for CVA calculations at major dealers. 5,000 (current default) is minimum viable; regulatory IMM approval often requires convergence analysis demonstrating stability.
- **Quasi-Monte Carlo (Sobol sequences)** reduces variance at O(log(N)^d / N) vs O(1/√N) for standard MC. Widely adopted in XVA engines (OpenGamma, Quaternion Risk Engine).
- **Antithetic variates** are nearly free and typically halve variance; almost universally used as a baseline variance reduction technique.
- **Longstaff-Schwartz** (2001) is the industry standard for American/Bermudan option pricing. Regression basis functions are typically monomials or Laguerre polynomials in the state variable.
- **GPU acceleration** (CUDA via CuPy or custom kernels) is used by specialised XVA vendors (Numerix, FinCAD) for near-real-time CVA on large portfolios.

---

## Layer 5 — Exposure (`exposure/`)

### What to build
- **Realistic CSA:** threshold (TH), MTA, independent amount (IA), rounding, dispute resolution delay
- **Margin Period of Risk (MPOR):** 10-day cleared / 20-day bilateral; exposure = max V over [t, t+MPOR]
- **Initial Margin models:**
  - ISDA SIMM: sensitivity-based; delta/vega/curvature per risk class
  - Schedule-based IM: simplified regulatory approach
  - CCP IM (VaR-based): for cleared trades
- **PFE profiles:** 95th / 99th percentile; regulatory PFE for internal limits
- **EPE / EEPE / Effective EPE:** Basel III regulatory exposure definitions
- **Wrong-way risk (WWR):** positive correlation between exposure and counterparty credit quality
- **Incremental exposure attribution:** trade-level EE contribution to netting set (for XVA allocation)

### Industry practice
- **SA-CCR** replaced CEM as the Basel III standard for EAD calculation from January 2022. Every bank must compute SA-CCR for regulatory capital; internal models (IMM) require separate regulatory approval.
- **ISDA SIMM** is the industry standard for bilateral IM calculation (UMR Phases 1–6, fully phased in by 2022). It covers six risk classes (IR, Credit, Equity, Commodity, FX, Rates Vol) and is re-licensed annually.
- **MPOR of 10 business days** for cleared OTC and 20 business days for threshold CSAs is mandated by BCBS Basel III (CRE50). This is the key driver of MVA for cleared portfolios.
- **Wrong-way risk** is a known deficiency in standard CVA models. Basel III FRTB-CVA requires add-ons for general WWR; specific WWR (e.g., CDS on own stock as collateral) requires explicit modelling.

---

## Layer 6 — XVA (`xva/`)

### CVA / DVA improvements
- Bilateral first-to-default CVA (joint default adjustment)
- CVA semi-analytic formula for HW models (closed-form for ZCB/IRS without full MC)
- Wrong-way risk multiplier; correlation between EE and counterparty hazard rate
- Stressed CVA (SA-CVA): FRTB regulatory sensitivity-based approach
- CVA Greeks: delta to IR curves, credit spreads, FX; vega to vol surface; theta

### FVA improvements
- Asymmetric FVA: separate FBA (funding benefit) vs FCA (funding cost)
- Netting-set level funding need across positive and negative exposures
- Collateral optionality: cheapest-to-deliver across currencies

### MVA improvements
- Full ISDA SIMM MVA: nested MC for SIMM sensitivities per time step
- Dynamic IM schedule with proper discounting through life of trade
- Cleared vs bilateral IM cost comparison

### KVA improvements
- SA-CCR-based EAD replaces current simple RWA proxy
- IMM (Internal Model Method): EEPE-based RWA; requires regulatory approval
- CVA capital (SA-CVA): separate charge for CVA risk volatility
- KVA under Basel IV capital floor (72.5% output floor)
- Hurdle rate optimisation per business line

### Additional XVA metrics

| Metric                    | Definition                                                                            |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **ColVA**                 | Collateral valuation adjustment; cheapest-to-deliver optionality in multicurrency CSA |
| **HVA**                   | Hedging valuation adjustment; transaction cost of dynamic CVA hedge                   |
| **XVA P&L decomposition** | Daily P&L split: theta, market risk, credit risk, funding components                  |
| **Incremental XVA**       | Marginal XVA of adding a new trade (key for pre-trade pricing)                        |
| **XVA allocation**        | Trade-level Euler allocation from netting-set XVA back to individual trades           |

### Industry practice
- **FVA** is still debated academically (Hull & White argue it should be zero; Burgard & Kjaer argue it is real). In practice, every major dealer charges FVA to clients; the Burgard-Kjaer replication framework (2011) is the standard reference.
- **MVA** became critical after UMR (Uncleared Margin Rules). ISDA SIMM v2.6 (2024) is the current version. MVA is now comparable in size to CVA for many cleared portfolios.
- **KVA** (Green & Kenyon, 2014) is not yet universally charged but is becoming standard at tier-1 banks. The hurdle ROE rate is typically 10–12% for trading desks.
- **SA-CVA** under FRTB (Basel January 2016, revised 2019) replaces the current exposure method for CVA capital and requires sensitivity-based Greeks. EU implementation: CRR3 (January 2025).
- Key references: *XVA: Credit, Funding and Capital Valuation Adjustments* (Green, 2015); *Counterparty Credit Risk* (Gregory, 4th ed.).

---

## Layer 7 — Explainability (`explain/`)

### What to build
- Interactive self-contained HTML report with embedded Plotly charts; shareable without a server
- Full audit trail: input hash → model params hash → seed → output hash (content addressing)
- Sensitivity attribution report: CVA decomposed by IR delta, credit spread delta, FX delta, vol vega
- Scenario comparison: base vs stressed side-by-side with diff highlighting
- Regulatory report templates: FRTB-CVA sensitivity table; SA-CCR EAD table; Pillar 3 disclosures
- LaTeX export: `report.tex` → compiled PDF
- Excel workbook export: openpyxl; contribution tables + embedded charts; audit-ready

### Industry practice
- **Model validation** is a regulatory requirement (SR 11-7 in the US; ECB guide on internal models). Every model must have independent validation, documented assumptions, and a limitation register.
- **Audit trails** with hashed inputs/outputs are required by MiFID II / Dodd-Frank record-keeping rules. Many banks use content-addressed storage (similar to Git objects) for model run artifacts.
- **P&L attribution** (PLA) is an FRTB market risk requirement: daily P&L must be explained by risk factor sensitivities within tolerance. The same infrastructure applies to XVA P&L explain.

---

## Layer 8 — Engine Orchestration (`engine.py`)

### What to build
- Parallel netting set processing via `ProcessPoolExecutor`
- Incremental computation: cache curve/grid; recompute only affected XVA on market data change
- Trade-level XVA allocation: Euler decomposition; full additive consistency to netting-set total
- Day-one P&L booking: XVA at t=0 is the upfront charge booked at trade inception
- REST API: FastAPI wrapper; JSON in / JSON out; Docker-deployable microservice
- Message queue integration: Kafka/RabbitMQ consumer; trigger revaluation on market data event
- Real-time / streaming mode: linearised Greeks-based revaluation for sub-second pre-trade XVA

### Industry practice
- **Nightly batch CVA** is standard for regulatory reporting. Intraday / pre-trade CVA uses linearised approximations (CVA ≈ CVA₀ + ΔS × ∂CVA/∂S). Full MC revaluation is reserved for end-of-day.
- **Incremental XVA** (marginal impact of a new trade on portfolio XVA) is the day-to-day sales desk requirement. It drives pricing and trade acceptance decisions.
- **Pre-trade XVA** is typically computed in under 1 second using cached Greeks. Full revaluation for a large portfolio (100 netting sets, 5,000 paths) typically takes minutes in batch mode.
- Major industry platforms (Murex MX.3, Calypso, OpenGamma) expose XVA as a service with REST/FIX adapters.

---

## Layer 9 — IO & Data (`io/`)

### What to build
- FpML / FIXML parsers: map ISDA FpML v5.12 to internal schema objects
- Bloomberg BLPAPI / Refinitiv Eikon adapters: live market data pull; curve bootstrapping from raw quotes
- Database backends: SQLAlchemy ORM; trade blotter, market data store, results warehouse
- Arrow / Parquet streaming: memory-mapped columnar trade store; vectorised schema validation
- Pydantic v2 strict mode: discriminated unions per trade type; stricter field validation
- Regulatory data formats: CFTC / EMIR trade repository format; UTI/USI linking

### Industry practice
- **FpML** (Financial products Markup Language) is the ISDA standard for trade representation, required by all trade repositories (DTCC, CME TR, REGIS-TR). Version 5.12 covers all OTC derivative product types.
- **Bloomberg BLPAPI / Refinitiv Eikon API** are the de-facto sources for live market data. Curve construction from raw market quotes (deposits, futures, swaps) is called bootstrapping and is a core quant library function.
- **Apache Arrow / Parquet** is the modern standard for analytical data exchange, replacing CSV for large datasets. Used internally at major banks for risk data aggregation under BCBS 239.

---

## Layer 10 — Testing & CI

### What to build
- Property-based testing: Hypothesis library; fuzz test curve/schedule/pricing with random valid inputs
- Golden-file regression tests: lock numeric outputs for reference portfolio; fail on drift > tolerance
- Convergence tests: CVA standard error as N → ∞; assert convergence rate O(1/√N)
- Performance benchmarks: pytest-benchmark; assert runtime < threshold for reference portfolio
- Mutation testing: mutmut or cosmic-ray; verify test suite kills all single-line mutations
- Contract tests: interface layer contracts (curve returns scalar, pricer returns array [T, N])
- Load / stress tests: 1M paths, 10 netting sets, assert no memory blowup

### Industry practice
- **Model validation** (SR 11-7 / ECB guide) requires: conceptual soundness review, independent implementation, outcome analysis (backtesting), sensitivity analysis, and a limitation register. Test suites form part of the validation evidence package.
- **Backtesting CVA** means comparing ex-ante CVA with realised credit losses over a historical window. This is required for IMM approval.
- **Convergence analysis** (CVA as a function of path count) is a standard exhibit in any MC model validation package submitted to regulators.
- **Regression / golden files** are used by all major quant libraries (QuantLib, OpenGamma Strata) to detect unintended numerical changes across releases.

---

## Layer 11 — UI (`ui/`)

### What to build
- Portfolio-level dashboard: aggregate XVA across all netting sets; heatmap by counterparty
- Real-time slider updates: debounced live recalc without needing a "Run" button
- 3D surface charts: EE surface (time × path percentile) as Plotly 3D surface
- Scenario manager: save/load named scenarios; compare up to 4 scenarios side-by-side
- Trade blotter: editable DataGrid for adding/removing trades in the UI
- Market data editor: inline curve pillar editing with immediate re-bootstrap
- XVA attribution waterfall: drill down from portfolio → netting set → trade → risk factor
- Counterparty credit dashboard: survival curve, hazard term structure, PD term structure
- Report export: one-click HTML/PDF generation from current session state

### Industry practice
- **Bloomberg Terminal** is the reference UI standard in fixed income: tabbed layout, keyboard-driven, high information density. Risk systems (Murex, Calypso) replicate this density for traders.
- **Pre-trade XVA blotters** at major banks show live CVA/FVA/KVA as traders structure a deal, using cached Greeks for real-time approximations.
- **Self-service risk dashboards** (Tableau, Power BI, internal React apps) are increasingly used for senior management XVA P&L reporting. Streamlit is a lightweight equivalent for internal analytics teams.

---

## Layer 12 — Code Quality & Architecture

### What to build
- `Protocol` / interface definitions: `Pricer`, `RateModel`, `CollateralModel`, `CapitalModel` as formal `typing.Protocol` classes; enforces pluggability
- Full type annotations: `mypy --strict` clean; `numpy.typing.NDArray` for all arrays
- Dependency injection: all dependencies passed explicitly; no module-level globals
- Async engine: `asyncio`-native for REST API and streaming; `await engine.run_async()`
- Plugin registry: `@register_product("XCCY_SWAP")` decorator system; dynamic plugin loading
- Structured logging: `structlog` JSON logs; per-run_id correlation; configurable level per module
- OpenTelemetry tracing: distributed tracing for REST API mode; trace each engine stage
- Container / cloud deployment: `Dockerfile` + `docker-compose`; Helm chart for Kubernetes; GitHub Actions CD on tag

### Industry practice
- **Clean / hexagonal architecture** (ports and adapters) is widely adopted in quant libraries. The core principle: pricing logic (domain layer) has zero dependency on IO, UI, or infrastructure.
- **Quant library design patterns** (QuantLib, OpenGamma Strata): `Instrument` + `PricingEngine` separation; `MarketData` as an injectable context; `Calculator` as a pure function with no side effects.
- **Docker + Kubernetes** is the standard deployment target for risk microservices at tier-1 banks. XVA engines are typically deployed as gRPC or REST services behind an API gateway.
- **Structured logging + distributed tracing** (Datadog, Splunk, Grafana/Tempo) is mandatory for production risk systems to diagnose calculation failures and audit specific trade valuations.

---

## Suggested Phased Roadmap

| Phase  | Theme                           | Key deliverables                                             |
| ------ | ------------------------------- | ------------------------------------------------------------ |
| **4**  | Multi-curve + vol surface       | SOFR/OIS curve, swaption vol cube, cap/floor pricing         |
| **5**  | G2++ model + variance reduction | Better swaption fit, antithetic / QMC, automatic calibration |
| **6**  | FX + equity products            | XCCY CCS, equity options, hybrid HW+GBM model                |
| **7**  | Realistic collateral + SIMM     | Full CSA terms, MPOR, ISDA SIMM MVA                          |
| **8**  | SA-CCR + regulatory capital     | KVA under SA-CCR/IMM, FRTB-CVA capital charge                |
| **9**  | Wrong-way risk                  | JCIR++ correlated credit/rates, WWR-adjusted CVA             |
| **10** | REST API + production packaging | FastAPI, Docker, async engine, structured logs               |
| **11** | Advanced UI + reporting         | Portfolio dashboard, scenario manager, HTML/PDF export       |
| **12** | ML for IM / exposure            | Neural network SIMM proxy, signature-based exposure models   |

---

## Key Reference Texts

| Book / Document                                                          | Relevance                                                 |
| ------------------------------------------------------------------------ | --------------------------------------------------------- |
| Brigo & Mercurio — *Interest Rate Models: Theory and Practice* (2nd ed.) | G2++, HW2F calibration, swaption pricing                  |
| Gregory — *Counterparty Credit Risk* (4th ed.)                           | CVA/DVA methodology, wrong-way risk, regulatory framework |
| Green — *XVA: Credit, Funding and Capital Valuation Adjustments* (2015)  | FVA, MVA, KVA, full XVA framework                         |
| Andersen & Piterbarg — *Interest Rate Modelling* (3 vols.)               | LMM, SABR, Bermudan pricing reference                     |
| Glasserman — *Monte Carlo Methods in Financial Engineering*              | Variance reduction, QMC, LSM                              |
| ISDA SIMM Methodology (v2.6, 2024)                                       | SIMM sensitivity buckets, IM calculation                  |
| BCBS CRE50 (SA-CCR, 2019)                                                | Standardised EAD formula                                  |
| BCBS FRTB (January 2019, revised 2020)                                   | SA-CVA capital, sensitivity-based approach                |
