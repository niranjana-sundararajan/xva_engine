## Plan: Unified XVA Engine Implementation

This plan describes the technical implementation of the XVA Engine. It uses a modern, high-performance Python stack: structural validation with `Pydantic`, parallel data frames with `Polars`, core mathematics accelerated by `Numba`, and test-driven development via `pytest`. The engine covers deterministic pricing, batched Monte Carlo simulations, exposure aggregations, and full XVA analytics as an integrated stack.

**Steps**

**1. Foundations & Schemas**
1. **Repository Setup**: Initialize a `uv` project with core dependencies (numpy, numba, polars, pydantic, scipy, pytest, streamlit). Create the overarching structure under `src/xva_engine/` and `tests/`.
2. **IO Schemas (`io`)**: Create [src/xva_engine/io/schemas.py](src/xva_engine/io/schemas.py) defining strict `Pydantic` models for `Trade`, `NettingSet`, `CSA`, `CurveSnapshot`, `CreditCurve`, and `ModelConfig`.

**2. Market Data & Pricing Models**
3. **Curve Interpolation & Credit (`market`)**: Implement [src/xva_engine/market/curve.py](src/xva_engine/market/curve.py) & [src/xva_engine/market/credit.py](src/xva_engine/market/credit.py). Establish discount algorithms with log-linear interpolation and discrete PD ($\Delta PD(t_i)$) extraction routines over tabular data.
4. **Time Schedules (`products`)**: Build [src/xva_engine/products/schedule.py](src/xva_engine/products/schedule.py) handling dates, ACT/365F fractions, and payment frequencies dynamically using Polars.
5. **Valuation Models (`products`)**: Build path-compatible vector processors `ZcbPricer` ([src/xva_engine/products/zcb.py](src/xva_engine/products/zcb.py)) and `IrsPricer` ([src/xva_engine/products/irs.py](src/xva_engine/products/irs.py)).
6. **Hull-White 1F Math Models (`models`)**: Create [src/xva_engine/models/hw1f.py](src/xva_engine/models/hw1f.py) establishing Gaussian factor $B(t,T)$ and vectorized consistent bond pricing $P(t,T)$.

**3. Monte Carlo Simulation Engine**
7. **Time Grid Orchestration (`sim`)**: Create [src/xva_engine/sim/timegrid.py](src/xva_engine/sim/timegrid.py) extracting unique timestamps for cash flows alongside grid simulation intervals.
8. **JIT Path Generation (`sim`)**: Implement [src/xva_engine/sim/batching.py](src/xva_engine/sim/batching.py) with batched RNGs mapped directly to `@njit` Numba routines spanning out massive simulations of arrays natively.
9. **Pathwise Valuation (`sim`)**: Connect simulation arrays back through product pricers for real-time un-batched vector evaluations traversing grid points across Monte Carlo paths.

**4. Collateral & Exposure Processing**
10. **Collateral Engine (`exposure`)**: Establish [src/xva_engine/exposure/collateral.py](src/xva_engine/exposure/collateral.py) margin rule abstractions $C(t, \omega)$ (primarily `none` and `perfect_vm`).
11. **Exposure Statistics (`exposure`)**: Aggregate arrays in [src/xva_engine/exposure/exposure.py](src/xva_engine/exposure/exposure.py) computing cross-sectional $E$, $NE$, $EE$, and $ENE$. Back outputs to uniform Polars dataframes.

**5. XVA Analytics & UI Demo**
12. **XVA Components (`xva`)**: Multiply time-grid exposures by state discount functions and counterparty credit. Implement discrete plugins:
    - [src/xva_engine/xva/cva.py](src/xva_engine/xva/cva.py)
    - [src/xva_engine/xva/dva.py](src/xva_engine/xva/dva.py)
    - [src/xva_engine/xva/fva.py](src/xva_engine/xva/fva.py)
    - [src/xva_engine/xva/mva.py](src/xva_engine/xva/mva.py)
    - [src/xva_engine/xva/kva.py](src/xva_engine/xva/kva.py)
13. **Explainability Aggregation (`explain`)**: Finalize [src/xva_engine/explain/bundle.py](src/xva_engine/explain/bundle.py) producing `.npz` storage outputs alongside human-readable `math_report.md` generating exact breakdowns.
14. **Demo Integration (`ui`)**: Assemble an interactive dashboard inside [src/xva_engine/ui/streamlit_app.py](src/xva_engine/ui/streamlit_app.py) taking all system elements end-to-end dynamically mapping configured IOs.

**Verification**
- Run `uv sync` to ensure fully locked, repeatable platform dependencies.
- Pass edge-cases and mathematical invariances executed via `uv run pytest tests/` spanning par-swap calculations and 0-average VM path generation validations.
- Launch `uv run streamlit run src/xva_engine/ui/streamlit_app.py` directly simulating the default sample minimum testcase set.
- Check `math_report.md` generation artifacts visually verifying outputs properly reflect configuration states.

**Decisions**
- **uv**: Specified for immediate high-speed project initialization mapping pip dependencies identically across environments.
- **Numba**: Placed strictly around the engine simulation loop routines to maximize parallel single-threaded performance mathematically without adding heavy multi-node complexity early on.
- **Pydantic / Polars**: Segregated bounds; validation strictly through structural type checks ahead of ingesting dataframes into Polars aggregation pipelines.
- **pytest**: Ensures structural mathematical determinism required per non-negotiables laid out in initial project documentation specs.
