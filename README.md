# XVA Engine

XVA engine using Hull-White 1F, built with NumPy, Numba, Polars, and Pydantic.

## Stack
- `uv` — dependency management
- `numba` — JIT-compiled Monte Carlo simulation
- `polars` — vectorized data handling for XVA tables
- `pydantic` — strict input schema validation
- `pytest` — test suite
- `streamlit` — demo UI

## Project Structure
```
src/xva_engine/
├── io/          # Pydantic input schemas
├── market/      # Discount curve, credit curve
├── products/    # ZCB, IRS pricers, schedule generation
├── models/      # Hull-White 1F math (B, V, bond price, OU simulation)
├── sim/         # Time grid builder, batched Monte Carlo engine
├── exposure/    # Collateral model, exposure statistics (EE, ENE)
├── xva/         # CVA, DVA, FVA, MVA, KVA with contribution tables
├── explain/     # Explainability bundle (math_report.md, arrays, parquet tables)
├── ui/          # Streamlit demo app
└── engine.py    # Unified XVA orchestrator
```

## Run Tests
```bash
uv sync
uv run pytest tests/
```

## Run Demo UI
```bash
uv run streamlit run src/xva_engine/ui/streamlit_app.py
```
