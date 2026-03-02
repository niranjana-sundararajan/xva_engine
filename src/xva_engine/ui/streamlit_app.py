"""
Streamlit UI for the XVA Engine.
Run with: uv run streamlit run src/xva_engine/ui/streamlit_app.py
"""
import sys
import json
from pathlib import Path

_src_dir = str(Path(__file__).resolve().parents[2])
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from xva_engine.io.schemas import (
    ZCBTa, IRSTrade, NettingSet, CurveSnapshot, CurvePoint,
    CreditCurve, CreditPoint, ModelConfig, HullWhiteParams, MonteCarloConfig,
)
from xva_engine.io.loaders import load_portfolio_csv, load_discount_curve_json, load_credit_curves_json
from xva_engine.market.curve import DiscountCurve
from xva_engine.market.credit import CreditCurveModel
from xva_engine.products.zcb import ZcbPricer
from xva_engine.products.irs import IrsPricer
from xva_engine.engine import XVAEngine
import uuid as _uuid

# ---------------------------------------------------------------------------
st.set_page_config(page_title="XVA Engine", layout="wide")
st.title("XVA Engine")


def _df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()


def _default_curve():
    return CurveSnapshot(currency="USD", points=[
        CurvePoint(tenor=0,    discount_factor=1.000),
        CurvePoint(tenor=182,  discount_factor=0.988),
        CurvePoint(tenor=365,  discount_factor=0.975),
        CurvePoint(tenor=730,  discount_factor=0.950),
        CurvePoint(tenor=1825, discount_factor=0.880),
    ])


def _default_credit_map():
    return {
        "CPTY-1": CreditCurve(entity_id="CPTY-1", recovery_rate=0.40, points=[
            CreditPoint(tenor=0,    survival_prob=1.00),
            CreditPoint(tenor=365,  survival_prob=0.98),
            CreditPoint(tenor=730,  survival_prob=0.96),
            CreditPoint(tenor=1825, survival_prob=0.90),
        ]),
        "BANK": CreditCurve(entity_id="BANK", recovery_rate=0.40, points=[
            CreditPoint(tenor=0,    survival_prob=1.00),
            CreditPoint(tenor=365,  survival_prob=0.99),
            CreditPoint(tenor=730,  survival_prob=0.985),
            CreditPoint(tenor=1825, survival_prob=0.96),
        ]),
    }


def _default_netting_sets():
    return [NettingSet(
        netting_set_id="NS-CPTY1", counterparty_id="CPTY-1",
        trades=[
            ZCBTa(trade_id="ZCB-1", netting_set_id="NS-CPTY1",
                  notional=1_000_000, maturity_date=730),
            IRSTrade(trade_id="IRS-1", netting_set_id="NS-CPTY1",
                     notional=1_000_000, start_date=0, maturity_date=1825,
                     receive_fixed=False, fixed_rate=0.025, payment_frequency=6),
        ],
    )]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio Inputs")
    portfolio_file = st.file_uploader("portfolio.csv",       type="csv",  key="port")
    netting_file   = st.file_uploader("netting_sets.csv",    type="csv",  key="ns")
    curve_file     = st.file_uploader("discount_curve.json", type="json", key="crv")
    credit_file    = st.file_uploader("credit_curves.json",  type="json", key="crd")

    st.divider()
    st.header("Model Parameters")
    hw_a      = st.number_input("HW Mean Reversion (a)", min_value=0.0001, value=0.05,  step=0.005, format="%.4f")
    hw_sigma  = st.number_input("HW Volatility (sigma)", min_value=0.0001, value=0.01,  step=0.001, format="%.4f")
    num_paths = st.slider("MC Paths", 500, 20_000, 5_000, 500)
    seed      = int(st.number_input("Seed", 0, 99999, 42))
    s_f       = st.number_input("Funding Spread s_f",    value=0.010, step=0.001, format="%.4f")
    c_cap     = st.number_input("Cost of Capital c_cap", value=0.080, step=0.005, format="%.4f")

    st.divider()
    run_btn = st.button("Run XVA Engine", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Parse inputs
# ---------------------------------------------------------------------------
def _parse_inputs():
    if portfolio_file and netting_file:
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as pt:
            pt.write(portfolio_file.read()); pp = pt.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as nt:
            nt.write(netting_file.read()); np_ = nt.name
        ns_list = load_portfolio_csv(pp, np_)
        os.unlink(pp); os.unlink(np_)
    else:
        ns_list = _default_netting_sets()

    if curve_file:
        d = json.loads(curve_file.read())
        curve = CurveSnapshot(currency=d["currency"],
                              points=[CurvePoint(**p) for p in d["points"]])
    else:
        curve = _default_curve()

    if credit_file:
        d = json.loads(credit_file.read())
        cmap = {i["entity_id"]: CreditCurve(
            entity_id=i["entity_id"], recovery_rate=i["recovery_rate"],
            points=[CreditPoint(**p) for p in i["points"]]) for i in d}
    else:
        cmap = _default_credit_map()

    cfg = ModelConfig(
        hw_params=HullWhiteParams(mean_reversion=hw_a, volatility=hw_sigma),
        mc_config=MonteCarloConfig(num_paths=num_paths, seed=seed, batch_size=1000),
    )
    return ns_list, curve, cmap, cfg


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_btn:
    ns_list, curve, cmap, cfg = _parse_inputs()
    bank_cr = cmap.get("BANK", list(cmap.values())[-1])
    all_res = {}
    with st.spinner("Running Monte Carlo simulation..."):
        for ns in ns_list:
            cc = cmap.get(ns.counterparty_id, bank_cr)
            eng = XVAEngine.__new__(XVAEngine)
            eng.netting_set       = ns
            eng.config            = cfg
            eng.funding_spread    = s_f
            eng.cost_of_capital   = c_cap
            eng.output_dir        = "output"
            eng.discount_curve    = DiscountCurve(curve)
            eng.cpty_credit_model = CreditCurveModel(cc)
            eng.bank_credit_model = CreditCurveModel(bank_cr)
            eng.pricers = []
            for t in ns.trades:
                if t.trade_type == "ZCB":
                    eng.pricers.append(ZcbPricer(t))
                elif t.trade_type == "IRS":
                    eng.pricers.append(IrsPricer(t))
            eng.csa_schema = None
            eng.run_id = str(_uuid.uuid4())[:8]
            all_res[ns.netting_set_id] = eng.run()
    st.session_state.update(
        all_results=all_res, ns_list=ns_list,
        curve=curve, cmap=cmap, cfg=cfg,
        bank_cr=bank_cr, s_f=s_f, c_cap=c_cap,
        currency=curve.currency,
    )
    st.success(f"Done — {len(all_res)} netting set(s) processed.")

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------
if "all_results" not in st.session_state:
    st.info("Configure inputs in the sidebar and press **Run XVA Engine** to begin.")
    st.stop()

all_results = st.session_state["all_results"]
curve_snap  = st.session_state["curve"]
cmap        = st.session_state["cmap"]
cfg_obj     = st.session_state["cfg"]
base_currency = st.session_state.get("currency", "USD")

# Netting-set selector
ns_sel   = st.selectbox("Netting Set", list(all_results.keys()))
r        = all_results[ns_sel]
grid_yrs = r["grid_days"] / 365.0

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Summary", "Market Data", "Simulation",
    "Exposure", "XVA Breakdown", "Sensitivity",
])

# ===========================================================================
# TAB 1 — SUMMARY
# ===========================================================================
with tab1:
    net = r["CVA"] - r["DVA"] + r["FVA"] + r["MVA"] + r["KVA"]
    cols = st.columns(6)
    for col, label, val, tip in zip(cols,
            ["CVA", "DVA", "FVA", "MVA", "KVA", "Net XVA"],
            [r["CVA"], r["DVA"], r["FVA"], r["MVA"], r["KVA"], net],
            ["Cost of expected counterparty default",
             "Benefit from own default risk",
             "Uncollateralised funding cost",
             "Initial margin funding cost",
             "Regulatory capital charge",
             "CVA - DVA + FVA + MVA + KVA"]):
        col.metric(label, f"{val:,.2f}", help=tip)

    # ---- Waterfall --------------------------------------------------------
    cl, cr = st.columns(2)
    with cl:
        st.subheader("XVA Waterfall")
        st.caption(
            "Each bar shows the contribution of one XVA component to the total "
            "valuation adjustment. Positive bars increase the cost (or reduce value) "
            "to the bank; DVA is negative because it reflects a discount for the "
            "bank's own default risk. The final bar, Net XVA, is the algebraic sum."
        )
        fig = go.Figure(go.Waterfall(
            measure=["relative","relative","relative","relative","relative","total"],
            x=["CVA", "-DVA", "FVA", "MVA", "KVA", "Net XVA"],
            y=[r["CVA"], -r["DVA"], r["FVA"], r["MVA"], r["KVA"], 0],
            increasing={"marker": {"color": "#EF553B"}},
            decreasing={"marker": {"color": "#00CC96"}},
            totals={"marker": {"color": "#636EFA"}},
            connector={"line": {"color": "grey"}},
        ))
        fig.update_layout(height=320, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- EE / ENE --------------------------------------------------------
    with cr:
        st.subheader("Exposure Profile: EE and ENE")
        st.caption(
            "Expected Exposure (EE) is the average positive mark-to-market of the "
            "netting set across all Monte Carlo paths at each future date — "
            "the amount at risk if the counterparty defaults. "
            "Expected Negative Exposure (ENE) is the mirror: the average "
            "negative MtM, representing what the counterparty is owed if the bank "
            "itself defaults. Higher EE drives CVA; higher ENE drives DVA."
        )
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["EE"],
            fill="tozeroy", name="EE",
            fillcolor="rgba(99,110,250,0.25)", line=dict(color="rgb(99,110,250)")))
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["ENE"],
            fill="tozeroy", name="ENE",
            fillcolor="rgba(239,85,59,0.25)", line=dict(color="rgb(239,85,59)")))
        fig.update_layout(height=320, xaxis_title="Years",
                          yaxis_title=f"USD", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # ---- Product breakdown -----------------------------------------------
    st.subheader("Expected Exposure by Product Type")
    ee_by_type = r.get("ee_by_type", {})
    if ee_by_type:
        fig = go.Figure()
        colors = {"ZCB": "steelblue", "IRS": "darkorange"}
        for ptype, ee_arr in ee_by_type.items():
            fig.add_trace(go.Scatter(
                x=grid_yrs, y=ee_arr, name=ptype,
                line=dict(color=colors.get(ptype, "grey"), width=2)))
        fig.add_trace(go.Scatter(
            x=grid_yrs, y=r["EE"], name="Total (NS)",
            line=dict(color="black", dash="dash", width=1.5)))
        fig.update_layout(height=280, xaxis_title="Years",
                          yaxis_title=f"EE (USD)",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        product_combo = " + ".join(sorted(ee_by_type.keys()))
        st.caption(f"Product mix in {ns_sel}: **{product_combo}** — "
                   "each line is the EE contribution from that product type alone "
                   "(collateral applied at netting-set level).")
    else:
        st.info("No per-product breakdown available.")

    st.caption(
        f"Run ID: `{r['run_id']}` | Grid: {len(r['grid_days'])} steps | "
        f"Paths: {cfg_obj.mc_config.num_paths:,} | Currency: {base_currency}"
    )
    summary_df = pd.DataFrame(
        {"Metric": ["CVA","DVA","FVA","MVA","KVA","Net XVA"],
         "Value":  [r["CVA"],r["DVA"],r["FVA"],r["MVA"],r["KVA"],net]}
    )
    st.download_button("Download Summary CSV", _df_to_csv(summary_df),
                       "xva_summary.csv", "text/csv")

# ===========================================================================
# TAB 2 — MARKET DATA
# ===========================================================================
with tab2:
    cl, cr = st.columns(2)

    with cl:
        st.subheader("Discount Curve")
        st.caption(
            "The discount curve maps each tenor to a present-value factor P(0, t). "
            "Discount factors are interpolated log-linearly between pillar points, "
            "which preserves positivity and ensures a smooth zero-rate curve. "
            "The continuously-compounded zero rate r(t) = -ln P(0,t) / t is "
            "shown on the right axis."
        )
        t_arr = [p.tenor / 365 for p in curve_snap.points]
        d_arr = [p.discount_factor for p in curve_snap.points]
        z_arr = [-np.log(max(d, 1e-9)) / (t + 1e-9) * 100 for t, d in zip(t_arr, d_arr)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_arr, y=d_arr, name="Discount Factor",
                                  line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=t_arr, y=z_arr, name="Zero Rate (%)",
                                  line=dict(color="orange", dash="dash"), yaxis="y2"))
        fig.update_layout(height=300,
            yaxis=dict(title="Discount Factor"),
            yaxis2=dict(title="Zero Rate (%)", overlaying="y", side="right"),
            margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Interpolation formula"):
            st.caption(
                "Between pillar points t_i and t_{i+1}, the log discount factor "
                "is linearly interpolated. This corresponds to a piecewise-constant "
                "instantaneous forward rate between pillars."
            )
            st.latex(
                r"\ln P(0,t)=\ln P(0,t_i)"
                r"+\frac{t-t_i}{t_{i+1}-t_i}"
                r"\bigl[\ln P(0,t_{i+1})-\ln P(0,t_i)\bigr]"
            )
        dc_dl = pd.DataFrame({
            "tenor_days": [p.tenor for p in curve_snap.points],
            "discount_factor": d_arr, "zero_rate_pct": z_arr,
        })
        st.download_button("Download Discount Curve CSV", _df_to_csv(dc_dl),
                           "discount_curve.csv", "text/csv")

    with cr:
        st.subheader("Credit / Survival Curves")
        st.caption(
            "Each survival curve S(t) gives the probability that an entity has "
            "not defaulted by time t. Curves are derived from CDS or bond spreads "
            "and are used to compute marginal default probabilities in each time bucket. "
            "A steeper decline in S(t) indicates higher credit risk."
        )
        fig = go.Figure()
        for eid, cc in cmap.items():
            fig.add_trace(go.Scatter(
                x=[p.tenor / 365 for p in cc.points],
                y=[p.survival_prob for p in cc.points], name=eid))
        fig.update_layout(height=300, xaxis_title="Years",
                          yaxis_title="Survival Probability", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Marginal default probability"):
            st.caption(
                "The marginal probability of default in bucket [t_{i-1}, t_i] "
                "is simply the loss in survival probability over that interval. "
                "This is used directly when computing CVA and DVA bucket contributions."
            )
            st.latex(r"\Delta PD(t_{i-1},t_i)=S(t_{i-1})-S(t_i)")

# ===========================================================================
# TAB 3 — SIMULATION
# ===========================================================================
with tab3:
    st.subheader("Time Grid")
    st.caption(
        "The simulation grid is built from the union of all trade cashflow and "
        "reset dates across the netting set, plus optional dense points. "
        "Each dot below is a valuation date. Pricing at every grid point "
        "allows the engine to compute trade mark-to-market along each path "
        "and then aggregate to netting-set exposure."
    )
    gd = r["grid_days"]
    fig = go.Figure(go.Scatter(
        x=gd / 365, y=[0] * len(gd), mode="markers+text",
        text=[f"{int(d)}d" for d in gd], textposition="top center",
        marker=dict(size=8, color=["green"] + ["steelblue"] * (len(gd) - 1)),
    ))
    fig.update_layout(height=140, xaxis_title="Years",
                      yaxis=dict(visible=False), showlegend=False,
                      margin=dict(t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hull-White Factor Paths")
    st.caption(
        "The Hull-White 1-factor model decomposes the short rate as r(t) = phi(t) + x(t), "
        "where phi(t) fits the initial discount curve exactly, and x(t) is a "
        "mean-reverting Ornstein-Uhlenbeck process driven by Brownian shocks. "
        "The chart shows 50 sample paths of x(t). Mean reversion pulls paths back "
        "toward zero at rate 'a'; wider fans correspond to higher volatility sigma."
    )
    xs = r["x_sample"]
    fig = go.Figure()
    for j in range(xs.shape[1]):
        fig.add_trace(go.Scatter(x=grid_yrs, y=xs[:, j] * 100,
            mode="lines", opacity=0.25, line=dict(width=0.8, color="steelblue"),
            showlegend=False))
    fig.add_trace(go.Scatter(x=grid_yrs, y=xs.mean(axis=1) * 100,
        line=dict(width=2.5, color="navy"), name="Mean x(t)"))
    fig.update_layout(height=300, xaxis_title="Years",
                      yaxis_title="x(t) (%)", margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Exact simulation step"):
        st.caption(
            "The OU process is simulated exactly (no Euler discretisation error) "
            "using the conditional distribution of x at t + dt given x at t. "
            "The variance of the increment depends on a and sigma only."
        )
        st.latex(
            r"x_{t+\Delta t}=x_t e^{-a\Delta t}"
            r"+\sigma\sqrt{\frac{1-e^{-2a\Delta t}}{2a}}\,Z,\quad Z\sim\mathcal{N}(0,1)"
        )

# ===========================================================================
# TAB 4 — EXPOSURE
# ===========================================================================
with tab4:
    st.caption(
        "Exposure measures how much is at risk at each future date. "
        "Each path represents one possible market scenario; the wide fan "
        "of MtM values reflects interest-rate uncertainty. "
        "EE and ENE are path averages of the positive and negative MtM after "
        "applying any collateral agreement."
    )

    Vs = r["V_ns_sample"]
    cl, cr = st.columns(2)

    with cl:
        st.subheader("Mark-to-Market Path Fan")
        fig = go.Figure()
        for j in range(Vs.shape[1]):
            fig.add_trace(go.Scatter(x=grid_yrs, y=Vs[:, j],
                mode="lines", opacity=0.2, line=dict(width=0.8, color="teal"),
                showlegend=False))
        fig.add_trace(go.Scatter(x=grid_yrs, y=Vs.mean(axis=1),
            line=dict(width=2.5, color="darkgreen"), name="Mean V(t)"))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(height=300, xaxis_title="Years",
                          yaxis_title="V_NS (USD)", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("EE / ENE Profile")
        with st.expander("What these curves represent"):
            st.latex(r"EE(t_i)=\mathbb{E}[\max(V_{NS}(t_i)-C(t_i),\,0)]")
            st.latex(r"ENE(t_i)=\mathbb{E}[\max(C(t_i)-V_{NS}(t_i),\,0)]")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["EE"],
            fill="tozeroy", name="EE",
            fillcolor="rgba(0,204,150,0.3)", line=dict(color="rgb(0,204,150)")))
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["ENE"],
            fill="tozeroy", name="ENE",
            fillcolor="rgba(239,85,59,0.3)", line=dict(color="rgb(239,85,59)")))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(height=300, xaxis_title="Years",
                          yaxis_title="USD", margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Integrated exposure metrics
    st.subheader("Aggregated Exposure (All Time Steps)")
    dt_arr = np.diff(r["grid_days"]) / 365.0
    df_arr = np.array([curve_snap.points[0].discount_factor] +
                      [float(DiscountCurve(curve_snap).df(t)) for t in r["grid_days"][1:]])
    EPE  = float(np.sum(r["EE"][1:]  * df_arr[1:] * dt_arr))
    ENE_int = float(np.sum(r["ENE"][1:] * df_arr[1:] * dt_arr))
    peak_ee = float(np.max(r["EE"]))
    peak_t  = grid_yrs[int(np.argmax(r["EE"]))]
    m1, m2, m3 = st.columns(3)
    m1.metric("Expected Positive Exposure (EPE)",
              f"{EPE:,.2f}",
              help="Discount-weighted integral of EE over the simulation horizon.")
    m2.metric("Expected Negative Exposure (ENE integral)",
              f"{ENE_int:,.2f}",
              help="Discount-weighted integral of ENE over the simulation horizon.")
    m3.metric("Peak EE",
              f"{peak_ee:,.2f}",
              help=f"Maximum EE reached at t = {peak_t:.2f} yr.")

    st.caption(
        "EPE = integral of P(0,t) x EE(t) dt — the single number that enters "
        "the CVA formula when the hazard rate is constant. "
        "Peak EE indicates the worst-case exposure date."
    )

    # Distribution
    st.subheader("MtM Distribution at Selected Time Step")
    idx = st.slider("Grid step", 0, len(gd) - 1, min(4, len(gd) - 1))
    fig = go.Figure(go.Histogram(x=Vs[idx, :], nbinsx=40,
                                 marker_color="steelblue", opacity=0.75))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=200, xaxis_title="V_NS (USD)",
        title=f"t = {gd[idx]:.0f}d / {grid_yrs[idx]:.2f} yr",
        margin=dict(t=35, b=10))
    st.plotly_chart(fig, use_container_width=True)

    ee_dl = pd.DataFrame({"grid_days": gd, "grid_years": grid_yrs,
                          "EE": r["EE"], "ENE": r["ENE"]})
    st.download_button("Download EE/ENE CSV", _df_to_csv(ee_dl),
                       "exposure_profile.csv", "text/csv")

# ===========================================================================
# TAB 5 — XVA BREAKDOWN
# ===========================================================================
with tab5:
    # Currency filter
    avail_ccy = sorted({st.session_state.get("currency", "USD")})
    sel_ccy = st.selectbox("Currency", avail_ccy, key="ccy5")

    METRIC_META = [
        ("CVA", "cva_table", "cva_contribution", "#EF553B",
         "Cost of expected loss if the counterparty defaults before maturity.",
         r"CVA=(1-R_c)\sum_i P(0,t_i)\cdot EE(t_i)\cdot\Delta PD_c(t_{i-1},t_i)"),
        ("DVA", "dva_table", "dva_contribution", "#00CC96",
         "Benefit arising from the bank's own credit risk — reduces net XVA.",
         r"DVA=(1-R_b)\sum_i P(0,t_i)\cdot ENE(t_i)\cdot\Delta PD_b(t_{i-1},t_i)"),
        ("FVA", "fva_table", "fva_contribution", "#AB63FA",
         "Cost of funding uncollateralised positive exposure at the bank's spread.",
         r"FVA=\sum_i P(0,t_i)\cdot s_f\cdot EE_{net}(t_i)\cdot\Delta t_i"),
        ("MVA", "mva_table", "mva_contribution", "#FFA15A",
         "Funding cost of posting initial margin over the life of the trade.",
         r"MVA=\sum_i P(0,t_i)\cdot s_f\cdot\mathbb{E}[IM(t_i)]\cdot\Delta t_i"),
        ("KVA", "kva_table", "kva_contribution", "#19D3F3",
         "Return on regulatory capital required to support the exposure.",
         r"KVA=\sum_i P(0,t_i)\cdot c_{cap}\cdot\mathbb{E}[K(t_i)]\cdot\Delta t_i"),
    ]

    # Combined chart — all metrics on one figure as grouped bars
    st.subheader("Bucket Contributions — All Metrics")
    fig = go.Figure()
    all_tbl_frames = []
    for name, tbl_k, col, colour, _desc, _formula in METRIC_META:
        tbl = r[tbl_k].to_pandas()
        tbl["metric"] = name
        renamed = tbl.rename(columns={col: "contribution"})
        keep = [c for c in ["metric", "t_end_days", "df", "dt_years", "contribution"]
                if c in renamed.columns]
        all_tbl_frames.append(renamed[keep])
        fig.add_trace(go.Bar(
            name=name,
            x=tbl["t_end_days"].values / 365,
            y=tbl[col].values,
            marker_color=colour,
        ))
    fig.update_layout(
        barmode="group",
        height=380,
        xaxis_title="Years",
        yaxis_title=f"Contribution (USD)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        margin=dict(t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metric summaries
    totals_row = {name: float(r[tbl_k].to_pandas()[col].sum())
                  for name, tbl_k, col, *_ in METRIC_META}
    tot_cols = st.columns(5)
    for tc, (name, *_) in zip(tot_cols, METRIC_META):
        tc.metric(name, f"{totals_row[name]:,.4f}")

    # Formulas and one-line explanations
    st.markdown("---")
    st.subheader("Definitions and Formulas")
    for name, _tbl, _col, colour, description, formula in METRIC_META:
        fc, lc = st.columns([1, 3])
        with fc:
            st.markdown(f"**{name}**")
            st.caption(description)
        with lc:
            st.latex(formula)

    # Data preview + download
    st.markdown("---")
    combined_df = pd.concat(all_tbl_frames, ignore_index=True)
    st.caption("Data preview (first 10 rows):")
    st.dataframe(combined_df.head(10), use_container_width=True, height=220)
    st.download_button(
        "Download All Bucket Data CSV",
        _df_to_csv(combined_df),
        "xva_buckets_all.csv", "text/csv",
    )

# ===========================================================================
# TAB 6 — SENSITIVITY
# ===========================================================================
with tab6:
    # Currency filter
    avail_ccy6 = sorted({st.session_state.get("currency", "USD")})
    sel_ccy6 = st.selectbox("Currency", avail_ccy6, key="ccy6")

    st.subheader("Sensitivity Analysis")

    ca, cb, cc_ = st.columns(3)
    hz  = ca.slider("Hazard rate multiplier", 0.5, 3.0, 1.0, 0.1, key="hz_sl")
    ca.caption("Scales the implied hazard rate of the counterparty survival curve — "
               "a multiplier >1 increases credit spread, raising CVA.")
    sfb = cb.slider("Funding spread bump (bps)", -50, 200, 0, 10, key="sfb_sl")
    cb.caption("Adds a flat shift in basis points to the funding spread s_f — "
               "positive bumps increase FVA and MVA proportionally.")
    svp = cc_.slider("HW sigma scaling (%)", 50, 200, 100, 10, key="svp_sl")
    cc_.caption("Rescales the Hull-White volatility parameter sigma — "
                "higher values widen the MtM distribution and increase all exposure-driven XVAs.")

    if st.button("Run Sensitivity"):
        ns_obj  = next(ns for ns in st.session_state["ns_list"] if ns.netting_set_id == ns_sel)
        bank_cr = st.session_state["bank_cr"]
        base_cc = cmap.get(ns_obj.counterparty_id, bank_cr)

        # Bump credit — scale hazard rate
        bpts = []
        for pt in base_cc.points:
            if pt.tenor == 0:
                bpts.append(CreditPoint(tenor=0, survival_prob=1.0))
            else:
                t_y = pt.tenor / 365
                h   = -np.log(max(pt.survival_prob, 1e-9)) / t_y * hz
                bpts.append(CreditPoint(tenor=pt.tenor,
                                        survival_prob=float(np.exp(-h * t_y))))
        bcc = CreditCurve(entity_id=base_cc.entity_id,
                          recovery_rate=base_cc.recovery_rate, points=bpts)

        bcfg = ModelConfig(
            hw_params=HullWhiteParams(
                mean_reversion=cfg_obj.hw_params.mean_reversion,
                volatility=cfg_obj.hw_params.volatility * svp / 100),
            mc_config=cfg_obj.mc_config,
        )
        bsf = st.session_state["s_f"] + sfb / 10_000

        with st.spinner("Running sensitivity..."):
            e2 = XVAEngine.__new__(XVAEngine)
            e2.netting_set = ns_obj
            e2.config = bcfg
            e2.funding_spread = bsf
            e2.cost_of_capital = st.session_state["c_cap"]
            e2.output_dir = "output"
            e2.discount_curve    = DiscountCurve(curve_snap)
            e2.cpty_credit_model = CreditCurveModel(bcc)
            e2.bank_credit_model = CreditCurveModel(bank_cr)
            e2.pricers = []
            for t in ns_obj.trades:
                if t.trade_type == "ZCB":
                    e2.pricers.append(ZcbPricer(t))
                elif t.trade_type == "IRS":
                    e2.pricers.append(IrsPricer(t))
            e2.csa_schema = None
            e2.run_id = str(_uuid.uuid4())[:8]
            rs = e2.run()
        st.session_state["rs"] = rs

    if "rs" in st.session_state:
        rs = st.session_state["rs"]
        metrics = ["CVA", "DVA", "FVA", "MVA", "KVA"]
        bv = [r[m]  for m in metrics]
        sv = [rs[m] for m in metrics]

        fig = go.Figure()
        fig.add_trace(go.Bar(name="Base",   x=metrics, y=bv, marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Bumped", x=metrics, y=sv, marker_color="#EF553B"))
        fig.update_layout(barmode="group", height=320, yaxis_title=f"USD ({sel_ccy6})",
                          margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

        sens_df = pd.DataFrame({
            "Metric":    metrics,
            "Currency":  [sel_ccy6] * len(metrics),
            "Base":      bv,
            "Bumped":    sv,
            "Delta_USD": [s - b for s, b in zip(sv, bv)],
            "Delta_pct": [(s - b) / (abs(b) + 1e-12) * 100 for s, b in zip(sv, bv)],
        })
        st.dataframe(
            sens_df.style.format(
                {"Base": "{:,.4f}", "Bumped": "{:,.4f}",
                 "Delta_USD": "{:+,.4f}", "Delta_pct": "{:+.2f}%"}),
            use_container_width=True)
        st.download_button("Download Sensitivity CSV", _df_to_csv(sens_df),
                           "sensitivity.csv", "text/csv")
