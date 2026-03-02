"""
Helper script — writes stream lit_app.py.
Run once: .venv/Scripts/python.exe scripts/write_ui.py
Then delete this file.
"""
from pathlib import Path

APP = Path(__file__).parent.parent / "src" / "xva_engine" / "ui" / "streamlit_app.py"

CONTENT = '''\
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
from xva_engine.io.loaders import (
    load_portfolio_csv, load_discount_curve_json,
    load_credit_curves_json,
)
from xva_engine.market.curve import DiscountCurve
from xva_engine.market.credit import CreditCurveModel
from xva_engine.products.zcb import ZcbPricer
from xva_engine.products.irs import IrsPricer
from xva_engine.engine import XVAEngine
import uuid as _uuid

# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio Inputs")
    portfolio_file = st.file_uploader("portfolio.csv",       type="csv",  key="port")
    netting_file   = st.file_uploader("netting_sets.csv",    type="csv",  key="ns")
    curve_file     = st.file_uploader("discount_curve.json", type="json", key="crv")
    credit_file    = st.file_uploader("credit_curves.json",  type="json", key="crd")

    st.divider()
    st.header("Model Parameters")
    hw_a      = st.number_input("HW Mean Reversion (a)", min_value=0.0001, value=0.05,  step=0.005, format="%.4f")
    hw_sigma  = st.number_input("HW Volatility (\\u03c3)",    min_value=0.0001, value=0.01,  step=0.001, format="%.4f")
    num_paths = st.slider("MC Paths", 500, 20_000, 5_000, 500)
    seed      = int(st.number_input("Seed", 0, 99999, 42))
    s_f       = st.number_input("Funding Spread s_f",    value=0.010, step=0.001, format="%.4f")
    c_cap     = st.number_input("Cost of Capital c_cap", value=0.080, step=0.005, format="%.4f")

    st.divider()
    run_btn = st.button("Run XVA Engine", type="primary", use_container_width=True)


# -----------------------------------------------------------------------
# Parse inputs
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# Run
# -----------------------------------------------------------------------
if run_btn:
    ns_list, curve, cmap, cfg = _parse_inputs()
    bank_cr = cmap.get("BANK", list(cmap.values())[-1])
    all_res = {}
    with st.spinner("Running Monte Carlo simulation\\u2026"):
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
    )
    st.success(f"\\u2705 Done \\u2014 {len(all_res)} netting set(s) processed.")

# -----------------------------------------------------------------------
# Guard
# -----------------------------------------------------------------------
if "all_results" not in st.session_state:
    st.info("Configure inputs in the sidebar and press **\\u25b6 Run XVA Engine** to begin.")
    st.stop()

all_results = st.session_state["all_results"]
curve_snap  = st.session_state["curve"]
cmap        = st.session_state["cmap"]
cfg_obj     = st.session_state["cfg"]

ns_sel = st.selectbox("Netting Set", list(all_results.keys()))
r = all_results[ns_sel]
grid_yrs = r["grid_days"] / 365.0

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Summary", "Market Data", "Simulation",
    "Exposure", "XVA Breakdown", "Sensitivity",
])

# ======================== TAB 1 — SUMMARY ================================
with tab1:
    net = r["CVA"] - r["DVA"] + r["FVA"] + r["MVA"] + r["KVA"]
    cols = st.columns(6)
    for col, label, val, tip in zip(cols,
            ["CVA","DVA","FVA","MVA","KVA","Net XVA"],
            [r["CVA"],r["DVA"],r["FVA"],r["MVA"],r["KVA"],net],
            ["Cpty default cost","Own default benefit","Funding cost",
             "IM funding","Capital charge","CVA-DVA+FVA+MVA+KVA"]):
        col.metric(label, f"{val:,.2f}", help=tip)

    cl, cr = st.columns(2)
    with cl:
        st.subheader("XVA Waterfall")
        fig = go.Figure(go.Waterfall(
            measure=["relative","relative","relative","relative","relative","total"],
            x=["CVA","\\u2212DVA","FVA","MVA","KVA","Net XVA"],
            y=[r["CVA"],-r["DVA"],r["FVA"],r["MVA"],r["KVA"],0],
            increasing={"marker":{"color":"#EF553B"}},
            decreasing={"marker":{"color":"#00CC96"}},
            totals={"marker":{"color":"#636EFA"}},
            connector={"line":{"color":"grey"}},
        ))
        fig.update_layout(height=340, margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("EE / ENE Profile")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["EE"],
            fill="tozeroy", name="EE", fillcolor="rgba(99,110,250,0.25)",
            line=dict(color="rgb(99,110,250)")))
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["ENE"],
            fill="tozeroy", name="ENE", fillcolor="rgba(239,85,59,0.25)",
            line=dict(color="rgb(239,85,59)")))
        fig.update_layout(height=340, xaxis_title="Years", yaxis_title="USD", margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Run ID: `{r[\'run_id\']}` | Grid: {len(r[\'grid_days\'])} pts | Paths: {cfg_obj.mc_config.num_paths:,}")
    summary_df = pd.DataFrame({"Metric":["CVA","DVA","FVA","MVA","KVA","Net XVA"],
                                "Value":[r["CVA"],r["DVA"],r["FVA"],r["MVA"],r["KVA"],net]})
    st.download_button("\\u2b07 Download Summary CSV", _df_to_csv(summary_df),
                       "xva_summary.csv", "text/csv")

# ======================== TAB 2 — MARKET DATA =============================
with tab2:
    cl, cr = st.columns(2)
    with cl:
        st.subheader("Discount Curve")
        t_arr = [p.tenor/365 for p in curve_snap.points]
        d_arr = [p.discount_factor for p in curve_snap.points]
        z_arr = [-np.log(max(d,1e-9))/(t+1e-9)*100 for t,d in zip(t_arr,d_arr)]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t_arr, y=d_arr, name="Discount Factor",
                                  line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=t_arr, y=z_arr, name="Zero Rate (%)",
                                  line=dict(color="orange", dash="dash"), yaxis="y2"))
        fig.update_layout(height=320,
            yaxis=dict(title="Discount Factor"),
            yaxis2=dict(title="Zero Rate (%)", overlaying="y", side="right"),
            margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Log-linear interpolation"):
            st.latex(r"\\ln P(0,t)=\\ln P(0,t_i)+\\frac{t-t_i}{t_{i+1}-t_i}[\\ln P(0,t_{i+1})-\\ln P(0,t_i)]")
        dc_dl = pd.DataFrame({"tenor_days":[p.tenor for p in curve_snap.points],
                               "discount_factor":d_arr,"zero_rate_pct":z_arr})
        st.download_button("\\u2b07 Discount Curve CSV", _df_to_csv(dc_dl), "discount_curve.csv","text/csv")

    with cr:
        st.subheader("Credit / Survival Curves")
        fig = go.Figure()
        for eid, cc in cmap.items():
            fig.add_trace(go.Scatter(
                x=[p.tenor/365 for p in cc.points],
                y=[p.survival_prob for p in cc.points], name=eid))
        fig.update_layout(height=320, xaxis_title="Years",
                           yaxis_title="Survival Probability", margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("Marginal PD formula"):
            st.latex(r"\\Delta PD(t_{i-1},t_i)=S(t_{i-1})-S(t_i)")

# ======================== TAB 3 — SIMULATION ==============================
with tab3:
    st.subheader("Time Grid Structure")
    gd = r["grid_days"]
    fig = go.Figure(go.Scatter(
        x=gd/365, y=[0]*len(gd), mode="markers+text",
        text=[f"{int(d)}d" for d in gd], textposition="top center",
        marker=dict(size=8, color=["green"]+["steelblue"]*(len(gd)-1)),
    ))
    fig.update_layout(height=150, xaxis_title="Years",
                       yaxis=dict(visible=False), showlegend=False,
                       margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hull-White Factor Paths (50 samples)")
    xs = r["x_sample"]
    fig = go.Figure()
    for j in range(xs.shape[1]):
        fig.add_trace(go.Scatter(x=grid_yrs, y=xs[:,j]*100,
            mode="lines", opacity=0.25, line=dict(width=0.8, color="steelblue"),
            showlegend=False))
    fig.add_trace(go.Scatter(x=grid_yrs, y=xs.mean(axis=1)*100,
        line=dict(width=2.5, color="navy"), name="Mean x_t"))
    fig.update_layout(height=320, xaxis_title="Years",
                       yaxis_title="x_t (%)", margin=dict(t=10,b=10))
    st.plotly_chart(fig, use_container_width=True)
    with st.expander("Exact OU step formula"):
        st.latex(r"x_{t+\\Delta t}=x_t e^{-a\\Delta t}+\\sigma\\sqrt{\\frac{1-e^{-2a\\Delta t}}{2a}}\\,Z")

# ======================== TAB 4 — EXPOSURE ================================
with tab4:
    Vs = r["V_ns_sample"]
    cl, cr = st.columns(2)
    with cl:
        st.subheader("MtM Path Fan")
        fig = go.Figure()
        for j in range(Vs.shape[1]):
            fig.add_trace(go.Scatter(x=grid_yrs, y=Vs[:,j],
                mode="lines", opacity=0.2, line=dict(width=0.8, color="teal"),
                showlegend=False))
        fig.add_trace(go.Scatter(x=grid_yrs, y=Vs.mean(axis=1),
            line=dict(width=2.5, color="darkgreen"), name="Mean V(t)"))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(height=320, xaxis_title="Years",
                           yaxis_title="V_NS (USD)", margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

    with cr:
        st.subheader("EE / ENE Decomposition")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["EE"],
            fill="tozeroy", name="EE",
            fillcolor="rgba(0,204,150,0.3)", line=dict(color="rgb(0,204,150)")))
        fig.add_trace(go.Scatter(x=grid_yrs, y=r["ENE"],
            fill="tozeroy", name="ENE",
            fillcolor="rgba(239,85,59,0.3)", line=dict(color="rgb(239,85,59)")))
        fig.add_hline(y=0, line_dash="dash", line_color="grey")
        fig.update_layout(height=320, xaxis_title="Years",
                           yaxis_title="USD", margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)
        with st.expander("EE / ENE formulas"):
            st.latex(r"EE(t_i)=\\mathbb{E}[\\max(V_{NS}(t_i)-C(t_i),0)]")
            st.latex(r"ENE(t_i)=\\mathbb{E}[\\max(C(t_i)-V_{NS}(t_i),0)]")

    st.subheader("V(t) Distribution at Selected Step")
    idx = st.slider("Grid step", 0, len(gd)-1, min(4,len(gd)-1))
    fig = go.Figure(go.Histogram(x=Vs[idx,:], nbinsx=40,
                                  marker_color="steelblue", opacity=0.75))
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=220, xaxis_title="V_NS (USD)",
        title=f"t = {gd[idx]:.0f}d / {grid_yrs[idx]:.2f}yr",
        margin=dict(t=40,b=10))
    st.plotly_chart(fig, use_container_width=True)

    ee_dl = pd.DataFrame({"grid_days":gd,"grid_years":grid_yrs,
                           "EE":r["EE"],"ENE":r["ENE"]})
    st.download_button("\\u2b07 EE/ENE CSV", _df_to_csv(ee_dl),
                       "exposure_profile.csv", "text/csv")

# ======================== TAB 5 — XVA BREAKDOWN ===========================
with tab5:
    META = [
        ("CVA","cva_table","cva_contribution","#EF553B",
         r"CVA=(1-R_c)\\sum_i P(0,t_i)\\cdot EE(t_i)\\cdot\\Delta PD_c(t_{i-1},t_i)"),
        ("DVA","dva_table","dva_contribution","#00CC96",
         r"DVA=(1-R_b)\\sum_i P(0,t_i)\\cdot ENE(t_i)\\cdot\\Delta PD_b(t_{i-1},t_i)"),
        ("FVA","fva_table","fva_contribution","#AB63FA",
         r"FVA=\\sum_i P(0,t_i)\\cdot s_f(t_i)\\cdot EE_{net}(t_i)\\cdot\\Delta t_i"),
        ("MVA","mva_table","mva_contribution","#FFA15A",
         r"MVA=\\sum_i P(0,t_i)\\cdot s_f(t_i)\\cdot\\mathbb{E}[IM(t_i)]\\cdot\\Delta t_i"),
        ("KVA","kva_table","kva_contribution","#19D3F3",
         r"KVA=\\sum_i P(0,t_i)\\cdot c_{cap}(t_i)\\cdot\\mathbb{E}[K(t_i)]\\cdot\\Delta t_i"),
    ]
    for name, tbl_k, col, colour, formula in META:
        tbl = r[tbl_k].to_pandas()
        total = tbl[col].sum()
        with st.expander(f"**{name}** = {total:,.4f} USD", expanded=(name=="CVA")):
            fl, fr = st.columns([1,1])
            with fl:
                fig = go.Figure(go.Bar(
                    x=tbl["t_end_days"]/365, y=tbl[col], marker_color=colour))
                fig.update_layout(height=260, xaxis_title="Years",
                                   yaxis_title="USD", margin=dict(t=10,b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.latex(formula)
            with fr:
                st.dataframe(tbl.style.highlight_max(subset=[col], color="#ffe0b2"),
                             use_container_width=True, height=260)
                st.download_button(f"\\u2b07 {name} CSV", _df_to_csv(tbl),
                                   f"{name.lower()}_buckets.csv","text/csv",key=f"dl{name}")

# ======================== TAB 6 — SENSITIVITY =============================
with tab6:
    st.subheader("What-If Sensitivity Analysis")
    ca, cb, cc_ = st.columns(3)
    hz  = ca.slider("Hazard rate multiplier", 0.5, 3.0, 1.0, 0.1)
    sfb = cb.slider("Funding spread bump (bps)", -50, 200, 0, 10)
    svp = cc_.slider("HW \\u03c3 scaling (%)", 50, 200, 100, 10)

    if st.button("\\u25b6 Run Sensitivity"):
        ns_obj   = next(ns for ns in st.session_state["ns_list"] if ns.netting_set_id==ns_sel)
        bank_cr  = st.session_state["bank_cr"]
        base_cc  = cmap.get(ns_obj.counterparty_id, bank_cr)

        # Bump credit
        bpts = []
        for pt in base_cc.points:
            if pt.tenor == 0:
                bpts.append(CreditPoint(tenor=0, survival_prob=1.0))
            else:
                t_y = pt.tenor/365
                h   = -np.log(max(pt.survival_prob,1e-9))/t_y * hz
                bpts.append(CreditPoint(tenor=pt.tenor, survival_prob=float(np.exp(-h*t_y))))
        bcc = CreditCurve(entity_id=base_cc.entity_id,
                          recovery_rate=base_cc.recovery_rate, points=bpts)

        bcfg = ModelConfig(
            hw_params=HullWhiteParams(mean_reversion=cfg_obj.hw_params.mean_reversion,
                                      volatility=cfg_obj.hw_params.volatility*svp/100),
            mc_config=cfg_obj.mc_config,
        )
        bsf = st.session_state["s_f"] + sfb/10_000

        with st.spinner("Running sensitivity\\u2026"):
            e2 = XVAEngine.__new__(XVAEngine)
            e2.netting_set = ns_obj; e2.config = bcfg
            e2.funding_spread = bsf; e2.cost_of_capital = st.session_state["c_cap"]
            e2.output_dir = "output"
            e2.discount_curve    = DiscountCurve(curve_snap)
            e2.cpty_credit_model = CreditCurveModel(bcc)
            e2.bank_credit_model = CreditCurveModel(bank_cr)
            e2.pricers = []
            for t in ns_obj.trades:
                if t.trade_type=="ZCB": e2.pricers.append(ZcbPricer(t))
                elif t.trade_type=="IRS": e2.pricers.append(IrsPricer(t))
            e2.csa_schema = None; e2.run_id = str(_uuid.uuid4())[:8]
            rs = e2.run()
        st.session_state["rs"] = rs

    if "rs" in st.session_state:
        rs = st.session_state["rs"]
        metrics = ["CVA","DVA","FVA"]
        bv = [r[m] for m in metrics]
        sv = [rs[m] for m in metrics]
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Base",   x=metrics, y=bv, marker_color="#636EFA"))
        fig.add_trace(go.Bar(name="Bumped", x=metrics, y=sv, marker_color="#EF553B"))
        fig.update_layout(barmode="group", height=320, yaxis_title="USD",
                           margin=dict(t=10,b=10))
        st.plotly_chart(fig, use_container_width=True)

        sens_df = pd.DataFrame({
            "Metric": metrics, "Base": bv, "Bumped": sv,
            "Delta_USD": [s-b for s,b in zip(sv,bv)],
            "Delta_pct": [(s-b)/(abs(b)+1e-12)*100 for s,b in zip(sv,bv)],
        })
        st.dataframe(sens_df.style.format(
            {"Base":"{:,.4f}","Bumped":"{:,.4f}",
             "Delta_USD":"{:+,.4f}","Delta_pct":"{:+.2f}%"}),
            use_container_width=True)
        st.download_button("\\u2b07 Sensitivity CSV", _df_to_csv(sens_df),
                           "sensitivity.csv", "text/csv")
'''

APP.write_text(CONTENT, encoding="utf-8")
print(f"Written {len(CONTENT.splitlines())} lines to {APP}")
