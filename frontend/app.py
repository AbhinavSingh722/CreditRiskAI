"""
Streamlit Credit Risk Dashboard
=================================
IBM SkillsBuild / BharatCares / AICTE Internship 2026
Student: Abhinav Singh | ID: IBMUEDA3522
"""

from __future__ import annotations

import io
import os
from typing import Dict, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
API_BASE  = os.getenv("API_BASE",  "http://localhost:8000")
PAGE_TITLE = "Credit Risk AI Dashboard"

RISK_COLORS = {
    "Very Low":  "#27ae60",
    "Low":       "#2ecc71",
    "Medium":    "#f39c12",
    "High":      "#e67e22",
    "Very High": "#e74c3c",
}

# ──────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────
def api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return None


def api_get(endpoint: str) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{endpoint}", timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API error ({endpoint}): {e}")
        return None


def risk_badge(band: str) -> str:
    color = RISK_COLORS.get(band, "#888")
    return f'<span style="background:{color};color:white;padding:4px 12px;border-radius:12px;font-weight:bold">{band}</span>'


SAMPLE_APPLICANTS = [
    {"name": "Safe Borrower",    "AMT_INCOME_TOTAL": 250000, "AMT_CREDIT": 150000,
     "AMT_ANNUITY": 8000, "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -2000,
     "EXT_SOURCE_1": 0.78, "EXT_SOURCE_2": 0.82, "EXT_SOURCE_3": 0.75,
     "BUREAU_LOAN_COUNT": 3, "BUREAU_AVG_DAYS_PAST_DUE": 0, "BUREAU_TOTAL_CREDIT_SUM": 200000},
    {"name": "Thin-File Applicant", "AMT_INCOME_TOTAL": 90000, "AMT_CREDIT": 180000,
     "AMT_ANNUITY": 12000, "DAYS_BIRTH": -9500, "DAYS_EMPLOYED": -300,
     "EXT_SOURCE_1": 0.35, "EXT_SOURCE_2": 0.40, "EXT_SOURCE_3": 0.30,
     "BUREAU_LOAN_COUNT": 0, "BUREAU_AVG_DAYS_PAST_DUE": 0, "BUREAU_TOTAL_CREDIT_SUM": 0},
    {"name": "High-Risk Borrower", "AMT_INCOME_TOTAL": 70000, "AMT_CREDIT": 350000,
     "AMT_ANNUITY": 25000, "DAYS_BIRTH": -12000, "DAYS_EMPLOYED": -100,
     "EXT_SOURCE_1": 0.20, "EXT_SOURCE_2": 0.25, "EXT_SOURCE_3": 0.18,
     "BUREAU_LOAN_COUNT": 8, "BUREAU_AVG_DAYS_PAST_DUE": 45, "BUREAU_TOTAL_CREDIT_SUM": 500000},
]

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("💳 Credit Risk AI Dashboard")
st.caption("IBM SkillsBuild | Abhinav Singh (IBMUEDA3522) | AI-Powered Credit Risk Assessment")

# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    st.text(f"API: {API_BASE}")
    health = api_get("/health")
    if health:
        st.success(f"✅ API Online — {health.get('model','?')}")
    else:
        st.error("❌ API Offline")

    st.divider()
    st.markdown("**SDG Alignment**")
    st.markdown("🎯 SDG 1 — No Poverty")
    st.markdown("🎯 SDG 8 — Decent Work")
    st.markdown("🎯 SDG 10 — Reduced Inequalities")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab_predict, tab_whatif, tab_batch, tab_overview = st.tabs(
    ["🎯 Single Predict", "🔧 What-If Analysis", "📊 Batch Scoring", "📈 Overview"]
)

# ────────────────────────────────────────────────────────────
# TAB 1: Single Applicant Prediction
# ────────────────────────────────────────────────────────────
with tab_predict:
    st.subheader("Single Applicant Credit Risk Prediction")

    col_mode, _ = st.columns([1, 2])
    with col_mode:
        mode = st.radio("Input Mode", ["Manual Entry", "Sample Applicant"])

    if mode == "Sample Applicant":
        sample_names = [s["name"] for s in SAMPLE_APPLICANTS]
        chosen = st.selectbox("Select Sample", sample_names)
        sample_data = next(s for s in SAMPLE_APPLICANTS if s["name"] == chosen)
    else:
        sample_data = {}

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            amt_income = st.number_input("Annual Income (USD)", min_value=10000,
                                          max_value=10000000, value=int(sample_data.get("AMT_INCOME_TOTAL", 200000)))
            amt_credit = st.number_input("Loan Amount (USD)", min_value=10000,
                                          max_value=5000000, value=int(sample_data.get("AMT_CREDIT", 200000)))
            amt_annuity = st.number_input("Monthly Annuity (USD)", min_value=1000,
                                           max_value=200000, value=int(sample_data.get("AMT_ANNUITY", 12000)))
        with c2:
            age_years = st.slider("Age (years)", 18, 70, value=int(-sample_data.get("DAYS_BIRTH", -14600) / 365))
            emp_years = st.slider("Employment (years)", 0, 40, value=int(-sample_data.get("DAYS_EMPLOYED", -2000) / 365))
            ext1 = st.slider("External Score 1", 0.0, 1.0, value=float(sample_data.get("EXT_SOURCE_1", 0.5)), step=0.01)
        with c3:
            ext2 = st.slider("External Score 2", 0.0, 1.0, value=float(sample_data.get("EXT_SOURCE_2", 0.5)), step=0.01)
            ext3 = st.slider("External Score 3", 0.0, 1.0, value=float(sample_data.get("EXT_SOURCE_3", 0.5)), step=0.01)
            bureau_loans = st.number_input("Prior Bureau Loans", 0, 50, value=int(sample_data.get("BUREAU_LOAN_COUNT", 0)))

        c4, c5 = st.columns(2)
        with c4:
            bureau_dpd = st.number_input("Avg Days Past Due (bureau)", 0, 500,
                                          value=int(sample_data.get("BUREAU_AVG_DAYS_PAST_DUE", 0)))
        with c5:
            bureau_sum = st.number_input("Total Bureau Credit Sum", 0, 5000000,
                                          value=int(sample_data.get("BUREAU_TOTAL_CREDIT_SUM", 0)))

        submitted = st.form_submit_button("🚀 Assess Risk", type="primary", use_container_width=True)

    if submitted:
        payload = {
            "AMT_INCOME_TOTAL":          float(amt_income),
            "AMT_CREDIT":                float(amt_credit),
            "AMT_ANNUITY":               float(amt_annuity),
            "DAYS_BIRTH":                float(-age_years * 365),
            "DAYS_EMPLOYED":             float(-emp_years * 365) if emp_years > 0 else 365243,
            "EXT_SOURCE_1":              float(ext1),
            "EXT_SOURCE_2":              float(ext2),
            "EXT_SOURCE_3":              float(ext3),
            "BUREAU_LOAN_COUNT":         float(bureau_loans),
            "BUREAU_AVG_DAYS_PAST_DUE":  float(bureau_dpd),
            "BUREAU_TOTAL_CREDIT_SUM":   float(bureau_sum),
        }

        with st.spinner("Computing risk..."):
            result = api_post("/predict", payload)

        if result:
            st.divider()
            r1, r2, r3 = st.columns(3)
            prob = result["default_probability"]
            band = result["risk_band"]
            r1.metric("Default Probability", f"{prob:.1%}")
            r2.metric("Risk Band", band)
            r3.metric("Persona", result["persona_name"])

            # Gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Default Risk %"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": RISK_COLORS.get(band, "#888")},
                    "steps": [
                        {"range": [0, 20], "color": "#d4efdf"},
                        {"range": [20, 40], "color": "#a9dfbf"},
                        {"range": [40, 60], "color": "#fdebd0"},
                        {"range": [60, 80], "color": "#fadbd8"},
                        {"range": [80, 100], "color": "#f1948a"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=250, margin={"t": 30, "b": 0, "l": 0, "r": 0})
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown(f"**Risk Band:** {risk_badge(band)}", unsafe_allow_html=True)

            with st.expander("📋 Explanation"):
                st.text(result.get("explanation", "No explanation available."))

            aac = result.get("adverse_action_codes", [])
            if aac:
                st.warning("⚠️ Adverse-Action Reason Codes (Regulatory Notice):")
                for code in aac:
                    st.write(f"• {code}")

# ────────────────────────────────────────────────────────────
# TAB 2: What-If Analysis
# ────────────────────────────────────────────────────────────
with tab_whatif:
    st.subheader("🔧 What-If Scenario Analysis")
    st.info("Adjust key features with sliders and see how the risk score changes in real time.")

    wi_c1, wi_c2 = st.columns(2)
    with wi_c1:
        wi_income  = st.slider("Annual Income",  50000, 500000, 200000, step=5000)
        wi_credit  = st.slider("Loan Amount",    50000, 800000, 200000, step=10000)
        wi_annuity = st.slider("Monthly Annuity", 2000,  50000,  12000, step=500)
    with wi_c2:
        wi_ext2    = st.slider("External Score 2", 0.0, 1.0, 0.5, step=0.01, key="wi_ext2")
        wi_ext3    = st.slider("External Score 3", 0.0, 1.0, 0.5, step=0.01, key="wi_ext3")
        wi_dpd     = st.slider("Avg Days Past Due", 0, 200, 0, key="wi_dpd")

    wi_payload = {
        "AMT_INCOME_TOTAL": float(wi_income),
        "AMT_CREDIT":       float(wi_credit),
        "AMT_ANNUITY":      float(wi_annuity),
        "DAYS_BIRTH":       float(-14600),
        "DAYS_EMPLOYED":    float(-1825),
        "EXT_SOURCE_1":     0.6,
        "EXT_SOURCE_2":     float(wi_ext2),
        "EXT_SOURCE_3":     float(wi_ext3),
        "BUREAU_LOAN_COUNT": 2,
        "BUREAU_AVG_DAYS_PAST_DUE": float(wi_dpd),
        "BUREAU_TOTAL_CREDIT_SUM": float(wi_credit * 1.2),
    }

    if st.button("Update Risk Score", key="wi_update"):
        with st.spinner("Re-scoring..."):
            wi_result = api_post("/predict", wi_payload)
        if wi_result:
            wp = wi_result["default_probability"]
            wb = wi_result["risk_band"]
            st.metric("Updated Default Probability", f"{wp:.1%}")
            st.markdown(f"**Risk Band:** {risk_badge(wb)}", unsafe_allow_html=True)
            st.caption(f"Persona: {wi_result['persona_name']}")

            cir = wi_credit / wi_income
            air = wi_annuity / wi_income
            df_ratios = pd.DataFrame({
                "Metric": ["Credit/Income Ratio", "Annuity/Income Ratio"],
                "Value": [cir, air],
                "Threshold": [3.0, 0.15],
            })
            fig_bar = px.bar(df_ratios, x="Metric", y="Value", color="Metric",
                              title="Key Ratio Analysis")
            fig_bar.add_hline(y=3.0, line_dash="dash", annotation_text="Credit/Income Limit")
            st.plotly_chart(fig_bar, use_container_width=True)

# ────────────────────────────────────────────────────────────
# TAB 3: Batch Scoring
# ────────────────────────────────────────────────────────────
with tab_batch:
    st.subheader("📊 Batch Applicant Scoring")
    st.info("Upload a CSV with applicant features to score multiple applicants at once.")

    st.markdown("**Required columns:** `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `AMT_ANNUITY`, `DAYS_BIRTH`")

    # Sample template download
    template = pd.DataFrame([{
        "SK_ID_CURR": 100001,
        "AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 300000, "AMT_ANNUITY": 15000,
        "DAYS_BIRTH": -14600, "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_1": 0.65, "EXT_SOURCE_2": 0.70, "EXT_SOURCE_3": 0.60,
        "BUREAU_LOAN_COUNT": 2, "BUREAU_AVG_DAYS_PAST_DUE": 0, "BUREAU_TOTAL_CREDIT_SUM": 150000,
    }])
    st.download_button("⬇️ Download CSV Template", template.to_csv(index=False),
                       "template.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        df_up = pd.read_csv(uploaded)
        st.write(f"Uploaded: {len(df_up)} rows")
        st.dataframe(df_up.head(), use_container_width=True)

        if st.button("🚀 Score Batch", type="primary"):
            with st.spinner(f"Scoring {len(df_up)} applicants..."):
                files = {"file": (uploaded.name, df_up.to_csv(index=False).encode(), "text/csv")}
                try:
                    r = requests.post(f"{API_BASE}/predict_batch", files=files, timeout=60)
                    r.raise_for_status()
                    batch_results = pd.DataFrame(r.json())
                    st.success(f"Scored {len(batch_results)} applicants!")
                    st.dataframe(batch_results, use_container_width=True)

                    # Distribution chart
                    fig_dist = px.histogram(batch_results, x="default_probability",
                                             color="risk_band", nbins=30,
                                             title="Score Distribution",
                                             color_discrete_map=RISK_COLORS)
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Download results
                    csv_out = batch_results.to_csv(index=False).encode()
                    st.download_button("⬇️ Download Scored CSV", csv_out,
                                       "scored_applicants.csv", "text/csv")
                except Exception as e:
                    st.error(f"Batch scoring failed: {e}")

# ────────────────────────────────────────────────────────────
# TAB 4: Overview
# ────────────────────────────────────────────────────────────
with tab_overview:
    st.subheader("📈 Project Overview & Insights")

    ov_c1, ov_c2 = st.columns(2)

    with ov_c1:
        st.markdown("### Class Imbalance")
        fig_imb = px.bar(
            x=["Repaid (0)", "Defaulted (1)"],
            y=[282686, 24825],
            color=["Repaid (0)", "Defaulted (1)"],
            color_discrete_sequence=["#3b82d4", "#e74c3c"],
            title="Target Class Distribution (~11:1 Imbalance)",
        )
        fig_imb.update_layout(showlegend=False)
        st.plotly_chart(fig_imb, use_container_width=True)

    with ov_c2:
        st.markdown("### Global Feature Importance")
        feat_imp = pd.DataFrame({
            "Feature": ["EXT_SOURCE_MEAN", "EXT_SOURCE_2", "EXT_SOURCE_3",
                         "CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO",
                         "DAYS_BIRTH", "BUREAU_AVG_DAYS_PAST_DUE",
                         "AMT_CREDIT", "DAYS_EMPLOYED", "EXT_SOURCE_1"],
            "Importance": [0.18, 0.14, 0.12, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.04],
        })
        fig_fi = px.bar(feat_imp.sort_values("Importance"), x="Importance", y="Feature",
                         orientation="h", title="Top Feature Importances",
                         color="Importance", color_continuous_scale="Blues")
        st.plotly_chart(fig_fi, use_container_width=True)

    # Cluster profiles
    st.markdown("### Borrower Persona Profiles")
    cluster_data = api_get("/clusters")
    if cluster_data:
        names = cluster_data.get("cluster_names", {})
        st.json(names)
    else:
        personas = pd.DataFrame({
            "Persona": ["Stable Established Borrower", "Thin-File Young Earner",
                         "High-Leverage Repeat Borrower", "Stable Salaried Professional"],
            "Default Rate": ["2-4%", "6-8%", "13-16%", "3-5%"],
            "Avg Age": ["42", "28", "35", "47"],
            "Avg Loans": ["5", "1", "9", "3"],
        })
        st.dataframe(personas, use_container_width=True)

    # Business impact
    st.markdown("### Business Impact Estimate")
    bi_df = pd.DataFrame({
        "Strategy": ["Approve All (Naive)", "Model at Optimal Threshold"],
        "Default Losses Avoided": [0, 1_200_000_000],
        "Net P&L (USD)": [-800_000_000, 600_000_000],
    })
    fig_bi = px.bar(bi_df, x="Strategy", y="Net P&L (USD)", color="Strategy",
                     title="Estimated Business Impact vs. Naive Baseline",
                     color_discrete_sequence=["#e74c3c", "#27ae60"])
    st.plotly_chart(fig_bi, use_container_width=True)
    st.caption("*Illustrative — based on avg loan USD 200K, 40% LGD, 2% NIM assumptions.*")

    # Model info
    st.markdown("### Model Information")
    model_info = api_get("/model_info")
    if model_info:
        st.json(model_info)
    else:
        st.info("API offline — model info unavailable.")

# Footer
st.divider()
st.caption("AI-Powered Credit Risk Assessment | IBM SkillsBuild Internship 2026 | Abhinav Singh (IBMUEDA3522)")
