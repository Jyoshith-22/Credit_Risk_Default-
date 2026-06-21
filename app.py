# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(
    page_title="Credit Risk Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Single, clean CSS block
st.markdown("""
<style>
    /* ── Global ── */
    body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

    /* ── Title block ── */
    .page-title {
        font-size: 28px;
        font-weight: 700;
        color: #111827;
        margin-bottom: 2px;
    }
    .page-sub {
        font-size: 14px;
        color: #6B7280;
        margin-bottom: 20px;
    }

    /* ── Metric cards ── */
    .kpi-card {
        background: #ffffff;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 16px 20px;
        margin-bottom: 12px;
    }
    .kpi-label {
        font-size: 11px;
        font-weight: 600;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 26px;
        font-weight: 700;
        color: #111827;
    }

    /* ── Decision cards ── */
    .card-approve {
        background: #F0FDF4;
        border-left: 5px solid #22C55E;
        border-radius: 8px;
        padding: 16px 20px;
        color: #14532D;
        margin-bottom: 14px;
    }
    .card-review {
        background: #FEFCE8;
        border-left: 5px solid #EAB308;
        border-radius: 8px;
        padding: 16px 20px;
        color: #713F12;
        margin-bottom: 14px;
    }
    .card-decline {
        background: #FFF1F2;
        border-left: 5px solid #F43F5E;
        border-radius: 8px;
        padding: 16px 20px;
        color: #881337;
        margin-bottom: 14px;
    }
    .card-approve h4, .card-review h4, .card-decline h4 {
        margin: 0 0 6px 0;
        font-size: 16px;
    }
    .card-approve p, .card-review p, .card-decline p {
        margin: 0;
        font-size: 13px;
        line-height: 1.5;
    }

    /* ── Section headers ── */
    .section-header {
        font-size: 16px;
        font-weight: 600;
        color: #374151;
        border-bottom: 2px solid #F3F4F6;
        padding-bottom: 6px;
        margin-bottom: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ─────────────────────────────────────────────────────────────
model = None
prep = {
    'cat_cols': [],
    'num_cols': [],
    'encoded_cols_order': [],
    'scaler': None,
}
assets_loaded = False

@st.cache_resource
def load_ml_assets():
    m = joblib.load("best_credit_risk_model.pkl")
    p = joblib.load("preprocessing_artifacts.pkl")
    return m, p

@st.cache_data
def load_dataset():
    if os.path.exists("credit_risk_dataset.csv"):
        return pd.read_csv("credit_risk_dataset.csv")
    return None

try:
    model, prep = load_ml_assets()
    assets_loaded = True
except Exception as exc:
    st.error(f"Could not load ML artifacts: {exc}")

df = load_dataset()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Credit Risk Dashboard")
    st.markdown("Innovexa Catalyst · ML Division")
    st.markdown("---")
    navigation = st.radio(
        "Navigate",
        ["Executive Dashboard", "Model Performance", "Underwriting Calculator"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Dataset: credit_risk_dataset.csv")
    if df is not None:
        st.caption(f"{len(df):,} applications loaded")

# ── Page Header ────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>Credit Risk Analytics & Scoring Portal</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Default prediction · Risk segmentation · Underwriting decisions</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – EXECUTIVE DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if navigation == "Executive Dashboard":

    if df is None:
        st.warning("credit_risk_dataset.csv not found in the project directory.")
        st.stop()

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Total Applications</div><div class='kpi-value'>{len(df):,}</div></div>", unsafe_allow_html=True)
    with k2:
        rate = df["loan_status"].mean() * 100
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Default Rate</div><div class='kpi-value'>{rate:.2f}%</div></div>", unsafe_allow_html=True)
    with k3:
        avg_inc = df["person_income"].mean()
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg. Annual Income</div><div class='kpi-value'>${avg_inc:,.0f}</div></div>", unsafe_allow_html=True)
    with k4:
        avg_loan = df["loan_amnt"].mean()
        st.markdown(f"<div class='kpi-card'><div class='kpi-label'>Avg. Loan Amount</div><div class='kpi-value'>${avg_loan:,.0f}</div></div>", unsafe_allow_html=True)

    st.markdown("")

    # Row 2 – charts
    rc1, rc2 = st.columns(2)

    with rc1:
        st.markdown("<div class='section-header'>Default Rate by Credit Score Band</div>", unsafe_allow_html=True)
        if os.path.exists("risk_segmentation.png"):
            st.image("risk_segmentation.png", use_container_width=True)
        else:
            st.info("Run train_models.py to generate risk_segmentation.png")

    with rc2:
        st.markdown("<div class='section-header'>Income Distribution by Loan Status</div>", unsafe_allow_html=True)
        df_plot = df[df["person_income"] <= 150_000]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        colors = ["#6366F1", "#F43F5E"]
        sns.boxplot(data=df_plot, x="loan_status", y="person_income",
                    palette=colors, ax=ax)
        ax.set_xticklabels(["Non-Default", "Default"])
        ax.set_xlabel("Loan Status", fontsize=11)
        ax.set_ylabel("Annual Income ($)", fontsize=11)
        ax.set_title("Annual Income vs Default Status", fontsize=12, fontweight='bold')
        ax.spines[['top', 'right']].set_visible(False)
        fig.patch.set_facecolor('white')
        st.pyplot(fig)
        plt.close(fig)

    # Row 3 – charts
    rc3, rc4 = st.columns(2)

    with rc3:
        st.markdown("<div class='section-header'>Applications by Loan Intent</div>", unsafe_allow_html=True)
        fig2, ax2 = plt.subplots(figsize=(8, 4.5))
        sns.countplot(data=df, y="loan_intent", hue="loan_status",
                      palette=["#6366F1", "#F43F5E"], ax=ax2)
        ax2.set_xlabel("Number of Applications", fontsize=11)
        ax2.set_ylabel("Loan Purpose", fontsize=11)
        ax2.set_title("Defaults by Loan Intent", fontsize=12, fontweight='bold')
        ax2.spines[['top', 'right']].set_visible(False)
        ax2.legend(title="Default", labels=["No", "Yes"])
        fig2.patch.set_facecolor('white')
        st.pyplot(fig2)
        plt.close(fig2)

    with rc4:
        st.markdown("<div class='section-header'>Default Rate by Loan Grade</div>", unsafe_allow_html=True)
        grade_def = df.groupby("loan_grade")["loan_status"].mean().reset_index()
        grade_def["default_pct"] = grade_def["loan_status"] * 100
        fig3, ax3 = plt.subplots(figsize=(8, 4.5))
        bars = ax3.bar(grade_def["loan_grade"], grade_def["default_pct"],
                       color=["#22C55E","#84CC16","#EAB308","#F97316","#EF4444","#DC2626","#991B1B"])
        ax3.set_xlabel("Loan Grade (A-G)", fontsize=11)
        ax3.set_ylabel("Default Rate (%)", fontsize=11)
        ax3.set_title("Default Rate by Loan Grade", fontsize=12, fontweight='bold')
        ax3.spines[['top', 'right']].set_visible(False)
        for bar in bars:
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=10)
        fig3.patch.set_facecolor('white')
        st.pyplot(fig3)
        plt.close(fig3)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "Model Performance":
    st.markdown("<div class='section-header'>Model Evaluation & Comparison</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.2])

    with col1:
        st.markdown("**Performance Metrics**")
        if os.path.exists("model_evaluation_metrics.csv"):
            metrics_df = pd.read_csv("model_evaluation_metrics.csv")
            st.dataframe(
                metrics_df.style.highlight_max(
                    subset=["ROC-AUC", "PR-AUC", "F1-Score", "KS-Statistic"],
                    color="#D1FAE5"
                ),
                use_container_width=True
            )
            st.markdown("""
> **LightGBM (Tuned)** is the best model: ROC-AUC **0.9515**, PR-AUC **0.9082**, KS **0.7650**.  
> Ensemble models significantly outperform the Logistic Regression baseline.
            """)
        else:
            st.info("model_evaluation_metrics.csv not found. Run train_models.py first.")

        st.markdown("")
        st.markdown("**Top 15 Feature Importances**")
        if os.path.exists("feature_importance.png"):
            st.image("feature_importance.png", use_container_width=True)
        else:
            st.info("feature_importance.png not found.")

    with col2:
        st.markdown("**ROC Curves**")
        if os.path.exists("model_roc_curves.png"):
            st.image("model_roc_curves.png", use_container_width=True)
        else:
            st.info("model_roc_curves.png not found.")

        st.markdown("**SHAP Summary Plot**")
        if os.path.exists("shap_summary.png"):
            st.image("shap_summary.png", use_container_width=True)
        else:
            st.info("shap_summary.png not found.")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – UNDERWRITING CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif navigation == "Underwriting Calculator":
    st.markdown("<div class='section-header'>Real-time Credit Risk Underwriter</div>", unsafe_allow_html=True)
    st.markdown("Enter applicant details to calculate default probability and receive an underwriting recommendation.")

    if not assets_loaded:
        st.error("ML artifacts not found. Run train_models.py to generate the model files.")
        st.stop()

    with st.form("underwriter_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Applicant Demographics**")
            age            = st.slider("Age", 18, 85, 28)
            income         = st.number_input("Annual Income ($)", 2_000, 500_000, 55_000, step=1_000)
            emp_length     = st.slider("Employment Length (yrs)", 0, 40, 4)
            home_ownership = st.selectbox("Home Ownership", ["RENT", "MORTGAGE", "OWN", "OTHER"])

        with c2:
            st.markdown("**Loan Characteristics**")
            loan_amount    = st.number_input("Loan Amount ($)", 500, 50_000, 10_000, step=500)
            loan_intent    = st.selectbox("Loan Purpose",
                                ["PERSONAL", "EDUCATION", "MEDICAL",
                                 "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
            loan_grade     = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
            loan_int_rate  = st.slider("Interest Rate (%)", 4.0, 25.0, 11.5, step=0.1)

        with c3:
            st.markdown("**Credit History**")
            hist_default     = st.radio("Prior Default on File?", ["N", "Y"])
            cred_hist_length = st.slider("Credit History Length (yrs)", 1, 30, 5)

        submitted = st.form_submit_button("Evaluate Credit Risk", use_container_width=True)

    if submitted:
        # Build raw input row
        input_df = pd.DataFrame([{
            "person_age":                age,
            "person_income":             income,
            "person_home_ownership":     home_ownership,
            "person_emp_length":         float(emp_length),
            "loan_intent":               loan_intent,
            "loan_grade":                loan_grade,
            "loan_amnt":                 loan_amount,
            "loan_int_rate":             loan_int_rate,
            "loan_percent_income":       loan_amount / income,
            "cb_person_default_on_file": hist_default,
            "cb_person_cred_hist_length": cred_hist_length,
        }])

        # Feature engineering
        input_df["interest_to_income_ratio"] = (
            input_df["loan_amnt"] * input_df["loan_int_rate"] / 100
        ) / input_df["person_income"]
        input_df["emp_length_to_age_ratio"]  = input_df["person_emp_length"] / input_df["person_age"]
        input_df["cred_hist_to_age_ratio"]   = input_df["cb_person_cred_hist_length"] / input_df["person_age"]

        # Categorical alignment (prevents missing dummy columns on single-row input)
        cat_map = {
            "person_home_ownership":     ["RENT", "MORTGAGE", "OWN", "OTHER"],
            "loan_intent":               ["PERSONAL", "EDUCATION", "MEDICAL",
                                          "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
            "loan_grade":                ["A", "B", "C", "D", "E", "F", "G"],
            "cb_person_default_on_file": ["N", "Y"],
        }
        for col, cats in cat_map.items():
            input_df[col] = pd.Categorical(input_df[col], categories=cats)

        cat_cols      = prep["cat_cols"]
        num_cols      = prep["num_cols"]
        encoded_order = prep["encoded_cols_order"]

        input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        for col in encoded_order:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[encoded_order]

        dummy_cols = [c for c in encoded_order if any(cat in c for cat in cat_cols)]
        input_encoded[dummy_cols] = input_encoded[dummy_cols].astype(int)

        scaler = prep["scaler"]
        input_scaled = input_encoded.copy()
        input_scaled[num_cols] = scaler.transform(input_encoded[num_cols])

        # Predict
        default_prob = float(model.predict_proba(input_scaled)[0, 1])
        credit_score = int(850 - (default_prob * 550))

        # Decision logic
        if credit_score >= 800:
            band, decision, css = "Excellent (800-850)", "APPROVE", "card-approve"
            detail = "Very low risk. Recommend immediate approval with prime rate pricing."
        elif credit_score >= 740:
            band, decision, css = "Very Good (740-799)", "APPROVE", "card-approve"
            detail = "Strong credit profile. Recommend approval under standard terms."
        elif credit_score >= 670:
            band, decision, css = "Good (670-739)", "MANUAL REVIEW", "card-review"
            detail = "Moderate risk. Manual verification of income and employment recommended."
        elif credit_score >= 580:
            band, decision, css = "Fair (580-669)", "MANUAL REVIEW", "card-review"
            detail = "Elevated risk. Consider requiring a co-signer or adjusting interest rate."
        else:
            band, decision, css = "Poor (300-579)", "DECLINE", "card-decline"
            detail = "High default probability. Recommend automatic rejection."

        # Display results
        st.markdown("---")
        st.markdown("#### Assessment Results")

        res1, res2 = st.columns([1, 1.5])

        with res1:
            st.markdown(
                f"<div class='{css}'>"
                f"<h4>Decision: {decision}</h4>"
                f"<p>{detail}</p>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.markdown(f"**Default Probability:** `{default_prob * 100:.2f}%`")
            st.progress(default_prob)

        with res2:
            m1, m2 = st.columns(2)
            m1.metric("Credit Score", credit_score)
            m2.metric("Risk Band", band)

            if df is not None:
                st.markdown("**Applicant vs Portfolio Average**")
                summary = pd.DataFrame({
                    "Metric": ["Annual Income", "Loan Amount", "Loan % of Income", "Interest Rate"],
                    "Applicant": [
                        f"${income:,.0f}",
                        f"${loan_amount:,.0f}",
                        f"{loan_amount / income * 100:.1f}%",
                        f"{loan_int_rate:.1f}%",
                    ],
                    "Portfolio Avg": [
                        f"${df['person_income'].mean():,.0f}",
                        f"${df['loan_amnt'].mean():,.0f}",
                        f"{df['loan_percent_income'].mean() * 100:.1f}%",
                        f"{df['loan_int_rate'].mean():.1f}%",
                    ],
                })
                st.table(summary)
