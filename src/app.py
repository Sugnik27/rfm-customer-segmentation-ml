"""
Streamlit App for E-commerce Customer Segmentation
- Single customer prediction via RFM manual input
- Batch prediction via CSV upload
- Dark professional theme
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st
from src.deployment import predict_single, predict_batch, SEGMENT_ADVICE


# PAGE CONFIG


st.set_page_config(
    page_title = "Customer Segmentation",
    page_icon  = "🛒",
    layout     = "wide"
)


# CUSTOM CSS


st.markdown("""
<style>
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    .metric-card {
        background-color: #1E2130;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        border-left: 4px solid;
    }
    .segment-card {
        background-color: #1E2130;
        border-radius: 12px;
        padding: 25px;
        margin: 15px 0;
        text-align: center;
    }
    .tip-card {
        background-color: #262B3D;
        border-radius: 8px;
        padding: 12px 18px;
        margin: 8px 0;
        border-left: 3px solid #3498DB;
    }
    .main-header {
        text-align: center;
        padding: 20px 0;
        border-bottom: 1px solid #2E3347;
        margin-bottom: 30px;
    }
    .stButton > button {
        background-color: #3498DB;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 30px;
        font-size: 16px;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #2980B9;
    }
    .stTabs [data-baseweb="tab"] {
        color: #AAAAAA;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        color: #3498DB !important;
        border-bottom: 2px solid #3498DB !important;
    }
</style>
""", unsafe_allow_html=True)


# HEADER


st.markdown("""
<div class="main-header">
    <h1>🛒 E-commerce Customer Segmentation</h1>
    <p style="color:#AAAAAA; font-size:16px">
        Predict customer segments using RFM Analysis and Machine Learning
    </p>
</div>
""", unsafe_allow_html=True)


# SIDEBAR


with st.sidebar:
    st.markdown("## 📖 How It Works")
    st.markdown("""
    This app uses **RFM Analysis** combined with
    **Machine Learning** to segment customers.

    ---
    **RFM stands for:**
    - 🕐 **Recency** — Days since last purchase
    - 🔁 **Frequency** — Number of orders placed
    - 💰 **Monetary** — Total amount spent

    ---
    **Customer Segments:**
    """)

    for segment, info in SEGMENT_ADVICE.items():
        st.markdown(f"{info['emoji']} **{segment}**")

    st.markdown("---")
    st.markdown("**Model:** Logistic Regression")
    st.markdown("**Accuracy:** 99.31%")
    st.markdown("**F1 Score:** 99.31%")
    st.markdown("**Dataset:** UCI Online Retail")
    st.markdown("**Customers:** 4,338")

# -----------------------------------------------------------------------
# TABS
# -----------------------------------------------------------------------

tab1, tab2 = st.tabs(["🔍 Single Customer Prediction", "📊 Bulk Prediction"])


# TAB 1 — SINGLE PREDICTION


with tab1:
    st.markdown("### Enter Customer RFM Values")
    st.markdown(
        "Input the customer's purchasing behavior to predict their segment."
    )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🕐 Recency")
        st.markdown(
            "<p style='color:#AAAAAA;font-size:13px'>Days since last purchase</p>",
            unsafe_allow_html=True
        )
        recency = st.number_input(
            "Recency",
            min_value=0,
            max_value=365,
            value=30,
            step=1,
            label_visibility="collapsed"
        )
        st.markdown(
            f"<div class='metric-card' style='border-color:#3498DB'>"
            f"<h2 style='color:#3498DB;margin:0'>{recency}</h2>"
            f"<p style='color:#AAAAAA;margin:0'>Days</p></div>",
            unsafe_allow_html=True
        )

    with col2:
        st.markdown("#### 🔁 Frequency")
        st.markdown(
            "<p style='color:#AAAAAA;font-size:13px'>Number of orders placed</p>",
            unsafe_allow_html=True
        )
        frequency = st.number_input(
            "Frequency",
            min_value=1,
            max_value=500,
            value=5,
            step=1,
            label_visibility="collapsed"
        )
        st.markdown(
            f"<div class='metric-card' style='border-color:#2ECC71'>"
            f"<h2 style='color:#2ECC71;margin:0'>{frequency}</h2>"
            f"<p style='color:#AAAAAA;margin:0'>Orders</p></div>",
            unsafe_allow_html=True
        )

    with col3:
        st.markdown("#### 💰 Monetary")
        st.markdown(
            "<p style='color:#AAAAAA;font-size:13px'>Total amount spent</p>",
            unsafe_allow_html=True
        )
        monetary = st.number_input(
            "Monetary",
            min_value=0.0,
            max_value=300000.0,
            value=1000.0,
            step=100.0,
            label_visibility="collapsed"
        )
        st.markdown(
            f"<div class='metric-card' style='border-color:#F39C12'>"
            f"<h2 style='color:#F39C12;margin:0'>{monetary:,.0f}</h2>"
            f"<p style='color:#AAAAAA;margin:0'>Total Spend</p></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Segment", key="single_predict")

    if predict_btn:
        try:
            with st.spinner("Predicting..."):
                result = predict_single(recency, frequency, monetary)

            segment    = result["segment"]
            confidence = result["confidence"]
            advice     = result["advice"]

            st.markdown("---")
            st.markdown("### 🎯 Prediction Result")

            # Segment Card
            st.markdown(f"""
            <div class='segment-card' style='border:2px solid {advice["color"]}'>
                <h1 style='color:{advice["color"]};margin:0'>
                    {advice["emoji"]} {segment}
                </h1>
                <p style='color:#AAAAAA;margin:10px 0'>
                    {advice["description"]}
                </p>
                <h3 style='color:{advice["color"]};margin:0'>
                    Confidence: {confidence}%
                </h3>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")

            # RFM Summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Recency",   f"{recency} days")
            with col2:
                st.metric("Frequency", f"{frequency} orders")
            with col3:
                st.metric("Monetary",  f"{monetary:,.2f}")

            st.markdown("---")

            # Business Tips
            st.markdown("### 💡 Business Recommendations")
            for tip in advice["tips"]:
                st.markdown(f"""
                <div class='tip-card'>✅ {tip}</div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")


# TAB 2 — BULK PREDICTION


with tab2:
    st.markdown("### Upload Customer Data for Bulk Prediction")
    st.markdown(
        "Upload a CSV with **Recency**, **Frequency**, and **Monetary** columns."
    )
    st.markdown("---")

    # Sample template download
    sample_df = pd.DataFrame({
        "CustomerID": ["C001", "C002", "C003"],
        "Recency"   : [10, 120, 250],
        "Frequency" : [15, 3, 1],
        "Monetary"  : [5000, 800, 150]
    })

    st.markdown("#### 📥 Download Sample CSV Template")
    csv_template = sample_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label     = "Download Template",
        data      = csv_template,
        file_name = "rfm_template.csv",
        mime      = "text/csv"
    )

    st.markdown("---")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file)

        st.markdown("#### 👀 Preview of Uploaded Data")
        st.dataframe(df_upload.head(), use_container_width=True)
        st.markdown(f"**Total Rows:** {len(df_upload):,}")

        if st.button("📊 Run Bulk Prediction", key="bulk_predict"):
            try:
                with st.spinner("Running predictions..."):
                    result_df = predict_batch(df_upload)

                st.markdown("---")
                st.markdown("### 📊 Prediction Results")

                # Segment Distribution Cards
                seg_counts = result_df["Segment"].value_counts()
                col1, col2, col3, col4 = st.columns(4)
                cols     = [col1, col2, col3, col4]
                segments = [
                    ("Champions",         "#2ECC71"),
                    ("Loyal Customers",   "#3498DB"),
                    ("At-Risk Customers", "#F39C12"),
                    ("Lost Customers",    "#E74C3C")
                ]

                for (seg, color), col in zip(segments, cols):
                    count = seg_counts.get(seg, 0)
                    with col:
                        st.markdown(
                            f"<div class='metric-card' style='border-color:{color};text-align:center'>"
                            f"<h2 style='color:{color};margin:0'>{count}</h2>"
                            f"<p style='color:#AAAAAA;margin:0;font-size:12px'>{seg}</p>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

                st.markdown("---")
                st.markdown("#### 📋 Full Results Table")
                st.dataframe(result_df, use_container_width=True)

                # Download results
                csv_result = result_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label     = "📥 Download Predictions",
                    data      = csv_result,
                    file_name = "customer_segments.csv",
                    mime      = "text/csv"
                )

            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")


# FOOTER


st.markdown("---")
st.markdown("""
<div style='text-align:center;color:#AAAAAA;padding:20px'>
    <p>E-commerce Customer Segmentation | Boston Institute of Analytics | Sugnik Mondal</p>
</div>
""", unsafe_allow_html=True)