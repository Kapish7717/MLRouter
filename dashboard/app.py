# dashboard/app.py
import streamlit as st
import requests, time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

API = "http://localhost:8000"
EXP = "exp_001"

st.set_page_config(
    page_title="MLRouter",
    page_icon="🧪", layout="wide"
)

st.title("🧪 MLRouter")
st.caption("Live A/B Testing Dashboard")

# ── Auto-refresh ──────────────────────────────────────────────────
if st.sidebar.button("🔄 Refresh"):
    st.rerun()
auto = st.sidebar.checkbox("Auto-refresh (5s)", value=False)

# ── Fetch data ────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def fetch_metrics():
    try:
        r = requests.get(
            f"{API}/experiments/{EXP}/metrics", timeout=3
        )
        return r.json().get("metrics", {})
    except:
        return {}

@st.cache_data(ttl=5)
def fetch_predictions():
    try:
        r = requests.get(
            f"{API}/experiments/{EXP}/predictions"
            f"?limit=200", timeout=3
        )
        return r.json()
    except:
        return []

metrics = fetch_metrics()
preds   = fetch_predictions()
df      = pd.DataFrame(preds) if preds else pd.DataFrame()

# ── KPI Cards ─────────────────────────────────────────────────────
st.subheader("📊 Live Experiment Metrics")

ma = metrics.get("model_a", {})
mb = metrics.get("model_b", {})

col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric("Model A Requests",
            ma.get("total_requests", 0))
col2.metric("Model A Avg Confidence",
            f"{ma.get('avg_confidence', 0):.3f}")
col3.metric("Model A Latency",
            f"{ma.get('avg_latency_ms', 0):.1f}ms")
col4.metric("Model B Requests",
            mb.get("total_requests", 0))
col5.metric("Model B Avg Confidence",
            f"{mb.get('avg_confidence', 0):.3f}")
col6.metric("Model B Latency",
            f"{mb.get('avg_latency_ms', 0):.1f}ms")

st.divider()

if not df.empty:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Charts ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        # Request distribution
        req_counts = df["variant"].value_counts().reset_index()
        req_counts.columns = ["Variant", "Count"]
        fig = px.pie(
            req_counts, values="Count",
            names="Variant",
            title="Traffic Split — Model A vs B",
            color_discrete_map={"A": "#1F77B4", "B": "#FF7F0E"}
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Confidence comparison
        fig2 = px.box(
            df, x="variant", y="confidence",
            color="variant",
            title="Confidence Score Distribution",
            labels={"variant": "Model", "confidence": "Confidence"},
            color_discrete_map={"A": "#1F77B4", "B": "#FF7F0E"}
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Latency over time ─────────────────────────────────────────
    fig3 = px.line(
        df.sort_values("timestamp"),
        x="timestamp", y="latency_ms",
        color="variant",
        title="Latency Over Time (ms)",
        color_discrete_map={"A": "#1F77B4", "B": "#FF7F0E"}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ── Recent predictions table ──────────────────────────────────
    st.subheader("🕐 Recent Predictions")
    display_df = df[[
        "timestamp", "variant", "model_id",
        "prediction", "confidence", "latency_ms"
    ]].head(20)
    st.dataframe(display_df, use_container_width=True)

# ── Control Panel ─────────────────────────────────────────────────
st.divider()
st.subheader("⚙️ Experiment Controls")

c1, c2 = st.columns(2)

with c1:
    st.markdown("**Adjust Traffic Split**")
    new_split = st.slider(
        "% traffic to Model A", 0, 100, 50
    )
    if st.button("Update Split"):
        requests.patch(
            f"{API}/experiments/{EXP}/split",
            json={"traffic_split": new_split / 100}
        )
        st.success(
            f"Split updated: {new_split}% → A, "
            f"{100-new_split}% → B"
        )
        st.cache_data.clear()

with c2:
    st.markdown("**Promote Winner**")
    winner = st.radio("Select winner", ["A", "B"],
                      horizontal=True)
    if st.button("🏆 Promote & End Experiment",
                 type="primary"):
        requests.post(
            f"{API}/experiments/{EXP}/promote"
            f"?winner={winner}"
        )
        st.success(f"Model {winner} promoted to 100% traffic!")
        st.balloons()

@st.cache_data(ttl=60)
def fetch_auc_metrics():
    try:
        r = requests.get(f"{API}/models/auc", timeout=3)
        return r.json().get("models", [])
    except:
        return []

# ── Training Performance ──────────────────────────────────────────
st.divider()
st.subheader("📈 Model Training Performance")
st.caption("Training-time ROC AUC and Accuracy from currently active model versions.")

auc_metrics = fetch_auc_metrics()
if auc_metrics:
    m1, m2 = st.columns(2)
    
    for idx, m in enumerate(auc_metrics):
        col = m1 if idx % 2 == 0 else m2
        with col:
            st.markdown(f"#### {m['model_id'].replace('_', ' ').title()}")
            mc1, mc2 = st.columns(2)
            mc1.metric("Training ROC AUC", f"{m['roc_auc']:.4f}")
            mc2.metric("Training Accuracy", f"{m['accuracy']:.4f}")
            st.caption(f"Version: {m['version']}")

    st.markdown("---")
    # Display the ROC Curve image from the API
    st.markdown("**ROC AUC Curve Comparison**")
    try:
        # We use a timestamp query param to bypass browser cache when model version changes
        st.image(f"{API}/models/roc_curve?t={int(time.time())}", 
                 caption="Active Models Performance (Champion vs Challenger)",
                 use_container_width=True)
    except:
        st.warning("ROC curve image not found on server.")

# ── Test prediction ───────────────────────────────────────────────
st.divider()
st.subheader("🔬 Test a Prediction")
st.caption("Send a test request and see which model responds")

if st.button("Send Test Request"):
    test_features = {
        "Age": 65, "Gender": 1, "Family History": 1,
        "Prior Fractures": 0, "Calcium Intake": 1,
        "Physical Activity": 0, "Smoking": 0,
        "Alcohol Consumption": 0, "Hormonal Changes": 1,
        "Body Weight": 1, "Vitamin D Intake": 1,
        "Medical Conditions": 0, "Medications": 0,
        "Race/Ethnicity": 1
    }
    resp = requests.post(f"{API}/predict", json={
        "features":      test_features,
        "experiment_id": EXP
    }).json()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prediction",
                "Positive" if resp["prediction"] else "Negative")
    col2.metric("Confidence", f"{resp['confidence']:.3f}")
    col3.metric("Model Used", resp["model_id"])
    col4.metric("Variant", resp["variant"])
    st.caption(f"Latency: {resp['latency_ms']}ms")

# Auto refresh
if auto:
    time.sleep(5)
    st.cache_data.clear()
    st.rerun()