"""
PathSnap — Streamlit Frontend (Laptop 2)
ui/app.py

Run with:
    streamlit run app.py

Set the backend IP via environment variable:
    $env:LAPTOP1_IP = "192.168.x.x"   (Windows PowerShell)
    export LAPTOP1_IP="192.168.x.x"   (Linux/Mac)
"""

import os
import requests
import streamlit as st
import folium
from streamlit_folium import st_folium

# ─────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────

LAPTOP1_IP = os.environ.get("LAPTOP1_IP", "localhost")
API_BASE = f"http://{LAPTOP1_IP}:8000"

st.set_page_config(
    page_title="PathSnap — Map Matching Evaluator",
    page_icon="🗺️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Sidebar — connection & model settings
# ─────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown(f"**Backend:** `{API_BASE}`")

    # Health check
    try:
        resp = requests.get(f"{API_BASE}/", timeout=3)
        if resp.status_code == 200:
            st.success("✅ Backend connected")
            loaded_models = resp.json().get("models_loaded", [])
        else:
            st.error("❌ Backend unreachable")
            loaded_models = []
    except Exception:
        st.error(f"❌ Cannot reach `{API_BASE}`\n\nMake sure Laptop 1 is running:\n`uvicorn main:app --host 0.0.0.0 --port 8000`")
        loaded_models = []
        st.stop()

    # Fetch trips
    try:
        trips_resp = requests.get(f"{API_BASE}/trips", timeout=5).json()
        trip_ids = trips_resp.get("trips", [])
    except Exception:
        trip_ids = []

    selected_trip = st.selectbox("🚗 Select Evaluator Trip", trip_ids)
    selected_model = st.selectbox("🤖 Select Model", loaded_models if loaded_models else ["(none)"])
    use_hmm = st.toggle("Enable HMM / Viterbi Decoding", value=True)
    run_btn = st.button("▶️ Run Prediction", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────

st.title("🗺️ PathSnap — Map-Matching Evaluator")
st.caption(f"Trip: **{selected_trip}** | Model: **{selected_model}** | HMM: **{'ON' if use_hmm else 'OFF'}**")

col_map, col_metrics = st.columns([3, 1])

if run_btn and selected_trip:
    with st.spinner(f"Running prediction for **{selected_trip}**..."):
        # --- Get prediction ---
        try:
            pred_resp = requests.post(
                f"{API_BASE}/predict/{selected_trip}",
                params={"model_name": selected_model, "use_hmm": use_hmm},
                timeout=120
            ).json()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        # --- Get metrics ---
        try:
            metrics_resp = requests.get(
                f"{API_BASE}/metrics/{selected_trip}",
                params={"model_name": selected_model, "use_hmm": use_hmm},
                timeout=120
            ).json()
        except Exception as e:
            metrics_resp = {}
            st.warning(f"Metrics unavailable: {e}")

    # ─── Map ───
    with col_map:
        gps_coords = pred_resp.get("gps_track_geojson", {}).get("geometry", {}).get("coordinates", [])
        pred_coords = pred_resp.get("predicted_route_geojson", {}).get("geometry", {}).get("coordinates", [])

        if gps_coords:
            center_lat = sum(c[1] for c in gps_coords) / len(gps_coords)
            center_lon = sum(c[0] for c in gps_coords) / len(gps_coords)
        else:
            center_lat, center_lon = 12.9716, 77.5946  # Default: Bangalore

        m = folium.Map(location=[center_lat, center_lon], zoom_start=15,
                       tiles="CartoDB positron")

        # GPS track (blue)
        if gps_coords:
            folium.PolyLine(
                locations=[[c[1], c[0]] for c in gps_coords],
                color="blue", weight=2, opacity=0.6,
                tooltip="GPS Track"
            ).add_to(m)
            # Start/end markers
            folium.Marker([gps_coords[0][1], gps_coords[0][0]],
                          popup="Start", icon=folium.Icon(color="green")).add_to(m)
            folium.Marker([gps_coords[-1][1], gps_coords[-1][0]],
                          popup="End", icon=folium.Icon(color="red")).add_to(m)

        # Predicted route (red)
        if pred_coords:
            folium.PolyLine(
                locations=[[c[1], c[0]] for c in pred_coords],
                color="red", weight=4, opacity=0.85,
                tooltip="Predicted Route (HMM)" if use_hmm else "Predicted Route"
            ).add_to(m)

        st_folium(m, width=800, height=550, key="main_map")

        st.caption("🔵 Blue = Raw GPS Track  |  🔴 Red = Predicted Road Route")

    # ─── Metrics panel ───
    with col_metrics:
        st.subheader("📊 Metrics")

        if metrics_resp:
            st.metric("Top-1 Accuracy", f"{metrics_resp.get('top1_accuracy', 0)*100:.1f}%")
            st.metric("Macro F1", f"{metrics_resp.get('macro_f1', 0):.4f}")
            st.metric("Route Continuity", f"{metrics_resp.get('route_continuity', 0)*100:.1f}%")
            st.metric("Inference Time", f"{metrics_resp.get('inference_time_sec', 0):.2f}s")
            st.metric("GPS Fixes", metrics_resp.get("n_fixes", "—"))
        else:
            st.info("Metrics not available.")

        st.subheader("🧠 Model Info")
        st.write(f"**Model:** {pred_resp.get('model_used', '—')}")
        st.write(f"**HMM:** {'Enabled' if use_hmm else 'Disabled'}")
        st.write(f"**Trip:** {pred_resp.get('trip_id', '—')}")

else:
    with col_map:
        st.info("👈 Select a trip and model from the sidebar, then click **Run Prediction**.")
        # Empty map
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=12,
                       tiles="CartoDB positron")
        st_folium(m, width=800, height=550, key="empty_map")

    with col_metrics:
        st.subheader("📊 Metrics")
        st.info("Run a prediction to see metrics here.")
