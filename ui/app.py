"""
PathSnap — Streamlit Frontend (Laptop 2)
ui/app.py

Run with:
    python -m streamlit run ui/app.py

Set the backend IP via environment variable:
    $env:LAPTOP1_IP = "192.168.x.x"   (Windows PowerShell)
    export LAPTOP1_IP="192.168.x.x"   (Linux/Mac)
"""

import os
import requests
import streamlit as st
import streamlit.components.v1 as components
import folium

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
            info = resp.json()
            st.success(f"✅ Backend connected (`{info.get('active_split', 'eval_test')}`)")
            loaded_models = info.get("models_loaded", [])
        else:
            st.error("❌ Backend unreachable")
            loaded_models = []
    except Exception:
        st.error(
            f"❌ Cannot reach `{API_BASE}`\n\n"
            "Make sure Laptop 1 is running:\n"
            "`python -m uvicorn main:app --host 0.0.0.0 --port 8000`"
        )
        st.stop()

    # Put lgbm_tuned first by default if available
    if "lgbm_tuned" in loaded_models:
        loaded_models = ["lgbm_tuned"] + [m for m in loaded_models if m != "lgbm_tuned"]

    # Fetch trips
    try:
        trips_resp = requests.get(f"{API_BASE}/trips", timeout=5).json()
        trip_ids = trips_resp.get("trips", [])
    except Exception:
        trip_ids = []

    selected_trip = st.selectbox("🚗 Select Trip", trip_ids)
    selected_model = st.selectbox("🤖 Select Model", loaded_models if loaded_models else ["(none)"])
    use_hmm = st.toggle("Enable HMM / Viterbi Decoding", value=True)
    run_btn = st.button("▶️ Run Prediction", type="primary", use_container_width=True)

# ─────────────────────────────────────────────
#  Fetch prediction on button click (persist in session_state)
# ─────────────────────────────────────────────

if run_btn and selected_trip and loaded_models:
    with st.spinner(f"Running **{selected_model}** on trip **{selected_trip}**..."):
        try:
            r_pred = requests.post(
                f"{API_BASE}/predict/{selected_trip}",
                params={"model_name": selected_model, "use_hmm": use_hmm},
                timeout=120,
            )
            r_pred.raise_for_status()
            st.session_state["pred_resp"] = r_pred.json()
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.stop()

        try:
            r_met = requests.get(
                f"{API_BASE}/metrics/{selected_trip}",
                params={"model_name": selected_model, "use_hmm": use_hmm},
                timeout=120,
            )
            r_met.raise_for_status()
            st.session_state["metrics_resp"] = r_met.json()
        except Exception as e:
            st.session_state["metrics_resp"] = {}
            st.warning(f"Metrics unavailable: {e}")

# ─────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────

st.title("🗺️ PathSnap — Map-Matching Evaluator")

pred_resp = st.session_state.get("pred_resp")
metrics_resp = st.session_state.get("metrics_resp", {})

if pred_resp:
    st.caption(
        f"Trip: **{pred_resp.get('trip_id')}** | "
        f"Model: **{pred_resp.get('model_used')}** | "
        f"HMM: **{'ON' if pred_resp.get('use_hmm') else 'OFF'}**"
    )
else:
    st.caption(f"Trip: **{selected_trip}** | Model: **{selected_model}** | HMM: **{'ON' if use_hmm else 'OFF'}**")

col_map, col_metrics = st.columns([3, 1])

if pred_resp:
    # ─── Map (rendered via components.html to avoid streamlit-folium 0px bug) ───
    with col_map:
        gps_coords = pred_resp.get("gps_track_geojson", {}).get("geometry", {}).get("coordinates", [])
        pred_coords = pred_resp.get("predicted_route_geojson", {}).get("geometry", {}).get("coordinates", [])

        if gps_coords:
            lats = [c[1] for c in gps_coords]
            lons = [c[0] for c in gps_coords]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
        else:
            center_lat, center_lon = 49.85, 8.60

        m = folium.Map(location=[center_lat, center_lon], zoom_start=14, tiles="OpenStreetMap")

        # GPS track layer (blue halo so it remains visible behind the red route)
        if gps_coords:
            gps_layer = folium.FeatureGroup(name="🔵 Raw GPS Track", show=True)
            gps_latlon = [[c[1], c[0]] for c in gps_coords]
            folium.PolyLine(
                locations=gps_latlon,
                color="#0077ff", weight=7, opacity=0.45,
                tooltip="Raw GPS Track",
            ).add_to(gps_layer)
            folium.Marker(
                gps_latlon[0],
                popup="Start", icon=folium.Icon(color="green", icon="play"),
            ).add_to(gps_layer)
            folium.Marker(
                gps_latlon[-1],
                popup="End", icon=folium.Icon(color="red", icon="stop"),
            ).add_to(gps_layer)
            gps_layer.add_to(m)
            m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

        # Predicted route layer (crisp red line on top)
        if pred_coords:
            pred_layer = folium.FeatureGroup(name="🔴 Predicted Road Route", show=True)
            folium.PolyLine(
                locations=[[c[1], c[0]] for c in pred_coords],
                color="#e63946", weight=3, opacity=0.95,
                tooltip="Predicted Route (HMM)" if pred_resp.get("use_hmm") else "Predicted Route (Classifier Only)",
            ).add_to(pred_layer)
            pred_layer.add_to(m)

        folium.LayerControl(collapsed=False).add_to(m)
        components.html(m.get_root().render(), height=580, scrolling=False)
        st.caption("🔵 Blue = Raw GPS Track  |  🔴 Red = Predicted Road Route (Ground truth stays strictly server-side)")

    # ─── Metrics panel ───
    with col_metrics:
        st.subheader("📊 Server-Side Metrics")

        if metrics_resp and "top1_accuracy" in metrics_resp:
            st.metric("Fix Hit Accuracy", f"{metrics_resp.get('top1_accuracy', 0)*100:.1f}%")
            st.metric("Route F1 Score", f"{metrics_resp.get('macro_f1', 0):.4f}")
            if "route_precision" in metrics_resp:
                st.metric(
                    "Route Precision / Recall",
                    f"{metrics_resp['route_precision']*100:.1f}% / {metrics_resp['route_recall']*100:.1f}%",
                )
            st.metric("Route Continuity", f"{metrics_resp.get('route_continuity', 0)*100:.1f}%")
            st.metric("Inference Time", f"{metrics_resp.get('inference_time_sec', 0):.2f}s")
            st.metric("GPS Fixes", metrics_resp.get("n_fixes", "—"))
        elif metrics_resp:
            st.metric("Route Continuity", f"{metrics_resp.get('route_continuity', 0)*100:.1f}%")
            st.metric("Inference Time", f"{metrics_resp.get('inference_time_sec', 0):.2f}s")
            st.info(metrics_resp.get("note", "Ground truth unavailable."))
        else:
            st.info("Metrics not available.")

        st.subheader("🧠 Run Info")
        st.write(f"**Model:** `{pred_resp.get('model_used', '—')}`")
        st.write(f"**HMM:** {'Enabled ✅' if pred_resp.get('use_hmm') else 'Disabled'}")
        st.write(f"**Trip:** `{pred_resp.get('trip_id', '—')}`")

else:
    with col_map:
        st.info("👈 Select a trip and model from the sidebar, then click **Run Prediction**.")
        m = folium.Map(location=[49.85, 8.60], zoom_start=12, tiles="OpenStreetMap")
        components.html(m.get_root().render(), height=580, scrolling=False)

    with col_metrics:
        st.subheader("📊 Metrics")
        st.info("Run a prediction to see metrics here.")
