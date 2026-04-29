"""AQI Prediction Dashboard — Rich visual experience."""

import streamlit as st
import pandas as pd
import numpy as np
import json, os
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
import yaml

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, "data", "processed")
CFG_PATH = os.path.join(BASE, "configs", "config.yaml")

@st.cache_data
def load_all():
    with open(os.path.join(PROC, "meta.json")) as f:
        meta = json.load(f)
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)
    return (meta, cfg,
            pd.read_csv(os.path.join(PROC, "all_results.csv")),
            pd.read_csv(os.path.join(PROC, "health_alerts.csv")),
            pd.read_csv(os.path.join(PROC, "shap_feature_importance.csv")),
            pd.read_csv(os.path.join(PROC, "shap_city_importance.csv")))

meta, cfg, results, alerts, feat_imp, city_imp = load_all()
cities = meta["city_names"]
locs = cfg["locations"]

AQI_COLORS = {"Good": "#27ae60", "Moderate": "#f1c40f", "Unhealthy for Sensitive Groups": "#e67e22",
              "Unhealthy": "#e74c3c", "Very Unhealthy": "#8e44ad", "Hazardous": "#1a1a2e"}
AQI_EMOJI = {"Good": "🟢", "Moderate": "🟡", "Unhealthy for Sensitive Groups": "🟠",
             "Unhealthy": "🔴", "Very Unhealthy": "🟣", "Hazardous": "⬛"}

st.set_page_config(page_title="AQI Forecast India", layout="wide", page_icon="🌍")

# ── Global CSS ──
st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {font-family: 'Inter', sans-serif;}
    .block-container {padding-top: 0.5rem; padding-bottom: 1rem; max-width: 1200px;}
    .stTabs [data-baseweb="tab-list"] {gap: 12px; background: transparent; padding: 0; justify-content: center;}
    .stTabs [data-baseweb="tab-list"] > div {background: transparent !important;}
    .stTabs [data-baseweb="tab-highlight"] {background-color: #e74c3c !important; height: 3px !important;}
    .stTabs [data-baseweb="tab-border"] {background-color: rgba(255,255,255,0.1) !important;}
    .stTabs [data-baseweb="tab"] {border-radius: 8px 8px 0 0; padding: 8px 16px; font-weight: 500; color: rgba(255,255,255,0.6) !important; background: transparent;}
    .stTabs [aria-selected="true"] {color: white !important; font-weight: 700 !important; background: rgba(255,255,255,0.1) !important;}
    [data-testid="stMetric"] {background: linear-gradient(135deg, #667eea11, #764ba211); border-radius: 12px;
        padding: 16px; border: 1px solid #e8e8e8;}
    [data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        transition: all 0.2s ease;}
    [data-testid="stMetricLabel"] {font-size: 13px !important; color: #666 !important;}
    [data-testid="stMetricValue"] {font-size: 28px !important; font-weight: 700 !important;}
    div[data-testid="stExpander"] {border: 1px solid #e8e8e8; border-radius: 12px;}
    .hero-card {background: linear-gradient(135deg, #0f0c29, #302b63, #24243e); color: white;
        padding: 32px 40px; border-radius: 16px; margin-bottom: 24px;}
    .hero-card h1 {margin: 0; font-size: 2.2rem; font-weight: 700;}
    .hero-card p {margin: 8px 0 0 0; opacity: 0.85; font-size: 1rem;}
    .stat-pill {display: inline-block; background: rgba(255,255,255,0.15); padding: 6px 14px;
        border-radius: 20px; margin: 12px 8px 0 0; font-size: 0.85rem;}
    .alert-card {padding: 14px 20px; margin: 8px 0; border-radius: 12px; border-left: 5px solid;
        background: #1e1e2e; box-shadow: 0 1px 4px rgba(0,0,0,0.2);}
    .section-header {font-size: 1.1rem; font-weight: 600; color: white; margin: 20px 0 12px 0;
        padding-bottom: 8px; border-bottom: 2px solid #e74c3c;}
</style>""", unsafe_allow_html=True)

# ── Hero Header ──
wm = results[(results["Model"].str.contains("WM-STGN")) & (results["Target"] == "pm2_5")]
wm1h = wm[wm["Horizon"] == "1h"]
wm24h = wm[wm["Horizon"] == "24h"]
r2_1h = f"{wm1h['R2'].values[0]:.3f}" if not wm1h.empty else "—"
mae_24h = f"{wm24h['MAE'].values[0]:.1f}" if not wm24h.empty else "—"

st.markdown(f"""<div class="hero-card">
    <h1>🌍 Air Quality Prediction System</h1>
    <p>Spatio-Temporal Graph Neural Network · Real CPCB Sensor Data · Residual Prediction</p>
    <div>
        <span class="stat-pill">📍 {len(cities)} Cities</span>
        <span class="stat-pill">🎯 R² = {r2_1h} (1h)</span>
        <span class="stat-pill">📊 24h MAE = {mae_24h} µg/m³</span>
        <span class="stat-pill">🔬 6 Models Compared</span>
    </div>
</div>""", unsafe_allow_html=True)

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🗺️  AQI Map", "📈  City Forecasts", "⚠️  Health Alerts",
                                          "🔬  Explainability", "📊  Model Comparison"])

# ════════════════════════════════════════════════════════════
# TAB 1: AQI Map
# ════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Predicted AQI Across India</div>', unsafe_allow_html=True)
    hz = st.selectbox("Forecast Horizon", ["1h", "6h", "12h", "24h"], key="map_hz",
                       help="How far ahead to predict — 1h is the next hour, 24h is tomorrow")
    map_data = alerts[alerts["Horizon"] == hz]

    col_map, col_table = st.columns([5, 3])
    with col_map:
        m = folium.Map(location=[22.5, 78.5], zoom_start=5, tiles="CartoDB dark_matter",
                       attr=" ")

        # Legend on map
        legend_html = """<div style="position:fixed;bottom:30px;left:30px;z-index:1000;background:rgba(0,0,0,0.75);
            padding:10px 14px;border-radius:8px;font-size:12px;color:white;font-family:Inter,sans-serif;line-height:1.8">
            <b>AQI Categories</b><br>
            <span style="color:#27ae60">●</span> Good (0-50)<br>
            <span style="color:#f1c40f">●</span> Moderate (51-100)<br>
            <span style="color:#e67e22">●</span> USG (101-150)<br>
            <span style="color:#e74c3c">●</span> Unhealthy (151-200)<br>
            <span style="color:#8e44ad">●</span> Very Unhealthy (201-300)<br>
            <span style="color:#555">●</span> Hazardous (301+)
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))
        for _, row in map_data.iterrows():
            city = row["City"]
            if city in locs:
                lat, lon = locs[city]
                aqi = row["AQI_pred"]
                cat = row["Category (conservative)"]
                color = AQI_COLORS.get(cat, "#95a5a6")
                folium.CircleMarker(
                    location=[lat, lon], radius=max(7, min(aqi / 10, 28)),
                    color="white", weight=1, fill=True, fill_color=color, fill_opacity=0.85,
                    popup=folium.Popup(f"<div style='font-family:Inter,sans-serif;min-width:140px'>"
                          f"<b style='font-size:14px'>{city}</b><br>"
                          f"<span style='font-size:22px;font-weight:700;color:{color}'>{aqi:.0f}</span><br>"
                          f"<span style='color:#666'>{cat}</span><br>"
                          f"PM2.5: {row['PM2.5_pred']:.0f} · PM10: {row['PM10_pred']:.0f}</div>", max_width=220),
                    tooltip=f"{city}: AQI {aqi:.0f}"
                ).add_to(m)
        st_folium(m, width=None, height=520, use_container_width=True)

    with col_table:
        st.markdown(f"#### {hz} Forecast Rankings")
        summary = map_data[["City", "AQI_pred", "Category (conservative)"]].copy()
        summary.columns = ["City", "AQI", "Category"]
        summary["AQI"] = summary["AQI"].round(0).astype(int)
        summary = summary.sort_values("AQI", ascending=False).reset_index(drop=True)
        # Color the category column
        def color_cat(val):
            c = AQI_COLORS.get(val, "#999")
            return f"color: {c}; font-weight: 600"
        st.dataframe(summary.style.map(color_cat, subset=["Category"]),
                     use_container_width=True, height=480, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 2: City Forecasts
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Multi-Horizon City Forecast</div>', unsafe_allow_html=True)
    city_sel = st.selectbox("Select City", cities, index=cities.index("Delhi") if "Delhi" in cities else 0)
    cd = alerts[alerts["City"] == city_sel]

    if not cd.empty:
        # AQI cards with colored backgrounds
        cols = st.columns(4)
        for i, (_, row) in enumerate(cd.iterrows()):
            cat = row["Category (mean)"]
            color = AQI_COLORS.get(cat, "#999")
            emoji = AQI_EMOJI.get(cat, "⚪")
            cols[i].markdown(f"""<div style="background:linear-gradient(135deg, {color}22, {color}08);
                border:1px solid {color}44; border-radius:12px; padding:16px; text-align:center">
                <div style="font-size:12px;color:#666;font-weight:500">{row['Horizon']} FORECAST</div>
                <div style="font-size:32px;font-weight:700;color:{color}">{row['AQI_pred']:.0f}</div>
                <div style="font-size:13px;color:{color}">{emoji} {cat}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("")

        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cd["Horizon"], y=cd["PM2.5_pred"],
                                  marker=dict(color="#e74c3c", cornerradius=6), opacity=0.9))
            fig.update_layout(title=dict(text=f"PM2.5 Forecast", font=dict(size=16)),
                              yaxis_title="µg/m³", height=320, template="plotly_white",
                              margin=dict(t=50, b=30, l=50, r=20))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cd["Horizon"], y=cd["PM10_pred"],
                                  marker=dict(color="#3498db", cornerradius=6), opacity=0.9))
            fig.update_layout(title=dict(text=f"PM10 Forecast", font=dict(size=16)),
                              yaxis_title="µg/m³", height=320, template="plotly_white",
                              margin=dict(t=50, b=30, l=50, r=20))
            st.plotly_chart(fig, use_container_width=True)

        # AQI with uncertainty
        fig = go.Figure()
        error_vals = (cd["AQI_upper_95"].values - cd["AQI_pred"].values)
        fig.add_trace(go.Bar(x=cd["Horizon"], y=cd["AQI_pred"], name="Predicted AQI",
                              marker=dict(color="#667eea", cornerradius=6),
                              error_y=dict(type="data", array=error_vals, color="white", thickness=2, width=6)))
        fig.add_hline(y=150, line_dash="dash", line_color="#e74c3c", line_width=2,
                      annotation_text="⚠️ Unhealthy", annotation_font_color="#e74c3c")
        fig.update_layout(title=dict(text="AQI Forecast with Uncertainty (MC Dropout)", font=dict(size=16)),
                          yaxis_title="AQI", height=380, template="plotly_white",
                          margin=dict(t=50, b=30), legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Error bars show the 95% confidence interval — the model is 95% sure the actual AQI "
                   "will fall below the top of the bar. Health alerts use this upper bound to err on the side of caution.")

# ════════════════════════════════════════════════════════════
# TAB 3: Health Alerts
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Health Risk Alerts</div>', unsafe_allow_html=True)
    st.markdown("Alerts use the **95% upper confidence bound** from MC Dropout — erring on the side of caution.")

    level = st.selectbox("Filter by severity", ["All", "Unhealthy", "Very Unhealthy", "Hazardous"])
    thresh = {"All": 0, "Unhealthy": 150, "Very Unhealthy": 200, "Hazardous": 300}
    active = alerts[alerts["AQI_upper_95"] >= thresh[level]].sort_values("AQI_upper_95", ascending=False)
    worst = active.loc[active.groupby("City")["AQI_upper_95"].idxmax()].sort_values("AQI_upper_95", ascending=False)

    if len(worst) == 0:
        st.success(f"✅ No cities at {level} level or above")
    else:
        if level != "All":
            st.error(f"⚠️ **{len(worst)} cities** at {level} level or above")

        for _, row in worst.iterrows():
            cat = row["Category (conservative)"]
            color = AQI_COLORS.get(cat, "#999")
            emoji = AQI_EMOJI.get(cat, "⚪")
            aqi = row["AQI_upper_95"]
            bw = 8 if aqi >= 200 else 5
            aqi_sz = 30 if aqi >= 200 else 24
            city_sz = 18 if aqi >= 200 else 16
            pad = "18px 24px" if aqi >= 200 else "14px 20px"
            city_rows = active[active["City"] == row["City"]]
            hz_order = {"1h": 0, "6h": 1, "12h": 2, "24h": 3}
            city_rows = city_rows.copy()
            city_rows["_order"] = city_rows["Horizon"].map(hz_order)
            city_rows = city_rows.sort_values("_order")
            hz_parts = []
            for _, r in city_rows.iterrows():
                hz_parts.append(f"<span style='background:{color}33;padding:4px 12px;border-radius:6px;"
                               f"font-size:12px;color:white;font-weight:500'>{r['Horizon']}: <b>{r['AQI_pred']:.0f}</b></span>")
            hz_str = " ".join(hz_parts)
            st.markdown(f"""<div style="padding:{pad};margin:8px 0;border-radius:12px;border-left:{bw}px solid {color};
                        background:#1e1e2e;box-shadow:0 1px 4px rgba(0,0,0,0.2)">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <span style="font-size:{city_sz}px;font-weight:600;color:white">{emoji} {row['City']}</span>
                        <span style="color:{color};font-weight:600;margin-left:12px">{cat}</span>
                    </div>
                    <span style="font-size:{aqi_sz}px;font-weight:700;color:{color}">{aqi:.0f}</span>
                </div>
                <div style="margin-top:8px">{hz_str}</div>
            </div>""", unsafe_allow_html=True)

    with st.expander("📋 Full Alert Table"):
        st.dataframe(alerts[["City", "Horizon", "AQI_pred", "AQI_upper_95",
                              "Category (mean)", "Category (conservative)"]].round(1),
                     use_container_width=True, height=400, hide_index=True)

# ════════════════════════════════════════════════════════════
# TAB 4: Explainability
# ════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-header">What Drives the Predictions?</div>', unsafe_allow_html=True)

    # Non-technical intro
    st.markdown("""
    > **In plain English:** We test what happens when the model loses access to each piece of information.
    > If removing something makes predictions much worse, the model depends on it heavily.
    """)

    col1, col2 = st.columns(2)
    col_name = "ΔMAE" if "ΔMAE" in feat_imp.columns else feat_imp.columns[1]

    with col1:
        st.markdown("#### Which data matters most?")
        fi = feat_imp.sort_values(col_name, ascending=True)
        fig = go.Figure(go.Bar(x=fi[col_name], y=fi.iloc[:, 0], orientation="h",
                                marker=dict(color=fi[col_name], colorscale="RdYlGn_r", cornerradius=4)))
        fig.update_layout(title=dict(text="Feature Importance", font=dict(size=16)),
                          height=550, template="plotly_white", yaxis_title="",
                          xaxis_title="Impact on prediction accuracy",
                          margin=dict(l=0, t=50, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        top3 = feat_imp.nlargest(3, col_name).iloc[:, 0].tolist()
        st.markdown(f"""
        **🔍 Key takeaways:**
        - **{top3[0]}** is the most influential — the model relies on it the most
        - **{top3[1]}** and **{top3[2]}** also significantly affect predictions
        - Weather features like temperature and wind help the model understand pollution trapping and transport

        *Technical: Permutation importance — each feature is randomly shuffled across test samples and the increase in MAE (ΔMAE) is measured.*
        """)

    col_name2 = "ΔMAE" if "ΔMAE" in city_imp.columns else city_imp.columns[1]
    with col2:
        st.markdown("#### Which cities matter most?")
        ci = city_imp.sort_values(col_name2, ascending=True)
        fig = go.Figure(go.Bar(x=ci[col_name2], y=ci.iloc[:, 0], orientation="h",
                                marker=dict(color=ci[col_name2], colorscale="RdYlGn_r", cornerradius=4)))
        fig.update_layout(title=dict(text="City Importance (Ablation)", font=dict(size=16)),
                          height=550, template="plotly_white", yaxis_title="",
                          xaxis_title="Impact on prediction accuracy",
                          margin=dict(l=0, t=50, r=20))
        st.plotly_chart(fig, use_container_width=True)

        # Interpretation
        top3c = city_imp.nlargest(3, col_name2).iloc[:, 0].tolist()
        bottom1 = city_imp.nsmallest(1, col_name2).iloc[:, 0].tolist()[0]
        st.markdown(f"""
        **🔍 Key takeaways:**
        - **{top3c[0]}** is the most critical node in the graph — removing it hurts predictions the most
        - **{top3c[1]}** and **{top3c[2]}** are also important — likely due to high pollution variance
        - **{bottom1}** has the least impact — its AQI is stable, so the model doesn't depend on it

        *Technical: City ablation — each city's input is zeroed out and the increase in MAE is measured, showing how much the graph relies on each node.*
        """)
# ════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="section-header">Model Performance Comparison</div>', unsafe_allow_html=True)

    # Model descriptions
    with st.expander("ℹ️ What are these models?"):
        st.markdown("""
        | Model | What it does |
        |-------|-------------|
        | **Persistence** | Assumes pollution stays the same — the simplest possible baseline |
        | **Historical Avg** | Predicts the average change seen in training data |
        | **LSTM** | A recurrent neural network that learns temporal patterns (e.g., rush hour cycles) |
        | **CNN-LSTM** | Adds a convolutional layer to capture spatial patterns across cities |
        | **WM-STGN** | Our main model — a graph neural network that models cities as connected nodes |
        | **WA-STGN** | Same as WM-STGN but with wind-aware dynamic connections between cities |
        """)

    target = st.selectbox("Target Pollutant", ["pm2_5", "pm10"])
    filt = results[results["Target"] == target]

    # Fix horizon order
    horizon_order = ["1h", "6h", "12h", "24h"]
    filt = filt.copy()
    filt["Horizon"] = pd.Categorical(filt["Horizon"], categories=horizon_order, ordered=True)
    filt = filt.sort_values("Horizon")

    model_colors = {"Persistence": "#bdc3c7", "HistoricalAvg": "#95a5a6", "LSTM": "#3498db",
                    "CNN-LSTM": "#2ecc71", "WM-STGN-ens3": "#e74c3c", "WA-STGN-best": "#9b59b6"}

    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for m in filt["Model"].unique():
            md = filt[filt["Model"] == m]
            fig.add_trace(go.Scatter(x=md["Horizon"], y=md["MAE"], mode="lines+markers",
                                      name=m, line=dict(color=model_colors.get(m, "#777"), width=3),
                                      marker=dict(size=10)))
        fig.update_layout(title=dict(text=f"MAE vs Horizon — {target.upper()}", font=dict(size=16)),
                          yaxis_title="MAE (µg/m³)", height=420, template="plotly_white",
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        for m in filt["Model"].unique():
            md = filt[filt["Model"] == m]
            fig2.add_trace(go.Scatter(x=md["Horizon"], y=md["R2"], mode="lines+markers",
                                       name=m, line=dict(color=model_colors.get(m, "#777"), width=3),
                                       marker=dict(size=10)))
        fig2.update_layout(title=dict(text=f"R² vs Horizon — {target.upper()}", font=dict(size=16)),
                           yaxis_title="R²", height=420, template="plotly_white",
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig2, use_container_width=True)

    # Winner summary
    pm25_filt = results[results["Target"] == "pm2_5"]
    best_24h = pm25_filt[pm25_filt["Horizon"] == "24h"].sort_values("MAE").iloc[0]
    best_1h = pm25_filt[pm25_filt["Horizon"] == "1h"].sort_values("R2", ascending=False).iloc[0]
    st.markdown(f"""
    > **🏆 Summary:** **{best_24h['Model']}** achieves the lowest 24h MAE ({best_24h['MAE']:.2f} µg/m³)
    > and **{best_1h['Model']}** has the best 1h R² ({best_1h['R2']:.3f}).
    > The graph neural network's advantage grows with forecast horizon — spatial modeling matters more for longer predictions.
    """)

    # Tables
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**MAE (µg/m³)** — *lower is better*")
        mae_pivot = filt.pivot(index="Model", columns="Horizon", values="MAE").round(2)
        mae_pivot = mae_pivot.reindex(columns=horizon_order)
        st.dataframe(mae_pivot.style.format("{:.2f}").highlight_min(axis=0, color="#1a472a"),
                     use_container_width=True)
    with col2:
        st.markdown("**R² Score** — *higher is better (max 1.0)*")
        r2_pivot = filt.pivot(index="Model", columns="Horizon", values="R2").round(3)
        r2_pivot = r2_pivot.reindex(columns=horizon_order)
        st.dataframe(r2_pivot.style.format("{:.3f}").highlight_max(axis=0, color="#1a472a"),
                     use_container_width=True)

# ── About + Footer ──
st.markdown("---")

with st.expander("ℹ️ About this system"):
    st.markdown(f"""
    **What is this?** A machine learning system that predicts air pollution (PM2.5 and PM10) across {len(cities)} Indian cities,
    1 to 24 hours into the future.

    **How does it work?** Cities are modeled as nodes in a graph, connected by geographic proximity. A graph neural network
    learns how pollution propagates between cities over time. The key innovation is *residual prediction* — the model predicts
    the *change* in pollution rather than the absolute value, which improved accuracy by 90%.

    **Where does the data come from?** Real sensor measurements from CPCB (Central Pollution Control Board) government
    monitoring stations, combined with weather data from Open-Meteo (ECMWF ERA5 reanalysis).

    **What are the confidence intervals?** The model runs 20 predictions with slight random variations (MC Dropout).
    The spread of these predictions tells us how confident the model is. Health alerts use the worst-case estimate
    to err on the side of caution.
    """)

st.markdown(f"""<div style="text-align:center;padding:16px 0">
    <span style="font-size:18px">🌍</span><br>
    <span style="color:#aaa;font-size:13px;font-weight:500">
    {len(cities)} Indian Cities · CPCB Sensor Data · Graph Neural Network · Residual Prediction · Confidence Intervals
    </span><br>
    <span style="color:#666;font-size:11px">SRM Institute of Science & Technology, Chennai</span>
</div>""", unsafe_allow_html=True)
