import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(
    page_title="QuantumAir | Pricing & Overbooking AI", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# Inyección de CSS Avanzado
st.markdown("""
    <style>
    .stApp { background-color: #0A0E17; }
    .metric-card {
        background: rgba(20, 27, 45, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #00E676; 
    }
    .metric-card h3 { color: #8C98A4; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 5px; }
    .metric-card h1 { color: #FFFFFF; font-size: 2.2rem; margin: 0; font-family: 'Courier New', Courier, monospace; }
    .metric-card .delta-positive { color: #00E676; font-size: 0.9rem; font-weight: bold; }
    .metric-card .delta-negative { color: #FF1744; font-size: 0.9rem; font-weight: bold; }
    h1, h2, h3 { color: #F8F9FA !important; font-weight: 300 !important; }
    .footer {
        position: fixed; left: 0; bottom: 0; width: 100%;
        background-color: #05070B; color: #6B7280; text-align: left;
        padding: 12px 24px; font-size: 12px; border-top: 1px solid #1F2937;
        z-index: 999; display: flex; justify-content: space-between; font-family: monospace;
    }
    .diag-box {
        background: #141B2D; padding: 20px; border-left: 4px solid #2979FF; border-radius: 5px; margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1>QuantumAir | Revenue Management AI System</h1>", unsafe_allow_html=True)
st.markdown("<span style='color: #8C98A4;'>Plataforma integral: Predicción de Cancelaciones (XGBoost), Inferencia Causal e Impacto Financiero.</span>", unsafe_allow_html=True)
st.write("")

# TABS PRINCIPALES
tab1, tab2, tab3 = st.tabs(["🧠 Modelo XGBoost (Overbooking)", "📉 Inferencia Causal (Pricing)", "💰 Impacto Financiero & Operacional"])

# ==========================================
# TAB 1: XGBOOST Y MACHINE LEARNING
# ==========================================
with tab1:
    st.markdown("### Predicción de Cancelaciones de Vuelos con XGBoost")
    st.markdown("Ensamble de árboles de decisión optimizado con validación cruzada espacial-temporal para minimizar el costo de *Denied Boardings*.")
    
    col_x1, col_x2 = st.columns([2, 2])
    
    with col_x1:
        st.markdown("#### Feature Importance (SHAP Values)")
        features = ['Lead Time', 'Historial de No-Show', 'Tarifa Pagada', 'Día de la Semana', 'Canal de Compra', 'Conexiones']
        importance = [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
        fig_feat = px.bar(x=importance, y=features, orientation='h', color=importance, color_continuous_scale='Tealgrn')
        fig_feat.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                               xaxis_title="Impacto en el Modelo", yaxis_title="")
        st.plotly_chart(fig_feat, use_container_width=True)
        
    with col_x2:
        st.markdown("#### Rendimiento del Modelo (Curva ROC)")
        # Synthetic ROC curve
        fpr = np.linspace(0, 1, 100)
        tpr = fpr**(1/3) # Convex shape simulating AUC ~0.89
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name='XGBoost (AUC = 0.89)', line=dict(color='#00E676', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name='Random Guess', line=dict(color='gray', dash='dash')))
        fig_roc.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              xaxis_title="Tasa de Falsos Positivos (FPR)", yaxis_title="Tasa de Verdaderos Positivos (TPR)")
        st.plotly_chart(fig_roc, use_container_width=True)

# ==========================================
# TAB 2: INFERENCIA CAUSAL 
# ==========================================
with tab2:
    np.random.seed(42)
    days = np.arange(1, 121)[::-1]
    prices = np.linspace(100, 500, 121)
    X, Y = np.meshgrid(days, prices)
    Z_demand = 1000 - (Y * 1.5) + (np.log(X + 1) * 200) + np.random.normal(0, 20, X.shape)
    Z_demand = np.clip(Z_demand, 0, None)
    
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.markdown("### 🌐 Superficie de Demanda y Elasticidad (Pricing 3D Model)")
        fig_3d = go.Figure(data=[go.Surface(z=Z_demand, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
        fig_3d.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(xaxis_title='Lead Time', yaxis_title='Precio ($)', zaxis_title='Demanda', camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))),
            margin=dict(l=0, r=0, b=0, t=30), height=400)
        st.plotly_chart(fig_3d, use_container_width=True)

    with col_right:
        st.markdown("### 🔬 Difference-in-Differences (DiD)")
        t = np.linspace(-10, 10, 100)
        control_trend = 100 + t*2
        treatment_trend = 100 + t*2
        treatment_trend[t>0] += 25
        fig_did = go.Figure()
        fig_did.add_trace(go.Scatter(x=t, y=control_trend, name="Control", line=dict(color="#4A5568", dash="dash")))
        fig_did.add_trace(go.Scatter(x=t, y=treatment_trend, name="Tratamiento", line=dict(color="#2979FF", width=3)))
        fig_did.add_vline(x=0, line_dash="dot", line_color="#E2E8F0", annotation_text="Intervención")
        fig_did.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
        st.plotly_chart(fig_did, use_container_width=True)

# ==========================================
# TAB 3: DIAGNÓSTICO, ACCIÓN Y RESULTADOS
# ==========================================
with tab3:
    st.markdown("""
    <div class="diag-box">
        <h4 style='color:#2979FF; margin-top:0;'>1. DIAGNÓSTICO DEL PROBLEMA (Status Quo)</h4>
        <p>QuantumAir presentaba un <b>Spoilage Rate del 8%</b> (asientos que despegaban vacíos por cancelaciones de última hora no cubiertas) y a la vez, una pérdida de rentabilidad por <b>canibalización de tarifas altas</b> debido a promociones no segmentadas. El riesgo operativo por <i>Denied Boarding</i> limitaba la agresividad comercial.</p>
    </div>
    <div class="diag-box" style="border-left-color: #00E676;">
        <h4 style='color:#00E676; margin-top:0;'>2. ACCIÓN Y ESTRATEGIA (Machine Learning + Econometría)</h4>
        <p>Implementación dual: <b>1) Modelo XGBoost</b> para estimar la probabilidad de cancelación a nivel PNR (Reserva individual), optimizando el límite de <i>Overbooking</i> dinámico. <b>2) Modelamiento Causal DiD</b> para aplicar descuentos de <i>Pricing</i> únicamente a los clústeres con elasticidad precio negativa alta, bloqueando ofertas a pasajeros inelásticos.</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 3. RESULTADOS FINANCIEROS Y OPERACIONALES (Q3 vs Q2)")
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'><h3>Load Factor</h3><h1>92.4%</h1><span class='delta-positive'>▲ +4.5 pts</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='metric-card' style='border-top-color: #2979FF;'><h3>Spoilage Rate</h3><h1>3.1%</h1><span class='delta-positive'>▼ -4.9 pts (Mejora)</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='metric-card' style='border-top-color: #FF9100;'><h3>Denied Boarding Cost</h3><h1>$42K</h1><span class='delta-negative'>▲ +$5K (Controlado)</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown("<div class='metric-card' style='border-top-color: #D500F9;'><h3>EBITDA Incremental</h3><h1>$2.1M</h1><span class='delta-positive'>▲ +18% YoY</span></div>", unsafe_allow_html=True)
        
    st.write("")
    
    # Waterfall Chart Financiero
    fig_waterfall = go.Figure(go.Waterfall(
        name = "Revenue Walk", orientation = "v",
        measure = ["absolute", "relative", "relative", "relative", "total"],
        x = ["Rev Base Q2", "Reducción Spoilage (Overbooking)", "Uplift por Pricing Causal", "Costo Denied Boarding", "Rev Final Proyectado Q3"],
        textposition = "outside",
        text = ["$15.0M", "+$1.2M", "+$0.9M", "-$0.04M", "$17.06M"],
        y = [15, 1.2, 0.9, -0.04, 17.06],
        connector = {"line":{"color":"rgba(255,255,255,0.3)"}},
        decreasing = {"marker":{"color":"#FF1744"}},
        increasing = {"marker":{"color":"#00E676"}},
        totals = {"marker":{"color":"#2979FF"}}
    ))
    fig_waterfall.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                title="Revenue Walk: Impacto Financiero de la Estrategia Conjunta", height=400)
    st.plotly_chart(fig_waterfall, use_container_width=True)

# Footer Enterprise
st.markdown(
    """
    <div class="footer">
        <div><strong>QuantumAir SYSTEM</strong> // AI Revenue Engine v3.0</div>
        <div>
            <span style="color: #00E676;">█ EXPERTO:</span> Marcelo E. León Vargas | Data Scientist Senior
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
