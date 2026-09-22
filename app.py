import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página (Tema oscuro y profesional Enterprise B2B)
st.set_page_config(
    page_title="DRIMO | Pricing & Causal AI Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Inyección de CSS Avanzado para simular un dashboard de grado empresarial (B2B SaaS)
st.markdown("""
    <style>
    /* Estilos base */
    .stApp {
        background-color: #0A0E17;
    }
    
    /* Contenedores de métricas tipo tarjeta de vidrio (Glassmorphism sutil) */
    .metric-card {
        background: rgba(20, 27, 45, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border-top: 3px solid #00E676; /* Accent color neon green */
    }
    
    .metric-card h3 {
        color: #8C98A4;
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 5px;
    }
    
    .metric-card h1 {
        color: #FFFFFF;
        font-size: 2.2rem;
        margin: 0;
        font-family: 'Courier New', Courier, monospace;
    }
    
    .metric-card .delta-positive {
        color: #00E676;
        font-size: 0.9rem;
        font-weight: bold;
    }
    .metric-card .delta-negative {
        color: #FF1744;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #F8F9FA !important;
        font-weight: 300 !important;
    }
    
    /* Divider */
    hr {
        border-color: #2D3748 !important;
    }
    
    /* Custom Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #05070B;
        color: #6B7280;
        text-align: left;
        padding: 12px 24px;
        font-size: 12px;
        border-top: 1px solid #1F2937;
        z-index: 999;
        display: flex;
        justify-content: space-between;
        font-family: monospace;
    }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# GENERACIÓN DE DATOS (MOCK) ALTA FIDELIDAD
# ---------------------------------------------------------
np.random.seed(42)
days = np.arange(1, 121)[::-1]
prices = np.linspace(100, 500, 121)

# Superficie 3D: Elasticidad Precio vs Lead Time vs Demanda
X, Y = np.meshgrid(days, prices)
Z_demand = 1000 - (Y * 1.5) + (np.log(X + 1) * 200) + np.random.normal(0, 20, X.shape)
Z_demand = np.clip(Z_demand, 0, None)

# Heatmap: Matriz de Transición/Canibalización
classes = ['Basic Economy', 'Main Cabin', 'Premium Economy', 'Business']
cannibalization_matrix = np.array([
    [0.85, 0.12, 0.03, 0.00],
    [0.10, 0.78, 0.10, 0.02],
    [0.01, 0.15, 0.80, 0.04],
    [0.00, 0.02, 0.08, 0.90]
])

# ---------------------------------------------------------
# UI MAIN DASHBOARD
# ---------------------------------------------------------
st.markdown("<h1>DRIMO | Causal Pricing & Yield Optimization Engine</h1>", unsafe_allow_html=True)
st.markdown("<span style='color: #8C98A4;'>Plataforma prescriptiva B2B para inferencia causal, modelamiento de elasticidad cruzada y A/B Testing continuo.</span>", unsafe_allow_html=True)
st.write("")

# KPIs B2B Style
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("""
    <div class="metric-card">
        <h3>Average Treatment Effect (ATE)</h3>
        <h1>+14.2%</h1>
        <span class="delta-positive">▲ p-value < 0.001 (Stat Sig)</span>
    </div>
    """, unsafe_allow_html=True)
with c2:
    st.markdown("""
    <div class="metric-card" style="border-top-color: #2979FF;">
        <h3>Elasticidad Precio (Promedio)</h3>
        <h1>-2.14</h1>
        <span class="delta-negative">▼ Highly Elastic Segment</span>
    </div>
    """, unsafe_allow_html=True)
with c3:
    st.markdown("""
    <div class="metric-card" style="border-top-color: #FF9100;">
        <h3>Cannibalization Index</h3>
        <h1>3.8%</h1>
        <span class="delta-positive">▼ -0.5% vs Q2 Benchmark</span>
    </div>
    """, unsafe_allow_html=True)
with c4:
    st.markdown("""
    <div class="metric-card" style="border-top-color: #D500F9;">
        <h3>Expected Incremental Rev (Uplift)</h3>
        <h1>$892.4K</h1>
        <span class="delta-positive">▲ +12.1% WoW</span>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.write("")

# Main Charts Area
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown("### 🌐 Superficie de Demanda y Elasticidad (Pricing 3D Model)")
    # 3D Surface Plot
    fig_3d = go.Figure(data=[go.Surface(z=Z_demand, x=X, y=Y, colorscale='Viridis', opacity=0.9)])
    fig_3d.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(
            xaxis_title='Lead Time (Días)',
            yaxis_title='Precio Objetivo ($)',
            zaxis_title='Volumen Demanda',
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=0.5))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=500
    )
    st.plotly_chart(fig_3d, use_container_width=True)

with col_right:
    st.markdown("### 📉 Matriz de Efectos de Sustitución (Cross-Elasticity)")
    # Heatmap para canibalización
    fig_heat = px.imshow(
        cannibalization_matrix, 
        labels=dict(x="Tarifa Destino (Post-Tratamiento)", y="Tarifa Origen", color="Probabilidad de Migración"),
        x=classes, y=classes,
        color_continuous_scale='Tealgrn',
        text_auto=".0%"
    )
    fig_heat.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, b=0, t=30),
        height=500,
        coloraxis_showscale=False
    )
    st.plotly_chart(fig_heat, use_container_width=True)

st.divider()

# Bottom Section: Causal Model Output
st.markdown("### 🔬 Inferencia Causal: Difference-in-Differences (DiD) Estimation")
col_did1, col_did2 = st.columns([2, 3])

with col_did1:
    st.markdown("""
    **Especificación del Modelo Econométrico:**
    `log(Demand_{it}) = α + β_1(Treatment_i) + β_2(Post_t) + δ(Treatment_i × Post_t) + γ(Controls_{it}) + ε_{it}`
    
    *   **Tratamiento:** Aplicación de algoritmo dinámico de descuentos por *segmento elástico*.
    *   **Controles:** Efectos fijos de ruta, estacionalidad y DOW (Day of Week).
    *   **Resultado:** El estimador $δ$ (Uplift) muestra un incremento causal robusto en el volumen de reservas neto de canibalización.
    """)

with col_did2:
    # Simulated DID plot
    t = np.linspace(-10, 10, 100)
    control_trend = 100 + t*2
    treatment_trend = 100 + t*2
    treatment_trend[t>0] += 25 # Uplift effect
    
    fig_did = go.Figure()
    fig_did.add_trace(go.Scatter(x=t, y=control_trend, name="Grupo Control", line=dict(color="#4A5568", width=3, dash="dash")))
    fig_did.add_trace(go.Scatter(x=t, y=treatment_trend, name="Grupo Tratamiento", line=dict(color="#00E676", width=3)))
    fig_did.add_vline(x=0, line_width=2, line_dash="dot", line_color="#E2E8F0", annotation_text="Intervención")
    fig_did.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        height=250, margin=dict(l=0, r=0, b=0, t=0),
        yaxis_title="Índice de Demanda Normalizado", xaxis_title="Días relativos a la Intervención"
    )
    st.plotly_chart(fig_did, use_container_width=True)

# Footer Enterprise
st.markdown(
    """
    <div class="footer">
        <div><strong>DRIMO SYSTEM</strong> // Causal Pricing Module v2.4.1</div>
        <div>
            <span style="color: #00E676;">█ EXPERTO:</span> Data Scientist Senior (Econometría) | 
            <span style="color: #2979FF;">█ GRADO:</span> PhD / MSc Analytics
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
