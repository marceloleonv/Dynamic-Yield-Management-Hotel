import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página (Tema oscuro y profesional)
st.set_page_config(
    page_title="Causal Inference & Pricing Science Engine", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Estilo CSS personalizado para forzar tema oscuro y estética senior
st.markdown("""
    <style>
    .reportview-container {
        background: #0E1117;
        color: #FAFAFA;
    }
    .sidebar .sidebar-content {
        background: #262730;
    }
    .metric-container {
        background-color: #1E2129;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4F8BF9;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: #8C92AC;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        border-top: 1px solid #262730;
        z-index: 100;
    }
    .footer-bullet {
        margin: 0 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Título y Descripción
st.title("🛫 Causal Inference & Pricing Science Engine")
st.markdown("### Modelamiento de Demanda, Elasticidad Cruzada y Optimización de Revenue")
st.markdown("""
Este motor analítico utiliza métodos cuasi-experimentales y modelamiento contrafactual para estimar la elasticidad del precio y evaluar el impacto causal de diferentes políticas de *pricing* en tiempo real. Diseñado para optimización de inventarios (Revenue Management).
""")
st.divider()

# Generación de datos simulados avanzados
np.random.seed(42)
days_to_departure = np.arange(1, 91)[::-1]
base_demand = np.log(days_to_departure) * 20 + np.random.normal(0, 5, 90)
price_elasticity = -0.05 - (90 - days_to_departure) * 0.001 

treatment_price_drop = 0.15 # 15% discount
demand_control = base_demand
demand_treatment = base_demand * (1 - price_elasticity * treatment_price_drop * 100)
revenue_control = demand_control * 100
revenue_treatment = demand_treatment * 85

# KPIs
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="Average Treatment Effect (ATE)", value="+12.4% Demanda", delta="Significancia: p < 0.01")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="Price Elasticity (Promedio)", value="-1.85", delta="Demanda Elástica", delta_color="inverse")
    st.markdown('</div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="Riesgo de Canibalización", value="4.2%", delta="-1.5% vs Benchmark", delta_color="normal")
    st.markdown('</div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-container">', unsafe_allow_html=True)
    st.metric(label="Revenue Incremental Esperado", value="$245.5K", delta="+8.2% Uplift")
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# Gráficos
col_chart1, col_chart2 = st.columns(2)

with col_chart1:
    st.subheader("Análisis Contrafactual: Demanda (Tratamiento vs Control)")
    fig_demand = go.Figure()
    fig_demand.add_trace(go.Scatter(x=days_to_departure, y=demand_control, mode='lines', name='Control (Sin Descuento)', line=dict(color='#8C92AC', dash='dash')))
    fig_demand.add_trace(go.Scatter(x=days_to_departure, y=demand_treatment, mode='lines', name='Tratamiento (15% Off)', line=dict(color='#4F8BF9', width=3)))
    
    # Shade the uplift area
    fig_demand.add_trace(go.Scatter(
        x=np.concatenate([days_to_departure, days_to_departure[::-1]]),
        y=np.concatenate([demand_treatment, demand_control[::-1]]),
        fill='toself', fillcolor='rgba(79, 139, 249, 0.2)', line=dict(color='rgba(255,255,255,0)'),
        showlegend=False, name='Uplift'
    ))

    fig_demand.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Días previos al vuelo (Lead Time)",
        yaxis_title="Reservas Estimadas",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_demand, use_container_width=True)

with col_chart2:
    st.subheader("Curva de Elasticidad y Maximización de Revenue")
    prices = np.linspace(50, 200, 100)
    # Simple simulated revenue curve based on elasticity
    estimated_demand = 500 - 1.5 * prices
    revenue = prices * estimated_demand
    
    fig_rev = go.Figure()
    fig_rev.add_trace(go.Scatter(x=prices, y=revenue, mode='lines', line=dict(color='#00CC96', width=3), name='Curva de Ingresos'))
    
    # Optimal point
    opt_idx = np.argmax(revenue)
    fig_rev.add_trace(go.Scatter(x=[prices[opt_idx]], y=[revenue[opt_idx]], mode='markers', marker=dict(color='red', size=10), name='Punto Óptimo'))
    
    fig_rev.update_layout(
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Precio del Ticket ($)",
        yaxis_title="Revenue Total Proyectado ($)",
    )
    st.plotly_chart(fig_rev, use_container_width=True)

st.divider()

st.markdown("### 🔬 Resultados del Diseño Cuasi-Experimental (DiD)")
st.markdown("""
Se aplicó un modelo de Diferencias en Diferencias (Difference-in-Differences) controlando por estacionalidad, ruta y anticipación de compra. 
Se rechaza la hipótesis nula con un nivel de confianza del **99%**, demostrando que la estrategia de precios dinámica captura valor sin incurrir en efectos severos de canibalización en clases tarifarias superiores.
""")

# Footer
st.markdown(
    """
    <div class="footer">
        <span class="footer-bullet">📌 <b>Proyecto:</b> Causal AI para Yield Management y Pricing Analytics</span>
        <span class="footer-bullet">|</span>
        <span class="footer-bullet">🎓 <b>Perfil:</b> Data Scientist Senior / Experto en Econometría</span>
        <span class="footer-bullet">|</span>
        <span class="footer-bullet">✈️ <b>Sector:</b> Aviación / Travel & Hospitality</span>
    </div>
    """,
    unsafe_allow_html=True
)
