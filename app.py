import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Simulador de Cancelaciones Hoteleras", layout="wide")

st.title("🛎️ Simulador de Riesgo de Cancelación")
st.markdown("Ajusta los parámetros de la reserva para predecir en tiempo real la probabilidad de que el cliente cancele. Esta información permite optimizar el *overbooking* de forma segura.")

# Cargar el modelo entrenado
@st.cache_resource
def load_model():
    return pickle.load(open('xgboost_hotel_model.pkl', 'rb'))

modelo = load_model()

# Interfaz de usuario con columnas
col1, col2 = st.columns(2)

with col1:
    st.header("Datos de la Reserva")
    lead_time = st.slider("Días de anticipación (Lead Time)", 0, 365, 30)
    special_requests = st.selectbox("Peticiones especiales", [0, 1, 2, 3, 4, 5])
    parking = st.selectbox("Espacios de estacionamiento requeridos", [0, 1, 2])
    changes = st.number_input("Cambios realizados en la reserva", min_value=0, max_value=10, value=0)

with col2:
    st.header("Historial del Cliente")
    prev_cancels = st.number_input("Cancelaciones previas", min_value=0, max_value=20, value=0)
    is_repeated = st.selectbox("¿Huésped frecuente?", [0, 1])
    weekend_nights = st.slider("Noches de fin de semana", 0, 10, 1)
    week_nights = st.slider("Noches de semana", 0, 20, 2)

# Crear DataFrame con el input
input_data = pd.DataFrame([[lead_time, special_requests, parking, changes, 
                            prev_cancels, is_repeated, weekend_nights, week_nights]], 
                          columns=['lead_time', 'total_of_special_requests', 'required_car_parking_spaces', 
                                   'booking_changes', 'previous_cancellations', 'is_repeated_guest', 
                                   'stays_in_weekend_nights', 'stays_in_week_nights'])

# Predicción
if st.button("Calcular Riesgo de Cancelación", type="primary"):
    probabilidad = modelo.predict_proba(input_data)[0][1]
    
    st.divider()
    if probabilidad > 0.6:
        st.error(f"⚠️ Alto Riesgo de Cancelación: {probabilidad:.1%}")
        st.markdown("**Acción recomendada:** Habilitar la habitación para *overbooking* en los canales de venta.")
    elif probabilidad > 0.3:
        st.warning(f"🟡 Riesgo Medio de Cancelación: {probabilidad:.1%}")
    else:
        st.success(f"✅ Reserva Segura: Riesgo del {probabilidad:.1%}")
        st.markdown("**Acción recomendada:** Mantener inventario cerrado para esta unidad para asegurar la satisfacción del cliente.")
