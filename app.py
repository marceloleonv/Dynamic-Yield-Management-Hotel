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
