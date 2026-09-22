import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Simulador de Cancelaciones Hoteleras", layout="wide")

# Inyección de CSS para forzar tema oscuro y crear la viñeta inferior
st.markdown("""
