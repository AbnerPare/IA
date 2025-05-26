import streamlit as st
import pickle

# Charger le modèle
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Prédiction de maladie cardiaque")
# ... (ajoutez ici vos widgets Streamlit)
