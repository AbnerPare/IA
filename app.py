import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Configuration de la page
st.set_page_config(page_title="Prédiction de Maladies Cardiaques", layout="wide")

# Titre
st.title("🫀 Prédiction de Maladies Cardiaques")
st.markdown("""
Cette application prédit le risque de maladie cardiaque en fonction des caractéristiques médicales.
""")

# Fonction de chargement des artefacts
@st.cache_resource
def load_artifacts():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()

# Sidebar pour les inputs
st.sidebar.header("📋 Paramètres du Patient")

def get_user_input():
    age = st.sidebar.slider("Âge", 20, 100, 50)
    sex = st.sidebar.radio("Sexe", ["Femme", "Homme"])
    cp = st.sidebar.selectbox("Type de douleur thoracique", [
        "Typique angine", 
        "Angine atypique", 
        "Douleur non-angineuse", 
        "Asymptomatique"
    ], index=0)
    trestbps = st.sidebar.slider("Pression artérielle (mm Hg)", 80, 200, 120)
    chol = st.sidebar.slider("Cholestérol (mg/dl)", 100, 600, 200)
    
    # Conversion des entrées
    cp_mapping = {
        "Typique angine": 0,
        "Angine atypique": 1,
        "Douleur non-angineuse": 2,
        "Asymptomatique": 3
    }
    
    return pd.DataFrame([[
        age,
        1 if sex == "Homme" else 0,
        cp_mapping[cp],
        trestbps,
        chol,
        0,  # fbs
        0,  # restecg
        150,  # thalach
        0,  # exang
        1.0,  # oldpeak
        1,  # slope
        0,  # ca
        3   # thal
    ]], columns=[
        'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
    ])

# Récupération des inputs
user_input = get_user_input()

# Affichage des entrées
st.subheader("📊 Données du Patient")
st.dataframe(user_input)

# Prédiction
if st.button("🔍 Lancer la prédiction"):
    # Normalisation
    input_scaled = scaler.transform(user_input)
    
    # Prédiction
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0]
    
    # Affichage des résultats
    st.subheader("📌 Résultats")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Prédiction", 
                 "🟢 Risque faible" if prediction == 0 else "🔴 Risque élevé",
                 f"{proba[1]*100:.1f}% de probabilité")
    
    with col2:
        st.write(f"**Probabilités :**")
        st.write(f"- Sain: {proba[0]*100:.1f}%")
        st.write(f"- Malade: {proba[1]*100:.1f}%")
    
    # Explication
    st.info("""
    Note : Ce modèle a une précision d'environ 85%. 
    Consultez toujours un médecin pour un diagnostic précis.
    """)

# Footer
st.markdown("---")
st.caption("Projet IA - Classification des maladies cardiaques | Dr Arthur Sawadogo")
