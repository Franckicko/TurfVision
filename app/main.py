import streamlit as st
import sys
import os

# 🔧 Ajout du dossier racine au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configuration générale
st.set_page_config(
    page_title="TurfVision",
    layout="centered",
    page_icon="🏇"
)

# Titre principal
st.title("🏇 TurfVision")

# --- Menu de navigation ---
page = st.sidebar.radio("📚 Menu", [
    "🔮 Prédire une course",
    "🏁 Ajouter les arrivées",
    "📊 Voir les statistiques",
    "🕒 Courses en attente"
])

# --- Routage ---
if page == "🔮 Prédire une course":
    try:
        from app.ui_predict import show_prediction_ui
        show_prediction_ui()
    except Exception as e:
        st.error(f"❌ Erreur dans la page de prédiction : {e}")

elif page == "🏁 Ajouter les arrivées":
    try:
        from app.ui_resultats import ajouter_resultats_ui
        ajouter_resultats_ui()
    except Exception as e:
        st.error(f"❌ Erreur dans la page d'ajout des résultats : {e}")

elif page == "📊 Voir les statistiques":
    try:
        from app.ui_stats import show_stats_ui
        show_stats_ui()
    except Exception as e:
        st.error(f"❌ Erreur dans la page des statistiques : {e}")

elif page == "🕒 Courses en attente":
    try:
        from app.ui_en_attente import afficher_courses_en_attente
        afficher_courses_en_attente()
    except Exception as e:
        st.error(f"❌ Erreur dans la page des courses en attente : {e}")
