# app/main.py

import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Permet d'importer le dossier parent (TurfVision) pour accéder à model/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.predictor import TurfPredictor

st.set_page_config(page_title="TurfVision", layout="centered")
st.title("🌎 TurfVision – Prédiction du gagnant (A1)")
st.markdown("Saisissez les informations de la course et les chevaux pronostiqués.")

# --- Chargement des hippodromes ---
try:
    hippodrome_list = pd.read_csv("data/Hippodromes.csv", header=None)[0].sort_values().tolist()
except Exception as e:
    st.error(f"❌ Erreur de chargement du fichier Hippodromes.csv : {e}")
    hippodrome_list = []

# --- Formulaire d'entrée ---
date_input = st.date_input("📅 Date de la course")
discipline_input = st.selectbox("Discipline", ["trot", "galop", "obstacle"])
hippodrome = st.selectbox("🏟️ Hippodrome", options=hippodrome_list)
num_course = st.selectbox("🆕 Numéro de course", options=[f"C{i}" for i in range(1, 11)])

# Formatage de la date JJ/MM/AA
formatted_date = date_input.strftime("%d/%m/%y")
course_id = f"{formatted_date}_{hippodrome}_{num_course}"
discipline = "trot" if discipline_input == "trot" else "galop"

st.write("🆔 ID de course généré :", course_id)
st.write("🏇 Discipline utilisée :", discipline)
st.write("📆 Date formatée :", formatted_date)

# --- Saisie des chevaux pronostiqués ---
st.subheader("🐴 Chevaux pronostiqués")
pronos = []
for i in range(1, 9):
    prono = st.number_input(f"Prono {i}", min_value=1, max_value=20, step=1, key=f"prono_{i}")
    pronos.append(prono)

# --- Lancement de la prédiction ---
if st.button("🔮 Prédire le gagnant (A1)"):
    with st.spinner("📊 Entraînement du modèle..."):
        predictor = TurfPredictor("data/chevaux_par_course.csv")
        predictor.train()

        input_rows = []
        for i, cheval in enumerate(pronos):
            row = {
                'course_id': course_id,
                'cheval_num': cheval,
                'prono_rank': i + 1,
                'top3_1': 0,
                'top3_2': 0,
                'top4_1': 0,
                'top4_2': 0,
                'a1_imp': 0,
                'a2_imp': 0,
                'a1_inf9': int(cheval < 9),
                'a2_inf9': 0,
                'cplg_top3': 0,
                'cplg_top4': 0,
                'top3_1_top4': 0,
                'top3_2_top4': 0,
                'top3_3_top4': 0,
                'siprono1<9-A1<9': int(i == 0 and cheval < 9),
                'siprono2<9-A1<9': int(i == 1 and cheval < 9),
                'siprono3<9-A1<9': int(i == 2 and cheval < 9),
                'is_A1': 0
            }
            input_rows.append(row)

        df_course = pd.DataFrame(input_rows)
        top4 = predictor.predict_top4_A1(df_course)
        couples = predictor.predict_couple_gagnant(df_course)

    # --- Affichage résultats ---
    st.subheader("✅ Top 4 chevaux les plus probables A1 :")
    st.dataframe(top4[['cheval_num', 'proba_A1']].reset_index(drop=True))

    st.subheader("🏆 6 couples gagnants les plus probables :")
    st.table(couples.rename(columns={
        "cheval_1": "Cheval 1",
        "cheval_2": "Cheval 2",
        "proba_couple": "Probabilité Moyenne"
    }).reset_index(drop=True))

    # --- Sauvegarde prédictions ---
    historique_path = "data/historique_predictions.csv"
    top4 = top4.copy()
    top4['date'] = formatted_date
    top4['hippodrome'] = hippodrome
    top4['course_id'] = course_id
    top4['discipline'] = discipline
    top4['num_course'] = num_course

    try:
        historique = pd.read_csv(historique_path)
        historique = historique[~(
            (historique["course_id"] == course_id) &
            (historique["date"] == formatted_date) &
            (historique["discipline"] == discipline)
        )]
        historique = pd.concat([historique, top4], ignore_index=True)
    except FileNotFoundError:
        historique = top4

    historique.to_csv(historique_path, index=False)
    st.success("📁 Prédiction sauvegardée dans l'historique.")

    # --- Sauvegarde des couples ---
    couples['date'] = formatted_date
    couples['hippodrome'] = hippodrome
    couples['course_id'] = course_id
    couples['discipline'] = discipline
    couples['num_course'] = num_course

    historique_couples_path = "data/historique_couples.csv"
    try:
        hist_couples = pd.read_csv(historique_couples_path)
        hist_couples = hist_couples[~(
            (hist_couples["course_id"] == course_id) &
            (hist_couples["date"] == formatted_date) &
            (hist_couples["discipline"] == discipline)
        )]
        hist_couples = pd.concat([hist_couples, couples], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        hist_couples = couples

    hist_couples.to_csv(historique_couples_path, index=False)
    st.success("📝 Couples gagnants sauvegardés dans l'historique.")
