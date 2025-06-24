import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Pour importer le dossier parent et accéder à model/
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

# --- Formulaire d'entrée utilisateur ---
date_input = st.date_input("📅 Date de la course")
discipline_input = st.selectbox("Discipline", ["trot", "galop", "obstacle"])
hippodrome = st.selectbox("🏟️ Hippodrome", options=hippodrome_list)
num_course = st.selectbox("🆕 Numéro de course", options=[f"C{i}" for i in range(1, 11)])
distance = st.number_input("📏 Distance de la course (en mètres)", min_value=800, max_value=5000, step=100)

# Formatage
discipline = discipline_input.lower()
formatted_date = date_input.strftime("%d/%m/%y")
course_id = f"{formatted_date}_{hippodrome}_{num_course}"

st.write("🆔 ID de course généré :", course_id)
st.write("🏇 Discipline utilisée :", discipline)
st.write("📆 Date formatée :", formatted_date)

# --- Saisie des chevaux ---
st.subheader("🏈 Chevaux pronostiqués")
pronos = [st.number_input(f"Prono {i}", min_value=1, max_value=20, step=1, key=f"prono_{i}") for i in range(1, 9)]

# --- Prédiction ---
if st.button("🔮 Prédire le gagnant (A1)"):
    with st.spinner("📊 Entraînement du modèle..."):
        predictor = TurfPredictor("data/chevaux_par_course.csv")
        predictor.train()

        input_rows = []
        for i, cheval in enumerate(pronos):
            input_rows.append({
                'course_id': course_id,
                'cheval_num': cheval,
                'prono_rank': i + 1,
                'top3_1': 0, 'top3_2': 0, 'top4_1': 0, 'top4_2': 0,
                'a1_imp': 0, 'a2_imp': 0,
                'a1_inf9': int(cheval < 9), 'a2_inf9': 0,
                'cplg_top3': 0, 'cplg_top4': 0,
                'top3_1_top4': 0, 'top3_2_top4': 0, 'top3_3_top4': 0,
                'siprono1<9-A1<9': int(i == 0 and cheval < 9),
                'siprono2<9-A1<9': int(i == 1 and cheval < 9),
                'siprono3<9-A1<9': int(i == 2 and cheval < 9),
                'is_A1': 0
            })

        df_course = pd.DataFrame(input_rows)

        # Colonnes _ecart par défaut
        ecart_cols = [
            'is_A1_ecart', 'is_A2_ecart', 'top3_1_ecart', 'top3_2_ecart',
            'top4_1_ecart', 'top4_2_ecart', 'a1_imp_ecart', 'a2_imp_ecart',
            'a1_inf9_ecart', 'a2_inf9_ecart', 'cplg_top3_ecart', 'cplg_top4_ecart',
            'top3_1_top4_ecart', 'top3_2_top4_ecart', 'top3_3_top4_ecart'
        ]
        for col in ecart_cols:
            df_course[col] = 0

        top4 = predictor.predict_top4_A1(df_course)

        if 'is_A1_ecart' in top4.columns:
            top4 = top4.sort_values(['proba_A1', 'is_A1_ecart'], ascending=[False, True])

        couples = predictor.predict_couple_gagnant(df_course)

    # --- Résultats ---
    st.subheader("✅ Top 4 chevaux les plus probables A1 :")
    st.dataframe(top4[['cheval_num', 'proba_A1'] + [col for col in top4.columns if col.endswith("_ecart")]].reset_index(drop=True))

    st.subheader("🏆 6 couples gagnants les plus probables :")
    st.table(couples.rename(columns={
        "cheval_1": "Cheval 1",
        "cheval_2": "Cheval 2",
        "proba_couple": "Probabilité Moyenne"
    }).reset_index(drop=True))

    # --- Historique Top 4 ---
    top4['date'] = formatted_date
    top4['hippodrome'] = hippodrome
    top4['course_id'] = course_id
    top4['discipline'] = discipline
    top4['num_course'] = num_course
    top4['distance'] = distance

    histo_path = "data/historique_predictions.csv"
    try:
        historique = pd.read_csv(histo_path)
        historique = historique[~(
            (historique["course_id"] == course_id) &
            (historique["date"] == formatted_date) &
            (historique["discipline"] == discipline)
        )]
        historique = pd.concat([historique, top4], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        historique = top4

    historique.to_csv(histo_path, index=False)
    st.success("📁 Prédiction A1 sauvegardée.")

    # --- Historique Couples ---
    couples['date'] = formatted_date
    couples['hippodrome'] = hippodrome
    couples['course_id'] = course_id
    couples['discipline'] = discipline
    couples['num_course'] = num_course
    couples['distance'] = distance

    couples_path = "data/historique_couples.csv"
    try:
        hist_couples = pd.read_csv(couples_path)
        hist_couples = hist_couples[~(
            (hist_couples["course_id"] == course_id) &
            (hist_couples["date"] == formatted_date) &
            (hist_couples["discipline"] == discipline)
        )]
        hist_couples = pd.concat([hist_couples, couples], ignore_index=True)
    except Exception:
        hist_couples = couples

    hist_couples.to_csv(couples_path, index=False, sep=";")
    st.success("📝 Couples gagnants sauvegardés.")

    # --- Taux de réussite ---
    st.subheader("📈 Taux de réussite A1 dans les historiques")
    try:
        historique_all = pd.read_csv(histo_path)
        if 'is_A1' in historique_all.columns and 'cheval_num' in historique_all.columns:
            historique_all['is_A1_in_top4'] = (historique_all['is_A1'] == 1)
            taux = historique_all['is_A1_in_top4'].mean() * 100
            st.markdown(f"✅ **Taux global : {taux:.2f}%**")

            if "discipline" in historique_all.columns:
                st.markdown("### 📊 Taux par discipline :")
                stats = historique_all.groupby("discipline")["is_A1_in_top4"].mean() * 100
                for d, val in stats.round(2).items():
                    st.markdown(f"- {d.capitalize()} : **{val:.2f}%**")
        else:
            st.warning("ℹ️ Les colonnes nécessaires au calcul du taux de réussite ne sont pas présentes.")
    except Exception as e:
        st.warning(f"❌ Impossible de calculer le taux de réussite : {e}")