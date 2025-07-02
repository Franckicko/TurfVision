import streamlit as st
import pandas as pd
from datetime import datetime
import sys
import os

# Accès au dossier parent pour importer model/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.predictor import TurfPredictor
from app.confiance import get_confiance_A1, get_confiance_A1_par_distance


def show_prediction_ui():
    st.title("🌎 TurfVision – Prédiction du gagnant (A1)")
    st.markdown("Saisissez les informations de la course et les chevaux pronostiqués.")

    try:
        hippodrome_list = pd.read_csv("data/Hippodromes.csv", header=None)[0].sort_values().tolist()
    except Exception as e:
        st.error(f"❌ Erreur de chargement du fichier Hippodromes.csv : {e}")
        hippodrome_list = []

    date_input = st.date_input("🗕️ Date de la course")
    discipline_input = st.selectbox("Discipline", ["trot", "galop"])
    hippodrome = st.selectbox("🏟️ Hippodrome", options=hippodrome_list)
    num_course = st.selectbox("🆑️ Numéro de course", options=[f"C{i}" for i in range(1, 11)])
    distance = st.number_input("📏 Distance (en mètres)", min_value=800, max_value=5000, step=100)

    discipline = discipline_input.lower()
    distance_class = "longue" if distance >= 2200 else "courte"
    formatted_date = date_input.strftime("%d/%m/%y")
    course_id = f"{formatted_date}_{hippodrome}_{num_course}"
    st.write("🆔 ID généré :", course_id)

    attente_path = "data/courses_en_attente.csv"
    df_attente = pd.read_csv(attente_path) if os.path.exists(attente_path) else pd.DataFrame()

    if not df_attente.empty and course_id in df_attente["id_course"].values:
        st.warning("📌 Une course avec cet ID est déjà en attente.")
        if st.button("🪑 Vider pour saisir une nouvelle course"):
            df_attente = df_attente[df_attente["id_course"] != course_id]
            df_attente.to_csv(attente_path, index=False)
            st.success("✅ Course retirée. Vous pouvez maintenant en saisir une nouvelle.")
            st.experimental_rerun()
        return

    st.subheader("🏈 Chevaux pronostiqués")
    cols = st.columns(8)
    pronos = [cols[i].number_input(f"P{i+1}", min_value=1, max_value=20, step=1, key=f"prono_{i+1}") for i in range(8)]

    if st.button("🔮 Prédire le gagnant (A1)"):
        with st.spinner("📊 Entraînement du modèle..."):
            predictor = TurfPredictor()
            predictor.load("model/checkpoints/model_a1.joblib", "model/checkpoints/scaler_a1.joblib")

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
            for col in predictor.required_ecart_columns():
                df_course[col] = 0

            top4 = predictor.predict_top4_A1(df_course)
            if 'is_A1_ecart' in top4.columns:
                top4 = top4.sort_values(['proba_A1', 'is_A1_ecart'], ascending=[False, True])

            couples = predictor.predict_couple_gagnant(df_course)

        st.subheader("✅ Top 4 chevaux les plus probables A1 :")
        st.dataframe(top4[['cheval_num', 'proba_A1'] + [col for col in top4.columns if col.endswith("_ecart")]].reset_index(drop=True))

        st.subheader("🏆 6 couples gagnants les plus probables :")
        st.table(couples.rename(columns={
            "cheval_1": "Cheval 1",
            "cheval_2": "Cheval 2",
            "proba_couple": "Probabilité Moyenne"
        }).reset_index(drop=True))

        st.subheader("🔐 Score de confiance pour cette course")
        score_conf_crossed = get_confiance_A1_par_distance(discipline, distance_class)
        if score_conf_crossed is not None:
            st.success(f"{discipline.capitalize()} + {distance_class} → {int(score_conf_crossed * 100)} % de réussite A1 dans l'historique")
            if score_conf_crossed < 0.6:
                st.warning("⚠️ Prédiction peu fiable dans ce type de course.")
        else:
            st.info("Pas assez de données pour cette configuration.")

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

        couples['date'] = formatted_date
        couples['hippodrome'] = hippodrome
        couples['course_id'] = course_id
        couples['discipline'] = discipline
        couples['num_course'] = num_course
        couples['distance'] = distance

        couples_path = "data/historique_couples.csv"
        try:
            hist_couples = pd.read_csv(couples_path, sep=";")
            hist_couples = hist_couples[~(
                (hist_couples["course_id"] == course_id) &
                (hist_couples["date"] == formatted_date) &
                (hist_couples["discipline"] == discipline)
            )]
            hist_couples = pd.concat([hist_couples, couples], ignore_index=True)
        except Exception:
            hist_couples = couples

        hist_couples.to_csv(couples_path, index=False, sep=";")
        st.success("🦘️ Couples gagnants sauvegardés.")

        try:
            course_data = {
                "id_course": course_id,
                "date": formatted_date,
                "discipline": discipline,
                "hippodrome": hippodrome,
                "numcourse": num_course,
                "distance": distance,
                "prono1": pronos[0], "prono2": pronos[1], "prono3": pronos[2],
                "prono4": pronos[3], "prono5": pronos[4], "prono6": pronos[5],
                "prono7": pronos[6], "prono8": pronos[7],
            }

            if os.path.exists(attente_path):
                df_attente = pd.read_csv(attente_path)
                df_attente = df_attente[df_attente["id_course"] != course_id]
                df_attente = pd.concat([df_attente, pd.DataFrame([course_data])], ignore_index=True)
            else:
                df_attente = pd.DataFrame([course_data])

            df_attente.to_csv(attente_path, index=False)
            st.success("📌 Course enregistrée dans la file d'attente (provisoire).")
        except Exception as e:
            st.warning(f"⚠️ Impossible d’enregistrer dans courses_en_attente.csv : {e}")

        st.subheader("📊 Score global de confiance (discipline uniquement)")
        confiance = get_confiance_A1(discipline)
        if confiance is not None:
            st.info(f"Confiance globale pour {discipline.upper()} : **{int(confiance * 100)}%**")
        else:
            st.warning("Pas assez de données pour établir un score de confiance global.")
