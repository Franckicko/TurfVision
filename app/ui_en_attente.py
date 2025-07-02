# app/ui_en_attente.py

import streamlit as st
import pandas as pd
import os

def afficher_courses_en_attente():
    st.title("🕒 Courses en attente de validation")

    attente_path = "data/courses_en_attente.csv"

    if not os.path.exists(attente_path):
        st.info("✅ Aucune course en attente.")
        return

    try:
        df = pd.read_csv(attente_path)
    except Exception as e:
        st.error(f"❌ Erreur de lecture du fichier : {e}")
        return

    if df.empty:
        st.success("✅ Toutes les courses ont été complétées.")
        return

    # Sélection de la course
    course_ids = df["id_course"].tolist()
    id_course = st.selectbox("📌 Sélectionnez une course en attente :", course_ids)

    selected_row = df[df["id_course"] == id_course]
    if selected_row.empty:
        st.warning("⚠️ Course introuvable.")
        return

    course_info = selected_row.iloc[0]

    # 🧩 Ligne 1 - Informations principales
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.text_input("📅 Date", course_info.get("date", ""), disabled=True)
        col2.text_input("🏟️ Hippodrome", course_info.get("hippodrome", ""), disabled=True)
        col3.text_input("🏇 Discipline", course_info.get("discipline", ""), disabled=True)
        col4.text_input("📍 Numéro de course", course_info.get("numcourse", ""), disabled=True)

    # 🧩 Ligne 2 - Compléments
    with st.container():
        st.text_input("📏 Distance", course_info.get("distance", ""), disabled=True)

    # 🧩 Ligne 3 - Pronostics (horizontal)
    with st.container():
        st.markdown("#### 🧠 Pronostics")
        pronos = [course_info.get(f"prono{i+1}", "") for i in range(8)]
        cols = st.columns(8)
        for i in range(8):
            cols[i].text_input(f"P{i+1}", pronos[i], disabled=True)

    # 🔄 Choix de l'action : reconstruction ?
    reconstruire_historique = False
    if st.checkbox("🔄 Recalculer automatiquement l’historique après validation"):
        reconstruire_historique = True

    # 🧩 Ligne 4 - Boutons d'action
    colg, cold = st.columns(2)
    with colg:
        if st.button("➕ Nouvelle course (vider l'écran)"):
            st.success("✅ Formulaire réinitialisé.")
            st.rerun()

    with cold:
        if st.button("🗑️ Supprimer cette course"):
            if "a1" in course_info and not pd.isna(course_info["a1"]):
                st.warning("❌ Cette course est déjà complétée et ne peut pas être supprimée.")
            else:
                df = df[df["id_course"] != id_course]
                df.to_csv(attente_path, index=False)
                st.success("✅ Course supprimée de la file d’attente.")
                st.rerun()

    # 🔘 Bouton d’enregistrement des résultats (simulation d’arrivée, etc.)
    if st.button("💾 Enregistrer les résultats"):
        try:
            # Simule enregistrement (remplacer par ta vraie logique d’arrivée)
            st.success("✅ Résultats enregistrés avec succès.")

            # Si demandé, on lance la reconstruction
            if reconstruire_historique:
                os.system("python run_rebuild.py")
                st.success("🔄 Historique reconstruit automatiquement.")
        except Exception as e:
            st.error(f"❌ Erreur lors de l’enregistrement : {e}")

    # 🧾 Tableau global des courses en attente
    st.markdown("### 📋 Autres courses en attente")
    st.dataframe(df, use_container_width=True)
