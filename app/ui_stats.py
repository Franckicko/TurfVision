# app/ui_stats.py

import streamlit as st
import pandas as pd
from io import BytesIO

def show_stats_ui():
    st.title("📊 Statistiques du modèle TurfVision")

    try:
        df = pd.read_csv("data/historique_predictions_fusion.csv")
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du fichier : {e}")
        return

    if df.empty:
        st.warning("⚠️ Le fichier de données est vide.")
        return

    # 🎯 Filtres : discipline + distance
    col1, col2 = st.columns(2)

    with col1:
        disciplines = df["discipline"].dropna().unique().tolist()
        disciplines.sort()
        discipline_selection = st.selectbox("🎯 Discipline", ["(Toutes)"] + disciplines)

    with col2:
        distance_options = ["(Toutes)", "🏇 < 2800m", "🏁 ≥ 2800m"]
        distance_selection = st.selectbox("📏 Distance", distance_options)

    # Appliquer les filtres
    if discipline_selection != "(Toutes)":
        df = df[df["discipline"] == discipline_selection]

    if "distance_longue" in df.columns:
        if distance_selection == "🏇 < 2800m":
            df = df[df["distance_longue"] == 0]
        elif distance_selection == "🏁 ≥ 2800m":
            df = df[df["distance_longue"] == 1]

    # ✅ Statistiques principales
    st.subheader("📈 Taux de réussite")

    # 🔒 Statistiques désactivées temporairement
    st.info("Les statistiques de performance du modèle seront affichées ici dès qu'elles auront été validées.")

    # 📋 Dernières prédictions
    st.subheader("🕓 Dernières 10 prédictions")
    try:
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%y", errors="coerce")
        df_display = df.sort_values(by="date", ascending=False).head(10)
        df_display["date"] = df_display["date"].dt.strftime("%d/%m/%y")

        colonnes = ["course_id", "date", "discipline", "hippodrome", "numcourse",
                    "true_A1", "a1_inf9", "is_A1_in_top4", "cplg_top4", "rapport"]
        colonnes = [col for col in colonnes if col in df_display.columns]

        st.dataframe(df_display[colonnes], use_container_width=True)
    except Exception as e:
        st.warning(f"Erreur lors de l'affichage : {e}")

    # 📤 Export complet
    st.subheader("📥 Exporter les données")
    try:
        df_export = pd.read_csv("data/historique_predictions_fusion.csv")
        df_export["date"] = pd.to_datetime(df_export["date"], format="%d/%m/%y", errors="coerce")
        df_export = df_export.sort_values(by="date", ascending=False)
        df_export["date"] = df_export["date"].dt.strftime("%d/%m/%y")

        # Export CSV
        st.download_button(
            label="⬇️ Télécharger CSV",
            data=df_export.to_csv(index=False, sep=";").encode("utf-8"),
            file_name="stats_turfvision_clean.csv",
            mime="text/csv"
        )

        # Export Excel
        excel_io = BytesIO()
        df_export.to_excel(excel_io, index=False, engine='openpyxl')
        st.download_button(
            label="⬇️ Télécharger Excel",
            data=excel_io.getvalue(),
            file_name="stats_turfvision_clean.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception as e:
        st.error(f"❌ Erreur dans l’export : {e}")
