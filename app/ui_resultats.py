import streamlit as st
import pandas as pd
import os
from app.rebuild_historique import reconstruire_historique


def ajouter_resultats_ui():
    st.title("📥 Ajouter les arrivées officielles")

    attente_path = "data/courses_en_attente.csv"
    complet_path = "data/Courses_CompletesTurfVision_id.csv"
    fusion_path = "data/historique_predictions_fusion.csv"

    if not os.path.exists(attente_path):
        pd.DataFrame(columns=[
            "id_course", "date", "discipline", "hippodrome", "numcourse", "distance",
            "prono1", "prono2", "prono3", "prono4", "prono5", "prono6", "prono7", "prono8"
        ]).to_csv(attente_path, index=False)
        st.info("✅ Fichier 'courses_en_attente.csv' créé (vide).")
        return

    try:
        df = pd.read_csv(attente_path)
    except Exception as e:
        st.error(f"❌ Erreur lors de la lecture : {e}")
        return

    if df.empty or "id_course" not in df.columns:
        st.warning("Aucune course en attente à valider.")
        return

    course_ids = df["id_course"].unique()
    id_course = st.selectbox("📌 Choisissez une course à compléter", sorted(course_ids, reverse=True))

    st.markdown("### 🐎 Arrivées officielles")
    cols = st.columns(5)
    a1 = cols[0].number_input("A1", min_value=1, max_value=20, step=1, key="a1")
    a2 = cols[1].number_input("A2", min_value=1, max_value=20, step=1, key="a2")
    a3 = cols[2].number_input("A3", min_value=1, max_value=20, step=1, key="a3")
    a4 = cols[3].number_input("A4", min_value=1, max_value=20, step=1, key="a4")
    a5 = cols[4].number_input("A5", min_value=1, max_value=20, step=1, key="a5")

    st.markdown("### 💶 Rapport (facultatif)")
    rapport_a1 = st.text_input("Rapport de A1 (€)", value="")

    if st.button("💾 Enregistrer les résultats"):
        try:
            if len({a1, a2, a3, a4, a5}) < 5:
                st.warning("⚠️ Les arrivées doivent être toutes différentes.")
                return

            if os.path.exists(complet_path):
                df_final = pd.read_csv(complet_path)
            else:
                df_final = pd.DataFrame()

            ligne = df[df["id_course"] == id_course].iloc[0].to_dict()
            ligne.update({
                "a1": int(a1), "a2": int(a2), "a3": int(a3),
                "a4": int(a4), "a5": int(a5),
                "rapport": float(rapport_a1.replace(",", ".")) if rapport_a1 else None
            })

            df_final = df_final[df_final["id_course"] != id_course]
            df_final = pd.concat([df_final, pd.DataFrame([ligne])], ignore_index=True)
            df_final.to_csv(complet_path, index=False)
            st.success("✅ Résultats enregistrés avec succès.")

            # 🔁 Mise à jour automatique historique_predictions_fusion.csv
            if os.path.exists(fusion_path):
                df_fusion = pd.read_csv(fusion_path)
                for col, val in zip(["true_A1", "true_A2", "true_A3", "true_A4", "true_A5", "rapport_A1"],
                                    [a1, a2, a3, a4, a5, rapport_a1]):
                    if col in df_fusion.columns:
                        df_fusion.loc[df_fusion["course_id"] == id_course, col] = val
                df_fusion.to_csv(fusion_path, index=False)

            # ✅ Mise à jour de l’historique complet
            reconstruire_historique()

            df = df[df["id_course"] != id_course]
            df.to_csv(attente_path, index=False)
            st.info("📤 Course retirée de la file d'attente.")

        except Exception as e:
            st.error(f"❌ Erreur lors de l’enregistrement : {e}")

    st.markdown("### 📜 Dernières courses complètes")
    if os.path.exists(complet_path):
        try:
            df_recent = pd.read_csv(complet_path)
            if not df_recent.empty:
                if "date" in df_recent.columns:
                    df_recent["date"] = pd.to_datetime(df_recent["date"], format="%d/%m/%y", errors="coerce")

                df_recent = df_recent.sort_values(by="date", ascending=False).head(10)
                colonnes = ["id_course", "date", "discipline", "a1", "a2", "a3", "a4", "a5", "rapport"]
                df_recent = df_recent[[col for col in colonnes if col in df_recent.columns]]
                df_recent["date"] = df_recent["date"].dt.strftime("%d/%m/%y")

                st.dataframe(df_recent, use_container_width=True)
        except Exception as e:
            st.warning(f"⚠️ Impossible d’afficher les dernières courses : {e}")

