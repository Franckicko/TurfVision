import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="📊 Évaluation TurfVision")
st.title("📈 Évaluation des performances du modèle")

# ------------------------------
# 📁 Chargement de l'historique
# ------------------------------
historique_path = "data/historique_predictions.csv"
historique_complet_path = "data/historique_predictions_complet.csv"

try:
    df = pd.read_csv(historique_path)
    st.success(f"✅ Fichier chargé avec succès ({len(df)} lignes)")

    # Calculer is_A1_in_top4 si nécessaire
    if "is_A1_in_top4" not in df.columns and "true_A1" in df.columns:
        df["is_A1_in_top4"] = (df["cheval_num"] == df["true_A1"]).astype(int)

    # ------------------------------
    # 📊 Taux de réussite par discipline
    # ------------------------------
    if "discipline" in df.columns:
        st.markdown("### 📊 Taux de réussite par discipline")
        stats = df.groupby("discipline")["is_A1_in_top4"].agg(["mean", "count"])
        stats["% de réussite"] = stats["mean"] * 100
        stats = stats.rename(columns={"count": "Nb de courses"})
        st.dataframe(stats[["% de réussite", "Nb de courses"]].round(2))
    else:
        st.info("ℹ️ Colonne 'discipline' manquante pour les statistiques.")

    # ------------------------------
    # ❌ Dernières erreurs de prédiction
    # ------------------------------
    st.markdown("### ❌ Dernières erreurs de prédiction (A1 non présent dans top 4)")
    if "is_A1_in_top4" in df.columns:
        df_echecs = df[df["is_A1_in_top4"] == 0].sort_values(by="date", ascending=False)
        if df_echecs.empty:
            st.success("Aucune erreur récente dans les prédictions 👌")
        else:
            st.dataframe(df_echecs[["course_id", "date", "cheval_num", "true_A1", "discipline"]])
    else:
        st.warning("⚠️ Colonne 'is_A1_in_top4' non disponible.")

except Exception as e:
    st.error(f"❌ Erreur lors de la lecture du fichier : {e}")
    df = None


# ------------------------------
# 🔧 Fonction : Score discipline + distance
# ------------------------------
def get_confiance_A1_par_distance(discipline: str, distance_class: str, historique_path=historique_complet_path) -> float:
    if not os.path.exists(historique_path):
        return None

    try:
        df_hist = pd.read_csv(historique_path)

        if "discipline" not in df_hist.columns or "is_A1_in_top4" not in df_hist.columns or "distance" not in df_hist.columns:
            return None

        # Filtrer selon la classe de distance
        if distance_class == "courte":
            df_hist = df_hist[df_hist["distance"] < 2200]
        elif distance_class == "longue":
            df_hist = df_hist[df_hist["distance"] >= 2200]
        else:
            return None

        df_filtered = df_hist[df_hist["discipline"] == discipline]
        if len(df_filtered) < 20:
            return None

        taux_reussite = df_filtered["is_A1_in_top4"].mean()
        return round(taux_reussite, 3)

    except Exception as e:
        print(f"Erreur dans get_confiance_A1_par_distance : {e}")
        return None


# ------------------------------
# 🔍 Score de confiance par discipline et distance
# ------------------------------
st.markdown("### 🤖 Score de confiance par discipline × distance")
if isinstance(df, pd.DataFrame) and "discipline" in df.columns and "distance" in df.columns:
    for discipline in df["discipline"].dropna().unique():
        for classe in ["courte", "longue"]:
            taux = get_confiance_A1_par_distance(discipline, classe)
            if taux is not None:
                st.markdown(f"- **{discipline.capitalize()} + {classe}** → {int(taux * 100)} %")
            else:
                st.markdown(f"- **{discipline.capitalize()} + {classe}** → Données insuffisantes")
else:
    st.warning("⚠️ Colonnes 'discipline' et/ou 'distance' manquantes pour afficher les scores de confiance.")
