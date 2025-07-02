import os
import pandas as pd

def get_confiance_A1(discipline: str, historique_path="data/historique_predictions_complet.csv") -> float:
    """
    Calcule un score de confiance global A1 pour une discipline,
    basé sur le taux de réussite dans l'historique.
    """
    if not os.path.exists(historique_path):
        return None

    try:
        df_hist = pd.read_csv(historique_path)

        if "discipline" not in df_hist.columns or "is_A1_in_top4" not in df_hist.columns:
            return None

        df_filtered = df_hist[df_hist["discipline"] == discipline]

        if len(df_filtered) < 20:
            return None  # Pas assez de données fiables

        taux_reussite = df_filtered["is_A1_in_top4"].mean()
        return round(taux_reussite, 3)

    except Exception as e:
        print(f"❌ Erreur dans get_confiance_A1 : {e}")
        return None


def get_confiance_A1_par_distance(discipline: str, distance_class: str, historique_path="data/historique_predictions_complet.csv") -> float:
    """
    Calcule un score de confiance A1 pour une combinaison discipline + distance (courte/longue).
    Retourne une valeur entre 0 et 1, ou None si pas assez de données.
    """
    if not os.path.exists(historique_path):
        return None

    try:
        df_hist = pd.read_csv(historique_path)

        if "discipline" not in df_hist.columns or "distance" not in df_hist.columns or "is_A1_in_top4" not in df_hist.columns:
            return None

        # Sélection par distance
        if distance_class == "courte":
            df_hist = df_hist[df_hist["distance"] < 2200]
        elif distance_class == "longue":
            df_hist = df_hist[df_hist["distance"] >= 2200]
        else:
            return None

        # Sélection par discipline
        df_filtered = df_hist[df_hist["discipline"] == discipline]

        if len(df_filtered) < 20:
            return None  # Pas assez de cas pour juger

        taux_reussite = df_filtered["is_A1_in_top4"].mean()
        return round(taux_reussite, 3)

    except Exception as e:
        print(f"❌ Erreur dans get_confiance_A1_par_distance : {e}")
        return None
