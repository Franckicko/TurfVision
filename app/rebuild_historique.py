import pandas as pd
import os
import sys
import joblib

# ➕ Ajoute le dossier parent au chemin d'import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.predictor import TurfPredictor

def reconstruire_historique():
    source_path = "data/chevaux_par_course.csv"
    model_path = "model/checkpoints/model_a1.joblib"
    scaler_path = "model/checkpoints/scaler_a1.joblib"
    output_path = "data/historique_predictions.csv"

    if not os.path.exists(source_path):
        print("❌ Fichier source introuvable :", source_path)
        return

    try:
        # 📥 Chargement des données
        df = pd.read_csv(source_path)

        colonnes_min = ["id_course", "date", "discipline", "cheval_num", "a1"]
        for col in colonnes_min:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante : {col}")

        # 🔍 Chargement du modèle
        predictor = TurfPredictor()
        predictor.load(model_path, scaler_path)

        # ⚙️ Préparation des features pour la prédiction
        features = predictor.features
        df_pred = df.copy()
        for col in features:
            if col not in df_pred.columns:
                df_pred[col] = 0

        df_pred["proba_A1"] = predictor.model.predict_proba(
            predictor.scaler.transform(df_pred[features])
        )[:, 1]

        # 🧱 Construction de l'historique ligne par ligne
        lignes = []
        for _, row in df_pred.iterrows():
            cheval = row["cheval_num"]
            top4_arrives = [row.get(f"a{i}", None) for i in range(1, 5)]
            top4_arrives = [x for x in top4_arrives if pd.notna(x)]

            ligne = {
                "course_id": row["id_course"],
                "date": row["date"],
                "discipline": row["discipline"],
                "cheval_num": cheval,
                "true_A1": int(row["a1"]),
                "is_A1_in_top4": int(cheval in top4_arrives),
                "proba_A1": row["proba_A1"],
                "distance": row.get("distance", 0),
                "num_course": row.get("numcourse", ""),
                "hippodrome": row.get("hippodrome", "")
            }

            lignes.append(ligne)

        df_out = pd.DataFrame(lignes)
        df_out.to_csv(output_path, index=False)
        print(f"✅ Historique reconstruit avec prédictions : {output_path} ({len(df_out)} lignes)")

    except Exception as e:
        print("❌ Erreur :", e)

if __name__ == "__main__":
    reconstruire_historique()
