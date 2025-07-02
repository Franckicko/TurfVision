import pandas as pd
import os

def marquer_si_A1_dans_top4(df):
    """
    Pour chaque course, marque si le cheval 'true_A1' est dans le top 4 prédit (colonne 'cheval_num').
    """
    if 'true_A1' not in df.columns:
        print("❌ Colonne 'true_A1' manquante.")
        return df

    # On initialise la colonne
    df["is_A1_in_top4"] = 0

    # Pour chaque course
    for course_id in df["course_id"].unique():
        course_df = df[df["course_id"] == course_id]
        if course_df.empty:
            continue

        vrai_A1 = course_df["true_A1"].iloc[0]
        top4_chevaux = course_df.sort_values("proba_A1", ascending=False).head(4)["cheval_num"].tolist()

        # Marquer les lignes du top4 avec is_A1_in_top4 = 1 si cheval = true_A1
        df.loc[(df["course_id"] == course_id) & (df["cheval_num"] == vrai_A1), "is_A1_in_top4"] = int(vrai_A1 in top4_chevaux)

    return df

def main():
    chemin = "data/historique_predictions.csv"

    if not os.path.exists(chemin):
        print("❌ Le fichier historique_predictions.csv est introuvable.")
        return

    try:
        df = pd.read_csv(chemin)

        if "cheval_num" not in df.columns or "course_id" not in df.columns:
            print("❌ Colonnes nécessaires manquantes.")
            return

        df = marquer_si_A1_dans_top4(df)
        df.to_csv(chemin, index=False)
        print("✅ Fichier mis à jour avec is_A1_in_top4.")
    except Exception as e:
        print(f"❌ Erreur : {e}")

if __name__ == "__main__":
    main()
