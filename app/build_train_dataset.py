import pandas as pd
import os

def build_train_dataset():
    input_path = "./data/chevaux_par_course.csv"
    output_path = "./data/train_chevaux.csv"

    if not os.path.exists(input_path):
        print(f"❌ Fichier introuvable : {input_path}")
        return

    # Chargement des données enrichies
    df = pd.read_csv(input_path)
    print(f"📥 Dataset chargé : {len(df)} lignes – {len(df.columns)} colonnes")

    # Vérification présence de la cible
    if 'is_A1' not in df.columns:
        print("❌ Colonne 'is_A1' manquante : le modèle ne pourra pas être entraîné.")
        return

    # Sauvegarde du fichier d'entraînement prêt à l'emploi
    df.to_csv(output_path, index=False)
    print(f"✅ Dataset d'entraînement généré : {output_path}")

if __name__ == "__main__":
    build_train_dataset()
