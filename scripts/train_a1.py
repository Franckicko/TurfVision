# scripts/train_a1.py

import pandas as pd
from model.predictor import TurfPredictor

# 🔁 1. Charger les données
DATA_PATH = "data/chevaux_par_course.csv"
df = pd.read_csv(DATA_PATH)

# ✅ Vérification minimale
if "course_id" not in df.columns or "cheval_num" not in df.columns:
    raise ValueError("❌ Le fichier doit contenir les colonnes 'course_id' et 'cheval_num'.")

# ✅ Normalisation : transformer 'obstacle' en 'galop' si la colonne existe
if "discipline" in df.columns:
    df["discipline"] = df["discipline"].replace({"obstacle": "galop"})

# 🔧 2. Initialiser le modèle
predictor = TurfPredictor(DATA_PATH)

# 📚 3. Entraîner sur toutes les données
predictor.train()

# 🏇 4. Sélectionner une course pour la prédiction (ex : la première course)
first_course_id = df['course_id'].iloc[0]
df_course = df[df['course_id'] == first_course_id]

# 🔮 5. Prédire les 4 chevaux les plus probables comme gagnant A1
top4 = predictor.predict_top4_A1(df_course)

# 📤 6. Afficher les résultats
print(f"\n🎯 Top 4 des chevaux prédits comme A1 pour la course {first_course_id} :")
if 'proba_A1' in top4.columns:
    print(top4[['cheval_num', 'proba_A1']].reset_index(drop=True))
else:
    print(top4[['cheval_num']].reset_index(drop=True))

# 🏆 7. Afficher le couple gagnant
couple = predictor.predict_couple_gagnant(df_course)
print(f"\n👥 6 couples gagnants prédits pour la course {first_course_id} :")
print(couple.rename(columns={
    "cheval_1": "Cheval 1",
    "cheval_2": "Cheval 2",
    "proba_couple": "Probabilité"
}).reset_index(drop=True))
