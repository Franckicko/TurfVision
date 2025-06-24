# model/predictor.py

import pandas as pd
import xgboost as xgb
import joblib
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class TurfPredictor:
    def __init__(self, train_path=None):
        self.train_path = train_path
        self.model = xgb.XGBClassifier(eval_metric='logloss', use_label_encoder=False)
        self.scaler = StandardScaler()
        self.features = [
            'prono_rank', 'top3_1', 'top3_2', 'top4_1', 'top4_2',
            'a1_imp', 'a2_imp', 'a1_inf9', 'a2_inf9',
            'cplg_top3', 'cplg_top4',
            'top3_1_top4', 'top3_2_top4', 'top3_3_top4',
            'siprono1<9-A1<9', 'siprono2<9-A1<9', 'siprono3<9-A1<9',
            'is_A1_ecart', 'is_A2_ecart',
            'top3_1_ecart', 'top3_2_ecart', 'top4_1_ecart', 'top4_2_ecart',
            'a1_imp_ecart', 'a2_imp_ecart', 'a1_inf9_ecart', 'a2_inf9_ecart',
            'cplg_top3_ecart', 'cplg_top4_ecart',
            'top3_1_top4_ecart', 'top3_2_top4_ecart', 'top3_3_top4_ecart'
        ]

    def load_training_data(self):
        df = pd.read_csv(self.train_path)

        if 'is_A1' not in df.columns:
            raise ValueError("❌ La colonne 'is_A1' est manquante dans le fichier d'entraînement.")

        X = df[self.features]
        y = df['is_A1']
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, _, y_train, _ = self.load_training_data()
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)

    def save(self, model_path: str, scaler_path: str):
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

    def load(self, model_path: str, scaler_path: str):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

    def predict_top4_A1(self, df_course: pd.DataFrame):
        X = self.scaler.transform(df_course[self.features])
        df_course = df_course.copy()
        df_course["proba_A1"] = self.model.predict_proba(X)[:, 1]
        return df_course.sort_values("proba_A1", ascending=False).head(4)

    def predict_couple_gagnant(self, df_course: pd.DataFrame):
        X = self.scaler.transform(df_course[self.features])
        df_course = df_course.copy()
        df_course["proba_A1"] = self.model.predict_proba(X)[:, 1]

        top_chevaux = df_course.sort_values("proba_A1", ascending=False).head(4)

        couples = []
        for c1, c2 in combinations(top_chevaux.itertuples(index=False), 2):
            couples.append({
                "cheval_1": c1.cheval_num,
                "cheval_2": c2.cheval_num,
                "proba_couple": round((c1.proba_A1 + c2.proba_A1) / 2, 5)
            })

        return pd.DataFrame(couples).sort_values("proba_couple", ascending=False).head(6)
