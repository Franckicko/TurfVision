from model.predictor import TurfPredictor

def main():
    print("📚 Chargement des données d'entraînement...")

    predictor = TurfPredictor(train_path="./data/train_chevaux.csv")

    try:
        print("🏋️‍♂️ Entraînement du modèle en cours...")
        predictor.train()
        predictor.save("./model/turf_model.pkl", "./model/turf_scaler.pkl")
        print("✅ Modèle entraîné et sauvegardé avec succès.")
    except ValueError as e:
        print(f"❌ Erreur d'entraînement : {e}")
    except Exception as e:
        print(f"❌ Erreur inattendue : {e}")

if __name__ == "__main__":
    main()
