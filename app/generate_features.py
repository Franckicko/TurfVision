import pandas as pd
import os

# Fichier source des courses complètes
SOURCE_PATH = "data/Courses_CompletesTurfVision_id.csv"
# Fichier de sortie enrichi
OUTPUT_PATH = "data/historique_predictions_fusion.csv"


def charger_donnees():
    if not os.path.exists(SOURCE_PATH):
        raise FileNotFoundError("❌ Fichier source introuvable.")
    return pd.read_csv(SOURCE_PATH)


def evaluer_champs_calcules(row):
    """Calcule les features manuelles à partir des pronostics et arrivées."""
    top3 = [row.get(f"prono{i}") for i in range(1, 4) if pd.notna(row.get(f"prono{i}"))]
    top4 = [row.get(f"prono{i}") for i in range(1, 5) if pd.notna(row.get(f"prono{i}"))]
    top8 = [row.get(f"prono{i}") for i in range(1, 9) if pd.notna(row.get(f"prono{i}"))]

    a1, a2 = row.get("a1"), row.get("a2")

    features = {
        "top3_1": int(a1 in top3) if pd.notna(a1) else 0,
        "top3_2": int(a2 in top3) if pd.notna(a2) else 0,
        "top4_1": int(a1 in top4) if pd.notna(a1) else 0,
        "top4_2": int(a2 in top4) if pd.notna(a2) else 0,
        "a1_imp": int(a1 % 2 == 1) if pd.notna(a1) else 0,
        "a2_imp": int(a2 % 2 == 1) if pd.notna(a2) else 0,
        "a1_inf9": int(a1 < 9) if pd.notna(a1) else 0,
        "a2_inf9": int(a2 < 9) if pd.notna(a2) else 0,
        "cplg_top3": int((a1 in top3) + (a2 in top3)) if pd.notna(a1) and pd.notna(a2) else 0,
        "cplg_top4": int((a1 in top4) + (a2 in top4)) if pd.notna(a1) and pd.notna(a2) else 0,
        "top3_1_top4": int(row.get("prono1") in top4 and row.get("prono1") in [a1, a2]),
        "top3_2_top4": int(row.get("prono2") in top4 and row.get("prono2") in [a1, a2]),
        "top3_3_top4": int(row.get("prono3") in top4 and row.get("prono3") in [a1, a2]),
        "siprono1<9-A1<9": int(row.get("prono1", 99) < 9 and a1 < 9) if pd.notna(a1) else 0,
        "siprono2<9-A1<9": int(row.get("prono2", 99) < 9 and a1 < 9) if pd.notna(a1) else 0,
        "siprono3<9-A1<9": int(row.get("prono3", 99) < 9 and a1 < 9) if pd.notna(a1) else 0,
        "distance_longue": int(row.get("distance") >= 2800) if pd.notna(row.get("distance")) else 0,
        "top3_pred": str(top3),
        "top4_pred": str(top8),
    }

    try:
        features["prono_rank_a1"] = next((i for i in range(1, 9) if row.get(f"prono{i}") == a1), None)
        features["prono_rank_a2"] = next((i for i in range(1, 9) if row.get(f"prono{i}") == a2), None)
    except:
        features["prono_rank_a1"] = None
        features["prono_rank_a2"] = None

    return pd.Series(features)


def generer_features():
    df = charger_donnees()

    # Conserver uniquement les lignes avec A1 et A2 connus
    df = df.dropna(subset=["a1", "a2"], how="any")

    # Appliquer les features calculées ligne par ligne
    df_feats = df.apply(evaluer_champs_calcules, axis=1)

    # Fusion avec les données initiales
    df_final = pd.concat([df.reset_index(drop=True), df_feats], axis=1)

    # Export final
    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"✅ Données enrichies sauvegardées dans {OUTPUT_PATH} ({len(df_final)} lignes)")


if __name__ == "__main__":
    generer_features()
