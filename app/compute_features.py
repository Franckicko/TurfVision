import pandas as pd

def compute_ecart(series):
    """Calcule l'écart entre les 1 dans une série binaire."""
    ecart = []
    compteur = 0
    for val in series:
        if val == 1:
            ecart.append(compteur)
            compteur = 0
        else:
            compteur += 1
            ecart.append(compteur)
    return ecart

def calculer_features(df, include_target=True):
    pronos = [f'prono{i}' for i in range(1, 9)]

    # Sécurisation des colonnes manquantes
    for p in pronos:
        if p not in df.columns:
            df[p] = 0

    # Présence de A1 / A2 dans les pronos
    df['top3_1'] = df.apply(lambda x: int(x['a1'] in [x[p] for p in pronos[:3]]), axis=1) if 'a1' in df else 0
    df['top3_2'] = df.apply(lambda x: int(x['a2'] in [x[p] for p in pronos[:3]]), axis=1) if 'a2' in df else 0
    df['top4_1'] = df.apply(lambda x: int(x['a1'] in [x[p] for p in pronos[3:7]]), axis=1) if 'a1' in df else 0
    df['top4_2'] = df.apply(lambda x: int(x['a2'] in [x[p] for p in pronos[3:7]]), axis=1) if 'a2' in df else 0

    # Infos diverses sur A1/A2
    df['a1_imp'] = df['a1'] % 2 if 'a1' in df else 0
    df['a2_imp'] = df['a2'] % 2 if 'a2' in df else 0
    df['a1_inf9'] = (df['a1'] < 9).astype(int) if 'a1' in df else 0
    df['a2_inf9'] = (df['a2'] < 9).astype(int) if 'a2' in df else 0

    # Couples
    df['cplg_top3'] = df['top3_1'] + df['top3_2']
    df['cplg_top4'] = df['top4_1'] + df['top4_2']

    # Liens entre top3 et top4
    df['top3_1_top4'] = df.apply(
        lambda x: 2 if (x.get('prono1') in [x.get(p) for p in pronos[3:7]] and x.get('prono1') in [x.get('a1'), x.get('a2')])
        else 1 if x.get('prono1') in [x.get('a1'), x.get('a2')] else 0,
        axis=1
    )
    df['top3_2_top4'] = df.apply(lambda x: int(x.get('prono2') in [x.get('a1'), x.get('a2')]), axis=1)
    df['top3_3_top4'] = df.apply(lambda x: int(x.get('prono3') in [x.get('a1'), x.get('a2')]), axis=1)

    # Interactions < 9
    df['siprono1<9-A1<9'] = df.apply(lambda x: int(x.get('prono1', 99) < 9 and x.get('a1', 99) < 9), axis=1)
    df['siprono2<9-A1<9'] = df.apply(lambda x: int(x.get('prono2', 99) < 9 and x.get('a1', 99) < 9), axis=1)
    df['siprono3<9-A1<9'] = df.apply(lambda x: int(x.get('prono3', 99) < 9 and x.get('a1', 99) < 9), axis=1)

    # Rang des pronos associés à A1/A2
    df['prono_rank_a1'] = df.apply(lambda x: next((i+1 for i, p in enumerate(pronos) if x.get('a1') == x.get(p)), 9), axis=1)
    df['prono_rank_a2'] = df.apply(lambda x: next((i+1 for i, p in enumerate(pronos) if x.get('a2') == x.get(p)), 9), axis=1)

    # Groupe de distance
    if 'distance' in df:
        df['distance_group'] = df['distance'].apply(lambda x: 'longue' if x >= 2800 else 'courte')

    # Génération du cheval_num + cible
    if include_target:
        if 'cheval_num' not in df.columns and 'a1' in df.columns:
            df['cheval_num'] = df['a1']
        df['is_A1'] = (df['cheval_num'] == df['a1']).astype(int)
    else:
        df['is_A1'] = 0

    # ID course fallback
    if 'course_id' not in df.columns and {'date', 'hippodrome', 'numcourse'}.issubset(df.columns):
        df['course_id'] = df['date'].astype(str) + "_" + df['hippodrome'].astype(str) + "_" + df['numcourse'].astype(str)

    # Colonnes pour lesquelles on ajoute les écarts
    colonnes_base = [
        'top3_1', 'top3_2', 'top4_1', 'top4_2',
        'a1_imp', 'a2_imp', 'a1_inf9', 'a2_inf9',
        'cplg_top3', 'cplg_top4',
        'top3_1_top4', 'top3_2_top4', 'top3_3_top4'
    ]

    def _compute_group_ecart(group, col):
        flags = group[col].fillna(0) * group["is_A1"].fillna(0)
        return compute_ecart(flags)

    for col in colonnes_base:
        if include_target and 'course_id' in df.columns:
            df[f'{col}_ecart'] = (
                df.groupby('course_id', group_keys=False)
                .apply(lambda g: pd.Series(_compute_group_ecart(g, col), index=g.index))
                .reset_index(drop=True)
            )
        else:
            df[f'{col}_ecart'] = 0

    return df

def main():
    input_path = './data/Courses_CompletesTurfVision_id.csv'
    output_path = './data/chevaux_par_course.csv'

    df = pd.read_csv(input_path)
    print("\n📥 Colonnes chargées :", df.columns.tolist())

    # 🔁 Reconstruction : une ligne par cheval dans la course (pour entraînement)
    lignes = []
    for _, row in df.iterrows():
        for p in range(1, 9):  # prono1 à prono8
            cheval = row.get(f'prono{p}')
            if pd.isna(cheval):
                continue
            new_row = row.copy()
            new_row['cheval_num'] = cheval
            new_row['is_A1'] = int(cheval == row['a1'])
            lignes.append(new_row)

    df_all = pd.DataFrame(lignes)

    # Ajout des features sur toutes les lignes
    df_features = calculer_features(df_all, include_target=True)
    df_features.to_csv(output_path, index=False)

    print(f"✅ Fichier avec features sauvegardé : {output_path} ({len(df_features)} lignes)")

if __name__ == '__main__':
    main()
