# app/ui_helpers.py

import pandas as pd

def load_hippodromes(filepath="data/Hippodromes.csv"):
    """Charge la liste des hippodromes depuis un CSV"""
    try:
        df = pd.read_csv(filepath, header=None)
        return sorted(df[0].dropna().unique().tolist())
    except Exception as e:
        print(f"Erreur lors du chargement des hippodromes : {e}")
        return []

def get_num_chevaux():
    """Retourne les numéros possibles pour les chevaux"""
    return list(range(1, 21))  # de 1 à 20

def get_num_course():
    """Retourne les identifiants de course (C1 à C10)"""
    return [f"C{i}" for i in range(1, 11)]

def get_disciplines():
    """Retourne la liste des disciplines disponibles"""
    return ["trot", "galop"]  # "obstacle" est mappé à "galop" dans le code principal
