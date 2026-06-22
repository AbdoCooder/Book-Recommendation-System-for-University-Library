# Système de recommandation de livres - Bibliothèque universitaire

Projet Data Mining + DSS : recommander des livres aux étudiants à partir des interactions utilisateur-livre.

## Objectifs

- Comprendre les données : livres, utilisateurs, ratings.
- Prétraiter les données.
- Construire un système de recommandation par filtrage collaboratif.
- Regrouper les étudiants avec K-Means.
- Évaluer les résultats avec RMSE, Precision@K et Hit Rate@K.
- Interpréter les résultats pour l'aide à la décision DSS.
- Générer visualisations, tables et recommandations.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

## Données

Mets les fichiers Kaggle dans `data/raw/` :

```text
data/raw/Books.csv
data/raw/Ratings.csv
data/raw/Users.csv
```

Si tu n'as pas encore Kaggle, le notebook utilise automatiquement `data/sample/` pour tester le code.

## Exécution

Ouvre :

```text
notebooks/book_recommendation_university_library.ipynb
```

Puis exécute les cellules dans l'ordre.

## Livrables inclus

- Notebook complet : `notebooks/book_recommendation_university_library.ipynb`
- Code Python modulaire : `src/`
- Rapport modèle : `reports/rapport_projet.md` et `reports/rapport_projet.pdf`
- Plan présentation : `reports/presentation_orale.md`
- Slides PowerPoint : `slides/presentation_orale.pptx`
- Données test : `data/sample/`

## Ce qu'il faut dire à l'oral

Ce projet ne se limite pas au code. Il transforme les historiques de lecture en décisions pour la bibliothèque : recommandations personnalisées, segmentation des étudiants et priorités d'achat de livres.