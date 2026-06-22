# Rapport - Système de recommandation de livres pour une bibliothèque universitaire

## 1. Introduction

Les bibliothèques universitaires possèdent souvent un grand nombre de livres, mais les étudiants ne savent pas toujours quels ouvrages choisir. Ce projet applique les techniques de Data Mining pour recommander automatiquement des livres adaptés aux préférences des étudiants.

L'objectif principal est de construire un système de recommandation basé sur les interactions utilisateur-livre, puis d'exploiter les résultats pour aider la bibliothèque à prendre de meilleures décisions.

## 2. Problématique

Comment recommander les bons livres aux bons étudiants à partir de leurs historiques de lecture et de notation ?

La bibliothèque veut répondre à plusieurs questions :

- Quels livres recommander à chaque étudiant ?
- Quels profils de lecteurs existent dans la bibliothèque ?
- Quels livres sont les plus populaires ?
- Comment utiliser ces résultats pour améliorer le stock et le service ?

## 3. Description des données

Le projet utilise le dataset Kaggle **Book Recommendation Dataset**. Les principales tables sont :

- `Books.csv` : informations sur les livres : ISBN, titre, auteur, année, éditeur.
- `Ratings.csv` : interactions entre utilisateurs et livres : User-ID, ISBN, Book-Rating.
- `Users.csv` : informations sur les utilisateurs : User-ID, localisation, âge.

Dans ce projet, les utilisateurs sont interprétés comme des étudiants de la bibliothèque universitaire.

## 4. Prétraitement

Les étapes de prétraitement réalisées sont :

1. Suppression des doublons.
2. Conversion de l'année de publication en valeur numérique.
3. Suppression ou correction des années incohérentes.
4. Filtrage des ratings explicites entre 1 et 10.
5. Sélection des utilisateurs actifs et livres populaires.
6. Création d'une matrice utilisateur-livre.

La matrice utilisateur-livre est essentielle pour le filtrage collaboratif. Les lignes représentent les étudiants, les colonnes représentent les livres, et les valeurs représentent les ratings.

## 5. Méthodes utilisées

### 5.1 Filtrage collaboratif User-Based

Le filtrage collaboratif User-Based recommande des livres à un étudiant en cherchant d'autres étudiants ayant des goûts similaires. La similarité entre utilisateurs est calculée avec la similarité cosinus.

### 5.2 Filtrage collaboratif Item-Based

Le filtrage collaboratif Item-Based recommande des livres similaires à un livre donné. Deux livres sont considérés proches s'ils sont aimés par les mêmes utilisateurs.

### 5.3 Clustering K-Means

K-Means est utilisé pour regrouper les étudiants selon leurs comportements de lecture. Chaque cluster représente un profil de lecteur.

## 6. Évaluation

Les métriques utilisées sont :

- **RMSE** : mesure l'erreur entre les ratings réels et les ratings prédits.
- **Precision@K** : mesure la proportion de livres pertinents parmi les K recommandations.
- **Hit Rate@K** : mesure la proportion d'utilisateurs pour lesquels au moins un livre pertinent est recommandé.
- **Silhouette Score** : mesure la qualité des clusters K-Means.

## 7. Résultats attendus

Après exécution du notebook, les résultats sont exportés dans :

- `outputs/tables/rmse_results.csv`
- `outputs/tables/precision_hit_rate.csv`
- `outputs/tables/user_clusters.csv`
- `outputs/tables/cluster_interpretation.csv`
- `outputs/tables/recommendations_example.csv`

Les figures sont exportées dans :

- `outputs/figures/ratings_distribution.png`
- `outputs/figures/top_books.png`
- `outputs/figures/rmse_comparison.png`
- `outputs/figures/cluster_distribution.png`

## 8. Interprétation DSS

Le système aide la bibliothèque à prendre des décisions concrètes :

1. Proposer des recommandations personnalisées aux étudiants.
2. Identifier les groupes de lecteurs avec K-Means.
3. Acheter davantage de livres correspondant aux clusters dominants.
4. Mettre en avant les livres populaires mais peu visibles.
5. Suivre la qualité des recommandations avec RMSE et Precision@K.

## 9. Conclusion

Ce projet montre comment le Data Mining peut être utilisé dans une bibliothèque universitaire pour passer d'une simple base de données à un système intelligent d'aide à la décision. Le filtrage collaboratif permet de recommander des livres, tandis que K-Means permet de comprendre les profils des étudiants. L'évaluation et les visualisations rendent le projet mesurable, explicable et utile pour la décision.