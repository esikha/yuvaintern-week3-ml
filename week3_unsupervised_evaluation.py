"""
Week 3: Unsupervised Learning & Model Evaluation
YuvaIntern project — Binnala Esikha

Run:
    pip install -r requirements.txt
    python week3_unsupervised_evaluation.py

The included dataset is a small illustrative sample created for learning.
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_score, recall_score,
    f1_score, roc_auc_score, accuracy_score, silhouette_score
)

DATA_FILE = "student_learning_dataset.csv"

def main():
    data = pd.read_csv(DATA_FILE)
    features = ["Age", "Study_Hours_Per_Week", "Previous_Score"]
    X = data[features]
    y = data["High_Performance"]

    # Scale numerical features for clustering and distance-based methods.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means clustering and silhouette score.
    kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    print("K-Means silhouette score:", round(silhouette_score(X_scaled, kmeans_labels), 3))

    # Hierarchical (agglomerative) clustering.
    hierarchical = AgglomerativeClustering(n_clusters=2, linkage="ward")
    hierarchical_labels = hierarchical.fit_predict(X_scaled)
    print("Hierarchical clustering silhouette score:",
          round(silhouette_score(X_scaled, hierarchical_labels), 3))

    # PCA demonstrates dimensionality reduction from 3 features to 2.
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_.round(3))
    print("PCA output shape:", X_pca.shape)

    # Supervised evaluation example using the existing illustrative labels.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42))
    ])
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy")
    print("3-fold cross-validation accuracy:", cv_scores.round(3))
    print("Mean CV accuracy:", round(cv_scores.mean(), 3))

    # Small grid search for hyperparameter tuning.
    param_grid = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [None, 3, 5]
    }
    search = GridSearchCV(pipeline, param_grid, cv=3, scoring="f1", n_jobs=None)
    search.fit(X_train, y_train)
    best_model = search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]

    print("\nBest parameters:", search.best_params_)
    print("Accuracy:", round(accuracy_score(y_test, predictions), 3))
    print("Precision:", round(precision_score(y_test, predictions, zero_division=0), 3))
    print("Recall:", round(recall_score(y_test, predictions, zero_division=0), 3))
    print("F1-score:", round(f1_score(y_test, predictions, zero_division=0), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 3))
    print("Confusion matrix:\n", confusion_matrix(y_test, predictions))
    print("Classification report:\n", classification_report(y_test, predictions, zero_division=0))

if __name__ == "__main__":
    main()
