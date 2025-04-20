import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

# Define parameter grid
param_grid = {
    'mlp__hidden_layer_sizes': [(50,), (20, 30, 40), (64, 64)],
    'mlp__activation': ['relu', 'tanh'],
    'mlp__solver': ['adam', 'sgd'],
    'mlp__alpha': [0.0001, 0.001],
    'mlp__learning_rate': ['constant', 'adaptive']
}

# Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
pkl_path = os.path.join(script_dir, "../AllModels")
log_path = os.path.join(script_dir, "../mlp_metrics_log.txt")
df = pd.read_csv(f"{script_dir}/../raw_skels/new_raw_co-ordinates1.csv")

# Data prep
X = df.iloc[:, 1:-1].values
Y = df["label"].values

label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Y)

# Define pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),
    ('mlp', MLPClassifier(max_iter=500))
])

# GridSearch with CV
print("grid search started")
grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, verbose=2, n_jobs=-1, return_train_score=False)
grid_search.fit(X, encoded_labels)
print("grid search ended")

# Create log file
os.makedirs(pkl_path, exist_ok=True)
with open(log_path, "w") as f:
    f.write("MLP Classifier Evaluation with Cross-Validation\n\n")

    for i, params in enumerate(grid_search.cv_results_["params"]):
        # Recreate model with these params
        pipeline.set_params(**params)
        y_pred = cross_val_predict(pipeline, X, encoded_labels, cv=5)

        acc = accuracy_score(encoded_labels, y_pred)
        prec = precision_score(encoded_labels, y_pred, average='weighted', zero_division=0)
        recall = recall_score(encoded_labels, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(encoded_labels, y_pred, average='weighted', zero_division=0)

        # Log the result
        f.write(f"Model {i+1} Params: {params}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {prec:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("-" * 50 + "\n")

# Save best model and encoder
joblib.dump(grid_search.best_estimator_, f"{pkl_path}/best_mlp_model.pkl")
joblib.dump(label_encoder, f"{pkl_path}/label_encoder.pkl")

print("All metrics logged to:", log_path)
