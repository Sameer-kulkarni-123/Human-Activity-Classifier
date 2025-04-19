import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import os

# Get the script's directory and load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(f"{script_dir}/../raw_skels/new_raw_co-ordinates1.csv")

# Extract features and labels
X = df.iloc[:, 1:-1].values
Y = df["label"].values

# Preprocessing
label_encoder = LabelEncoder()
standard_scaler = StandardScaler()
pca = PCA(n_components=0.95)

# Transformations
encoded_labels = label_encoder.fit_transform(Y)
standardized_X = standard_scaler.fit_transform(X)
reduced_X = pca.fit_transform(standardized_X)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    reduced_X, encoded_labels, test_size=0.2, stratify=encoded_labels, random_state=42
)

# Define model
mlp = MLPClassifier(max_iter=500, verbose=True, random_state=42)

# Grid search parameter space
param_grid = {
    'hidden_layer_sizes': [(50,), (20, 30, 40), (64, 64)],
    'activation': ['relu', 'tanh'],
    'solver': ['adam', 'sgd'],
    'alpha': [0.0001, 0.001],
    'learning_rate': ['constant', 'adaptive']
}

# Cross-validation strategy
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# GridSearchCV setup
grid_search = GridSearchCV(estimator=mlp,
                           param_grid=param_grid,
                           cv=cv,
                           scoring='accuracy',
                           verbose=2,
                           n_jobs=-1)

# Run grid search (this doesn't train a final model, just searches)
grid_search.fit(X_train, Y_train)

# Output best parameters
print("Best Parameters Found:\n", grid_search.best_params_)

# Optional: Show best cross-validation score
print("Best CV Accuracy: ", grid_search.best_score_)