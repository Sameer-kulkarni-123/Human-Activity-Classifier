import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import mediapipe as mp
import cv2
import joblib
import os




df = pd.read_csv("new_raw_co-ordinates.csv")
cordsArray = df.iloc[:, 1:-1].values
Y = df["label"].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Y)


all_labels=["jump", "kick", "punch", "run", "sit", "squat", "stand", "walk", "wave"]

standard_scaler = StandardScaler()
standardized_X = standard_scaler.fit_transform(cordsArray)
print(standardized_X.shape)

pca = PCA(n_components=.95)
reduced_X = pca.fit_transform(standardized_X)
print(reduced_X.shape)


X_train, X_test, Y_train, Y_test = train_test_split(reduced_X, encoded_labels, test_size=0.2, random_state=42)
print(X_train)
model = MLPClassifier(hidden_layer_sizes=(20, 30, 40), max_iter=500)
model.fit(X_train, Y_train)



script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")
os.makedirs(pkl_path, exist_ok=True)


if False: # change to True to create the models
  joblib.dump(model, f"{pkl_path}/mlp_model.pkl")
  joblib.dump(label_encoder, f"{pkl_path}/label_encoder_model.pkl")
  joblib.dump(pca, f"{pkl_path}/pca_model.pkl")
  joblib.dump(standard_scaler, f"{pkl_path}/standard_scaler_model.pkl")





