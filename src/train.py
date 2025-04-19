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

#getting the pwd of the script
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")

#reading the file from dir raw_skels
df = pd.read_csv(f"{script_dir}/../raw_skels/new_raw_co-ordinates1.csv")

#getting the X and Y features from the csv
X = df.iloc[:, 1:-1].values
Y = df["label"].values


#initializing the model and preprocessing tools
label_encoder = LabelEncoder()
standard_scaler = StandardScaler()
pca = PCA(n_components=.95)
model = MLPClassifier(hidden_layer_sizes=(64, 64),activation="relu", alpha=0.001, learning_rate="constant", solver="adam" ,max_iter=500, verbose=True)


#preforming preprocessing and label encoding
encoded_labels = label_encoder.fit_transform(Y)
standardized_X = standard_scaler.fit_transform(X)
reduced_X = pca.fit_transform(standardized_X)


X_train, X_test, Y_train, Y_test = train_test_split(reduced_X, encoded_labels, test_size=0.2, random_state=42)
model.fit(X_train, Y_train)




os.makedirs(pkl_path, exist_ok=True)

#run to create a .pkl file of the models
if True: # change to True to create the models
  joblib.dump(model, f"{pkl_path}/mlp_model2.pkl")
  joblib.dump(label_encoder, f"{pkl_path}/label_encoder_model2.pkl")
  joblib.dump(pca, f"{pkl_path}/pca_model2.pkl")
  joblib.dump(standard_scaler, f"{pkl_path}/standard_scaler_model2.pkl")





