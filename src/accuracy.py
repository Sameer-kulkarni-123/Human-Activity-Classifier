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
from sklearn.model_selection import train_test_split

script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")

standard_scaler_model = joblib.load(f"{pkl_path}/standard_scaler_model.pkl")
loaded_model = joblib.load(f"{pkl_path}/mlp_model.pkl")
label_encoded_loaded_model = joblib.load(f"{pkl_path}/label_encoder_model.pkl")


df = pd.read_csv(f"{script_dir}/../raw_skels/new_raw_co-ordinates.csv")
cordsArray = df.iloc[:, 1:-1].values
Y = df["label"].values
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(Y)



standard_scaler = StandardScaler()
standardized_X = standard_scaler.fit_transform(cordsArray)

pca = PCA(n_components=.95)
reduced_X = pca.fit_transform(standardized_X)

X_train, X_test, Y_train, Y_test = train_test_split(reduced_X, encoded_labels, test_size=0.2, random_state=42)
Y_predict = loaded_model.predict(X_test)

correct_ans = 0


for i in range(len(Y_test)):
  if Y_test[i] == Y_predict[i]:
    correct_ans += 1

print(f"accuracy of the model: {round((correct_ans/len(Y_test))*100, 2)} %")
