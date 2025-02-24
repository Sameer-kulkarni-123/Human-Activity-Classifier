import mediapipe as mp
import cv2
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler


#finding the path of the folder model
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")

#importing all the necessary models
standard_scaler_model = joblib.load(f"{pkl_path}/standard_scaler_model.pkl")
loaded_model = joblib.load(f"{pkl_path}/mlp_model.pkl")
label_encoded_loaded_model = joblib.load(f"{pkl_path}/label_encoder_model.pkl")
pca_loaded_model = joblib.load(f"{pkl_path}/pca_model.pkl")

#getting the cords of the test image
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
currImage = cv2.imread("testImg.jpg")
results = pose.process(currImage)
testCords = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

#converting the cords of the test image to a 1d array
t1 = testCords.flatten()  #shape == (99, )

#converting the shape of the array to (1, 99)
t2 = t1.reshape(1, -1)  #shape == (1, 99)



standardized_X = standard_scaler_model.transform(t2)  # shape == (1, 99)

reduced_X = pca_loaded_model.transform(standardized_X) #shape == (1, 11)


prediction = loaded_model.predict(reduced_X)
final_label = label_encoded_loaded_model.inverse_transform(prediction)
print(final_label)
