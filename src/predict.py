import mediapipe as mp
import cv2
import numpy as np
import os
import joblib

#finding the path of the folder model
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")


#select the image to classify
# currImage = cv2.imread(f"{script_dir}/../testImgs/manStanding.jpg")
# currImage = cv2.imread(f"{script_dir}/../testImgs/manRunning.jpg")
# currImage = cv2.imread(f"{script_dir}/../testImgs/manSitting.jpg")
currImage = cv2.imread(f"{script_dir}/../testImgs/00113.jpg")
# currImage = cv2.imread(f"{script_dir}/../testImgs/humanKick.jpg")
# currImage = cv2.imread(f"{script_dir}/../testImgs/humanWave.jpg")
# currImage = cv2.imread(f"{script_dir}/../testImgs/testImg5.jpg")



#importing all the necessary models
standard_scaler_model = joblib.load(f"{pkl_path}/standard_scaler_model.pkl")
loaded_model = joblib.load(f"{pkl_path}/mlp_model.pkl")
label_encoded_loaded_model = joblib.load(f"{pkl_path}/label_encoder_model.pkl")
pca_loaded_model = joblib.load(f"{pkl_path}/pca_model.pkl")

#getting the cords of the test image
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
results = pose.process(currImage)

#drawing the skeleton on the test images
if True: #enable this to draw the skeletons on the test images
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing.draw_landmarks(currImage, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  cv2.imshow("test img", currImage)
  cv2.waitKey(2000)
  cv2.destroyAllWindows()


#check if a human can be identified in the picture
if not results.pose_landmarks:
  print("Could not find any humans in the picture")
  exit()

X = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

#converting the cords of the test image to a 1d array
X = X.flatten()  #shape == (99, )

#converting the shape of the array to (1, 99)
X = X.reshape(1, -1)  #shape == (1, 99)



standardized_X = standard_scaler_model.transform(X)  # shape == (1, 99)

reduced_X = pca_loaded_model.transform(standardized_X) #shape == (1, 11)


prediction = loaded_model.predict(reduced_X)
final_label = label_encoded_loaded_model.inverse_transform(prediction)
print(final_label)
