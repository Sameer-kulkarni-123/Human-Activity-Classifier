import mediapipe as mp
import cv2
import numpy as np
import os
import joblib

#finding the path of the folder model
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")



# imgNames = ["00160.jpg", "humanJumping.jpg", "humanKick.jpg", "humanRunning.jpg", "humanRunning1.jpg", "humanSitting.jpg", "humanSquatting.jpg", "humanWave.jpg", "humanWave1.jpg"]
imgNames = ["00152_40.jpg"]


#importing all the necessary models
standard_scaler_model = joblib.load(f"{pkl_path}/best_standard_scaler_model.pkl")
loaded_model = joblib.load(f"{pkl_path}/best_mlp_model.pkl")
label_encoded_loaded_model = joblib.load(f"{pkl_path}/best_label_encoder_model.pkl")
pca_loaded_model = joblib.load(f"{pkl_path}/best_pca_model.pkl")

#getting the cords of the test image
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)



for imgName in imgNames:
  currImage = cv2.imread(f"{script_dir}/../testImgs/{imgName}") 
  results = pose.process(currImage)

  #drawing the skeleton on the test images
  if True: #enable this to draw the skeletons on the test images
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing.draw_landmarks(currImage, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)


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

  position = (70, 20)  # (x, y) coordinates
  font = cv2.FONT_HERSHEY_SIMPLEX
  font_scale = 1
  color = (0, 0, 255)  # Green color (B, G, R)
  thickness = 2

  # Write text on the image
  cv2.putText(currImage, final_label[0], position, font, font_scale, color, thickness)

  # Show the image
  cv2.imshow("Image with Text", currImage)
  cv2.waitKey(000)
  cv2.destroyAllWindows()

  print(f"{imgNames.index(imgName)} : {final_label}")
  print("\n")
