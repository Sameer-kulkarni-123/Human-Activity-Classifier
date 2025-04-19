

#THIS FILE USES MEDIAPOSE'S HOLISTIC MODEL WHICH IS NOT BEING USED IN THE MAIN PROJECT
#THE MAIN PROJECT USES MEDIAPOSE'S BLAZEPOSE (OR JUST POSE)

#THIS FILE USES MEDIAPOSE'S HOLISTIC MODEL WHICH IS NOT BEING USED IN THE MAIN PROJECT
#THE MAIN PROJECT USES MEDIAPOSE'S BLAZEPOSE (OR JUST POSE)

#THIS FILE USES MEDIAPOSE'S HOLISTIC MODEL WHICH IS NOT BEING USED IN THE MAIN PROJECT
#THE MAIN PROJECT USES MEDIAPOSE'S BLAZEPOSE (OR JUST POSE)

# MediaPipe Pose Landmarker


import mediapipe as mp
import cv2
import numpy as np
import os
import joblib
import cv2
import time
import mediapipe as mp

#finding the path of the folder model
script_dir = os.path.dirname(os.path.abspath(__file__))  # Get the directory of the script
pkl_path = os.path.join(script_dir, "../model")

#importing all the necessary models
standard_scaler_model = joblib.load(f"{pkl_path}/standard_scaler_model1.pkl")
loaded_model = joblib.load(f"{pkl_path}/mlp_model1.pkl")
label_encoded_loaded_model = joblib.load(f"{pkl_path}/label_encoder_model1.pkl")
pca_loaded_model = joblib.load(f"{pkl_path}/pca_model1.pkl")

# Grabbing the Holistic Model from Mediapipe and
# Initializing the Model
# mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose
holistic_model = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Initializing the drawing utils for drawing the facial landmarks on image
mp_drawing = mp.solutions.drawing_utils

# (0) in VideoCapture is used to connect to your computer's default camera
capture = cv2.VideoCapture(0)

# Initializing current time and precious time for calculating the FPS
previousTime = 0
currentTime = 0
temp = 0

while capture.isOpened():
	# capture frame by frame
	ret, frame = capture.read()

	# resizing the frame for better view
	# frame = cv2.resize(frame, (1024, 576))
	frame = cv2.resize(frame, (800, 600))

	# Converting the from BGR to RGB
	image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

	# Making predictions using holistic model
	# To improve performance, optionally mark the image as not writeable to
	# pass by reference.
	image.flags.writeable = False
	results = holistic_model.process(image)
	if results.pose_landmarks:
		
		image.flags.writeable = True
		print("this is the frames test", temp," : ", results.pose_landmarks.landmark)
		temp += 1
		
		# X = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])
		X = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark])

		# #converting the cords of the test image to a 1d array
		# X = X.flatten()  #shape == (99, )
		X = X.flatten()

		# #converting the shape of the array to (1, 99)
		# X = X.reshape(1, -1)  #shape == (1, 99)
		X = X.reshape(1, -1)
		
		# standardized_X = standard_scaler_model.transform(X)  # shape == (1, 99)
		standardized_X = standard_scaler_model.transform(X)

		# reduced_X = pca_loaded_model.transform(standardized_X) #shape == (1, 11)
		reduced_X = pca_loaded_model.transform(standardized_X)


		# prediction = loaded_model.predict(reduced_X)
		prediction = loaded_model.predict(reduced_X)
		
		# final_label = label_encoded_loaded_model.inverse_transform(prediction)
		final_label = label_encoded_loaded_model.inverse_transform(prediction)

		# # Converting back the RGB image to BGR
		# image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
		# position = (70, 20)  # (x, y) coordinates
		position = (300, 50)
		# font = cv2.FONT_HERSHEY_SIMPLEX
		font = cv2.FONT_HERSHEY_SIMPLEX
		# font_scale = 1 
		font_scale = 3
		# color = (0, 0, 255)  # Green color (B, G, R)
		color = (255, 0, 0)
		# thickness = 2
		thickness = 3

		# Write text on the image
		# cv2.putText(image, final_label[0], position, font, font_scale, color, thickness)
		cv2.putText(image, final_label[0], position, font, font_scale, color, thickness)
	
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
	

	# Drawing the Facial Landmarks
	mp_drawing.draw_landmarks(
	image,
	results.pose_landmarks,
	mp_pose.POSE_CONNECTIONS
	)


	
	# Calculating the FPS
	currentTime = time.time()
	fps = 1 / (currentTime-previousTime)
	previousTime = currentTime
	
	# Displaying FPS on the image
	cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)

	# Display the resulting image
	cv2.imshow("Facial and Hand Landmarks", image)

	# Enter key 'q' to break the loop
	if cv2.waitKey(5) & 0xFF == ord('q'):
		break

# When all the process is done
# Release the capture and destroy all windows
capture.release()
cv2.destroyAllWindows()

