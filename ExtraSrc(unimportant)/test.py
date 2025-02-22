import cv2
import mediapipe as mp
import re
import os

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

str = "file_name, nose_x, nose_y, nose_z, left_eye_inner_x, left_eye_inner_y, left_eye_inner_z, left_eye_x, left_eye_y, left_eye_z, left_eye_outer_x, left_eye_outer_y, left_eye_outer_z, right_eye_inner_x, right_eye_inner_y, right_eye_inner_z, right_eye_x, right_eye_y, right_eye_z, right_eye_outer_x, right_eye_outer_y, right_eye_outer_z, left_ear_x, left_ear_y, left_ear_z, right_ear_x, right_ear_y, right_ear_z, mouth_left_x, mouth_left_y, mouth_left_z, mouth_right_x, mouth_right_y, mouth_right_z, left_shoulder_x, left_shoulder_y, left_shoulder_z, right_shoulder_x, right_shoulder_y, right_shoulder_z, left_elbow_x, left_elbow_y, left_elbow_z, right_elbow_x, right_elbow_y, right_elbow_z, left_wrist_x, left_wrist_y, left_wrist_z, right_wrist_x, right_wrist_y, right_wrist_z, left_pinky_x, left_pinky_y, left_pinky_z, right_pinky_x, right_pinky_y, right_pinky_z, left_index_x, left_index_y, left_index_z, right_index_x, right_index_y, right_index_z, left_thumb_x, left_thumb_y, left_thumb_z, right_thumb_x, right_thumb_y, right_thumb_z, left_hip_x, left_hip_y, left_hip_z, right_hip_x, right_hip_y, right_hip_z, left_knee_x, left_knee_y, left_knee_z, right_knee_x, right_knee_y, right_knee_z, left_ankle_x, left_ankle_y, left_ankle_z, right_ankle_x, right_ankle_y, right_ankle_z, left_heel_x, left_heel_y, left_heel_z, right_heel_x, right_heel_y, right_heel_z, left_foot_index_x, left_foot_index_y, left_foot_index_z, right_foot_index_x, right_foot_index_y, right_foot_index_z"

landmark_names = str.split(", ")


csv_filename="raw-coordinates.csv"

createFile = 0
createFile = 1 #uncomment this line to create the csv file
if createFile:
  with open(csv_filename, "a") as file:
    for i in range(len(landmark_names)):
      file.write(landmark_names[i] + ", ")
    file.write("\n")

currImagePath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../Action-Recognition-Dataset/source_images3/jump_2/00348.jpg")

currImage = cv2.imread(currImagePath)
results = pose.process(currImage)
if results.pose_landmarks:
  mp_drawing.draw_landmarks(currImage, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
  skels_coors = results.pose_landmarks.landmark
  cv2.imshow(currImagePath, currImage)
  cv2.waitKey(000)
  cv2.destroyAllWindows()
print("passed")

# skels_coors = results.pose_landmarks.landmark
# with open(csv_filename, "a") as file:
#   file.write(action_label + "_" + padded_num+ ", ")
#   for bodyPoint in range(len(skels_coors)):
#     point = skels_coors[bodyPoint]
#     file.write(f"{point.x}, {point.y}, {point.z}, ")
#   file.write("\n")