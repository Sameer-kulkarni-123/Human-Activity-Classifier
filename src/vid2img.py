import cv2
import os


#Create a folder named videos in the root dir and upload it the vid in that
#And Change the vid_name in the main funciton which is set to test.mp3 by default


script_dir = os.path.dirname(os.path.abspath(__file__))

def extract_frames(video_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)

    if not video.isOpened():
        print(f"Error: Unable to open video file '{video_path}'")
        return

    # Get video properties
    fps = int(video.get(cv2.CAP_PROP_FPS))  # Original frames per second
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = int(total_frames / fps)  # Duration in seconds

    print(f"Video FPS: {fps}")
    print(f"Total Duration: {duration} seconds")
    print(f"Expected Frames to Extract: {duration * 10}")

    frame_count = 0
    extracted_count = 0

    # Read frames from the video
    while True:
        ret, frame = video.read()

        if not ret:
            break  # End of video

        # Calculate the time-based extraction
        current_time_sec = frame_count / fps

        # Extract 10 frames per second
        if int(current_time_sec * 10) > extracted_count:
            frame_filename = os.path.join(output_folder, f"{extracted_count:05d}.jpg")
            cv2.imwrite(frame_filename, frame)
            extracted_count += 1

        frame_count += 1

    # Release video capture object
    video.release()
    print(f"Total frames extracted: {extracted_count}")
    print(f"Frames saved in folder: {output_folder}")

if __name__ == "__main__":
    vid_name = "test.mp3"
    video_path = os.path.join(script_dir, "../videos", vid_name)
    output_folder = os.path.join(script_dir, "../imagesFromVids")

    extract_frames(video_path, output_folder)