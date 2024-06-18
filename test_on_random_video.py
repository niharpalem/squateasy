import cv2
import numpy as np
import mediapipe as mp

def process_video(video_path):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Initialize variables to store joint points
    all_joint_points = []

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB for MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with MediaPipe
        results = pose.process(frame_rgb)

        # Check if pose landmarks are detected
        if results.pose_landmarks is not None:
            # Extract 3D joint points for 19 specific joints
            frame_joint_points = []
            for joint_id in range(19):
                landmark = results.pose_landmarks.landmark[joint_id]
                # Get X, Y, Z coordinates of the landmark (Z is optional, set to 0 if not available)
                x = landmark.x
                y = landmark.y
                z = landmark.z if landmark.HasField('z') else 0.0
                frame_joint_points.extend([x, y, z])
            
            all_joint_points.append(frame_joint_points)

    # Close MediaPipe and release the video capture
    cap.release()
    pose.close()

    # Convert joint points to numpy array
    joint_points_array = np.array(all_joint_points)

    # Resize array to have exactly 300 frames
    num_frames_needed = 300
    num_frames = len(joint_points_array)
    if num_frames < num_frames_needed:
        # Duplicate frames
        repetitions = num_frames_needed // num_frames + 1
        joint_points_array = np.tile(joint_points_array, (repetitions, 1))[:num_frames_needed]
    elif num_frames > num_frames_needed:
        # Interpolate frames
        indices = np.linspace(0, num_frames - 1, num=num_frames_needed, dtype=int)
        joint_points_array = joint_points_array[indices]

    # Reshape to match model input shape
    input_data = joint_points_array.reshape(1, num_frames_needed, 57)

    return input_data

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ")
    input_data = process_video(video_path)
    print("Video processing complete.")
