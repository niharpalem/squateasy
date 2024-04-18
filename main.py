import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import mediapipe as mp

class HolisticTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Convert the BGR image to RGB, flip the image for correct orientation
        img = cv2.cvtColor(cv2.flip(img, 1), cv2.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to pass by reference.
        img.flags.writeable = False
        results = self.holistic.process(img)

        # Draw the pose, face, hands annotations on the image.
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        mp.solutions.drawing_utils.draw_landmarks(img, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION)
        mp.solutions.drawing_utils.draw_landmarks(img, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(img, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(img, results.pose_landmarks, mp.solutions.holistic.POSE_CONNECTIONS)

        return cv2.flip(img, 1)

def main():
    st.header("Webcam Live Feed with MediaPipe Holistic")
    st.write("This app uses `streamlit-webrtc` and `mediapipe` to process and display the webcam video stream with pose, face, and hand landmarks.")

    webrtc_streamer(key="holistic", video_processor_factory=HolisticTransformer)

if __name__ == "__main__":
    main()
