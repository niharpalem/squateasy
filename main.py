import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoTransformerBase
import mediapipe as mp
import av

# MediaPipe setup
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class HolisticTransformer(VideoTransformerBase):
    def __init__(self):
        self.holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Process the image with MediaPipe Holistic
        img.flags.writeable = False
        results = self.holistic.process(img)
        img.flags.writeable = True

        # Draw the pose, face, and hand landmarks on the image
        img = av.VideoFrame.from_ndarray(img, format="bgr24")
        annotated_image = img.to_ndarray(format="bgr24")
        mp_drawing.draw_landmarks(
            annotated_image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
        mp_drawing.draw_landmarks(
            annotated_image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(
            annotated_image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        return av.VideoFrame.from_ndarray(annotated_image, format="bgr24")

def main():
    st.header("Webcam Live Feed with MediaPipe Holistic")
    st.write("This app uses `streamlit-webrtc` to capture the webcam video stream and `mediapipe` to process and display the video stream with pose, face, and hand landmarks.")

    # Define RTC configuration
    rtc_configuration = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})


    webrtc_streamer(key="holistic", 
                    video_processor_factory=HolisticTransformer, 
                    rtc_configuration=rtc_configuration)

if __name__ == "__main__":
    main()
