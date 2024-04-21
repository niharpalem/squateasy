import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2 as cv
import mediapipe as mp

RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class ImageTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

    def transform(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Convert the BGR image to RGB.
        image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        # Process the image and draw landmarks.
        results = self.face_mesh.process(image_rgb)
        image_with_landmarks = image_rgb.copy()

        # Draw facial landmarks on the image.
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    image=image_with_landmarks,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(255, 255, 255), thickness=1, circle_radius=1))

        return cv.cvtColor(image_with_landmarks, cv.COLOR_RGB2BGR)

st.title('Facial Landmarks Detection with MediaPipe')

# Button to capture the image
if st.button('Capture'):
    # Using webrtc_streamer to capture and process image from the webcam
    webrtc_streamer(key="example", video_processor_factory=ImageTransformer, rtc_configuration=RTC_CONFIGURATION)
