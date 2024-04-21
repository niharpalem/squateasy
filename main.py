import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image

# Initialize MediaPipe Face Mesh.
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5)

# Streamlit UI
st.title('Facial Landmarks Detection with MediaPipe and Streamlit')
st.sidebar.title('Upload Image')

# File Uploader
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))

    # Convert the BGR image to RGB.
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Process the image and draw landmarks.
    results = face_mesh.process(image_rgb)
    image_with_landmarks = image.copy()

    # Draw facial landmarks on the image.
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp.solutions.drawing_utils.draw_landmarks(
                image=image_with_landmarks,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1))

    # Display the original image and the image with landmarks.
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Image")
        st.image(image, use_column_width=True)
    with col2:
        st.header("Image with Facial Landmarks")
        st.image(image_with_landmarks, use_column_width=True)
else:
    st.write("Please upload an image to proceed.")

# Release resources
face_mesh.close()
