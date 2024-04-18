import streamlit as st
from streamlit_webrtc import webrtc_streamer

def main():
    st.header("Webcam Live Feed")
    st.write("This Streamlit app is using `streamlit-webrtc` to access the webcam and display the video stream.")

    # Start the webcam feed
    webrtc_streamer(key="example")

if __name__ == "__main__":
    main()
