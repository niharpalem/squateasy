import streamlit as st
from streamlit_webrtc import webrtc_streamer

def main():
    st.header("Webcam Live Feed")
    st.text("This Streamlit app accesses the webcam.")

    # Start the webcam feed
    webrtc_streamer(key="example")

if __name__ == "__main__":
    main()
