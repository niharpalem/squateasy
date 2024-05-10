import streamlit as st
from streamlit_webrtc import webrtc_streamer

def main():
    st.header("Webcam Live Feed")
    st.write("This application streams the webcam feed directly to the browser.")

    # Start the webcam stream
    webrtc_streamer(key="example")

if __name__ == "__main__":
    main()

