import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase
import cv2

# Define RTC configuration to use Google's STUN server.
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        # Calculate the target height to maintain a 16:9 aspect ratio.
        height, width = img.shape[:2]
        target_height = int(width * 9 / 16)

        # Resize the image to maintain a 16:9 aspect ratio.
        # Adjust the resizing method as needed.
        resized_img = cv2.resize(img, (width, target_height))

        return resized_img

def main():
    st.header("Webcam Live Feed in 16:9 Aspect Ratio")
    st.text("This Streamlit app accesses the webcam and displays the video in a 16:9 aspect ratio.")

    # Start the webcam feed with the specified RTC configuration and video processor.
    webrtc_streamer(key="example", rtc_configuration=RTC_CONFIGURATION, video_processor_factory=VideoProcessor)

if __name__ == "__main__":
    main()
