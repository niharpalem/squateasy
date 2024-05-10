import streamlit as st
import cv2
import tempfile
import numpy as np

def main():
    st.header("Upload and Process Video")

    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file is not None:
        # Create a temporary file to store the uploaded video
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())

        # Load the video with OpenCV
        cap = cv2.VideoCapture(tfile.name)

        # Check if camera opened successfully
        if (cap.isOpened() == False):
            st.write("Error opening video stream or file")

        # Read until video is completed
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                # Convert the colors from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the resulting frame
                st.image(frame)
                
                # Add a delay to mimic real time frame rate
                cv2.waitKey(25)

                # Adding a placeholder for further processing like detecting points
                # process_frame(frame) # You can define this function as needed

            # Break the loop
            else: 
                break

        # When everything done, release the video capture object
        cap.release()

        # Closes all the frames (used in native OpenCV, not needed in Streamlit)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
