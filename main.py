import cv2
import streamlit as st

def main():
    st.title("Webcam Stream (Using OpenCV)")

    # Video capture setup (replace 0 with your camera index if needed)
    cap = cv2.VideoCapture(0)

    # Streamlit app loop
    while True:
        ret, frame = cap.read()

        # Display the webcam frame in the app
        st.image(frame, channels="BGR")

        # Exit loop if 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    # Release capture resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
