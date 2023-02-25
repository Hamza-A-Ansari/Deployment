import streamlit as st
import cv2
import numpy as np

# Load the pre-trained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

@st.cache_data(ttl=600)
def load_glasses_images():
    glass = input("Enter image name: ")
    glasses = cv2.resize(cv2.imread(glass, -1), (10000, 5000)),
    # glasses = [
    #     cv2.resize(cv2.imread("images/glasses1.png", -1), (10000, 5000)),
    #     cv2.resize(cv2.imread("images/glasses2.png", -1), (10000, 5000)),
    #     cv2.resize(cv2.imread("images/glasses3.png", -1), (10000, 5000))
    # ]
    return glasses

# Define the function to switch between glasses
def switch_glasses(selected_glass_index, glasses):
    selected_glass_index += 1
    if selected_glass_index >= len(glasses):
        selected_glass_index = 0
    return selected_glass_index

# Define the main function to overlay glasses on a face
def overlay_glasses_on_face(image, glasses, selected_glass_index):
    # Convert the frame to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect the faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Resize the glasses image to fit the detected face
        glass_image = cv2.resize(glasses[selected_glass_index], (w, h))

        # Overlay the glasses on the face
        alpha_s = glass_image[:, :, 3] / 255.0
        alpha_l = 1.0 - alpha_s
        for c in range(0, 3):
            image[y:y+h, x:x+w, c] = (
                alpha_s * glass_image[:, :, c] + 
                alpha_l * image[y:y+h, x:x+w, c]
            )

    return image, selected_glass_index

# Define the main function to run the app
def run():
    # Set the title and page layout
    st.set_page_config(page_title='Glasses on Face', layout='wide')

    # Set the app title
    st.title('Glasses on Face')

    # Load the glasses images
    glasses = load_glasses_images()

    # Define the initial index of the selected glasses
    selected_glass_index = 0

    # Start the webcam
    cap = cv2.VideoCapture(0)

    # Create a streamlit image placeholder
    image_placeholder = st.empty()

    # Create a button to switch glasses
    if st.button("Switch Glasses"):
        selected_glass_index = switch_glasses(selected_glass_index, glasses)

    while True:
        # Read the frames from the webcam
        ret, frame = cap.read()

        if ret:
            # Overlay the glasses on the frame
            frame, selected_glass_index = overlay_glasses_on_face(frame, glasses, selected_glass_index)

            # Display the output
            image_placeholder.image(frame, channels="BGR")

            # Check for user input to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

        else:
            break

    # Release the resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()