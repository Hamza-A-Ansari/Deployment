import os
current_directory = os.getcwd()
import cv2
import numpy as np

# Load the pre-trained model for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load the glasses images
# Load the glasses images and resize them to a smaller size
glass_images = [    cv2.resize(cv2.imread("images/glasses1.png", -1), (10000, 5000)),
                    cv2.resize(cv2.imread("images/glasses2.png", -1), (10000, 5000)),
                    cv2.resize(cv2.imread("images/glasses3.png", -1), (10000, 5000))
                
                ]


# Define the initial index of the selected glasses
selected_glass_index = 0

# Define the function to switch between glasses
def switch_glasses():
    global selected_glass_index
    selected_glass_index += 1
    if selected_glass_index >= len(glass_images):
        selected_glass_index = 0

# Start the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read the frames from the webcam
    ret, frame = cap.read()
    
    if ret:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect the faces in the frame
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            # Resize the glasses image to fit the detected face
            glass_image = cv2.resize(glass_images[selected_glass_index], (w, h))

            # Overlay the glasses on the face
            alpha_s = glass_image[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s
            for c in range(0, 3):
                frame[y:y+h, x:x+w, c] = (
                    alpha_s * glass_image[:, :, c] + 
                    alpha_l * frame[y:y+h, x:x+w, c]
                )

        # Display the output
        cv2.imshow('frame', frame)

        # Check for user input to switch between glasses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            switch_glasses()

    else:
        break

# Release the resources
cap.release()
cv2.destroyAllWindows()
