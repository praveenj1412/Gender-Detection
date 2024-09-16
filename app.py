import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv

# Load the pre-trained gender detection model
model = load_model(r"D:\gender_detection.h5")
classes = ['male', 'female']

# Function to detect gender from an image frame
def gender_detection(frame):
    # Apply face detection
    face, confidence = cv.detect_face(frame)
    
    # Loop through detected faces
    for idx, f in enumerate(face):
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]

        # Draw rectangle around the face
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        # Crop the face from the frame
        face_crop = np.copy(frame[startY:endY, startX:endX])

        if face_crop.shape[0] < 10 or face_crop.shape[1] < 10:
            continue

        # Preprocess the face for the model
        face_crop = cv2.resize(face_crop, (96, 96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

        # Predict the gender
        conf = model.predict(face_crop)[0]
        idx = np.argmax(conf)
        label = classes[idx]
        label = "{}: {:.2f}%".format(label, conf[idx] * 100)

        Y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return frame

# Streamlit app function
def main():
    st.title("Gender Detection using Webcam")
    
    # Define the checkbox outside the loop and give it a unique key
    run = st.checkbox('Run Webcam', key="webcam_checkbox")
    FRAME_WINDOW = st.image([])

    # Capture webcam input when the checkbox is selected
    if run:
        webcam = cv2.VideoCapture(0)
        
        # Check if the webcam opened successfully
        if not webcam.isOpened():
            st.error("Could not open webcam.")
            return
        
        while run:
            status, frame = webcam.read()
            
            if not status:
                st.error("Webcam not accessible!")
                break
            
            # Process the frame for gender detection
            frame = gender_detection(frame)
            
            # Convert color (BGR to RGB) for Streamlit display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the Streamlit image display
            FRAME_WINDOW.image(frame_rgb)

            # Stop the loop if the checkbox is unchecked
            run = st.session_state.webcam_checkbox

        webcam.release()
        cv2.destroyAllWindows()

# Run the Streamlit app
if __name__ == '__main__':
    main()
