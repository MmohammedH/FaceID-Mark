# Import necessary libraries
import cv2
import pickle
import numpy as np
import os

# Initialize the video capture object using the default camera (Camera index 0)
video = cv2.VideoCapture(0)

# Load the Haar Cascade classifier for face detection
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

# Create an empty list to store face images
faces_data = []

# Initialize a counter variable
i = 0

# Prompt the user to enter their name
name = input("Enter Your Name: ")

# Start an infinite loop to capture and process video frames
while True:
    # Read a frame from the video feed
    ret, frame = video.read()

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    # Iterate over detected faces
    for (x, y, w, h) in faces:
        # Crop the face region from the frame
        crop_img = frame[y:y+h, x:x+w, :]

        # Resize the cropped face image to a fixed size (50x50 pixels)
        resized_img = cv2.resize(crop_img, (50, 50))

        # Add the resized face image to the faces_data list, up to a maximum of 100 samples
        if len(faces_data) <= 100 and i % 10 == 0:
            faces_data.append(resized_img)

        # Increment the counter
        i = i + 1

        # Draw text and rectangle around the detected face
        cv2.putText(frame, str(len(faces_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    # Display the frame with face detection results
    cv2.imshow("Frame", frame)

    # Wait for a key press for 1 millisecond and check if 'q' is pressed or 100 faces are collected
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()

# Convert the list of face images to a NumPy array and reshape it
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Save or update the 'names.pkl' file with the user's name
if 'names.pkl' not in os.listdir('data/'):
    names = [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Save or update the 'faces_data.pkl' file with the collected face data
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)
