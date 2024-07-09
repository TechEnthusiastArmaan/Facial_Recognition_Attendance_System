import cv2
import pickle
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier

# Load trained model and data
with open('attendence system\data\\faces_data.pkl', 'rb') as f:
    faces, labels = pickle.load(f)

with open('attendence system\data\\names.pkl', 'rb') as f:
    names = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, labels)

# Initialize video capture and face detection
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('attendence system\data\haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in detected_faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50))
        reshaped_img = resized_img.reshape(1, -1)  # Ensure the image is reshaped to match training data

        # Predict the name
        output = knn.predict(reshaped_img)
        name = output[0]

        # Display the name and rectangle around the face
        cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
