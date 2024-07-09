import cv2
import numpy as np
import os
import pickle

facedetect = cv2.CascadeClassifier(r'attendence system\data\haarcascade_frontalface_default.xml')
video = cv2.VideoCapture(0)

Id = input("ENTER YOUR NAME: ")
dataset_faces = []
dataset_labels = []

# Check if names.pkl exists
names_file = r'attendence system\data\names.pkl'
if not os.path.exists(names_file):
    with open(names_file, 'wb') as f:
        pickle.dump([], f)
        
# Load names.pkl
with open(names_file, 'rb') as f:
    names = pickle.load(f)
names.append(Id)
with open(names_file, 'wb') as f:
    pickle.dump(names, f)

# Check if faces_data.pkl exists
faces_data_file = r'attendence system\data\faces_data.pkl'
if not os.path.exists(faces_data_file):
    with open(faces_data_file, 'wb') as f:
        pickle.dump(([], []), f)

# Load faces_data.pkl
with open(faces_data_file, 'rb') as f:
    faces_data, labels = pickle.load(f)

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (50, 50))
        if len(dataset_faces) < 50:
            dataset_faces.append(face.flatten())
            dataset_labels.append(Id)
        else:
            break
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) == 13 or len(dataset_faces) >= 50:  # Enter key pressed or 50 faces collected
        break

video.release()
cv2.destroyAllWindows()

# Ensure dataset_faces and dataset_labels are numpy arrays
dataset_faces = np.array(dataset_faces)
dataset_labels = np.array(dataset_labels)

# Append new data
if len(faces_data) > 0:
    faces_data = np.concatenate((faces_data, dataset_faces), axis=0)
    labels = np.concatenate((labels, dataset_labels), axis=0)
else:
    faces_data = dataset_faces
    labels = dataset_labels

# Save updated faces_data.pkl
with open(faces_data_file, 'wb') as f:
    pickle.dump((faces_data, labels), f)
