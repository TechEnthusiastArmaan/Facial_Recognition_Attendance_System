from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(text):
    speak = Dispatch("SAPI.SpVoice")
    speak.Speak(text)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('attendence system\\data\\haarcascade_frontalface_default.xml')

# Load labels and faces data
with open('attendence system\\data\\names.pkl', 'rb') as w:
    labels = pickle.load(w)
with open('attendence system\\data\\faces_data.pkl', 'rb') as f:
    faces, face_labels = pickle.load(f)

# Convert faces to a numpy array
faces = np.array(faces)

print('Shape of Faces matrix --> ', faces.shape)
print('Shape of Labels array --> ', len(labels))

# Initialize and train KNN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(faces, face_labels)

img_background = cv2.imread("background.png")

col_names = ['NAME', 'TIME']

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected_faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in detected_faces:
        crop_img = frame[y:y + h, x:x + w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
        output = knn.predict(resized_img)
        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
        exist = os.path.isfile(f"Attendance\\Attendance_{date}.csv")
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
        cv2.putText(frame, str(output[0]), (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
        attendance = [str(output[0]), str(timestamp)]
    img_background[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", img_background)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        with open(f"attendence system\Attendance\\Attendance_{date}.csv", "a", newline='') as csvfile:
            writer = csv.writer(csvfile)
            if not exist:
                writer.writerow(col_names)
            writer.writerow(attendance)
    if k == ord('q'):
        break
video.release()
cv2.destroyAllWindows()
