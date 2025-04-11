import os
import cv2
import pickle
import time
import csv
import numpy as np
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from deepface import DeepFace
from win32com.client import Dispatch

# Speak function
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Load pre-trained data from .pkl files
with open('face_detection_attendence/data/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('face_detection_attendence/data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix:', FACES.shape)
print('Number of Labels:', len(LABELS))

# Ensure consistency of FACES and LABELS
min_length = min(len(FACES), len(LABELS))
FACES = FACES[:min_length]
LABELS = LABELS[:min_length]

# Normalize FACES for consistency
FACES = FACES / 255.0

# Train KNN model
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(FACES, LABELS)

# Background image and column names
imgBackground = cv2.imread("face_detection_attendence/image.png")
COL_NAMES = ['NAME', 'TIME', 'EMOTION', 'POINTS']

# Threshold for recognizing faces
THRESHOLD = 15  # Adjust threshold based on accuracy needs

# Emotion-to-points mapping
emotion_points = {
    "happy": 10,
    "sad": 2,
    "angry": 1,
    "surprise": 5,
    "neutral": 3,
    "fear": 1,
    "disgust": 0,
    "Unknown": 0  # Default points for unknown emotions
}

# Initialize video capture
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('face_detection_attendence\data\haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1) / 255.0  # Normalize input

        distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
        closest_distance = distances[0][0]
        predicted_label = LABELS[indices[0][0]]

        print(f"Predicted Label: {predicted_label}, Distance: {closest_distance}, Threshold: {THRESHOLD}")

        ts = time.time()
        date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
        exist = os.path.isfile(f"face_detection_attendence/Attendance_{date}.csv")

        # Emotion detection
        try:
            analysis = DeepFace.analyze(crop_img, actions=['emotion'], enforce_detection=False)
            emotion = analysis['dominant_emotion'] if isinstance(analysis, dict) else analysis[0]['dominant_emotion']
        except Exception as e:
            print(f"Emotion Detection Error: {e}")
            emotion = "Unknown"

        # Calculate points based on detected emotion
        points = emotion_points.get(emotion.lower(), 0)

        # Check if the predicted label matches and face distance is within the threshold
        if closest_distance <= THRESHOLD and predicted_label in LABELS:
            name = predicted_label
            print("Authorized: Face recognized")
        else:
            name = "Unauthorized"
            speak("Unauthorized person detected")
            print("Unauthorized: Face not recognized")

        # Draw rectangles and put text with emotion and name
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Draw bounding box
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)  # Background for text
        cv2.putText(frame, f"{name}, {emotion} ({points} points)", (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

        # Save attendance if authorized
        if name != "Unauthorized":
            attendance = [name, str(timestamp), emotion, points]
            if cv2.waitKey(1) == ord('o'):
                speak("Attendance Taken..")
                time.sleep(1)
                with open(f"face_detection_attendence/Attendance_{date}.csv", "a") as csvfile:
                    writer = csv.writer(csvfile)
                    if not exist:
                        writer.writerow(COL_NAMES)
                    writer.writerow(attendance)

    imgBackground[162:162 + 480, 55:55 + 640] = frame
    cv2.imshow("Frame", imgBackground)

    # Exit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
