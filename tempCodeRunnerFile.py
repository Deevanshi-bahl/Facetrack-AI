import streamlit as st
import pandas as pd
import time
from datetime import datetime
import os
import cv2
import pickle
import numpy as np
import csv
from deepface import DeepFace
from win32com.client import Dispatch
import qrcode
from io import BytesIO
from PIL import Image
from streamlit_autorefresh import st_autorefresh
import uuid  # For generating unique UID

# Speak function
def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Face Registration & Data Capture
def register_face_data():
    # Initialize video capture and face detector
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('face_detection_attendence/data/haarcascade_frontalface_default.xml')

    faces_data = []
    i = 0

    # Prompt for user name
    name = st.text_input("Enter Your Name:")

    if 'names.pkl' in os.listdir('face_detection_attendence/data/'):
        with open('face_detection_attendence/data/names.pkl', 'rb') as f:
            existing_names = pickle.load(f)
        if name in existing_names:
            st.warning("Name already exists in the dataset. Use a different name.")
            return

    # Collect face samples
    st.write("Press 'q' to quit or wait until 100 samples are collected.")
    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(faces_data) < 100 and i % 10 == 0:  # Capture every 10th frame
                faces_data.append(resized_img)
            i += 1
            cv2.putText(frame, f"Samples: {len(faces_data)}/100", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        
        # Remove image display during registration
        if len(faces_data) == 100:
            break

    video.release()
    cv2.destroyAllWindows()

    if len(faces_data) < 100:
        st.error("Capture incomplete. At least 100 samples are required.")
        return

    # Convert and reshape face data
    faces_data = np.asarray(faces_data)
    faces_data = faces_data.reshape(100, -1)

    # Generate a unique UID for the person
    user_uid = str(uuid.uuid4())

    # Save names and UID
    if 'names.pkl' not in os.listdir('face_detection_attendence/data/'):
        names = [(name, user_uid)] * 100
        with open('face_detection_attendence/data/names.pkl', 'wb') as f:
            pickle.dump(names, f)
    else:
        with open('face_detection_attendence/data/names.pkl', 'rb') as f:
            names = pickle.load(f)
        names += [(name, user_uid)] * 100
        with open('face_detection_attendence/data/names.pkl', 'wb') as f:
            pickle.dump(names, f)

    # Save face data
    if 'faces_data.pkl' not in os.listdir('face_detection_attendence/data/'):
        with open('face_detection_attendence/data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)
    else:
        with open('face_detection_attendence/data/faces_data.pkl', 'rb') as f:
            faces = pickle.load(f)
        faces = np.append(faces, faces_data, axis=0)
        with open('face_detection_attendence/data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces, f)

    st.success(f"Data for {name} has been successfully saved! Your UID is: {user_uid}")


# Emotion Recognition & Attendance
def track_attendance():
    # Load pre-trained data from .pkl files
    with open('face_detection_attendence/data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('face_detection_attendence/data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    st.write(f'Shape of Faces matrix: {FACES.shape}')
    st.write(f'Number of Labels: {len(LABELS)}')

    # Ensure consistency of FACES and LABELS
    min_length = min(len(FACES), len(LABELS))
    FACES = FACES[:min_length]
    LABELS = LABELS[:min_length]

    # Normalize FACES for consistency
    FACES = FACES / 255.0

    # Train KNN model
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(FACES, LABELS)

    # Background image and column names
    imgBackground = cv2.imread("face_detection_attendence/image.png")
    COL_NAMES = ['NAME', 'TIME', 'EMOTION', 'POINTS']

    # Emotion-to-points mapping
    emotion_points = {
        "happy": 10,
        "sad": 2,
        "angry": 1,
        "surprise": 5,
        "neutral": 3,
        "fear": 1,
        "disgust": 0,
        "Unknown": 0
    }

    # Initialize video capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('face_detection_attendence/data/haarcascade_frontalface_default.xml')

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            crop_img = frame[y:y + h, x:x + w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1) / 255.0  # Normalize input

            distances, indices = knn.kneighbors(resized_img, n_neighbors=1)
            closest_distance = distances[0][0]
            predicted_label = LABELS[indices[0][0]]

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
            if closest_distance <= 15 and predicted_label in LABELS:
                name = predicted_label
            else:
                name = "Unauthorized"
                speak("Unauthorized person detected")

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


# Display Dashboard
def show_dashboard():
    # Auto-refresh the dashboard
    count = st_autorefresh(interval=2000, limit=100, key="refresh_counter")

    # Title and description
    st.title("Face Recognition and Emotion Analysis Dashboard")
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
    timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
    st.write(f"Date: {date}")

    # Load the CSV file
    try:
        df = pd.read_csv(f"face_detection_attendence/Attendance_{date}.csv")
        st.write("Attendance Data Loaded Successfully!")
        st.dataframe(df.style.highlight_max(axis=0))
    except FileNotFoundError:
        st.error(f"Attendance file for {date} not found. Please ensure the file exists in the 'Attendance' folder.")
        st.stop()

    # Check if 'NAME' column exists
    if "NAME" in df.columns:
        # Dropdown to select a user
        selected_user = st.selectbox("Select a User", df["NAME"].unique())

        # Filter data for the selected user
        user_data = df[df["NAME"] == selected_user]

        # Display user-specific data
        st.subheader(f"User Data for: {selected_user}")
        st.dataframe(user_data)

        # Generate QR Code for Points
        total_points = user_data["POINTS"].sum()
        st.write(f"Total Points for {selected_user}: {total_points}")

        # Generate and display QR code
        qr_data = f"User: {selected_user}\nTotal Points: {total_points}\nDate: {date}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)

        img = qr.make_image(fill="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        st.image(buffer, caption="QR Code for Points", use_column_width=True)

        # Option to download QR code
        st.download_button(
            label="Download QR Code",
            data=buffer,
            file_name=f"{selected_user}_points_qr_{date}.png",
            mime="image/png",
        )
    else:
        st.error("The 'NAME' column is missing in the CSV file. Please check the file format.")

    # Footer
    st.write("---")
    st.write("Auto-refresh count:", count)

# Main App Navigation
def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Select an option", ("Register Face", "Track Attendance", "Dashboard"))

    if choice == "Register Face":
        register_face_data()
    elif choice == "Track Attendance":
        track_attendance()
    elif choice == "Dashboard":
        show_dashboard()

if __name__ == "__main__":
    main()
