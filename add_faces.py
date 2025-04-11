import cv2
import pickle
import numpy as np
import os

# Initialize video capture and face detector
video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data\haarcascade_frontalface_default.xml')

faces_data = []
i = 0

# Prompt for user name and check for duplicates
name = input("Enter Your Name: ")

if 'names.pkl' in os.listdir('Attendence'):
    with open('data/names.pkl', 'rb') as f:
        existing_names = pickle.load(f)
    if name in existing_names:
        print("Name already exists in the dataset. Use a different name.")
        exit()

# Collect face samples
print("Press 'q' to quit or wait until 100 samples are collected.")
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
    
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 100:
        break

video.release()
cv2.destroyAllWindows()

if len(faces_data) < 100:
    print("Capture incomplete. At least 100 samples are required.")
    exit()

# Convert and reshape face data
faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(100, -1)

# Save names
if 'names.pkl' not in os.listdir('data'):
    names = [name] * 100
    with open('face_detection_attendence/data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('face_detection_attendence/data/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names += [name] * 100
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

print(f"Data for {name} has been successfully saved!")
