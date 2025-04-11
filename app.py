import os
import cv2
import pickle
import time
import csv
import numpy as np
import random
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from deepface import DeepFace
from win32com.client import Dispatch
import streamlit as st
import pandas as pd
import qrcode
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_autorefresh import st_autorefresh

# Custom CSS for styling
import streamlit as st
import os
import json
import base64
import streamlit.components.v1 as components

########################################
# Utility Functions
########################################
def load_image(image_file):
    """Load and Base64-encode an image for CSS background usage."""
    if not os.path.exists(image_file):
        st.error(f"Image file not found: {image_file}")
        return None
    with open(image_file, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_lottiefile(filepath: str):
    """Load a local Lottie JSON file."""
    if not os.path.exists(filepath):
        st.error(f"Lottie file not found: {filepath}")
        return None
    with open(filepath, "r") as f:
        return json.load(f)

########################################
# File Paths (UPDATE THESE)
########################################
sidebar_image_path      = r"C:\c++ programs\face_detection_attendence\implement\gg.jpg"
registration_image_path = r"C:\c++ programs\face_detection_attendence\implement\bg.jpg"
lottie_file_path        = r"face_detection_attendence\Animation - 1740074679056.json"

########################################
# Load Assets
########################################
encoded_sidebar_image = load_image(sidebar_image_path)
encoded_bg_image      = load_image(registration_image_path)
lottie_data          = load_lottiefile(lottie_file_path)

########################################
# Main Background Styling
########################################
if encoded_bg_image:
    st.markdown(
        f"""
        <style>
        /* Main page background */
        body {{
            background-image: url("data:image/jpeg;base64,{encoded_bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
            margin: 0;
            padding: 0;
        }}
        /* Slightly translucent container for the main content */
        .stApp {{
            background-color: rgba(255, 255, 255, 0.8);
        }}
        /* Header styling */
        .header {{
            font-size: 3em;
            font-weight: bold;
            text-align: center;
            color: #4CAF50;
            font-family: 'Arial', sans-serif;
            margin-top: 0.5em;
            margin-bottom: 0.5em;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

########################################
# Sidebar Background & Animation
########################################
if encoded_sidebar_image:
    # We add a gradient overlay on top of the sidebar image so text is easier to read
    st.markdown(
        f"""
        <style>
        /* Target the sidebar area */
        [data-testid="stSidebar"] > div:first-child {{
            background: linear-gradient(
                          rgba(0, 0, 0, 0.4), 
                          rgba(0, 0, 0, 0.4)
                      ),
                      url("data:image/jpeg;base64,{encoded_sidebar_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        
        /* Increase text size in sidebar */
        [data-testid="stSidebar"] * {{
            font-size: 18px !important; /* Adjust the size as needed */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


########################################
# Display Main Title
########################################
st.markdown('<h1 class="header">FaceTrack AI</h1>', unsafe_allow_html=True)

########################################
# Lottie Animation in the Sidebar
########################################
if lottie_data:
    # Convert Lottie JSON to base64 so we can embed it in HTML
    lottie_json_str = json.dumps(lottie_data)
    lottie_base64   = base64.b64encode(lottie_json_str.encode()).decode()

    # Build an HTML snippet that places the Lottie animation in a smaller area
    # at the lower-left corner, with text overlaid.
    sidebar_html = f"""
    <html>
    <head>
      <!-- Lottie player script from CDN -->
      <script src="https://unpkg.com/@lottiefiles/lottie-player@latest/dist/lottie-player.js"></script>
    </head>
    <body style="margin:0; padding:0;">
      <div style="position: relative; width: 500%; height: 400px;">
        <!-- Lottie Animation (smaller, placed in bottom-left corner) -->
        <lottie-player 
            src="data:application/json;base64,{lottie_base64}"
            background="black"
            speed="1"
            style="width: 250px; height: 450px; position: absolute; bottom: 10px; left: 10px;"
            loop
            autoplay>
        </lottie-player>

        <!-- Overlaid text near the animation -->
        <div style="
            position: absolute;
            bottom: 60px; 
            left: 10px;
            color: #D64BFF;
            font-size: 22px;
            font-weight: bold;
            font-family: 'Arial', sans-serif;
            pointer-events: none; /* let clicks pass through to animation if needed */
        ">
          <br><br>SEARCH HERE!
        </div>
      </div>
    </body>
    </html>
    """

    with st.sidebar:
        # Render the HTML with the Lottie animation
        components.html(sidebar_html, height=400, scrolling=False)

else:
    st.sidebar.write("Lottie animation not found. Please check your file path.")






def speak(text):
    speaker = Dispatch("SAPI.SpVoice")
    speaker.Speak(text)

# Function to generate unique 6-digit UID
def generate_uid():
    return random.randint(100000, 999999)

# Registration Section
def registration():
    ADMIN_PASSWORD = "12345@"

    # Password input section
    if 'password_verified' not in st.session_state:
        st.session_state.password_verified = False

    if not st.session_state.password_verified:
        password = st.text_input("Enter Admin Password:", type="password")
        if st.button("Submit Password"):
            if password == ADMIN_PASSWORD:
                st.session_state.password_verified = True
                st.success("Password verified! You can now register.")
            else:
                st.error("Incorrect password. Please try again.")
        return  # Exit the function if password is not verified

    # Registration section
    st.title("Registration")
    st.write("REGISTER YOURSELF HERE!")
    st.markdown('<div class="header">🎮 Face Registration</div>', unsafe_allow_html=True)
    name = st.text_input("Enter Your Name:")
    if st.button("Register", key="register_button"):
        if not name:
            st.error("Name cannot be empty.")
            return
        
        # Check if the name already exists in the dataset
        if 'names.pkl' in os.listdir('face_detection_attendence/data/'):
            with open('face_detection_attendence/data/names.pkl', 'rb') as f:
                existing_names = pickle.load(f)
            #if name in existing_names:
               # st.error("Name already exists in the dataset. Use a different name.")
                #return
        
        # Initialize video capture and face detector
        video = cv2.VideoCapture(0)
        facedetect = cv2.CascadeClassifier('face_detection_attendence/data/haarcascade_frontalface_default.xml')
        faces_data = []
        i = 0
        
        # Collect face samples
        st.write("Press 'q' to quit or wait until 100 samples are collected.")
        while True:
            ret, frame = video.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = facedetect.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                crop_img = frame[y:y+h, x:x+w, :]
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
            st.error("Capture incomplete. At least 100 samples are required.")
            return
        
        # Convert and reshape face data
        faces_data = np.array(faces_data, dtype='float32')  # Ensure consistent dtype
        faces_data = faces_data.reshape(len(faces_data), -1)  # Flatten each image

        # Save names
        uid = generate_uid()
        if 'names.pkl' not in os.listdir('face_detection_attendence/data/'):
            names = [name] * len(faces_data)
            with open('face_detection_attendence/data/names.pkl', 'wb') as f:
                pickle.dump(names, f)
        else:
            with open('face_detection_attendence/data/names.pkl', 'rb') as f:
                names = pickle.load(f)
            names += [name] * len(faces_data)
            with open('face_detection_attendence/data/names.pkl', 'wb') as f:
                pickle.dump(names, f)

        # Save face data
        if 'faces_data.pkl' not in os.listdir('face_detection_attendence/data/'):
            with open('face_detection_attendence/data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces_data, f)
        else:
            with open('face_detection_attendence/data/faces_data.pkl', 'rb') as f:
                faces = pickle.load(f)
            faces = np.vstack((faces, faces_data))  # Use vstack for consistent shapes
            with open('face_detection_attendence/data/faces_data.pkl', 'wb') as f:
                pickle.dump(faces, f)
        
        # Save UID and Name in CSV for later use
        with open('face_detection_attendence/registered_users.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([uid, name])

        st.success(f"Registration successful! UID: {uid} for {name} has been saved.")

# Function to track attendance
def attendance_tracking():
    st.markdown('<div class="header">🎮 Attendance Tracking</div>', unsafe_allow_html=True)
    # Load pre-trained data from .pkl files
    with open('face_detection_attendence/data/names.pkl', 'rb') as w:
        LABELS = pickle.load(w)
    with open('face_detection_attendence/data/faces_data.pkl', 'rb') as f:
        FACES = pickle.load(f)

    # Ensure consistency of FACES and LABELS
    LABELS = np.array([str(label) for label in LABELS])  # Convert LABELS to strings
    FACES = np.array([face.flatten() for face in FACES], dtype='float32')  # Flatten faces
    LABELS = LABELS.flatten()

    # Ensure consistency in shapes
    if len(FACES) != len(LABELS):
        min_length = min(len(FACES), len(LABELS))
        FACES = FACES[:min_length]
        LABELS = LABELS[:min_length]

    # Normalize FACES
    FACES = FACES / 255.0

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(FACES, LABELS)
    print("KNN model trained successfully.")

    # Background image and column names
    imgBackground = cv2.imread("face_detection_attendence\design2.png")
    COL_NAMES = ['UID', 'NAME', 'TIME', 'EMOTION', 'POINTS']

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
        "Unknown": 0  # Default points for unknown emotions/person
    }

    # Initialize video capture
    video = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier('face_detection_attendence/data/haarcascade_frontalface_default.xml')

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

                # Retrieve UID from registered users CSV file
                uid = "Unknown"
                with open('face_detection_attendence/registered_users.csv', 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[1] == name:  # Assuming the second column contains names
                            uid = row[0]  # Assuming the first column contains UID
                            break

            else:
                name = "Unauthorized"
                uid = "N/A"
                print("Unauthorized: Face not recognized")

            # Draw rectangles and put text with emotion and name
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)  # Draw bounding box
            cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)  # Background for text
            cv2.putText(frame, f"{name}, {emotion} ({points} points)", (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)

            # Save attendance if authorized
            if name != "Unauthorized":
                attendance = [uid, name, str(timestamp), emotion, points]
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

# Dashboard Section
from web3 import Web3

# Connect to the Ganache blockchain
ganache_url = "HTTP://127.0.0.1:7545"  # Default Ganache URL
web3 = Web3(Web3.HTTPProvider(ganache_url))

# Check if connected
if not web3.is_connected():
    st.error("Failed to connect to the blockchain. Ensure Ganache is running.")
    st.stop()

metacoin_address = "0x7d09155dcE587c1Afd946019596e06AEbEBe1087"  # Replace with your MetaCoin contract address
metacoin_abi = [
    {
        "constant": False,
        "inputs": [{"name": "points", "type": "uint256"}],
        "name": "redeemPoints",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "getBalance",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": True,
        "inputs": [],
        "name": "getBalanceInEth",
        "outputs": [{"name": "", "type": "uint256"}],
        "payable": False,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": False,
        "inputs": [{"name": "receiver", "type": "address"}, {"name": "amount", "type": "uint256"}],
        "name": "sendCoin",
        "outputs": [{"name": "", "type": "bool"}],
        "payable": False,
        "stateMutability": "nonpayable",
        "type": "function"
    }
]

# Initialize contract
metacoin_contract = web3.eth.contract(address=metacoin_address, abi=metacoin_abi)

# Dashboard Function
def dashboard():
    st.markdown('<div class="header">🎮 Attendance Dashboard</div>', unsafe_allow_html=True)

    # Get current date
    ts = time.time()
    date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")

    # Load the CSV file
    try:
        df = pd.read_csv(f"face_detection_attendence/Attendance_{date}.csv")
        st.write("Attendance Data Loaded Successfully!")
        st.dataframe(df.style.highlight_max(axis=0))
    except FileNotFoundError:
        st.error(f"Attendance file for {date} not found. Please ensure the file exists.")
        st.stop()

    # Check if 'UID' column exists
    if "UID" in df.columns:
        # Dropdown to select a user
        selected_user = st.selectbox("Select a User", df["UID"].unique())

        # Filter data for the selected user
        user_data = df[df["UID"] == selected_user]

        # Display user-specific data
        st.subheader(f"User Data for: {selected_user}")
        st.dataframe(user_data)

        # Generate QR Code for Points
        goodies = {10: "🎟 Free Snack",5: "✏️ Extra Stationery",3: "☕ Free Coffee",2: "🎭 Small Discount Coupon",1: "🎲 eraser"}
        total_points = user_data["POINTS"].sum()
        st.write(f"Total Points for {selected_user}: {total_points}")
        earned_goodie = goodies.get(total_points, "No reward earned")
        st.write(f"🎁 You earned: {earned_goodie}")
        st.write("Goodies Redemption Options:", goodies)


        # Generate and display QR code
        qr_data = f"User: {selected_user}\nTotal Points: {total_points}\nEarned: {earned_goodie}\nDate: {date}"
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(qr_data)
        qr.make(fit=True)

        img = qr.make_image(fill="black", back_color="white")
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        st.image(buffer, caption="QR Code for Points", use_container_width=True)


        # Option to download QR code
        st.download_button(
            label="Download QR Code",
            data=buffer,
            file_name=f"{selected_user}_points_qr_{date}.png",
            mime="image/png",
        )

        # Redeem points through blockchain
        st.write("### Redeem Points on Blockchain")
        wallet_address = st.text_input("Enter Wallet Address", help="Enter your wallet address to redeem points.").strip()
        if st.button("Redeem Points"):
            if wallet_address:
                try:
                    points_to_redeem = int(total_points)

                    # Checking balance of the wallet
                    balance = web3.eth.get_balance(wallet_address)
                    st.write(f"Wallet Balance (in Wei): {balance}")

                    # Proceed with the transaction regardless of balance
                    txn = metacoin_contract.functions.redeemPoints(points_to_redeem).build_transaction({
                        "from": wallet_address,
                        "nonce": web3.eth.get_transaction_count(wallet_address),
                        "gas": 2000000,
                        "gasPrice": Web3.to_wei("50", "gwei")  # Corrected method name
                    })

                    # Sign the transaction with your private key (replace with your actual private key)
                    signed_txn = web3.eth.account.sign_transaction(txn, private_key="0x0d4f284325e1a170e5c1da0ac5b4120cfc4d8b6d7e8cf497c35cbde281a5bb40")

                    # Send the transaction
                    tx_hash = web3.eth.send_raw_transaction(signed_txn.raw_transaction)

                    # Show success message
                    st.success(f"Transaction successful! TX Hash: {tx_hash.hex()}")
                    st.write("Check your blockchain explorer or Ganache for details.")
                    st.write(f"User: {selected_user}\nTotal Points Redeemed: {total_points}\nDate: {date}")


                except Exception as e:
                    st.error(f"Error redeeming points: {str(e)}")
            else:
                st.error("Please enter a valid wallet address.")

    else:
        st.error("The 'UID' column is missing in the CSV file. Please check the file format.")

    # Footer
    st.markdown('<div class="footer">Smart Attendance System © 2025</div>', unsafe_allow_html=True)

# Main app flow
page = st.sidebar.radio("Select Page", ("Registration", "Attendance Tracking", "Dashboard"))

if page == "Registration":
    registration()  # Ensure you have this function defined
elif page == "Attendance Tracking":
    attendance_tracking()  # Ensure you have this function defined
elif page == "Dashboard":
    dashboard()


   