# Facetrack-AI

FaceTrack AI (SMART ATTENDENCE SYSTEM WITH EMOTION ANALYSIS)
FaceTrack AI presents a smart and efficient solution for facial recognition and emotion detection, integrating computer vision with AI-driven analytics. By leveraging real-time recognition and emotion-based scoring, the system not only enhances user interaction but also opens doors to personalized applications across education, security, and gaming. This project demonstrates the potential of combining technology with human behavior analysis to create systems that are intuitive, responsive, and adaptable. With further improvements, FaceTrack AI can be scaled and customized for diverse real-world scenarios, contributing meaningfully to the field of artificial intelligence and user-centric design.

FaceTrack AI: Modular Architecture Overview

FaceTrack AI is designed with a modular approach that combines computer vision, emotion detection, blockchain, and a responsive UI. Built using Python 3.10.5, it uses open-source libraries to deliver real-time performance and efficient data processing.

A. Facial Recognition and Emotion Detection
Face detection is achieved through OpenCV, which captures and processes live video. Emotional analysis is powered by DeepFace using VGG-Face and Facenet models to classify emotions like happy or sad. For improved accuracy, scikit-learn’s KNeighborsClassifier adds local decision-making.

B. Reward System and QR Code Generation
The system rewards users with points for positive emotions. These points are converted into QR codes using the qrcode library. These codes act as tokens for interaction and can be redeemed or verified.

C. Blockchain Integration
Ganache, a local Ethereum blockchain, is used to simulate a decentralized environment. Smart contracts written in Solidity handle QR code validation and reward redemption, ensuring secure and traceable transactions.

D. User Interface and Visualization
The interface is built using Streamlit for real-time interaction. Lottie animations enhance user experience, while Matplotlib and Pandas provide emotion data visualization and structured logging.

