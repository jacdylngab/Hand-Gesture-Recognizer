# Hand Gesture Recognizer using IMU Data

## Overview
This project implements a two-stage hand gesture recognition system using raw 6-axis IMU data. The system first discovers gesture patterns using unsupervised learning and then trains a supervised model to recognize these gestures in real time.

---

## Approach

### Stage 1: Unsupervised Gesture Discovery
- Collected continuous 6-axis IMU motion data from a wearable device
- Applied PCA for dimensionality reduction and motion visualization
- Used DBSCAN to automatically cluster motion patterns without predefined labels
- Interpreted the resulting clusters as five distinct gesture categories

### Stage 2: Supervised Gesture Recognition
- Extracted representative samples from each discovered cluster
- Labeled the samples and split them into training and testing sets
- Trained a supervised classifier (Random Forest) for gesture recognition
- Applied the same preprocessing and PCA transformation during inference
- Performed real-time gesture prediction from live or replayed sensor data

---

## Gestures Classes
The five recognized gestures are: Stationary, Waving, Punch Forward, Lift Upward, and Swinging/Striking.

--- 

## Demo

https://github.com/user-attachments/assets/ec0f5855-7adc-48eb-b058-f0db3b5ef2cd

---
## How to Run Locally

1. Clone the repo and navigate to the project:
   ```
   git clone https://github.com/jacdylngab/Hand-Gesture-Recognizer.git
   cd Hand-Gesture-Recognizer
   ```

2. Create and activate a virtual environment:
   ```
   python3 -m venv env
   source env/bin/activate    # On Windows use: env\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
   
4. Run the Flask server:
   ```
   python3 real-time_predictions.py # Use python if required by your system.
   ```
