from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import cv2
import pickle
import numpy as np
import os
import csv
import time
from datetime import datetime
from win32com.client import Dispatch

def speak(str1):
    speak = Dispatch(("SAPI.SpVoice"))
    speak.Speak(str1)

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

with open('newdata/names.pkl', 'rb') as w:
    LABELS = pickle.load(w)
with open('newdata/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

print('Shape of Faces matrix --> ', FACES.shape)
print('Number of labels:', len(LABELS))
print('Number of unique labels:', len(set(LABELS)))
print('Unique labels:', set(LABELS))

# Ensure FACES and LABELS have the same number of samples
if FACES.shape[0] != len(LABELS):
    min_samples = min(FACES.shape[0], len(LABELS))
    FACES = FACES[:min_samples]
    LABELS = LABELS[:min_samples]
    print(f"Adjusted data to {min_samples} samples to ensure consistency")

# Create a pipeline with StandardScaler and RandomForestClassifier
pipeline = make_pipeline(
    StandardScaler(),
    RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, min_samples_leaf=2)
)
pipeline.fit(FACES, LABELS)

COL_NAMES = ['NAME', 'TIME']

# Function to get the most confident prediction above a threshold
def get_prediction(probabilities, classes, threshold=0.7):
    max_prob = max(probabilities)
    if max_prob >= threshold:
        return classes[np.argmax(probabilities)]
    return None

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50,50)).flatten().reshape(1,-1)
        probabilities = pipeline.predict_proba(resized_img)[0]
        output = get_prediction(probabilities, pipeline.classes_)
        
        if output is not None:
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%H:%M-%S")
            exist = os.path.isfile("Attendance/Attendance_" + date + ".csv")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.rectangle(frame, (x,y-40), (x+w,y), (0,255,0), -1)
            cv2.putText(frame, str(output), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
            attendance = [str(output), str(timestamp)]
            
            # Add this for debugging
            print(f"Detected face: {output}")
            print(f"Probabilities: {dict(zip(pipeline.classes_, probabilities))}")
        else:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)

    cv2.imshow("Face Recognition", frame)
    k = cv2.waitKey(1)
    if k == ord('o'):
        speak("Attendance Taken..")
        time.sleep(5)
        if exist:
            with open("Attendance/Attendance_" + date + ".csv", "a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendance)
        else:
            with open("Attendance/Attendance_" + date + ".csv", "w") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendance)
    if k == ord('q'):
        break

video.release()
cv2.destroyAllWindows()

# After the main loop, add these lines:
print("Feature importances:")
feature_importance = pipeline.named_steps['randomforestclassifier'].feature_importances_
top_features = sorted(zip(feature_importance, range(len(feature_importance))), reverse=True)[:10]
for importance, index in top_features:
    print(f"Feature {index}: {importance}")