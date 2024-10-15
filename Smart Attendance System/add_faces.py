import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

faces_data = []

i = 0

name = input("Enter Your Name: ")

while True:
    ret, frame = video.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (50,50))
        if len(faces_data) <= 200 and i % 5 == 0:  # Changed from 10 to 5 to collect data faster
            faces_data.append(resized_img)
        i = i + 1
        cv2.putText(frame, f"Samples: {len(faces_data)}/200", (20,50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0,255,0), 2)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
    
    cv2.imshow("Face Data Collection", frame)
    k = cv2.waitKey(1)
    if k == ord('q') or len(faces_data) == 200:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)
faces_data = faces_data.reshape(200, -1)

if 'names.pkl' not in os.listdir('newdata/'):
    names = [name] * 200
    with open('newdata/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('newdata/names.pkl', 'rb') as f:
        names = pickle.load(f)
    names = names + [name] * 200
    with open('newdata/names.pkl', 'wb') as f:
        pickle.dump(names, f)

if 'faces_data.pkl' not in os.listdir('newdata/'):
    with open('newdata/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
else:
    with open('newdata/faces_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    faces = np.append(faces, faces_data, axis=0)
    with open('newdata/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)

print(f"Data collection completed. 200 samples collected for {name}.")
print(f"Total samples in dataset: {len(names)}")
print(f"Shape of faces data: {faces.shape if 'faces' in locals() else faces_data.shape}")