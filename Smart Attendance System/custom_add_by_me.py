import cv2
import pickle
import numpy as np
import os

def process_dataset(dataset_path):
    faces_data = []
    names = []
    
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                image = cv2.imread(image_path)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    for (x, y, w, h) in faces:
                        face = image[y:y+h, x:x+w]
                        resized_face = cv2.resize(face, (50, 50))
                        faces_data.append(resized_face.flatten())
                        names.append(person_name)

    faces_data = np.array(faces_data)
    
    # Save the processed data
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces_data, f)
    
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

    print(f"Processed {len(faces_data)} images.")

# Usage
dataset_path = 'Dataset'
process_dataset(dataset_path)