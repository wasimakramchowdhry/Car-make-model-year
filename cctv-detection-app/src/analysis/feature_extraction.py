import os
import numpy as np
from deepface import DeepFace
import cv2

def extract_features(image_path):
    features = DeepFace.represent(img_path=image_path, model_name='Facenet', enforce_detection=False)
    return np.array(features[0]["embedding"])

def process_dataset(data_dir):
    X, y = [], []
    print(f"Processing dataset in directory: {data_dir}")
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if os.path.isdir(person_dir):
            print(f"Processing person: {person_name}")
            for img_name in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_name)
                features = extract_features(img_path)
                X.append(features)  # Use only the 'embedding' part
                y.append(person_name)
    print(f"Processed {len(X)} images with {len(set(y))} classes.")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_dir = os.path.join(project_root, "data/face_data", "train")
    val_dir = os.path.join(project_root, "data/face_data", "val")
    
    print(f"Training directory: {train_dir}")
    print(f"Validation directory: {val_dir}")
    
    X_train, y_train = process_dataset(train_dir)
    X_val, y_val = process_dataset(val_dir)
    
    np.save(os.path.join(project_root, "data/face_train_model_data", "X_train.npy"), X_train)
    np.save(os.path.join(project_root, "data/face_train_model_data", "y_train.npy"), y_train)
    np.save(os.path.join(project_root, "data/face_train_model_data", "X_val.npy"), X_val)
    np.save(os.path.join(project_root, "data/face_train_model_data", "y_val.npy"), y_val)
