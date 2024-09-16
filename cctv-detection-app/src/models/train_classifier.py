import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import joblib
import os

def train_model():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    X_train_path = os.path.join(project_root, "data/face_train_model_data/X_train.npy")
    y_train_path = os.path.join(project_root, "data/face_train_model_data/y_train.npy")
    X_val_path = os.path.join(project_root, "data/face_train_model_data/X_val.npy")
    y_val_path = os.path.join(project_root, "data/face_train_model_data/y_val.npy")


    X_train = np.load(X_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(X_val_path)
    y_val = np.load(y_val_path)

    # Ensure y_train and y_val are 1-dimensional
    print(f"Unique classes in training data: {np.unique(y_train)}")
    print(f"Unique classes in validation data: {np.unique(y_val)}")
    
    label_encoder = LabelEncoder()
    y_train_enc = label_encoder.fit_transform(y_train.flatten())
    y_val_enc = label_encoder.transform(y_val.flatten())

    # Ensure there are at least two classes
    if len(np.unique(y_train_enc)) > 1:
        model = SVC(kernel='linear', probability=True)
        model.fit(X_train, y_train_enc)
        
        # Ensure the models directory exists
        models_dir = "data/models"
        os.makedirs(models_dir, exist_ok=True)

        joblib.dump(model, os.path.join(models_dir, "face_recognition_model.pkl"))
        joblib.dump(label_encoder, os.path.join(models_dir, "face_label_encoder.pkl"))

        print(f"Training Accuracy: {model.score(X_train, y_train_enc)}")
        print(f"Validation Accuracy: {model.score(X_val, y_val_enc)}")
    else:
        print("Error: Training data must contain at least two classes.")

if __name__ == "__main__":
    train_model()
