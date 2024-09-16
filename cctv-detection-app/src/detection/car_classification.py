import os
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
import joblib
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CarClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(CarClassificationModel, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

def recognize_car(car_classification_model, car_label_encoder, frame, bbox, device, threshold=0.9):
    logging.info(f"Processing bounding box: {bbox}")
    
    x1, y1, x2, y2 = bbox
    car_image = frame[y1:y2, x1:x2]
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    car_image = preprocess(car_image).unsqueeze(0).to(device)
    
    car_classification_model.eval()
    with torch.no_grad():
        outputs = car_classification_model(car_image)
        probabilities = nn.functional.softmax(outputs, dim=1)
        max_prob, preds = torch.max(probabilities, 1)
    
    max_prob = max_prob.item()
    preds = preds.item()
    
    logging.info(f"Max probability: {max_prob}, Predicted class: {preds}")
    
    if max_prob > threshold:
        car_type = car_label_encoder.inverse_transform([preds])[0]
    else:
        car_type = "Unknown"
    
    return car_type

if __name__ == "__main__":
    logging.info("Starting car classification script")
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    models_dir = os.path.join(project_root, "data/models")
    model_path = os.path.join(models_dir, "car_classification_model.pth")
    label_encoder_path = os.path.join(models_dir, "car_label_encoder.pkl")
    
    logging.info("Loading label encoder")
    label_encoder = joblib.load(label_encoder_path)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    logging.info("Loading car classification model")
    model = CarClassificationModel(len(label_encoder.classes_))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    #file_path = os.path.join(project_root, r"data\car_classification_data\train\Hyundai Creta I 2016-2020\12799e4cd556256f549a844b568a6cbc.jpg")
    
    #logging.info(f"Reading image file: {file_path}")
    #frame = cv2.imread(file_path)
    
    #if frame is None:
    #    logging.error(f"Error: Could not read the image file {file_path}")
    #else:
        # Manually set the bounding box coordinates
    #    x1, y1, x2, y2 = 100, 100, 500, 500  # Adjust these coordinates as needed
    #    bbox = (x1, y1, x2, y2)
        
    #    logging.info("Recognizing car")
    #    car_type = recognize_car(model, label_encoder, frame, bbox, device, threshold=0.2)
    #    logging.info(f"Recognized car type: {car_type}")
    #    print(car_type)
