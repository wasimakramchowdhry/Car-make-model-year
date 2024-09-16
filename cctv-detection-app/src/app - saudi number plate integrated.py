from detection.yolo_detection import initialize_model, detect_persons, detect_vehicles, recognize_number_plate
from analysis.deepface_analysis import analyze_faces
import cv2
import joblib
import numpy as np
import os
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
import config  # Import the configuration file


def get_face_region(body_coords):
    """Assume the face region is the top part of the bounding box."""
    x1, y1, x2, y2 = body_coords
    face_height = (y2 - y1) // 3  # Adjust this value as needed
    return (x1, y1, x2, y1 + face_height)


def draw_text(frame, text, position, font_scale=0.6, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text with a black fill."""
    x, y = position
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


class FastRCNNPredictor(torch.nn.Module):
    def __init__(self, in_channels, num_classes):
        super(FastRCNNPredictor, self).__init__()
        self.cls_score = torch.nn.Linear(in_channels, num_classes)
        self.bbox_pred = torch.nn.Linear(in_channels, num_classes * 4)

    def forward(self, x):
        if x.dim() == 4:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


def load_saudi_plate_model(model_path):
    model = fasterrcnn_resnet50_fpn(pretrained=False)
    num_classes = 30  # Number of classes in your trained model
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    state_dict = torch.load(model_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("Saudi plate model loaded successfully")
    return model



def recognize_number_plate_saudi(model, image):
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
    print("Saudi plate recognition outputs:", outputs)
    return outputs



def main():
    # Initialize the YOLO model
    model = initialize_model()

    # Load the face recognition model and label encoder
    models_dir = "data/models"
    face_recognition_model = joblib.load(os.path.join(models_dir, "face_recognition_model.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "label_encoder.pkl"))

    # Load the Saudi plate recognition model
    saudi_plate_model_path = os.path.join(models_dir, "saudi_plate_model_3.pth")
    saudi_plate_model = load_saudi_plate_model(saudi_plate_model_path)

    # Set video source from config
    video_source = config.VIDEO_SOURCE
    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_height, frame_width = frame.shape[:2]

        # Detect persons and vehicles
        persons = detect_persons(model, frame)
        vehicles = detect_vehicles(model, frame)

        # Prepare face coordinates for analysis
        faces = [(int(x[0]), int(x[1]), int(x[2]), int(x[3])) for x in persons]
        face_analyses = analyze_faces(frame, faces)

        # Display detection results on the frame
        for (x1, y1, x2, y2) in faces:
            # Display the face detection rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for face region

        for (x1, y1, x2, y2, analysis) in face_analyses:
            age = analysis['age']
            gender = max(analysis['gender'].items(), key=lambda x: x[1])[0]

            # Check if 'embedding' key exists and handle missing embeddings
            embedding = analysis.get('embedding')
            if embedding is not None:
                # Predict the person's name using the face recognition model
                features = np.array([embedding])
                prediction = face_recognition_model.predict(features)
                person_name = label_encoder.inverse_transform(prediction)[0]
            else:
                person_name = "Unknown"

            # Combine age, gender, and name information
            face_info_text = f"Age: {age}\nGender: {gender}\nName: {person_name}"

            # Draw the semi-transparent white overlay next to the face
            overlay_height = 70  # Adjust as needed
            overlay_width = 180  # Adjust as needed
            overlay_x = x2 + 10  # Position next to the face
            overlay_y = y1  # Align with the top of the face
            overlay = frame.copy()
            cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + overlay_width, overlay_y + overlay_height), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

            # Display the face information
            y_offset = overlay_y + 20
            face_info_lines = face_info_text.split("\n")
            for line in face_info_lines:
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                draw_text(frame, f"{key}: ", (overlay_x + 10, y_offset), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                draw_text(frame, value, (overlay_x + 70, y_offset), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                y_offset += 20

        for result in vehicles:
            x1, y1, x2, y2, conf, cls = map(int, result[:6])
            label = model.names[int(cls)].upper()

            # Crop the vehicle area and send it to the Plate Recognizer API
            vehicle_img = frame[y1:y2, x1:x2]
            number_plate_result = recognize_number_plate(vehicle_img)
            if number_plate_result and 'results' in number_plate_result:
                for plate_result in number_plate_result['results']:
                    plate = plate_result['plate'].upper()  # Convert plate to uppercase
                    vehicle_info = plate_result.get('vehicle', {})
                    vehicle_type = vehicle_info.get('type', 'Unknown')
                    # Combine vehicle type, plate, and additional type information
                    vehicle_info_text = f"Plate: {plate}\nType: {vehicle_type}\nVehicle Type: {label}"

                    # Draw the semi-transparent white overlay
                    overlay_height = 90  # Adjust as needed
                    overlay_width = 250  # Adjust as needed
                    overlay_x = frame_width - overlay_width - 10  # Position on the right
                    overlay_y = 10  # Adjust as needed
                    overlay = frame.copy()
                    cv2.rectangle(overlay, (overlay_x, overlay_y), (overlay_x + overlay_width, overlay_y + overlay_height), (255, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                    # Display the vehicle information
                    y_offset = overlay_y + 20
                    vehicle_info_lines = vehicle_info_text.split("\n")
                    for line in vehicle_info_lines:
                        key, value = line.split(":")
                        key = key.strip()
                        value = value.strip()
                        draw_text(frame, f"{key}: ", (overlay_x + 10, y_offset), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                        draw_text(frame, value, (overlay_x + 120, y_offset), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                        y_offset += 20

            # Recognize the number plate using the Saudi plate model
            saudi_plate_outputs = recognize_number_plate_saudi(saudi_plate_model, vehicle_img)
            for output in saudi_plate_outputs:
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
                    if score > 0.5:  # Adjust the threshold as needed
                        x1, y1, x2, y2 = map(int, box)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for Saudi plate
                        plate_text = f"Plate: {label} ({score:.2f})"
                        draw_text(frame, plate_text, (x1, y1 - 10))

        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
