import torch
import requests
import config
import cv2

def initialize_model():
    # Initialize YOLO model (assuming YOLOv5)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    return model

def detect_objects(model, frame):
    # Perform object detection
    results = model(frame)
    return results

def detect_persons(model, frame):
    results = detect_objects(model, frame)
    persons = [x for x in results.xyxy[0] if int(x[5]) == 0]  # Class 0 is 'person' in COCO dataset
    faces = []
    for person in persons:
        x1, y1, x2, y2 = map(int, person[:4])
        # Calculate the face bounding box within the person's bounding box
        face_height = (y2 - y1) // 3  # Assume face is in the upper third of the bounding box
        face_width = (x2 - x1) // 2   # Adjust this value as needed
        face_x1 = x1 + (x2 - x1 - face_width) // 2
        face_x2 = face_x1 + face_width
        faces.append([face_x1, y1, face_x2, y1 + face_height])
    return faces

def detect_vehicles(model, frame):
    results = detect_objects(model, frame)
    vehicles = [x for x in results.xyxy[0] if int(x[5]) in [2, 7]]  # Class 2 is 'car', Class 7 is 'truck' in COCO dataset
    return vehicles

def recognize_number_plate(image):
    _, img_encoded = cv2.imencode('.jpg', image)
    response = requests.post(
        config.PLATE_RECOGNIZER_API_URL,
        files=dict(upload=img_encoded.tobytes()),
        headers={'Authorization': f'Token {config.PLATE_RECOGNIZER_API_KEY}'}
    )
    if response.status_code in [200, 201]:  # Check for both 200 and 201 status codes
        print("Result:", response.status_code, response.text)
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

# Test detection with an image frame
if __name__ == "__main__":
    frame = cv2.imread('path/to/your/image.jpg')
    model = initialize_model()
    results = detect_objects(model, frame)
    
    # Extract person and car detections
    persons = detect_persons(model, frame)
    vehicles = detect_vehicles(model, frame)
    
    # Display results
    for result in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = result[:6]
        label = model.names[int(cls)]
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    cv2.imshow('Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
