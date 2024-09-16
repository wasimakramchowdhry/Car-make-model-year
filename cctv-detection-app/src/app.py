from venv import logger
from detection.yolo_detection import initialize_model, detect_persons, detect_vehicles, recognize_number_plate
from detection.car_classification import CarClassificationModel, recognize_car
from analysis.deepface_analysis import analyze_faces
import cv2
import joblib
import numpy as np
import os
import config  # Import the configuration file
import mysql.connector
import shutil
import torch
from torchvision import transforms


def clear_cache():
    cache_dir = os.path.expanduser('~/.cache')
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory {cache_dir} cleared.")
    else:
        print("Cache directory does not exist.")


def get_face_region(body_coords):
    """Assume the face region is the top part of the bounding box."""
    x1, y1, x2, y2 = body_coords
    face_height = (y2 - y1) // 3  # Adjust this value as needed
    return (x1, y1, x2, y1 + face_height)

def draw_text(frame, text, position, font_scale=0.6, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
    """Draw text with a black fill."""
    x, y = position
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)

def get_car_info_from_db(car_plate):
    """Fetch car information from the database."""
    print(f"Fetching info for car plate: {car_plate}")
    conn = mysql.connector.connect(
        host=config.DB_HOST,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
        database=config.DB_NAME
    )
    cursor = conn.cursor(dictionary=True)

    query = "SELECT * FROM car_info WHERE car_plate = %s"
    cursor.execute(query, (car_plate,))
    result = cursor.fetchone()
    print(f"Database result for {car_plate}: {result}")

    cursor.close()
    conn.close()
    return result

def recognize_face(face_recognition_model, label_encoder, embedding, threshold=0.8):
    if embedding is not None:
        features = np.array([embedding])
        prediction = face_recognition_model.predict(features)
        probabilities = face_recognition_model.predict_proba(features)[0]
        max_prob = np.max(probabilities)
        
        # Print all class names with their probabilities
        for class_idx, prob in enumerate(probabilities):
            class_name = label_encoder.inverse_transform([class_idx])[0]
            print(f"{class_name}: {prob:.4f}")

        print("Face Accuracy: ", max_prob)
        if max_prob > threshold:
            person_name = label_encoder.inverse_transform(prediction)[0]
        else:
            person_name = "Unknown"
    else:
        person_name = "Unknown"
    
    return person_name



def main():

    # Clear cache at the start
    #clear_cache()

    # Initialize the YOLO model
    yolo_model = initialize_model()

    # Load the face recognition model and label encoder
    models_dir = "data/models"
    face_recognition_model = joblib.load(os.path.join(models_dir, "face_recognition_model.pkl"))
    label_encoder = joblib.load(os.path.join(models_dir, "face_label_encoder.pkl"))


    # Load the car classification model and label encoder
    car_label_encoder_path = os.path.join(models_dir, "car_label_encoder.pkl")
    car_label_encoder = joblib.load(car_label_encoder_path)

    car_classification_model = CarClassificationModel(len(car_label_encoder.classes_))
    car_classification_model_path = os.path.join(models_dir, "car_classification_model.pth")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    car_classification_model.load_state_dict(torch.load(car_classification_model_path, map_location=device))
    car_classification_model = car_classification_model.to(device)



    # Set video sources from config
    video_sources = config.VIDEO_SOURCES
    #video_file_path = config.VIDEO_FILE_PATH

    # Initialize video capture objects
    caps = [cv2.VideoCapture(source) for source in video_sources]
    #if video_file_path:
    #    caps.append(cv2.VideoCapture(video_file_path))

    for cap in caps:
        if not cap.isOpened():
            logger.error(f"Could not open video source {cap}")

    # Prepare video writer to save output if processing a single video file
    """ if video_file_path:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        output_file = "output_video.avi"
        out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))
        print(f"Output video will be saved as {output_file}") """

    while True:
        for idx, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                continue

            frame_height, frame_width = frame.shape[:2]

            # Detect persons and vehicles
            persons = detect_persons(yolo_model, frame)
            vehicles = detect_vehicles(yolo_model, frame)

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
                emotion = max(analysis['emotion'].items(), key=lambda x: x[1])[0]
                
                # Check if 'embedding' key exists and handle missing embeddings
                embedding = analysis.get('embedding')
                person_name = recognize_face(face_recognition_model, label_encoder, embedding, threshold=0.8)

                # Combine age, gender, and name information
                face_info_text = f"Age: {age}\nGender: {gender}\nEmotion: {emotion}\nName: {person_name}"
                
                # Draw the semi-transparent white overlay next to the face
                overlay_height = 90  # Adjust as needed
                overlay_width = 190  # Adjust as needed
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
                    draw_text(frame, f"  {value}", (overlay_x + 70, y_offset), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)
                    y_offset += 20


            for result in vehicles:
                x1, y1, x2, y2, conf, cls = map(int, result[:6])
                label = yolo_model.names[int(cls)].upper()

                # Crop the vehicle area and send it to the Plate Recognizer API
                vehicle_img = frame[y1:y2, x1:x2]
                number_plate_result = recognize_number_plate(vehicle_img)

                if number_plate_result and 'results' in number_plate_result:
                    for plate_result in number_plate_result['results']:
                        plate = plate_result['plate'].upper()  # Convert plate to uppercase
                        vehicle_info = plate_result.get('vehicle', {})
                        vehicle_type = vehicle_info.get('type', 'Unknown')

                        # Combine initial vehicle info
                        vehicle_info_text = f"Plate: {plate}\nType: {vehicle_type}\nVehicle Type: {label}"
                        
                        # Get additional information from the database
                        db_info = get_car_info_from_db(plate)
                        if db_info:
                            membership_status = db_info['membership_status']
                            make = db_info['make']
                            model = db_info['model']
                            year = db_info['year']
                            color = db_info['color']
                            
                            # Append the database information to the vehicle info text
                            vehicle_info_text += f"\nMembership: {membership_status}\nMake: {make}\nModel: {model}\nYear: {year}\nColor: {color}"
                        
                        # Draw the semi-transparent white overlay
                        overlay_height = 90 if not db_info else 160  # Adjust as needed
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

                
                # Get car model prediction using the car classification model
                car_model = recognize_car(car_classification_model, car_label_encoder, frame, (x1, y1, x2, y2), device, threshold=0.5)

                # Draw the semi-transparent white overlay for the car model information
                car_model_overlay_height = 30  # Adjust as needed
                car_model_overlay_width = 500  # Adjust as needed
                car_model_overlay_x = 10  # Position on the left
                car_model_overlay_y = frame_height - car_model_overlay_height - 10  # Position at the bottom
                car_model_overlay = frame.copy()
                cv2.rectangle(car_model_overlay, (car_model_overlay_x, car_model_overlay_y), (car_model_overlay_x + car_model_overlay_width, car_model_overlay_y + car_model_overlay_height), (255, 255, 255), -1)
                cv2.addWeighted(car_model_overlay, 0.6, frame, 0.4, 0, frame)

                # Display the car model information
                draw_text(frame, f"Prediction Model: {car_model}", (car_model_overlay_x + 10, car_model_overlay_y + 20), font_scale=0.5, font_thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX)

            # Show the frame for each source in a separate window
            cv2.imshow(f'Frame {idx}', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
