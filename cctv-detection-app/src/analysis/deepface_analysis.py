from deepface import DeepFace
import cv2

def analyze_faces(frame, faces):
    analyses = []
    for (x1, y1, x2, y2) in faces:
        face = frame[y1:y2, x1:x2]
        try:
            # Perform age, gender, and emotion analysis
            analysis = DeepFace.analyze(face, actions=['age', 'gender', 'emotion'], enforce_detection=False)
            if isinstance(analysis, list):
                analysis = analysis[0]  # Take the first element if it's a list

            # Extract embedding
            embedding = DeepFace.represent(face, model_name='Facenet', enforce_detection=False)[0]['embedding']
            analysis['embedding'] = embedding

            # Select the emotion with the highest confidence
            emotion_confidence = max(analysis['emotion'].items(), key=lambda x: x[1])
            analysis['dominant_emotion'] = emotion_confidence[0]

        except ValueError as e:
            print(f"Face analysis error: {e}")
            analysis = {'age': 'unknown', 'gender': {'gender': 'unknown', 'confidence': 0}, 'emotion': {'dominant_emotion': 'unknown'}, 'embedding': None}

        analyses.append((x1, y1, x2, y2, analysis))
    return analyses

# Test facial analysis with an image frame
if __name__ == "__main__":
    frame = cv2.imread('path/to/your/image.jpg')
    faces = [(50, 50, 200, 200)]  # Example face coordinates
    face_analyses = analyze_faces(frame, faces)
    
    for analysis in face_analyses:
        print(analysis)
