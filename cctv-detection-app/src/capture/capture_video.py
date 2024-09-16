import cv2

def capture_video(source=0):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Change source to 0 for webcam, or provide the IP address/URL for CCTV camera
    source = 0  # Example: "http://192.168.0.100:8080/video"
    capture_video(source)
