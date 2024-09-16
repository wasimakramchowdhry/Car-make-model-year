# Configuration settings for the application
VIDEO_SOURCES = [
    0#,1
    ]  # Default camera Use 0 for webcam, or provide the IP address/URL for CCTV camera Example: "http://192.168.0.100:8080/video"

VIDEO_FILE_PATH = "C:/Users/ACER/Downloads/Recording 2024-07-31 234255.mp4"  # Path to your video file

MODEL_PATH = 'data/models/'  # Path to models

PLATE_RECOGNIZER_API_URL = "https://api.platerecognizer.com/v1/plate-reader/"
PLATE_RECOGNIZER_API_KEY = "085832d60d320abec28b8a78cc71854ff13cf8c5"  


#DATABASE CREDENTIALS
DB_HOST = "localhost"
DB_USER = "root"
DB_PASSWORD = "root"
DB_NAME = "station_db"

