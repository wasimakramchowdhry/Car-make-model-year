# General utility functions
def preprocess_image(image_path):
    # Example preprocessing steps
    import cv2
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    return image
