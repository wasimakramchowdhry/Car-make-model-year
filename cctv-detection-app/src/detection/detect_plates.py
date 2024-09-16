import torch
import cv2
import os
from torchvision import transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torch.utils.data import Dataset
from PIL import Image

class SaudiPlatesDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, self.imgs[idx]

    def __len__(self):
        return len(self.imgs)

def get_transform():
    return T.Compose([T.ToTensor()])

def load_model():
    num_classes = 22  # Number of classes (21 for our dataset + 1 for background)
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torch.nn.Linear(in_features, num_classes)  # Use nn.Linear for classification head
    model_path = "data/models/saudi_plate_model_3.pth"  # Update to the correct model path
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def detect_and_display(model, image):
    transform = get_transform()
    img = transform(image)
    with torch.no_grad():
        prediction = model([img])[0]
    for element in prediction['boxes']:
        xmin, ymin, xmax, ymax = element.int()
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
        cv2.putText(image, "Plate", (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Plate Detection", image)
    cv2.waitKey(0)

if __name__ == "__main__":
    model = load_model()
    cap = cv2.VideoCapture(0)  # Change the source as needed

    if not cap.isOpened():
        print("Error: Could not open video source.")
        exit()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detect_and_display(model, frame)

    cap.release()
    cv2.destroyAllWindows()
