import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
import os
from PIL import Image
from tqdm import tqdm

class SaudiPlatesDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.labels = list(sorted(os.listdir(os.path.join(root, "labels"))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        label_path = os.path.join(self.root, "labels", self.labels[idx])
        img = Image.open(img_path).convert("RGB")
        
        boxes = []
        labels = []
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Unexpected format in {label_path}: {line.strip()}")
                    continue
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:5])
                img_width, img_height = img.size
                xmin = (x_center - width / 2) * img_width
                ymin = (y_center - height / 2) * img_height
                xmax = (x_center + width / 2) * img_width
                ymax = (y_center + height / 2) * img_height
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
                else:
                    print(f"Invalid box {xmin, ymin, xmax, ymax} in {label_path}")
        
        if not boxes:
            return None, None
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([idx])
        target = {"boxes": boxes, "labels": labels, "image_id": image_id}
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transform_list = []
    transform_list.append(T.ToTensor())
    if train:
        transform_list.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transform_list)

def create_datasets(root):
    train_dataset = SaudiPlatesDataset(root=os.path.join(root, "train"), transforms=get_transform(train=True))
    valid_dataset = SaudiPlatesDataset(root=os.path.join(root, "valid"), transforms=get_transform(train=False))
    return train_dataset, valid_dataset

def collate_fn(batch):
    batch = list(filter(lambda x: x[0] is not None, batch))  # Filter out None entries
    return tuple(zip(*batch))

def count_unique_classes(root):
    labels_path = os.path.join(root, "labels")
    unique_classes = set()
    for label_file in os.listdir(labels_path):
        with open(os.path.join(labels_path, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 1:
                    class_id = int(parts[0])
                    unique_classes.add(class_id)
    return len(unique_classes)

# Use the function to count unique classes
num_classes = count_unique_classes("data/saudi_plates/train") + 1  # Include background class

# Custom FastRCNNPredictor
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

def train_model():
    root = "data/saudi_plates"
    train_dataset, valid_dataset = create_datasets(root)
    data_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
    data_loader_test = DataLoader(valid_dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=collate_fn)

    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
    model = fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    num_epochs = 2

    for epoch in range(num_epochs):
        model.train()
        for images, targets in tqdm(data_loader):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # Debugging statements to inspect shapes
            print("Images:", [img.shape for img in images])
            print("Targets:", targets)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {losses.item()}")

        torch.save(model.state_dict(), f"data/models/saudi_plate_model_{epoch+1}.pth")

    # Evaluate on the validation dataset
    model.eval()
    with torch.no_grad():
        for images, targets in tqdm(data_loader_test):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # Debugging statements to inspect shapes
            print("Validation Images:", [img.shape for img in images])
            print("Validation Targets:", targets)

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
        print(f"Validation Loss: {losses.item()}")

if __name__ == "__main__":
    train_model()
