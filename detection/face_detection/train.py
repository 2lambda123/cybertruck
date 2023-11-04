import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.datasets import YourDataset  # Replace with your dataset
import os
from FaceClassifier import FaceClassifier
from ultralytics import YOLO
from PIL import Image

def get_box_crops(results):
    '''
    Take the results from the YOLO detector and get the top detection for each image
    '''
    outputs = []

    for r in results:
        boxes = r.boxes
        # Get the best box
        try:
            # Get the best detection for the image
            bb_index = torch.argmax(boxes.conf)
            best_box_coords = boxes[bb_index]
            x, y, x2, y2 = best_box_coords.xyxy.squeeze().tolist()
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            # Get the Crop of that image
            box = r.orig_img[y:y2, x:x2]
            box_image = Image.fromarray(box[..., ::-1]) # # RGB PIL image

            outputs.append(box_image)
        except Exception:
            # Create a blank image if no face is detected
            outputs.append(Image.new("RGB", (500, 500)))

    return torch.tensor(outputs)

classifier_model = FaceClassifier()
preprocessor = classifier_model.preprocessor

batch_size = 32
learning_rate = 0.001
num_epochs = 10
checkpoint_dir = "checkpoints"

# Initialize the dataset and DataLoader
train_dataset = YourDataset(root="path_to_your_dataset", transform=preprocessor)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

writer = SummaryWriter()

os.makedirs(checkpoint_dir, exist_ok=True)

yolo_pretrained_weights_path = r"./pretrained_models/yolov8n-face.pt"
detector_model = YOLO(yolo_pretrained_weights_path)

# Define the loss function and optimizer for classification
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    detector_model.eval()
    classifier_model.train()  # Set the classification model to training mode
    running_loss = 0.0
    correct, total = 0, 0

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data

        optimizer.zero_grad()

        # YOLO object detection 
        with torch.no_grad():
            results = detector_model(inputs)
        
       # Get the face detections as input for the classifier
        face_images = get_box_crops(results)
        face_images = preprocessor(face_images)

        # Forward pass through the classification model
        outputs = classifier_model(face_images)

        # Calculate loss
        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss for the epoch
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Log loss to Tensorboard
    writer.add_scalar("Train/Loss", avg_loss, epoch)
    writer.add_scalar("Train/Accuracy", accuracy, epoch)

    # Save a checkpoint after each epoch
    checkpoint_name = f"checkpoint_epoch_{epoch + 1}.pt"
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    torch.save(classifier_model.state_dict(), checkpoint_path)

print("Training finished")

# Close the SummaryWriter
writer.close()
