import torch.nn as nn
import torch
# import torch.functional as F
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import v2, PILToTensor

from hands_cnn import get_transforms, visualize_roi

class Face_CNN(nn.Module):
    '''
    Face CNN:

    This is primarily a wrapper class for the Face CNN used for classifying the face image.

    This model assumes we get a face detection image from a YOLO Face Detector as input (assume it is resized, etc)

    The CNN takes the face image and performs feature extraction, and classification

    https://pytorch.org/vision/stable/models.html#classification
        
    '''

    def __init__(self, args, num_classes=10):
        super().__init__()

        base_config = {
            "model_name": "resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "freeze": False
        }

        # Update the base config with the args
        base_config.update(vars(args))

        # Store our config and initialize
        self.config = base_config
        self._weights = torch.hub.load("pytorch/vision", "get_weight", name=self.config['weights'])
        self.classifier = torch.hub.load("pytorch/vision", self.config['model_name'], weights=self._weights)
        self.preprocessor = self._weights.transforms()

        feature_extractor = nn.Sequential(*list(self.classifier.children())[:-1])
        num_frozen_params = len(list(feature_extractor.parameters()))
        if self.config['freeze']: feature_extractor = self.freeze(feature_extractor, num_frozen_params)

        # Set the model to predict the 10 classes
        num_features = self.classifier.fc.in_features
        mlp = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_classes)
        )

        self.model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            mlp
        )

    def forward(self, x):
        '''
        Run the model inference
        '''
        return self.classifier(x)

    def freeze(self, feature_extractor, num_frozen_params):
        for param in list(feature_extractor.parameters())[: num_frozen_params]:
            param.requires_grad = False
        return feature_extractor


def extract_face_detections(images, results, train_mode):
    '''
    Take the results from the YOLO detector and get the top detection for each image
    '''

    outputs = []
    device = 'cuda'

    resize, transform = get_transforms(model_type='face', train_mode=train_mode)

    for image_idx, r in enumerate(results):
        boxes = r.boxes
        # Get the best box
        try:
            # Get the best detection for the image
            bb_index = torch.argmax(boxes.conf)
            x, y, x2, y2 = boxes[bb_index].xyxy.squeeze().tolist()
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            # Get the Crop of that image
            box = r.orig_img[y:y2, x:x2]
            box_image = Image.fromarray(box[..., ::-1])  # RGB PIL image
            box_image = resize(PILToTensor()(box_image)).to(device)
            # box_image = box_image.to(device)
            #            box_image = F.resize(box_image, (500, 500))
            #            box_image.resize((500, 500)) # Resize

            image = images[image_idx]
            orig_image = resize(image)

            stacked_image = transform(torch.cat((orig_image, box_image), dim=1))
            stacked_image = stacked_image.to(device)
            # visualize_roi(stacked_image)
            outputs.append(stacked_image)
        except Exception:
            # Create a blank image if no face is detected
            #            outputs.append(Image.new("RGB", (500, 500)))
            blank_image = Image.new("RGB", (500, 500))
            blank_tensor = transform(blank_image)
            blank_tensor = blank_tensor.to(device)
            outputs.append(blank_tensor)

    #    return torch.tensor(outputs)

    outputs = torch.stack(outputs)
    
    return outputs.to(device)
