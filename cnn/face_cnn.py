import torch.nn as nn
import torch
import torch.functional as F
from PIL import Image

class Face_CNN(nn.Module):
    '''
    Face CNN:

    This is primarily a wrapper class for the Face CNN used for classifying the face image.
    
    This model assumes we get a face detection image from a YOLO Face Detector as input (assume it is resized, etc)

    The CNN takes the face image and performs feature extraction, and classification

    https://pytorch.org/vision/stable/models.html#classification
        
    '''
    def __init__(self, args, out_features=10):
        super().__init__()

        base_config = {
            "model_name" : "resnet50",
            "weights" : "ResNet50_Weights.IMAGENET1K_V2",
            "freeze" : False 
        }

        # Update the base config with the args
        base_config.update(vars(args))

        # Store our config and initialize
        self.config = base_config
        self._weights = torch.hub.load("pytorch/vision", "get_weight", weights=self.config['weights'])
        self.classifier = torch.hub.load("pytorch/vision", self.config['model_name'], weights=self._weights)
        self.preprocessor = self._weights.transforms()

        if self.config['freeze']: feature_extractor = self.freeze(feature_extractor, self.config['num_frozen_params'])  

        # Set the model to predict the 10 classes
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_features, out_features)

    def forward(self, x):
        '''
        Run the model inference
        '''
        return self.classifier(x)
    
    def freeze(self, feature_extractor, num_frozen_params):
        for param in list(feature_extractor.parameters())[: num_frozen_params]:
            param.requires_grad = False
        return feature_extractor

def extract_face_detections(results):
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
            x, y, x2, y2 = boxes[bb_index].xyxy.squeeze().tolist()
            x, y, x2, y2 = int(x), int(y), int(x2), int(y2)

            # Get the Crop of that image
            box = r.orig_img[y:y2, x:x2]
            box_image = Image.fromarray(box[..., ::-1]) # RGB PIL image
            box_image.resize((500, 500)) # Resize

            outputs.append(box_image)
        except Exception:
            # Create a blank image if no face is detected
            outputs.append(Image.new("RGB", (500, 500)))

    return torch.tensor(outputs)