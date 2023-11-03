import torch.nn as nn
import torch
import torch.functional as F

class FaceClassifier(nn.Module):
    '''
    Face CNN:

    This is primarily a wrapper class for the Face CNN used for classifying the face image.
    
    This model assumes we get a face detection image from a YOLO Face Detector as input (assume it is resized, etc)

    The CNN takes the face image and performs feature extraction, and classification

    https://pytorch.org/vision/stable/models.html#classification
        
    '''
    def __init__(self, config=None):
        super().__init__()

        if config is None:
            config = {
                "model" : "resnet50",
                "weights" : "ResNet50_Weights.IMAGENET1K_V2", 
                "num_classes" : 10,
                "freeze_backbone" : False
            }

        self.config = config
        self._weights = torch.hub.load("pytorch/vision", "get_weight", weights=self.config['weights'])
        self.classifier = torch.hub.load("pytorch/vision", self.config['model'], weights=self._weights)
        self.preprocessor = self._weights.transforms()

        # Option to freeze the backbone
        if self.config['freeze_backbone']:
            for param in self.classifier.parameters():
                param.requires_grad = False

        # Set the model to predict the 10 classes
        num_features = self.classifier.fc.in_features
        self.classifier.fc = nn.Linear(num_features, self.config['num_classes'])

    def forward(self, x):
        '''
        Run the model inference. Note: we assume the preprocessor has been run on the input already
        '''
        return self.classifier(x)