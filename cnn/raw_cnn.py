import torch.nn as nn
import torch
# import torch.functional as F
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.transforms import v2, PILToTensor

from hands_cnn import get_transforms, visualize_roi

class Raw_CNN(nn.Module):
    '''
    Raw CNN:        
    '''

    def __init__(self, args, num_classes=10):
        super().__init__()

        base_config = {
            "model_name": "resnet18",
            "weights": "ResNet18_Weights.IMAGENET1K_V1",
            "freeze": True
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
            nn.Dropout(args.dropout),
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