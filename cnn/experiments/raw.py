import torch.hub
from torch import nn

class Raw_Resnet18(nn.Module):
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
        self.classifier.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        '''
        Run the model inference
        '''
        return self.classifier(x)
    
class Raw_Resnet50(nn.Module):
    def __init__(self, args, num_classes=10):
        super().__init__()

        base_config = {
            "model_name": "resnet50",
            "weights": "ResNet50_Weights.IMAGENET1K_V1",
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
        self.classifier.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        '''
        Run the model inference
        '''
        return self.classifier(x)