import torch.nn as nn
from torchvision.models import vgg16, VGG16_Weights

class Hands_CNN(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_CNN, self).__init__()

        feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        if args.freeze: feature_extractor = self.freeze(feature_extractor, args.num_frozen_params)  

        in_features = feature_extractor[-3].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, args.hidden_units),
            nn.ReLU(),
            nn.Linear(args.hidden_units, out_features),
        )

        self.model = nn.Sequential(
            feature_extractor,
            nn.Flatten(),
            classifier,
        )
    
    def freeze(self, feature_extractor, num_frozen_params):
        for param in list(feature_extractor.parameters())[: num_frozen_params]:
            param.requires_grad = False
        return feature_extractor
    

    def forward(self, x):
        return self.model(x)
