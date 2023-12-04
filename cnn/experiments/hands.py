import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights
from torchvision.models._api import WeightsEnum

class Hands_Squeeze(nn.Module):
    '''
    Second best performing model
    '''
    def __init__(self, args, num_classes=10):
        super(Hands_Squeeze, self).__init__()

        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)
        self.model.classifier[1].out_channels = num_classes

        num_frozen_params = len(list(self.model.features.parameters()))
        if args.freeze: self.freeze(num_frozen_params)  

    
    def freeze(self, num_frozen_params):
        for param in list(self.model.parameters())[: num_frozen_params]:
            param.requires_grad = False
    

    def forward(self, x):
        return self.model(x)

class Hands_InceptionV3(nn.Module):
    '''Model used in the paper which had the best performance'''
    def __init__(self, args, num_classes=10):
        super(Hands_InceptionV3, self).__init__()

        self.inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
        self.inception_model.fc.out_features = num_classes

        # self.feature_extractor = inception_model = nn.Sequential(
        #             inception_model.Conv2d_1a_3x3,
        #             inception_model.Conv2d_2a_3x3,
        #             inception_model.Conv2d_2b_3x3,
        #             nn.MaxPool2d(kernel_size=3, stride=2),
        #             inception_model.Conv2d_3b_1x1,
        #             inception_model.Conv2d_4a_3x3,
        #             nn.MaxPool2d(kernel_size=3, stride=2),
        #             inception_model.Mixed_5b,
        #             inception_model.Mixed_5c,
        #             inception_model.Mixed_5d,
        #             inception_model.Mixed_6a,
        #             nn.AdaptiveAvgPool2d(output_size=(1, 1))
        # )


        num_frozen_params = len(list(self.inception_model.parameters())) - 3

        if args.freeze: self.freeze(num_frozen_params)  

        # self.classifier = nn.Sequential(
        #     nn.Linear(in_features=768, out_features=256, bias=True),
        #     nn.ReLU(),
        #     nn.Dropout(p=args.dropout),
        #     nn.Linear(256, num_classes, bias=True),
        # )
    
    def freeze(self, num_frozen_params):
        for param in list(self.inception_model.parameters())[: num_frozen_params]:
            param.requires_grad = False
    

    def forward(self, x):
        x = self.inception_model(x)
        # x = x.squeeze(2).squeeze(2)
        # x = self.classifier(x)
        return x
    

def get_state_dict(self, *args, **kwargs):
        kwargs.pop("check_hash")
        return load_state_dict_from_url(self.url, *args, **kwargs)



class Hands_Efficient(nn.Module):
    def __init__(self, args, num_classes=10):
        super(Hands_Efficient, self).__init__()

        # # Workaround for downloading weights from url without encountering hash issue. Only needed for initial download.
        # WeightsEnum.get_state_dict = get_state_dict

        feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

        num_frozen_params = len(list(feature_extractor.parameters()))

        if args.freeze: feature_extractor = self.freeze(feature_extractor, num_frozen_params)  

        in_features = feature_extractor[-1][0].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, num_classes),
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