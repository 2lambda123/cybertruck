import cv2
import torch
import torch.nn as nn
from torchvision.transforms import v2

from torchvision.models import vgg16, VGG16_Weights
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchvision.models import squeezenet1_1, SqueezeNet1_1_Weights

from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


class Hands_VGG16(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_VGG16, self).__init__()

        feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        if args.freeze: feature_extractor = self.freeze(feature_extractor, args.num_frozen_params)  

        in_features = feature_extractor[-3].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Linear(1000, out_features),
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
    

class Hands_Efficient(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_Efficient, self).__init__()


        def get_state_dict(self, *args, **kwargs):
            kwargs.pop("check_hash")
            return load_state_dict_from_url(self.url, *args, **kwargs)

        WeightsEnum.get_state_dict = get_state_dict

        feature_extractor = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT).features

        num_frozen_params = len(list(feature_extractor.parameters()))

        if args.freeze: feature_extractor = self.freeze(feature_extractor, num_frozen_params)  

        in_features = feature_extractor[-1][0].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(1000, out_features),
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
    
class Hands_Squeeze(nn.Module):
    def __init__(self, args, out_features=10):
        super(Hands_Squeeze, self).__init__()

        self.model = squeezenet1_1(weights=SqueezeNet1_1_Weights.DEFAULT)

        num_frozen_params = len(list(self.model.features.parameters()))
        if args.freeze: self.freeze(num_frozen_params)  

    
    def freeze(self, num_frozen_params):
        for param in list(self.model.parameters())[: num_frozen_params]:
            param.requires_grad = False
    

    def forward(self, x):
        return self.model(x)
    
class Hands_InceptionV3(nn.Module):
    '''Model used in the paper which had the best performance'''
    def __init__(self, args, out_features=10):
        super(Hands_InceptionV3, self).__init__()

        inception_model = inception_v3(weights=Inception_V3_Weights.DEFAULT)

        self.feature_extractor = inception_model = nn.Sequential(
                    inception_model.Conv2d_1a_3x3,
                    inception_model.Conv2d_2a_3x3,
                    inception_model.Conv2d_2b_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    inception_model.Conv2d_3b_1x1,
                    inception_model.Conv2d_4a_3x3,
                    nn.MaxPool2d(kernel_size=3, stride=2),
                    inception_model.Mixed_5b,
                    inception_model.Mixed_5c,
                    inception_model.Mixed_5d,
                    inception_model.Mixed_6a,
                    nn.AdaptiveAvgPool2d(output_size=(1, 1))
        )


        num_frozen_params = len(list(self.feature_extractor.parameters()))

        if args.freeze: self.freeze(num_frozen_params)  

        self.classifier = nn.Sequential(
            nn.Linear(in_features=768, out_features=256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=args.dropout),
            nn.Linear(256, out_features, bias=True),
        )
    
    def freeze(self, num_frozen_params):
        for param in list(self.feature_extractor.parameters())[: num_frozen_params]:
            param.requires_grad = False
    

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.squeeze(2).squeeze(2)
        x = self.classifier(x)
        return x    




def visualize_roi(roi):
    roi = roi.cpu().numpy().transpose(1, 2, 0)
    roi_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Display the ROI
    cv2.imshow("Region of Interest", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_transforms(model_type='hands_efficient'):

    if model_type == 'hands_efficient' or model_type == 'hands_vgg':
        resize = v2.Resize((224,224)) 

    elif model_type =="hands_inception":
        resize = v2.Resize((299,299)) 

    elif model_type == 'hands_squeeze':
        resize = v2.Resize((227,227))

    transform = v2.Compose([
        v2.ToPILImage(),
        resize,
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        # v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return resize, transform

def extract_hands_detection(images, results, target, use_orig_img=True):
    
    rois = []
    data_list = []
    target_list = []

    resize, transform = get_transforms(model_type='hands_efficient')

    for img_idx, result in enumerate(results):
        num_boxes = len(result.boxes)

        # image is not useful if no hands are detected
        if num_boxes == 0: continue

        # if more than 2 detections, select the top 2 to exclude false positives
        if num_boxes > 2:
            _, top_idxs = torch.topk(result.boxes.conf, k=2)
        else:
            top_idxs = None

        for box, cls in zip(result.boxes.xyxy[top_idxs].squeeze(0),result.boxes.cls[top_idxs].squeeze(0)):
            
            # Convert coordinates to absolute coordinates
            x_min, y_min, x_max, y_max = box
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)


            roi = images[img_idx][:,y_min:y_max, x_min:x_max] 
            rois.append(roi)

        # if the second element in tensor is a left hand, switch order of rois so that left hand is always first.
        if cls == 0: rois.reverse()


        # if multiple detections, resize, stack vertically, and transform
        if num_boxes > 1:
            transformed_rois = [resize(roi) for roi in rois]
            stacked_rois = resize(torch.cat(transformed_rois, dim=2))
        else: stacked_rois = resize(roi)

        rois.clear()

        # if True, horizontally concatenates the image with the rois
        if use_orig_img:
            orig_image = resize(images[img_idx])
            stacked_rois = transform(torch.cat((orig_image, stacked_rois), dim=1))
        else:
            stacked_rois = transform(stacked_rois)


        # visualize_roi(stacked_rois)
        
        data_list.append(stacked_rois)
        target_list.append(target[img_idx])


    # Upsample data to 224x224 (or 299x299 if Inception), normalize, and create tensors
    data = torch.stack(data_list)
    data = data.to('cuda')

    target = torch.stack(target_list)

    assert data.shape[0] == target.shape[0], 'Batch size of data must be equal to target length.'

    return data, target
