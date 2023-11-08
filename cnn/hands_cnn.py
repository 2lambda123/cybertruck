import cv2
import torch
import torch.nn as nn
from torchvision import transforms
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
    

def visualize_roi(roi):
    roi = roi.cpu().numpy().transpose(1, 2, 0)
    roi_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Display the ROI
    cv2.imshow("Region of Interest", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def extract_hands_detection(images, results, target):
    
    rois = []
    data_list = []
    target_list = []

    resize = transforms.Resize((224,224))

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    for idx, result in enumerate(results):
        num_boxes = len(result.boxes)

        if num_boxes == 0: continue

        for box in result.boxes.xyxy:
            
            # Convert normalized coordinates to absolute coordinates
            x_min, y_min, x_max, y_max = box
        
            # Convert normalized coordinates to absolute coordinates
            x_min = int(x_min)
            y_min = int(y_min)
            x_max = int(x_max)
            y_max = int(y_max)

            roi = images[idx][:,y_min:y_max, x_min:x_max] 
            rois.append(roi)

        # if multiple detections, resize, stack vertically, and transform
        if num_boxes > 1:
            transformed_rois = [resize(roi) for roi in rois]
            stacked_rois = transform(torch.cat(transformed_rois, dim=2))
        else: stacked_rois = transform(roi)

        # visualize_roi(stacked_rois)
        
        data_list.append(stacked_rois)
        target_list.append(target[idx])


    # Upsample data to 224x224, normalize, and create tensors
    data = torch.stack(data_list)
    data = data.to('cuda')

    target = torch.stack(target_list)

    assert data.shape[0] == target.shape[0], 'Batch sizes for data and target must be equal.'

    return data, target
