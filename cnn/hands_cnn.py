import cv2
import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision.models import vgg16, VGG16_Weights



class Hands_VGG16(nn.Module):
    '''
    Model which from the experiments we ran had the best performance

    args: The arguments from the command line from main.py
    num_classes: The number of classes in the dataset (10)
    '''
    def __init__(self, args, num_classes=10):
        super(Hands_VGG16, self).__init__()
        # We are only using the feature extractor from VGG16, and freezing it
        # to leverage the pre-trained ImageNet weights. We then add our own classifier.
        feature_extractor = vgg16(weights=VGG16_Weights.DEFAULT).features
        if args.freeze: feature_extractor = self.freeze(feature_extractor, 30)  

        in_features = feature_extractor[-3].out_channels 
        classifier = nn.Sequential(
            nn.Linear(in_features * 7 * 7, 4096),
            nn.ReLU(),
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
    


def visualize_roi(roi):
    '''
    Displays the region of interest (for debugging)
    '''
    roi = roi.cpu().numpy().transpose(1, 2, 0)
    roi_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Display the ROI
    cv2.imshow("Region of Interest", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_transforms(model_type='hands_efficient', train_mode=True):
    '''
    Returns the transforms for the given model type
    '''
    if model_type == 'hands_efficient' or model_type == 'hands_vgg':
        resize = v2.Resize((224,224))

    elif model_type == 'face':
        resize = v2.Resize((299,299))

    # elif model_type =="hands_inception":
    #     resize = v2.Resize((299,299)) 

    # elif model_type == 'hands_squeeze':
    #     resize = v2.Resize((227,227))

    if train_mode:
        transform = v2.Compose([
            v2.ToPILImage(),
            resize,
            v2.RandomHorizontalFlip(p=0.4),
            v2.RandomPerspective(distortion_scale=0.1, p=0.25),
            # v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.3102, 0.3102, 0.3102], std=[0.3151, 0.3151, 0.3151])
        ])
    else:
        transform = v2.Compose([
            v2.ToPILImage(),
            resize,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.3879, 0.3879, 0.3879], std=[0.3001, 0.3001, 0.3001])
        ])

    return resize, transform


def concatenate_boxes(result, image, num_boxes, resize, transform, use_orig_img):
    '''
    Helper function for extract_hands_detection. Performs the concatenation of the region of interest(s) and the original image
    as explained in extract_hands_detection.

    return stacked_rois: The result from the concatenations of all images
    '''
    rois = []

     # if more than 2 detections, select the top 2 to exclude false positives
    if num_boxes > 2:
        _, top_idxs = torch.topk(result.boxes.conf, k=2)
    else:
        top_idxs = None


    for box, cls in zip(result.boxes.xyxy[top_idxs].squeeze(0),result.boxes.cls[top_idxs].squeeze(0)):
        
        # Convert coordinates to absolute coordinates
        x_min, y_min, x_max, y_max = box
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
        
        # Extract the region of interest and append to detections list
        roi = image[:,y_min:y_max, x_min:x_max] 
        rois.append(roi)

    # if the second element in tensor is a left hand, switch order of rois so that left hand is always first.
    if cls == 0: rois.reverse()


    # if multiple detections, resize, stack vertically, and transform
    if num_boxes > 1:
        transformed_rois = [resize(roi) for roi in rois]
        stacked_rois = resize(torch.cat(transformed_rois, dim=2))
    else:
        stacked_rois = resize(roi)

    rois.clear()

    # if True, horizontally concatenates the original image with the rois
    if use_orig_img:
        orig_image = resize(image)
        stacked_rois = transform(torch.cat((orig_image, stacked_rois), dim=1))
    else:
        stacked_rois = transform(stacked_rois)

    return stacked_rois


def extract_hands_detection(images, results, target, model_name, use_orig_img=True, train_mode=True):
    '''
    Extracts the region of interest(s) from the image. If there are multiple hand detections,
    left hand detection will be concatenated to the left side, right hand to right.
    If there are no hand detections, the image will be skipped.

    images: The original tensor images
    results: The results from the detector
    target: The target labels. If target is None, we are extracting features for the ensemble model, and we
            need to ensure all models have the same number of images/targets.
    model_name: The name of the model (used in get_transforms)
    use_orig_img: If True, horizontally concatenates the image with the rois for performance boost.
    train_mode: If True, uses the training transforms. If False, uses the testing transforms.

    return data, target: The data and target tensors
    '''
    # target will only be None if we are extracting features for the ensemble model

    data_list = []
    target_list = []

    resize, transform = get_transforms(model_name, train_mode)

    # iterates through the results for every batch
    for img_idx, result in enumerate(results):
        num_boxes = len(result.boxes)

        # image is not useful if no hands are detected. If we are extracting features for the ensemble model, we still want to use the image
        # because we need the same number of images across all models.
        if num_boxes == 0:
            if target is None:
                data_list.append(transform(images[img_idx]))
            continue

        stacked_rois = concatenate_boxes(result, images[img_idx], num_boxes, resize, transform, use_orig_img)

        # visualize_roi(stacked_rois)
        
        data_list.append(stacked_rois)

        if target is not None:
            target_list.append(target[img_idx])


    # Upsample data to 224x224 (or 299x299 if Inception), normalize, and create tensors
    data = torch.stack(data_list)
    data = data.to('cuda')

    # if target is not None, we are training the model and need to return the target tensor, 
    # else we are extracting features for the ensemble so we need to return the same sized data/target tensors for all models
    if target is not None:
        target = torch.stack(target_list)

        assert data.shape[0] == target.shape[0], 'Batch size of data must be equal to target length.'
        return data, target
    
    else:
        return data