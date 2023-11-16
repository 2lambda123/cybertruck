import cv2
import torch
from torch import nn 
from torchvision.transforms import v2
from ultralytics import YOLO


class Hands_Inference_Wrapper(nn.Module):
    def __init__(self, model):
        super(Hands_Inference_Wrapper, self).__init__()
        self.resize = v2.Resize((640,640))
        self.detector = YOLO('detection/hands_detection/runs/detect/best/weights/best.pt')
        self.model = model


    def forward(self, x):
        x = self.resize(x)
        detections = self.detector(x, verbose=False)
        rois = extract_hands_detection(x, detections, model_name='hands_efficient', train_mode=False)
        return self.model(rois)



def visualize_roi(roi):
    roi = roi.cpu().numpy().transpose(1, 2, 0)
    roi_img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    # Display the ROI
    cv2.imshow("Region of Interest", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def get_transforms(model_type='hands_efficient', train_mode=True):

    if model_type == 'hands_efficient' or model_type == 'hands_vgg':
        resize = v2.Resize((224,224))

    elif model_type =="hands_inception":
        resize = v2.Resize((299,299)) 

    elif model_type == 'hands_squeeze':
        resize = v2.Resize((227,227))

    if train_mode:
        transform = v2.Compose([
            v2.ToPILImage(),
            resize,
            v2.RandomHorizontalFlip(p=0.4),
            v2.RandomPerspective(distortion_scale=0.15, p=0.35),
            v2.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = v2.Compose([
            v2.ToPILImage(),
            resize,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    return resize, transform

def extract_hands_detection(images, results, model_name, use_orig_img=True, train_mode=False):
    
    rois = []
    data_list = []

    resize, transform = get_transforms(model_name, train_mode)

    for img_idx, result in enumerate(results):
        num_boxes = len(result.boxes)

        # just pass original image if no hands detected
        if num_boxes == 0: 
            data_list.append(resize(images[img_idx]))
            continue

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


    # Upsample data to 224x224 (or 299x299 if Inception), normalize, and create tensors
    data = torch.stack(data_list)
    data = data.to('cuda')

    return data