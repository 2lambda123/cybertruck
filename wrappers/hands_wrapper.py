import cv2
import torch
from torch import nn 
from torchvision.transforms import v2
from ultralytics import YOLO

from cnn.hands_cnn import extract_hands_detection

class Hands_Inference_Wrapper(nn.Module):
    def __init__(self, model):
        super(Hands_Inference_Wrapper, self).__init__()
        self.detector = YOLO('../detection/hands_detection/runs/detect/best/weights/best.pt')
        self.model = model


    def forward(self, x):
        detections = self.detector(x, verbose=False)
        rois = extract_hands_detection(x, detections, None, model_name='hands_vgg', train_mode=False)
        return self.model(rois)



