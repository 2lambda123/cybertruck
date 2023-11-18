import cv2
import torch
from torch import nn 
from torchvision.transforms import v2
from ultralytics import YOLO

from cnn.hands_cnn import extract_hands_detection

class Hands_Inference_Wrapper(nn.Module):
    def __init__(self, model, detector_path):
        super(Hands_Inference_Wrapper, self).__init__()
        self.detector = YOLO(detector_path)
        self.model = model


    def forward(self, x):
        detections = self.detector(x, verbose=False)
        rois = extract_hands_detection(x, detections, None, model_name='hands_vgg', train_mode=False)
        return self.model(rois)



