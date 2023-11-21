import cv2
import torch
from torch import nn 
from torchvision.transforms import v2
from ultralytics import YOLO

from cnn.face_cnn import extract_face_detections

class Face_Inference_Wrapper(nn.Module):
    def __init__(self, model):
        super(Face_Inference_Wrapper, self).__init__()
        self.detector = YOLO('/home/ron/Classes/CV-Systems/cybertruck/detection/face_detection/pretrained_models/yolov8n-face.pt')
        self.model = model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'


    def forward(self, x):
        detections = self.detector(x, verbose=False)
        rois = extract_face_detections(detections)
        rois = self.model.preprocessor(rois)
        rois.to(self.device)
        return self.model(rois)



