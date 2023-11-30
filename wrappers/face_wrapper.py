import cv2
import torch
from torch import nn 
from torchvision.transforms import v2
from ultralytics import YOLO
from torchvision.transforms import v2

from cnn.face_cnn import extract_face_detections

class Face_Inference_Wrapper(nn.Module):
    def __init__(self, model):
        super(Face_Inference_Wrapper, self).__init__()
        self.detector = YOLO('/home/ron/Classes/CV-Systems/cybertruck/detection/face_detection/pretrained_models/yolov8n-face.pt')
        self.model = model
        self.model.eval()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.resize = v2.Resize((640,640))


    def forward(self, x):
        x = self.resize(x)
        detections = self.detector(x, verbose=False)
        rois = extract_face_detections(x, detections, train_mode=False)
        # rois = self.model.preprocessor(rois)
        rois.to(self.device)
        return self.model(rois)



