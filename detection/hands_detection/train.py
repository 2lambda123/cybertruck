from ultralytics import YOLO
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

model = YOLO('runs/detect/mds_model_w_aug(best)/weights/best.pt')

results = model.train(data='egohands.yaml', 
                      epochs=150, batch=124, save_period=10, freeze=7, augment=True, single_cls=True,
                      cos_lr=True, degrees=0.1, translate=0.15, perspective=0.0003, flipud=0.2, resume=True)



