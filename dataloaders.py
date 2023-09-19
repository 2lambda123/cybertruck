from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import ConcatDataset
import pandas as pd
import PIL
import os


class V1Dataset(Dataset):
    def __init__(self, csv_path, images_folder, transform = None):
        self.df = pd.read_csv(csv_path)
        self.images_folder = images_folder
        self.transform = transform
        self.classes = {"Drive Safe":0, "Text Left":1, "Talk Left":2,
                            "Text Right":3, "Talk Right":4, "Adjust Radio":5,
                            "Drink":6, "Hair & Makeup":7, "Reach Behind":8,
                            "Talk Passenger":9}

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        filename = self.df["Image"][index]
        label = self.df["Label"][index]
        filename = filename.replace("distracted.driver", "v1_cam1_no_split")
        root = 'data'
        path = os.path.join(self.images_folder, filename)
        image = PIL.Image.open(root + path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


class V2Dataset(Dataset):
    def __init__(self, cam1_path, cam2_path, transform=None):

        cam1_data = ImageFolder(root=cam1_path, transform=transform)
        cam2_data = ImageFolder(root=cam2_path, transform=transform)
        self.v2_data = ConcatDataset([cam1_data, cam2_data])
        self.classes = cam1_data.classes
        self.classes_dict = {"Safe Driving":"c0", "Text Right":"c1","Phone Right": "c2", "Text Left": "c3", 
                             "Phone Left":"c4", "Adjusting Radio": "c5", "Drinking":"c6", "Reaching Behind":"c7", 
                             "Hair or Makeup":"c8", "Talking to Passenger":"c9"}

    def __len__(self):
        return len(self.v2_data)
    
    def __getitem__(self, index):
        return self.v2_data[index]

