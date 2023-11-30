import torch.nn as nn
import timm


class Raw_CNN(nn.Module):
    '''
    Raw CNN:        
    '''

    def __init__(self, args, num_classes=10):
        super().__init__()


        self.model = timm.create_model('xception', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        '''
        Run the model inference
        '''

        return self.model(x)


        
        
        