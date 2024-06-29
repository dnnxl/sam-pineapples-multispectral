import torch

from torch import nn 
from .models.models_dwv import vit_base_patch16, vit_large_patch16

class DOFAFeatureExtractor(nn.Module):
    def __init__(self, model_size:str="base", checkpoint:str=None, wave_list=None):
        super(DOFAFeatureExtractor,self).__init__()
        self.input_size = (224, 224)
        self.wave_list = wave_list
        self.is_transformer = False

        if model_size == "base":
            self.vit_model = vit_base_patch16()
        elif model_size == "large":
            self.vit_model = vit_large_patch16()
        
        if checkpoint != None:
            check_point = torch.load(checkpoint)
            self.vit_model.load_state_dict(check_point, strict=False)

    def forward(self, x):
        return self.vit_model.forward(x, wave_list=self.wave_list)

    def forward_features(self, x):
        return self.vit_model.forward_features(x, wave_list=self.wave_list)