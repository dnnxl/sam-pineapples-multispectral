import torch

from torch import nn 
from .models.models_dwv import vit_base_patch16, vit_large_patch16

class DOFAFeatureExtractor(nn.Module):
    def __init__(self, model_size:str="base", checkpoint:bool=False, wave_list=None, device="cpu"):
        super(DOFAFeatureExtractor,self).__init__()
        self.input_size = (224, 224)
        self.wave_list = wave_list
        self.is_transformer = False
        self.device = device

        if model_size == "base":
            self.vit_model = vit_base_patch16()
            path_checkpoint = "weights/DOFA_ViT_base_e100.pth"
        elif model_size == "large":
            self.vit_model = vit_large_patch16()
            path_checkpoint = "weights/DOFA_ViT_large_e100.pth"

        if checkpoint:
            check_point = torch.load(path_checkpoint)
            self.vit_model.load_state_dict(check_point, strict=False)
            self.vit_model = self.vit_model.to(device)
        else:
            self.vit_model = vit_base_patch16()
            self.vit_model = self.vit_model.to(device)

    def forward(self, x):
        x = x.to(self.device)
        return self.vit_model.forward(x, wave_list=self.wave_list)

    def forward_features(self, x):
        x = x.to(self.device)
        return self.vit_model.forward_features(x, wave_list=self.wave_list)