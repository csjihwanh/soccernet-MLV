import os
from .utils.config import Config

import io

from .models import VideoChat2_it_mistral
from .utils.easydict import EasyDict
import torch

from transformers import StoppingCriteria, StoppingCriteriaList

from PIL import Image
import numpy as np
import numpy as np
from decord import VideoReader, cpu
import torchvision.transforms as T
from torchvision.transforms import PILToTensor
from torchvision import transforms
from videochat2.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from torchvision import transforms

import matplotlib.pyplot as plt

from IPython.display import Video, HTML

from peft import get_peft_model, LoraConfig, TaskType
import copy

import json
from collections import OrderedDict

from tqdm import tqdm

import time
import decord
decord.bridge.set_bridge("torch")

os.environ['TRANSFORMERS_CACHE'] = '/hub_data5/intern/Ask-Anything/video_chat2/cache'
config_file = "videochat2/configs/config_mistral.json"
cfg = Config.from_file(config_file)

snprompt= """
We need to watch soccer clips and classify them into two categories: the type of action and the level of severity. The definitions for both categories are as follows:

### Type of Action:
1. **Tackling**: The sliding movement of a player towards an opponent who is in possession of the ball and legally or illegally using his foot or leg to try to take the ball away.
2. **Standing Tackling**: The movement (not sliding) of a player towards an opponent who is in possession of the ball and legally or illegally using his foot or leg to try to take the ball away.
3. **Holding**: Occurs when a player’s contact with an opponent’s body or equipment impedes the opponent’s movement.
4. **Pushing**: The action of using the upper body to push an opponent away.
5. **Challenge**: Physical challenge against an opponent, using the shoulder and/or the upper arm.
6. **Elbowing**: The use of arms (and frequently the elbows) as a tool or a weapon to gain an unfair advantage in aerial challenges, physical battles, to create space or to intimidate other players.
7. **High Leg**: A movement where a player swings his foot close to and above the waist of an opponent.
8. **Dive**: An action which creates a wrong/false impression that something has occurred when it has not, committed by a player to gain an unfair advantage.
9. **Don’t Know**: Corresponds to anything which cannot be classified in one of the classes above.

### Level of Severity:
1. **Careless Foul**: A careless foul occurs when a player shows a lack of attention or consideration when making a challenge or acts without precaution. No disciplinary sanction is needed. (No card)
2. **Reckless Foul**: A reckless foul occurs when a player acts with disregard for the danger to, or consequences for, an opponent and must be cautioned. (Yellow card)
3. **Violent Foul**: A violent foul occurs when a player exceeds the necessary use of force and/or endangers the safety of an opponent and must be sent off. (Red card)

"""

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    # 3136 1024 16 4 784
    ''' Sinusoid position encoding table ''' 
    # TODO: make it with torch instead of numpy 
    def get_position_angle_vec(position): 
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)] 
    
    # generate checkpoint position embedding
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2]) # dim 2i 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2]) # dim 2i+1 
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    
    if n_position != pre_n_position:
        T = ckpt_num_frame # checkpoint frame
        P = 14 # checkpoint size
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5) # testing size
        if new_P != 14:
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            # BT, C, H, W -> BT, H, W, C ->  B, T, H, W, C
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
    
    if cur_frame != ckpt_num_frame:

        T = ckpt_num_frame # checkpoint frame
        new_T = cur_frame # testing frame
        # interpolate
        P = int((n_position // cur_frame) ** 0.5) # testing size
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)  # BHW, C, T
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3) # B, T, H, W, C
        sinusoid_table = sinusoid_table.flatten(1, 3)  # B, THW, C
        
    return sinusoid_table



import torch
import torch.nn as nn
import torch.nn.functional as F

class VideoChat2(nn.Module):
    def __init__(self, pre_weight):
        super(VideoChat2, self).__init__()
        
        cfg.model.vision_encoder.num_frames = 4
        self.model = VideoChat2_it_mistral(config=cfg.model)

        if pre_weight == False: 
            state_dict = torch.load("checkpoints/videochat2/videochat2_mistral_7b_stage3.pth", map_location='cuda')

            if 'model' in state_dict.keys():
                msg = self.model.load_state_dict(state_dict['model'], strict=False)
            else:
                msg = self.model.load_state_dict(state_dict, strict=False)

        del self.model.mistral_model
        
        

    def forward(self, vid):
        #b, tc, h, w
        resolution = 224

        num_frame = vid.shape[1]//3

        new_pos_emb = get_sinusoid_encoding_table(n_position=(resolution//16)**2*num_frame, cur_frame=num_frame)
        self.model.vision_encoder.encoder.pos_embed = new_pos_emb

        BV, TC, H, W = vid.shape
        video = vid.reshape(BV, TC//3, 3, H, W)

        img_list = []

        image_emb, _ = self.model.encode_img(video, [snprompt]*BV)

        return image_emb

