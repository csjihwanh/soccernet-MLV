from torch.utils.data import Dataset
from random import random
import torch
import random
from .data_loader import label2vectormerge, clips2vectormerge
from decord import VideoReader, cpu
import torchvision.transforms as T
from videochat2.dataset.video_transforms import (
    GroupNormalize, GroupScale, GroupCenterCrop, 
    Stack, ToTorchFormatTensor
)
from PIL import Image
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import random
import einops


class MultiViewDataset(Dataset):
    def __init__(self, path, start, end, fps, split, num_views, transform=None):

        if split != 'Chall':
            # To load the annotations
            self.labels_offence_severity, self.labels_action, self.distribution_offence_severity,self.distribution_action, not_taking, self.number_of_actions = label2vectormerge(path, split, num_views)
            self.clips = clips2vectormerge(path, split, num_views, not_taking)
            self.distribution_offence_severity = torch.div(self.distribution_offence_severity, len(self.labels_offence_severity))
            self.distribution_action = torch.div(self.distribution_action, len(self.labels_action))

            self.weights_offence_severity = torch.div(1, self.distribution_offence_severity)
            self.weights_action = torch.div(1, self.distribution_action)
        else:
            self.clips = clips2vectormerge(path, split, num_views, [])

        # INFORMATION ABOUT SELF.LABELS_OFFENCE_SEVERITY
        # self.labels_offence_severity => Tensor of size of the dataset. 
        # each element of self.labels_offence_severity is another tensor of size 4 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if it is not an offence, then the tensor is [1, 0, 0, 0]. 

        # INFORMATION ABOUT SELF.LABELS_ACTION
        # self.labels_action => Tensor of size of the dataset. 
        # each element of self.labels_action is another tensor of size 8 (the number of classes) where the value is 1 if it is the class and 0 otherwise
        # for example if the action is a tackling, then the tensor is [1, 0, 0, 0, 0, 0, 0, 0]. 

        # INFORMATION ABOUT SLEF.CLIPS
        # self.clips => list of the size of the dataset
        # each element of the list is another list of size of the number of views. The list contains the paths to all the views of that particular action.

        # The offence_severity groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_offence_severity tensor
        # The type of action groundtruth of the i-th action in self.clips, is the i-th element in the self.labels_action tensor
        
        self.split = split
        self.start = start
        self.end = end
        self.transform = transform
        self.num_views = num_views

        self.factor = (end - start) / (((end - start) / 25) * fps)

        self.length = len(self.clips)
        print(self.length)

    def getDistribution(self):
        return self.distribution_offence_severity, self.distribution_action, 
    def getWeights(self):
        return self.weights_offence_severity, self.weights_action, 


    # RETURNS
    #
    # self.labels_offence_severity[index][0] => tensor of size 4. Example [1, 0, 0, 0] if the action is not an offence
    # self.labels_action[index][0] => tensor of size 8.           Example [1, 0, 0, 0, 0, 0, 0, 0] if the type of action is a tackling
    # videos => tensor of shape V, C, N, H, W with V = number of views, C = number of channels, N = the number of frames, H & W = height & width
    # self.number_of_actions[index] => the id of the action
    #
    def __getitem__(self, index):

        prev_views = []
        target_views = 4

        for num_view in range(len(self.clips[index])):

            index_view = num_view

            # Add the current view to prev_views if it's not already present
            if index_view not in prev_views:
                prev_views.append(index_view)

            # As we use a batch size > 1 during training, we always randomly select two views even if we have more than two views.
            # As the batch size during validation and testing is 1, we can have 2, 3 or 4 views per action.
            
            video = load_video(self.clips[index][index_view], transform_aug=self.transform, start_frame=self.start, end_frame=self.end)
            
            if num_view == 0:
                videos = video.unsqueeze(0)
            else:
                video = video.unsqueeze(0)
                videos = torch.cat((videos, video), 0)
        
        if len(prev_views) == 2:
            video1 = videos[1].unsqueeze(0)
            videos = torch.cat((videos, video1), 0)
            videos = torch.cat((videos, video1), 0)
        elif len(prev_views) == 3:
            rand_idx = random.choice([1, 2])
            video1 = videos[rand_idx].unsqueeze(0)
            videos = torch.cat((videos, video1), 0)

        if self.num_views != 1 and self.num_views != 5:
            videos = videos.squeeze()   

        if self.split != 'Chall':
            return self.labels_offence_severity[index][0], self.labels_action[index][0], videos, self.number_of_actions[index]
        else:
            return -1, -1, videos, str(index)

    def __len__(self):
        return self.length


def get_frame_indices(start_frame, end_frame, num_segments):
    """Generate frame indices from start_frame to end_frame divided into num_segments."""
    indices = np.linspace(start_frame, end_frame, num_segments, dtype=int)
    return indices

def load_video(video_path, start_frame, end_frame, transform_aug= None, return_msg=False, resolution=224):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    
    num_frames = len(vr)

    frame_indices = range(start_frame, end_frame)

    # transform
    crop_size = resolution
    scale_size = resolution
    input_mean = [0.48145466, 0.4578275, 0.40821073]
    input_std = [0.26862954, 0.26130258, 0.27577711]

    transform = T.Compose([
        GroupScale(int(scale_size), interpolation=InterpolationMode.BICUBIC),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std) 
    ])

    images_group = list()
    for frame_index in frame_indices:
        img = Image.fromarray(vr[frame_index].numpy())
        images_group.append(img)
    torch_imgs = transform(images_group)

    if transform_aug is not None:
        TC, H, W = torch_imgs.shape
        torch_imgs = torch_imgs.reshape((TC//3, 3, H, W))
        torch_imgs = transform_aug(torch_imgs)
        torch_imgs = torch_imgs.reshape((TC,H,W))

    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        # " " should be added in the start and end
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return torch_imgs, msg
    else:
        return torch_imgs