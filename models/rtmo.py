import logging
from typing import Dict, List, Optional, Sequence, Union
from mmcv.image import imread
import mmcv

from mmengine.logging import print_log

from mmpose.apis import inference_bottomup, init_model
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData
from mmpose.apis.inferencers import MMPoseInferencer
from mmengine.dataset import Compose, pseudo_collate
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nearby_joints_nms, nms

from mmpose.registry import VISUALIZERS

import torch.nn as nn 
import torchvision.transforms as transforms
import torch.nn.functional as F

import matplotlib.pyplot as plt


try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import torch
import numpy  as np

register_all_modules()

test_img = '../test/test.jpeg'
out_path = 'output.jpg'
 # or device='cuda:0'

InstanceList = List[InstanceData]
InputType = Union[str, np.ndarray]
InputsType = Union[InputType, Sequence[InputType]]
PredType = Union[InstanceData, InstanceList]
ImgType = Union[np.ndarray, Sequence[np.ndarray]]
ConfigType = Union[Config, ConfigDict]
ResType = Union[Dict, List[Dict], InstanceData, List[InstanceData]]

def visualize_tensor(img_tensor, dest):
    img_np = img_tensor.permute(1, 2, 0).detach().cpu().numpy() 
    img_np = (img_np * 255).astype(np.uint8)

    plt.imshow(img_np)
    plt.title("Tensor Image Visualization")
    plt.axis('off')  
    output_path = (f"test/{dest}.jpg")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def inference_bottomup_modified(model: nn.Module, img: Union[np.ndarray, str]):
    """Inference image with a bottom-up pose estimator.

    Args:
        model (nn.Module): The bottom-up pose estimator
        img (np.ndarray | str): The loaded image or image file to inference

    Returns:
        List[:obj:`PoseDataSample`]: The inference results. Specifically, the
        predicted keypoints and scores are saved at
        ``data_sample.pred_instances.keypoints`` and
        ``data_sample.pred_instances.keypoint_scores``.
    """

    pipeline = Compose(model.cfg.test_dataloader.dataset.pipeline)

    # prepare data batch
    if isinstance(img, str):
        data_info = dict(img_path=img)
    else:
        data_info = dict(img=img)
    data_info.update(model.dataset_meta)
    data = pipeline(data_info)
    batch = pseudo_collate([data])
    x = torch.from_numpy(img)
    
    results = model.test_step(batch)
    return results


class RTMOBackbone(torch.nn.Module):
    def __init__(self, device='cuda', full_model = False, full_model_weight=False, mode='tiny'):
        super().__init__()
        
        rtmo_t_cfg = 'config/rtmo-t_8xb32-600e_body7-416x416.py'
        rtmo_s_cfg = 'config/rtmo-s_8xb32-600e_body7-640x640.py'
        rtmo_l_cfg = 'config/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py'
        
        rtmo_t_weight = 'checkpoints/rtmo-t_8xb32-600e_body7-416x416-f48f75cb_20231219.pth'
        rtmo_s_weight = 'checkpoints/rtmo-s_8xb32-600e_body7-640x640-dac2bf74_20231211.pth'
        rtmo_l_weight = 'checkpoints/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth'
        
        if mode == 'tiny':
            cfg = rtmo_t_cfg
            weight = rtmo_t_weight
        if mode == 'small':
            cfg = rtmo_s_cfg
            weight = rtmo_s_weight


        self.init_setting = {
            'pose2d': cfg,
            'pose2d_weights': weight,
            'pose3d': None,
            'pose3d_weights': None,
            'det_model': None,
            'det_weights': None,
            'det_cat_ids': 0,
            'scope': 'mmpose',
            'device': device,
            'show_progress': False,
        }
        if full_model_weight:
            self.init_setting['pose2d_weights']=None

        self.call_setting = {
            'show': False,
            'draw_bbox': False,
            'draw_heatmap': False,
            'bbox_thr': 0.1,  # setting for RTMO
            'nms_thr': 0.65,  # setting for RTMO
            'pose_based_nms': True, # setting for RTMO
            'show_kpt_idx':False,
            'kpt_thr': 0.65,
            'tracking_thr': 0.3,
            'use_oks_tracking': False,
            'disable_norm_pose_2d': False,
            'disable_rebase_keypoint': False,
            'num_instances': 1,
            'radius': 3,
            'thickness': 1,
            'skeleton_style': 'mmpose',
            'black_background': False,
            'output_root': 'test/result.jpg',
            'pred_out_dir': '',
            'show_interval':0,
            'show_alias': False
        }

        self.full_model = full_model

        print('rtmo init')

        rtmo = init_model(self.init_setting['pose2d'], self.init_setting['pose2d_weights'], device=self.init_setting['device'])
        self.model = rtmo
        self.backbone = self.model.backbone
        self.neck = self.model.neck

        del self.model.head
        del self.model.data_preprocessor

        self.visualizer = None

        
    def forward(self, x):
        if self.full_model:

            result = inference_bottomup_modified(self.model, x)

            # pose based nms 
            # reference: https://github.com/open-mmlab/mmpose/blob/main/mmpose/apis/inferencers/pose2d_inferencer.py
            #print('before :', len(result[0].pred_instances))
            for ds in result:
                if len(ds.pred_instances) == 0:
                    continue

                kpts = ds.pred_instances.keypoints
                scores = ds.pred_instances.bbox_scores
                num_keypoints = kpts.shape[-2]

                kept_indices = nearby_joints_nms(
                    [
                        dict(keypoints=kpts[i], score=scores[i])
                        for i in range(len(kpts))
                    ],
                    num_nearby_joints_thr=num_keypoints // 3,
                )
                ds.pred_instances = ds.pred_instances[kept_indices]
            #print('nms :', len(result[0].pred_instances))
            # keypoint threshold filtering 
            kpt_threshold = self.call_setting['kpt_thr']
            for ds in result:
                if len(ds.pred_instances) == 0:
                    continue

                kpt_scores = ds.pred_instances.keypoint_scores  
                kpt_scores_avg = np.mean(kpt_scores, axis=-1)  

                kept_indices = np.where(kpt_scores_avg > kpt_threshold)[0]

                ds.pred_instances = ds.pred_instances[kept_indices]
            #print('thresholding :', len(result[0].pred_instances))

            return result
        
        else:

            x = F.interpolate(x, size=(640,640), mode='nearest-exact')
            x = self.backbone(x)
            x = self.neck(x)
    
            return x
        
    def visualize(self, x):
        result = inference_bottomup(self.model, x)
        result = result[0]

        if self.visualizer is None :
            self.visualizer = VISUALIZERS.build(self.model.cfg.visualizer)
            self.visualizer.set_dataset_meta(self.model.dataset_meta)

        if self.visualizer is not None:
            self.visualizer.add_datasample(
                'result',
                x,
                data_sample=result,
                draw_gt=False,
                draw_bbox=False,
                draw_heatmap=self.call_setting['draw_heatmap'],
                show_kpt_idx=self.call_setting['show_kpt_idx'],
                show=self.call_setting['show'],
                wait_time=self.call_setting['show_interval'],
                kpt_thr=self.call_setting['kpt_thr'],
            )
            img_vis = self.visualizer.get_image()
            mmcv.imwrite(mmcv.rgb2bgr(img_vis), 'test/result.jpg')

if __name__ == '__main__':
    rtmo = RTMOBackbone()
    
    #result = rtmo.model(torch.randn(1,3,416,416).to('cuda:0'))
    #cprint(np.shape(result[0].cpu()), np.shape(result[1].cpu()))
    # torch.Size([1, 512, 26, 26]) torch.Size([1, 512, 13, 13])
