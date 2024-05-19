import logging
from typing import Dict, List, Optional, Sequence, Union
from mmcv.image import imread
import mmcv

from mmengine.logging import print_log

from mmpose.apis import inference_bottomup, init_model
from mmengine.config import Config, ConfigDict
from mmengine.structures import InstanceData
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.evaluation.functional import nearby_joints_nms, nms

from mmpose.registry import VISUALIZERS

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

class RTMOBackbone(torch.nn.Module):
    def __init__(self, device='cuda', full_model = False):
        super().__init__()
        self.init_setting = {
            'pose2d': 'config/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py',
            'pose2d_weights': 'checkpoints/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth',
            'pose3d': None,
            'pose3d_weights': None,
            'det_model': None,
            'det_weights': None,
            'det_cat_ids': 0,
            'scope': 'mmpose',
            'device': device,
            'show_progress': False,
        }
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

        rtmo = init_model(self.init_setting['pose2d'], self.init_setting['pose2d_weights'], device=self.init_setting['device'])
        self.model = rtmo

        self.visualizer = None

    def get_neck_output_hook(self, module, input, output):
        global neck_output
        neck_output = output
        
    def forward(self, x):
        if self.full_model:
            result = inference_bottomup(self.model, x)
            
            # pose based nms 
            # reference: https://github.com/open-mmlab/mmpose/blob/main/mmpose/apis/inferencers/pose2d_inferencer.py
            print('before :', len(result[0].pred_instances))
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
            print('nms :', len(result[0].pred_instances))
            # keypoint threshold filtering 
            kpt_threshold = self.call_setting['kpt_thr']
            for ds in result:
                if len(ds.pred_instances) == 0:
                    continue

                kpt_scores = ds.pred_instances.keypoint_scores  
                kpt_scores_avg = np.mean(kpt_scores, axis=-1)  

                kept_indices = np.where(kpt_scores_avg > kpt_threshold)[0]

                ds.pred_instances = ds.pred_instances[kept_indices]
            print('thresholding :', len(result[0].pred_instances))

            return result
        
        else:
            hook_handle = self.model.neck.register_forward_hook(self.get_neck_output_hook)
            result = inference_bottomup(self.model, x)
            return neck_output
        
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
