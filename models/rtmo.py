import logging
from mmcv.image import imread

from mmengine.logging import print_log

from mmpose.apis import inference_topdown, init_model
from mmpose.apis.inferencers import MMPoseInferencer
from mmpose.utils import register_all_modules
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples

register_all_modules()

test_img = '../test/test.jpeg'
out_path = 'output.jpg'
 # or device='cuda:0'

class RTMO:
    def __init__(self, device='cuda:0'):
        self.init_setting = {
            'pose2d': '../config/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py',
            'pose2d_weights': '../checkpoints/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth',
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
            'kpt_thr': 0.3,
            'tracking_thr': 0.3,
            'use_oks_tracking': False,
            'disable_norm_pose_2d': False,
            'disable_rebase_keypoint': False,
            'num_instances': 1,
            'radius': 3,
            'thickness': 1,
            'skeleton_style': 'mmpose',
            'black_background': False,
            'vis_out_dir': '',
            'pred_out_dir': '',
            'show_alias': False
        }
        self.model = init_model(self.init_setting['pose2d'], self.init_setting['pose2d_weights'], device=self.init_setting['device'])
        self.inferencer = MMPoseInferencer(**self.init_setting)

    def inference(self, input):
        for _ in self.inferencer(**self.call_setting):
            pass
        

