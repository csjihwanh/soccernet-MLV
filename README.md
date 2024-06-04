# soccernet-pose

the experiments are conducted in CUDA 11.7

## Installation

```
conda create -n snMLV python=3.9

conda activate snMLV

pip install -r requirements.txt



```

## Weights

VARS baseline weight: https://drive.google.com/drive/folders/1N0Lv-lcpW8w34_iySc7pnlQ6eFMSDvXn

download it into `weights/` directory and then change its name to 'vars_baseline.tar'

RTMO config: https://github.com/open-mmlab/mmpose/blob/dev-1.x/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py

RTMO weight: https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth



## command

```
python main.py \
--path path/to/your/dataset \
--model_name SNPOSE \
--path_to_model_weights path/to/user/checkpoint \
--start_frame 63 \
--end_frame 87 \
--fps 17 \
--pooling_type "attention" \
--pre_model "mvit_v2_s"
```

## RTMO visualization

```
python video_demo.py path/to/video \
--pose2d ../config/rtmo-l_16xb16-700e_body7-crowdpose-640x640.py \
--pose2d_weights ../weights/rtmo-l_16xb16-700e_body7-crowdpose-640x640-5bafdc11_20231219.pth \
--vis-out-dir result.mp4 
```



