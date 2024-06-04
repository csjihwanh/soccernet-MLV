# soccernet-MLV

This repository is build based on [VARS](https://github.com/SoccerNet/sn-mvfoul/tree/main/VARS%20model) and [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).


## Installation

the experiments are conducted in CUDA 11.7

```
conda create -n snMLV python=3.9

conda activate snMLV

pip install -r requirements.txt

pip install soccernet
```

## Checkpoints

Download a pretrained checkpoint file from our [drive](https://drive.google.com/file/d/1rM3im9uVysbFdD76zcvZckZahHp1VdzF/view?usp=sharing).

Then place the file in `checkpoints/` directory.

## command

### Evaluation
```
python main.py \
--path path/to/dataset \
--model_name your_model_name \
--start_frame 67 \
--end_frame 83 \
--path_to_model_weight path/to/your/checkpoint \
--only_evaluation type \
--multi_gpu
```

### Training 
```
python main.py \
--path path/to/dataset \
--model_name your_model_name \
--start_frame 67 \
--end_frame 83 \
--path_to_model_weight path/to/your/checkpoint \
--model_to_store path/to/store \
--multi_gpu
```

