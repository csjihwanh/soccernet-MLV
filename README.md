# soccernet-MLV

This repository is built based on [VARS](https://github.com/SoccerNet/sn-mvfoul/tree/main/VARS%20model) and [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).

## Architecture
<img width="1206" alt="architecture" src="https://github.com/Jordano-Jackson/soccernet-MLV/assets/19871043/5488b52b-6bdb-4e8b-bb2b-4cf61380855f">

E represents the VideoChat2 encoder module, A denotes the aggregation module, and C is the classification head.

See our [technical report]() for details.


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

### Training from scratch
```
python main.py \
--path path/to/dataset \
--model_name your_model_name \
--start_frame 67 \
--end_frame 83 \
--model_to_store path/to/store \
--multi_gpu
```

If you want to train the model from scratch, place the VideoChat2 stage3 weight at `videochat2/checkpoints/videochat2/videochat2_mistral_7b_stage3.pth`. It can be downloaded at [VideoChat2](https://github.com/OpenGVLab/Ask-Anything/tree/main/video_chat2).

Because of time constraints, we have not been able to train sufficiently in various settings, so our results may not be optimal. I recommend training in various ways.
