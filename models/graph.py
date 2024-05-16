from rtmo import RTMOBackbone
from torchvision.io.video import read_video

#from ..datasets.dataset import MultiViewDataset
import torch

def main():
    rtmo = RTMOBackbone(full_model=True)
    path = '/hub_data1/intern/SoccerNet1/mvfouls/Test/action_0/clip_1.mp4'
    start_frame= 63 
    end_frame =87 
    fps =17 

    video, _, _ = read_video(path, output_format="THWC")

    print(rtmo.backbone(video))

    #dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
    #    num_views = num_views, transform=transformAug, transform_model=transforms_model)
    #dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views = 5, 
    #    transform_model=transforms_model)
    #dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views = 5, 
    #    transform_model=transforms_model)

    # Create the dataloaders for train validation and test datasets
    #train_loader = torch.utils.data.DataLoader(dataset_Train,
    #    batch_size=batch_size, shuffle=True,
    #    num_workers=max_num_worker, pin_memory=True)

    #val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
    #    batch_size=1, shuffle=False,
    #    num_workers=max_num_worker, pin_memory=True)

    #test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
    #    batch_size=1, shuffle=False,
    #    num_workers=max_num_worker, pin_memory=True)


if __name__ == '__main__':
    main()