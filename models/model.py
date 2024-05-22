
import __future__
import torch
from .mvaggregate import MVAggregate
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights
from .rtmo import RTMOBackbone

class MVNetwork(torch.nn.Module):

    def __init__(self, net_name='r2plus1d_18', agr_type='max', lifting_net=torch.nn.Sequential(), multi_gpu=False, full_model_weight=False, device='cuda'):
        super().__init__()

        ######
        use_graph = False
        only_rtmo = False
        mode='small'
        #####

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        self.feat_dim = 512
        
        if net_name == "r3d_18":
            weights_model = R3D_18_Weights.DEFAULT
            network = r3d_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "r2plus1d_18":
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
        elif net_name == "mvit_v2_s":
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        else:
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)

        network.fc = torch.nn.Sequential()

        if not use_graph:
            pose_model = RTMOBackbone(full_model=False, device=device, mode=mode) # only feature
        else:
            pose_model = RTMOBackbone(full_model=True, device=device, mode=mode) # keypoints

        self.mvnetwork = MVAggregate(
            model=network,
            agr_type=self.agr_type, 
            feat_dim=self.feat_dim, 
            lifting_net=self.lifting_net,
            multi_gpu=multi_gpu,
            pose_model=pose_model,
            only_rtmo=only_rtmo,
            use_graph=use_graph,
            mode=mode,
        )

    def forward(self, mvimages):
        return self.mvnetwork(mvimages)
