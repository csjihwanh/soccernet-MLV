from utils.utils import batch_tensor, unbatch_tensor
import torch
from torch import nn
import einops
from .graph_utils import generate_intra_spatial_edge, generate_inter_spatial_edge, visualize_graph
from .head import MultiScaleTransformerHead
import numpy as np
import torchvision.transforms as transforms


class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, groups=1):
        super(ConvModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activate = nn.SiLU()

    def forward(self, x):
        return self.activate(self.bn(self.conv(x)))

class RTMOHead(nn.Module):
    def __init__(self, feat_dim,):
        super(RTMOHead, self).__init__()
        # torch.Size([B*V*16, 512, 40, 40]) torch.Size([B*V*D, 512, 20, 20])
        self.conv_feat_1 = nn.Sequential(
            ConvModule(512, 256, kernel_size=3, padding=1), 
            nn.AvgPool2d(kernel_size=3, stride=2,padding=1), # 20 
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.AvgPool2d(kernel_size=3, stride=2,padding=1), # 10
            ConvModule(256, 256, kernel_size=3, padding=1),
            ConvModule(256, 256, kernel_size=3, padding=1),
            ConvModule(256, 2, kernel_size=3, padding=1), # 200
            nn.Flatten(start_dim=1)
        )
        self.conv_feat_2 = nn.Sequential(
            ConvModule(512, 256, kernel_size=3, padding=1), 
            nn.AvgPool2d(kernel_size=3, stride=2,padding=1), # 10
            ConvModule(256, 256, kernel_size=3, padding=1),
            ConvModule(256, 256, kernel_size=3, padding=1),
            ConvModule(256, 256, kernel_size=3, padding=1),
            ConvModule(256, 2, kernel_size=3, padding=1), # 8*8*16*2=1024
            nn.Flatten(start_dim=1)
        )
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.frame_linear = nn.Linear(feat_dim*16, feat_dim)


    def forward(self, x, view, batch, depth):
        x = torch.cat([
            self.conv_feat_1(x[0]),
            self.conv_feat_2(x[1]),
        ], dim=1)
        x = self.linear(x)
        x = einops.rearrange(x, '(B V D) F -> B V (D F)', B=batch, V=view, D=depth)
        x = self.frame_linear(x)
        
        return x
    

class WeightedAggregate(nn.Module):
    def __init__(self,  model, feat_dim, pose_model, lifting_net=nn.Sequential(), multi_gpu=False, only_rtmo =False, mode='tiny'):
        super().__init__()
        self.model = model
        self.pose_model = pose_model
        self.multi_gpu = multi_gpu
        self.only_rtmo = only_rtmo

        self.MViT_transform = transforms.Compose([
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

        self.pose_head = MultiScaleTransformerHead(feat_dim=feat_dim, multi_gpu=multi_gpu, mode=mode).cuda()

        self.fuselinear = nn.Linear(2*feat_dim, feat_dim)

        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()

        if multi_gpu:
            self.model = self.model.to('cuda:0')
            self.pose_model.data_preprocessor = self.pose_model.data_preprocessor.to('cuda:1')
            self.pose_head = self.pose_head.to('cuda:2')
            

    def forward(self, mvimages):
        #self.pose_model.model.head.dcc.pose_to_kpts.eval() # to avoid BatchNorm1d Bug

        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width # 2 2 3 16 224 224 # depth == frame num

        ## transform for MViT input 
        mvimages_mvit_transform = einops.rearrange(mvimages, 'B V C D H W -> B V D C H W') # to apply transformation for MViT (B T C H W)
        mvimages_mvit_transform = mvimages_mvit_transform.float() / 255.0 # to scaling [0,1]
        for b in range(B):
            for v in range(V):
                for d in range(D): 
                    mvimages_mvit_transform[b,v,d,:] = self.MViT_transform(mvimages_mvit_transform[b,v,d,:])

        mvimages_mvit_transform = einops.rearrange(mvimages_mvit_transform, 'B V D C H W -> B V C D H W')
        ## transformation done 

        # MViT output 
        if not self.only_rtmo :
            aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages_mvit_transform, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))

        # RTMO output 
        # torch.Size([BVT, H, W, C]) torch.Size([BVT, H/2, W/2, C])
       
        pose_input = einops.rearrange(mvimages, 'B V C D H W -> (B V D) H W C')
        pose_input = pose_input.cpu().numpy()
        pose_results = [None, None]


        for i in range(B*V*D):
            result = self.pose_model(pose_input[i])
            if pose_results[0] is None:
                pose_results[0] = result[0]
                pose_results[1] = result[1]
            else :
                pose_results[0] = torch.cat([pose_results[0], result[0]], dim=0)
                pose_results[1] = torch.cat([pose_results[1], result[1]], dim=0)

    
        # Transform to torch.Size([BV, 16, 512, 40, 40]) 
        for i, _ in enumerate(pose_results):
            pose_results[i] = einops.rearrange(pose_results[i], '(b v t) c h w -> (b v) t h w c', b=B,v=V,t=D)        

        # pose_head input: (BV, t h w c), output: (BV, feat_dim)
        pose_result = self.pose_head(pose_results)
        pose_result = einops.rearrange(pose_result, '(b v) f -> b v f', b=B, v=V)

        if self.only_rtmo :
            aux = pose_result
        elif not self.only_rtmo: 
            aux = self.fuselinear(torch.cat([aux, pose_result],dim=-1))

        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights
    
class GraphAggregate(nn.Module):
    def __init__(self,  model, feat_dim, pose_model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.pose_model = pose_model

        self.MViT_transform = transforms.Compose([
            transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
        ])

        self.fuselinear = nn.Linear(2*feat_dim, feat_dim)

        self.lifting_net = lifting_net
        self.feature_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()


    def forward(self, mvimages):
        self.pose_model.model.head.dcc.pose_to_kpts.eval() # to avoid BatchNorm1d Bug

        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width # 2 2 3 16 224 224 # depth == frame num

        ## transform for MViT input 
        mvimages_mvit_transform = einops.rearrange(mvimages, 'B V C D H W -> B V D C H W') # to apply transformation for MViT (B T C H W)
        mvimages_mvit_transform = mvimages_mvit_transform.float() / 255.0 # to scaling [0,1]
        for b in range(B):
            for v in range(V):
                for d in range(D): 
                    mvimages_mvit_transform[b,v,d,:] = self.MViT_transform(mvimages_mvit_transform[b,v,d,:])

        mvimages_mvit_transform = einops.rearrange(mvimages_mvit_transform, 'B V D C H W -> B V C D H W')
        ## transformation done 

        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages_mvit_transform, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        
        pose_input = einops.rearrange(mvimages, 'B V C D H W -> (B V D) H W C')
        pose_input = pose_input.cpu().numpy()
        self.pose_model.visualize(pose_input[0])

        pose_results = [None, None]
        for i in range(B*V*D):
            result = self.pose_model(pose_input[i])[0].pred_instances
            
            person_graph = []
            keypoint_scores = torch.Tensor(result.keypoint_scores).unsqueeze(dim=2)
            keypoints = torch.Tensor(result.keypoints)
            keypoint_features = torch.cat([keypoints,keypoint_scores],dim=2) # (N, 14, 3)

            for keypoint_feature in keypoint_features:
                print('average confidence:', torch.sum(keypoint_feature[:,2])/14) 

            for keypoint_feature in keypoint_features:
                person_graph.append(generate_intra_spatial_edge(keypoint_feature))

            frame_graph = generate_inter_spatial_edge(person_graph)

            visualize_graph(frame_graph,'test/result_graph.jpg')
            raise NotImplementedError

            # (N, 14), (N, 14, 2), (N, 14)


        # torch.Size([32, 512, 40, 40]) torch.Size([32, 512, 20, 20])
        
        # torch.Size([B, V, 16, 512, 40, 40]) torch.Size([32, 512, 20, 20])
        pose_result = self.pose_head((pose_results[0],pose_results[1]), view=V, batch=B, depth=D)

        #aux = pose_result
        aux = self.fuselinear(torch.cat([aux, pose_result],dim=2))

        ##################### VIEW ATTENTION #####################

        # S = source length 
        # N = batch size
        # E = embedding dimension
        # L = target length

        aux = torch.matmul(aux, self.attention_weights)
        # Dimension S, E for two views (2,512)

        # Dimension N, S, E
        aux_t = aux.permute(0, 2, 1)

        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T

        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))

        final_attention_weights = torch.sum(final_attention_weights, 1)

        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))

        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self,  model, lifting_net=nn.Sequential()):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        aux = self.lifting_net(unbatch_tensor(self.model(batch_tensor(mvimages, dim=1, squeeze=True)), B, dim=1, unsqueeze=True))
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self,  model, pose_model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential(), multi_gpu=False, use_graph=False, only_rtmo=False, mode='tiny'):
        super().__init__()


        for param in pose_model.model.neck.parameters():
            param.requires_grad = False

        for param in model.parameters():
            param.requires_grad = False

        self.multi_gpu = multi_gpu

        self.agr_type = agr_type

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )


        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(model=model, lifting_net=lifting_net)
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(model=model, lifting_net=lifting_net)
        elif use_graph:
            self.aggregation_model = GraphAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net, pose_model=pose_model)
        else:
            self.aggregation_model = WeightedAggregate(model=model, feat_dim=feat_dim, lifting_net=lifting_net, pose_model=pose_model, multi_gpu=multi_gpu, only_rtmo=only_rtmo, mode=mode)

    def check_dim(self, tensor):
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        return tensor


    def forward(self, mvimages):
        
        pooled_view, attention = self.aggregation_model(mvimages)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        #if self.multi_gpu:
            #inter = self.check_dim(inter)
            #pred_action = self.check_dim(pred_action)
            #print('dimension unsqueeze')
            #pred_offence_severity = self.check_dim(pred_offence_severity)

        #print("output shape:", pred_offence_severity.shape, pred_action.shape, attention.shape)
        # output shape: torch.Size([2, 4]) torch.Size([2, 8]) torch.Size([2, 2])
        # output shape: torch.Size([4]) torch.Size([8]) torch.Size([1, 2])
        return pred_offence_severity, pred_action, attention
