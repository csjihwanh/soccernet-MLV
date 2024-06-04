import torch
from torch import nn
import einops
import numpy as np
import torchvision.transforms as transforms
from videochat2 import VideoChat2


class WeightedAggregate(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.vlm = VideoChat2()

        self.token_num = 96
        self.feat_dim = feat_dim

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()

        seq_len = 32

        nhead = 8
        num_layers = 6

    def forward(self, mvimages):
        # torch.Size([b, v, tc, h, w])

        B, V, TC, H, W = mvimages.shape

        mvimages = einops.rearrange(mvimages, 'B V TC H W -> (B V) TC H W', B=B, V=V)

        aux = self.vlm(mvimages) # 6, 96, 4096
        aux = aux.reshape(B, V, self.token_num, self.feat_dim) # b v 96 4096

        aux = aux.mean(dim=2) # b v 1 4096 
        aux = aux.to(torch.float32)

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
    

class Model(nn.Module):
    def __init__(self, feat_dim=768):
        super().__init__()

        inter_dim = feat_dim

        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, inter_dim),
            nn.Linear(inter_dim, inter_dim),
        )

        self.fc_offence = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, inter_dim),
            nn.Linear(inter_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(inter_dim),
            nn.Linear(inter_dim, inter_dim),
            nn.Linear(inter_dim, 8)
        )

        self.aggregation_model = WeightedAggregate(feat_dim)



    def forward(self, mvimages):
        
        pooled_view, attention = self.aggregation_model(mvimages)

        inter = self.inter(pooled_view)
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention
