import torch
from torch import nn
import math
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Union, List, Optional, Sequence, Tuple
import torch.fx

import einops
import numpy as np


class MultiScaleTransformerHead(nn.Module):
    def __init__(self, feat_dim, multi_gpu=False, mode='tiny'):
        super(MultiScaleTransformerHead, self).__init__()
        
        self.blocks = nn.ModuleList()

        if mode == 'tiny':
            input_dim = 192
            first_module = SpatioTemporalBlock(
                input_size = (16,12,12,input_dim),
                output_size = (8,10,10,feat_dim),
                in_channels=input_dim,
                out_channels=feat_dim,
                kernel_size= 3,
                stride = 1,
                padding=0,
                num_heads=16
            )

        elif mode == 'small':
            input_dim = 256
            first_module = SpatioTemporalBlock(
                input_size = (16,20,20,input_dim),
                output_size = (8,10,10,feat_dim),
                in_channels=input_dim,
                out_channels=feat_dim,
                kernel_size= 4,
                stride = 2,
                padding=1,
                num_heads=16
            )

        self.blocks.extend([
            first_module,
            SpatioTemporalBlock(
                input_size = (8,10,10,feat_dim),
                output_size = (4,4,4,feat_dim),
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size= 4,
                stride = 2,
                padding=[1,0,0],
                num_heads=16
            ),
            SpatioTemporalBlock(
                input_size = (4,4,4,feat_dim),
                output_size = (4,4,4,feat_dim),
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size= 3,
                stride = 1,
                padding= 1,
                num_heads=16
            ),
        ])
        self.maxpool3d = torch.nn.MaxPool3d(4, stride=1, padding=0, dilation=1)
        self.maxpool2d = torch.nn.MaxPool3d(3, stride=[2, 2, 1], padding=1, dilation=1)

    def forward(
            self, 
            x: Tuple[torch.Tensor, torch.Tensor],
        )-> torch.Tensor:
        x = torch.stack([self.maxpool2d(x[0]), x[1]], dim=-1)
        x, _ = x.max(dim=-1)

            
        for module in self.blocks:
            BV,T,H,W,C = x.shape
            x = module(x, (T,H,W))
        
        x = einops.rearrange(x, 'b t h w c-> b c t h w')
        x = self.maxpool3d(x)
        x = einops.rearrange(x, 'b c t h w -> b t h w c')
        x = x.squeeze()

        return x

def _interpolate(embedding: torch.Tensor, d: int) -> torch.Tensor:
    # code reference: https://pytorch.org/vision/main/_modules/torchvision/models/video/mvit.html#mvit_v2_s 
    if embedding.shape[0] == d:
        return embedding

    return (
        nn.functional.interpolate(
            embedding.permute(1, 0).unsqueeze(0),
            size=d,
            mode="linear",
        )
        .squeeze(0)
        .permute(1, 0)
    )

def _add_rel_pos(
    attn: torch.Tensor,
    q: torch.Tensor,
    q_thw: Tuple[int, int, int],
    k_thw: Tuple[int, int, int],
    rel_pos_h: torch.Tensor,
    rel_pos_w: torch.Tensor,
    rel_pos_t: torch.Tensor,
) -> torch.Tensor:
    # code reference: https://pytorch.org/vision/main/_modules/torchvision/models/video/mvit.html#mvit_v2_s 
    # Modified code from: https://github.com/facebookresearch/SlowFast/commit/1aebd71a2efad823d52b827a3deaf15a56cf4932
    q_t, q_h, q_w = q_thw
    k_t, k_h, k_w = k_thw
    dh = int(2 * max(q_h, k_h) - 1)
    dw = int(2 * max(q_w, k_w) - 1)
    dt = int(2 * max(q_t, k_t) - 1)

    # Scale up rel pos if shapes for q and k are different.
    q_h_ratio = max(k_h / q_h, 1.0)
    k_h_ratio = max(q_h / k_h, 1.0)
    dist_h = torch.arange(q_h)[:, None] * q_h_ratio - (torch.arange(k_h)[None, :] + (1.0 - k_h)) * k_h_ratio
    q_w_ratio = max(k_w / q_w, 1.0)
    k_w_ratio = max(q_w / k_w, 1.0)
    dist_w = torch.arange(q_w)[:, None] * q_w_ratio - (torch.arange(k_w)[None, :] + (1.0 - k_w)) * k_w_ratio
    q_t_ratio = max(k_t / q_t, 1.0)
    k_t_ratio = max(q_t / k_t, 1.0)
    dist_t = torch.arange(q_t)[:, None] * q_t_ratio - (torch.arange(k_t)[None, :] + (1.0 - k_t)) * k_t_ratio

    # Interpolate rel pos if needed.
    rel_pos_h = _interpolate(rel_pos_h, dh)
    rel_pos_w = _interpolate(rel_pos_w, dw)
    rel_pos_t = _interpolate(rel_pos_t, dt)
    Rh = rel_pos_h[dist_h.long()]
    Rw = rel_pos_w[dist_w.long()]
    Rt = rel_pos_t[dist_t.long()]

    B, n_head, _, dim = q.shape
    r_q = q[:, :, :].reshape(B, n_head, q_t, q_h, q_w, dim)
    rel_h_q = torch.einsum("bythwc,hkc->bythwk", r_q, Rh)  # [B, H, q_t, qh, qw, k_h]
    rel_w_q = torch.einsum("bythwc,wkc->bythwk", r_q, Rw)  # [B, H, q_t, qh, qw, k_w]
    # [B, H, q_t, q_h, q_w, dim] -> [q_t, B, H, q_h, q_w, dim] -> [q_t, B*H*q_h*q_w, dim]
    r_q = r_q.permute(2, 0, 1, 3, 4, 5).reshape(q_t, B * n_head * q_h * q_w, dim)
    # [q_t, B*H*q_h*q_w, dim] * [q_t, dim, k_t] = [q_t, B*H*q_h*q_w, k_t] -> [B*H*q_h*q_w, q_t, k_t]
    rel_q_t = torch.matmul(r_q, Rt.transpose(1, 2)).transpose(0, 1)
    # [B*H*q_h*q_w, q_t, k_t] -> [B, H, q_t, q_h, q_w, k_t]
    rel_q_t = rel_q_t.view(B, n_head, q_h, q_w, q_t, k_t).permute(0, 1, 4, 2, 3, 5)

    # Combine rel pos.
    rel_pos = (
        rel_h_q[:, :, :, :, :, None, :, None]
        + rel_w_q[:, :, :, :, :, None, None, :]
        + rel_q_t[:, :, :, :, :, :, None, None]
    ).reshape(B, n_head, q_t * q_h * q_w, k_t * k_h * k_w)

    # Add it to attention
    attn[:, :, :, :] += rel_pos

    return attn

class SpatioTemporalPooling(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, List[int]],
            stride: Union[int, List[int]],
            padding: Union[int, List[int]]
        ) -> None:
        super().__init__()
        self.pooling = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
    
    def forward(self, x:torch.Tensor, thw: Tuple[int,int,int]) -> Tuple[torch.Tensor, Tuple[int,int,int]]:
        #input shape: (B, num_heads, T * H * W, head_dim)
        B, num_heads, _, head_dim = x.shape
        x = x.permute(0,1,3,2)
        x = x.reshape((B* num_heads, head_dim) + thw)

        x = self.pooling(x)
        
        t, h, w = x.shape[2:]
        x = x.reshape(B, num_heads, head_dim, -1)
        x = x.transpose(2,3)
        
        return x, (t,h,w)

class SpatioTemporalAttention(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        num_heads: int,

        ) -> None:
        super().__init__()

        self.head_dim = out_channels // num_heads
        assert self.head_dim * num_heads == out_channels, \
            f"embed_dim must be divisible by num_heads, {self.head_dim, num_heads, out_channels}"
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads

        self.pooling = SpatioTemporalPooling(
            in_channels=self.head_dim,
            out_channels=self.head_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.qkv_proj = nn.Linear(in_channels, out_channels*3)

        # code reference: https://pytorch.org/vision/main/_modules/torchvision/models/video/mvit.html#mvit_v2_s 
        size = max(input_size[1:])

        if isinstance(stride, int):
            q_size = size // stride
            kv_size = size // stride
        elif isinstance(stride, list):
            q_size = size // stride[0]
            kv_size = size // stride[0]
            
        spatial_dim = 2 * max(q_size, kv_size) - 1
        temporal_dim = 2 * input_size[0] - 1
        self.rel_pos_h = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(spatial_dim, self.head_dim))
        self.rel_pos_t = nn.Parameter(torch.zeros(temporal_dim, self.head_dim))
        nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_t, std=0.02)
        

    def forward(self, x: torch.Tensor, thw: Tuple[int,int,int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        BV, T, H, W, C = x.shape # x shape: (B*V), T, H, W, C

        #assert (T*H*W) == self.out_channels, \
        #    "Size mismatch: T * H * W must be equal to out_channel size"
        assert (C) == self.in_channels, \
            "Size mismatch: C must be equal to embed_dim size"

        x = x.reshape(BV, T * H * W, C)
        qkv = self.qkv_proj(x)
        # BV, THW, 3out -> BV T H W num_head head_dim 3
        qkv = qkv.view(BV, T * H * W, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # shape: (3, B, num_heads, T * H * W, head_dim)
        q, k, v = qkv.unbind(dim=0) # shape: (BV, num_heads, T * H * W, head_dim)

        # Q, K, V pooling
        q, _ = self.pooling(q, thw)
        k, k_thw = self.pooling(k, thw)
        v, thw = self.pooling(v, thw)

        T, H, W = thw

        attn = torch.matmul(q, k.transpose(2,3)) / math.sqrt(self.head_dim)
        attn = _add_rel_pos(
                attn,
                q,
                thw,
                k_thw,
                self.rel_pos_h,
                self.rel_pos_w,
                self.rel_pos_t,
            )
        attn = attn.softmax(dim=-1)   
        
        x = torch.matmul(attn, v) # shape: (B, num_heads, THW, head_dim)
        x = x.transpose(2,3) # shape: (B, num_heads, head_dim, THW)
        x = einops.rearrange(x, 'B N D (T H W) -> B T H W (N D)', B= BV, N=self.num_heads, D=self.head_dim, T=T, H=H, W=W)
        return x



class SpatioTemporalBlock(nn.Module):
    def __init__(
        self,
        input_size: List[int],
        output_size: List[int],
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, List[int]],
        stride: Union[int, List[int]],
        padding: Union[int, List[int]],
        num_heads: int,
    ) -> None:
        super().__init__()
        self.out_channels = out_channels

        self.layernorm = nn.LayerNorm(output_size)
        self.st_attention = SpatioTemporalAttention(
            input_size=input_size,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            num_heads=num_heads)
        self.q_pooling = SpatioTemporalPooling(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.down_mlp = nn.Linear(in_channels, out_channels)
        self.mlp = nn.Linear(out_channels, out_channels)
        


    def forward(self, x: torch.Tensor, thw: Tuple[int,int,int]) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        # x shape: (BV, T, H, W, C)
        BV, T, H, W, C = x.shape
        x_res, x_res_size = self.q_pooling(x.reshape(BV, -1, C).unsqueeze(dim=1), thw)
        x_res = x_res.reshape(BV, *x_res_size ,self.out_channels)
        x = x_res + self.st_attention(x, thw)
        x = self.layernorm(x)

        x = x + self.mlp(x)
        x = self.layernorm(x)

        return x



def gaussian_weights(size, center, sigma=1.0):
    """
    Gaussian weights centered at the specified point of the vector.
    
    Parameters:
    size (int): The number of elements in the vector.
    center (float): The center point of the Gaussian distribution.
    sigma (float): The standard deviation of the Gaussian distribution.
    
    Returns:
    torch.Tensor: A tensor of Gaussian weights.
    """
    x = torch.arange(size, dtype=torch.float32)
    weights = torch.exp(-0.5 * ((x - center) / sigma) ** 2)
    return weights / weights.sum()

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
        self.feat_dim = feat_dim
        # torch.Size([B*V*16, 256, 40, 40]) torch.Size([B*V*D, 256, 20, 20])
        self.conv_feat_1 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2,padding=1), # 20 
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 1, kernel_size=3, padding=1),
            nn.Flatten(start_dim = 1)
        )
        self.conv_feat_2 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 1, kernel_size=3, padding=1), 
            nn.Flatten(start_dim = 1)
        )
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.frame_linear = nn.Linear(feat_dim*16, feat_dim)

    def forward(self, x, batch, view): # (B V) D H W C -> (B V) F 
        _, D, _, _, _ = x[0].shape
        BV = batch*view

        # (B V) D H W C -> (B V D) C H W 
        x = (
            einops.rearrange(x[0], 'bv d h w c -> (bv d) c h w', bv=BV),
            einops.rearrange(x[1], 'bv d h w c -> (bv d) c h w', bv=BV)
        ) 

        # (B V D) C H W -> (B V D) N F (N=2, F=400)
        x = torch.stack([
            self.conv_feat_1(x[0]),
            self.conv_feat_2(x[1]),
        ], dim=1)

        x = x.reshape(BV, D, 2, self.feat_dim) # bv d n f 
        # x.shape: bv, 16, 2, 400
        

        # Gaussian weight
        center = 8.5
        sigma = 1.5
        weights = gaussian_weights(D, center, sigma).cuda()

        weights = weights.view(1, D, 1, 1)

        x = x * weights

        x = x.sum(dim=1).squeeze() # bv 16 2 400 -> bv 2 400
        x = x.sum(dim=1).squeeze() # bv 2 400 -> bv 400
        
        return x

class RTMOResidualHead(nn.Module):
    def __init__(self, feat_dim,frame=16):
        super(RTMOResidualHead, self).__init__()
        self.feat_dim = feat_dim
        # torch.Size([B*V*16, 256, 40, 40]) torch.Size([B*V*D, 256, 20, 20])

        self.pooling2d = nn.AvgPool2d(kernel_size=3, stride=2,padding=1) # 20 
        self.conv_feat_11 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_feat_12 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_feat_13 = nn.Sequential(
            ConvModule(256, 1, kernel_size=3, padding=1),
            nn.Flatten(start_dim = 1)
        )
        self.conv_feat_21 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1), 
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_feat_22 = nn.Sequential(
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            ConvModule(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.conv_feat_23 = nn.Sequential(
            ConvModule(256, 1, kernel_size=3, padding=1), 
            nn.Flatten(start_dim = 1)
        )
        self.linear = nn.Linear(feat_dim, feat_dim)
        self.frame_linear = nn.Linear(feat_dim*frame, feat_dim)

    def forward(self, x, batch, view): # (B V) D H W C -> (B V) F 
        _, D, _, _, _ = x[0].shape
        BV = batch*view

        # (B V) D H W C -> (B V D) C H W 
        x = (
            einops.rearrange(x[0], 'bv d h w c -> (bv d) c h w', bv=BV),
            einops.rearrange(x[1], 'bv d h w c -> (bv d) c h w', bv=BV)
        ) 

        # (B V D) C H W -> (B V D) N F (N=2, F=400)
        x0 = self.pooling2d(x[0])
        x0 = self.conv_feat_11(x0)+ x0
        x0 = self.conv_feat_12(x0) + x0
        x1 = self.conv_feat_21(x[1])+x[1]
        x1 = self.conv_feat_22(x1)+x1

        x = torch.stack([
            self.conv_feat_13(x0),
            self.conv_feat_23(x1),
        ], dim=1)

        x = x.reshape(BV, D, 2, self.feat_dim) # bv d n f 
        # x.shape: bv, 16, 2, 400
        

        # Gaussian weight
        center = 8.5
        sigma = 1.5
        weights = gaussian_weights(D, center, sigma).cuda()

        weights = weights.view(1, D, 1, 1)

        x = x * weights

        x = x.sum(dim=1).squeeze() # bv 16 2 400 -> bv 2 400
        x = x.sum(dim=1).squeeze() # bv 2 400 -> bv 400
        
        return x

if __name__=='__main__':

    test = MultiScaleTransformerHead(feat_dim=400)
    x = torch.rand((4,16,40,40,512),dtype=torch.float)
    x2 = torch.rand((4,16,20,20,512),dtype=torch.float)

    x = test((x,x2))