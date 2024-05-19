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
    def __init__(self, feat_dim):
        super(MultiScaleTransformerHead, self).__init__()
        
        self.blocks_1 = nn.ModuleList()
        self.blocks_2 = nn.ModuleList()

        self.blocks_1.extend([
            SpatioTemporalBlock(
                input_size = (16,40,40,512),
                output_size = (8,20,20,feat_dim),
                in_channels=512,
                out_channels=feat_dim,
                kernel_size= 4,
                stride = 2,
                padding=1,
                num_heads=16
            ),
            SpatioTemporalBlock(
                input_size = (8,20,20,feat_dim),
                output_size = (4,10,10,feat_dim),
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size= 4,
                stride = 2,
                padding=1,
                num_heads=16
            ),
            SpatioTemporalBlock(
                input_size = (4,10,10,feat_dim),
                output_size = (4,4,4,feat_dim),
                in_channels=feat_dim,
                out_channels=feat_dim,
                kernel_size= 3,
                stride = [1,3,3],
                padding= 1,
                num_heads=16
            ),
        ])

        self.blocks_2.extend([
            SpatioTemporalBlock(
                input_size = (16,20,20,512),
                output_size = (8,10,10,feat_dim),
                in_channels=512,
                out_channels=feat_dim,
                kernel_size= 4,
                stride = 2,
                padding=1,
                num_heads=16
            ),
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
        self.fuseLinear = nn.Linear(feat_dim*2, feat_dim)

    def forward(
            self, 
            x: Tuple[torch.Tensor, torch.Tensor],
        )-> torch.Tensor:

        y = None

        for i in range(len(x)):
            
            x_i = x[i]
            modules = None
            if i == 0:
                modules = self.blocks_1
            elif i == 1:
                modules = self.blocks_2
            else :
                raise RuntimeError
            
            for module in modules:
                BV,T,H,W,C = x_i.shape
                x_i = module(x_i, (T,H,W))
            
            x_i = einops.rearrange(x_i, 'b t h w c-> b c t h w')
            x_i = self.maxpool3d(x_i)
            x_i = einops.rearrange(x_i, 'b c t h w -> b t h w c')
            x_i = x_i.squeeze()

            if y is None:
                y = x_i
            else :
                y = torch.cat([y,x_i],dim=-1)

        y = self.fuseLinear(y)

        return y

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
            "embed_dim must be divisible by num_heads"
        
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


if __name__=='__main__':

    test = MultiScaleTransformerHead(feat_dim=400)
    x = torch.rand((4,16,40,40,512),dtype=torch.float)
    x2 = torch.rand((4,16,20,20,512),dtype=torch.float)

    x = test((x,x2))
    print(x.shape)