import os
import math
import logging
import numpy as np
from functools import partial
from einops import rearrange, repeat, pack, unpack
from typing import Any, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


CLIP_FIRST_CHUNK = False
CACHE_T = 2
INT_MAX = 2**31


class CausalConv3d(nn.Conv3d):
    """
    Causal 3d convolusion.
    """
    PAD_MODE = "replicate"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._padding = (
            self.padding[2],
            self.padding[2],
            self.padding[1],
            self.padding[1],
            2 * self.padding[0],
            0
        )
        arg_list = list(args)
        if len(arg_list)>=5:
            arg_list[4] = 0
        elif 'padding' in kwargs:
            kwargs['padding']= 0
        super().__init__(*arg_list, **kwargs)
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, cache_x=None):
        padding = list(self._padding)
        if cache_x is not None and self._padding[4] > 0:
            cache_x = cache_x.to(x.device)
            x = torch.cat([cache_x, x], dim=2)
            padding[4] -= cache_x.shape[2]
        x = F.pad(x, padding, mode=self.PAD_MODE)

        if x.numel() > INT_MAX:
            t = x.shape[2]
            kernel_t = self.kernel_size[0]
            num_split = max(1, t - kernel_t + 1)
            out_list = []
            for i in range(num_split):
                x_s = x[:, :, i:i+kernel_t, :, :]
                out_list.append(super().forward(x_s))
            out = torch.cat(out_list, dim=2)
            del out_list
        else:
            out = super().forward(x)

        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


class Fp32Upsample(nn.Upsample):

    def forward(self, x):
        """
        Fix bfloat16 support for nearest neighbor interpolation.
        """
        return super().forward(x.float()).type_as(x)


class Upsample(nn.Module):
    def __init__(self, dim_in, dim_out=None, new_upsample=False):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        if new_upsample:
            self.up = Fp32Upsample(scale_factor=(1.0, 2.0, 2.0), mode='nearest-exact')
            self.conv = CausalConv3d(dim_in, dim_out, (1, 5, 5), padding=(0, 2, 2))
        else:
            self.up = None
            self.conv = nn.ConvTranspose3d(dim_in, dim_out, (1, 5, 5), (1, 2, 2), (0, 2, 2),
                                output_padding=(0, 1, 1))

    def forward(self, x):
        if self.up is not None:
            x = self.up(x)
        x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, dim_in, dim_out=None):
        super().__init__()
        if dim_out is None:
            dim_out = dim_in
        self.conv = CausalConv3d(dim_in, dim_out, (1, 3, 3), (1, 2, 2), (0, 1, 1))

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.conv(x)
        return x


class TimeDownsample(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        kernel_size = 3,
        antialias = False
    ):
        super().__init__()
        dim_out = dim_out or dim
        self.time_causal_padding = (kernel_size - 1, 0)
        self.conv = nn.Conv1d(dim, dim_out, kernel_size, stride = 2)
        nn.init.zeros_(self.conv.bias)
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[range(dim_out), range(dim), -2] = 1.0
        self.conv.weight.data[range(dim_out), range(dim), -1] = 1.0
        self.cache_t = 1

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = rearrange(x, 'b c t h w -> b h w c t').contiguous()
        x, ps = pack([x], '* c t')

        if feat_cache is not None:
            idx = feat_idx[0]
            cache_x = x[..., -self.cache_t:].clone()

            if feat_cache[idx] is not None and self.time_causal_padding[0] > 0:
                x = torch.cat([feat_cache[idx], x], dim=-1)
            else:
                x = F.pad(x, self.time_causal_padding, mode=CausalConv3d.PAD_MODE)
            out = self.conv(x)
            feat_cache[idx] = cache_x
            feat_idx[0] += 1
        else:
            x = F.pad(x, self.time_causal_padding, mode=CausalConv3d.PAD_MODE)
            out = self.conv(x)

        out = unpack(out, ps, '* c t')[0]
        out = rearrange(out, 'b h w c t -> b c t h w').contiguous()
        return out


class TimeUpsample(nn.Module):
    def __init__(
        self,
        dim
    ):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim*2, 1)
        self.init_conv_(self.conv)

    def init_conv_(self, conv):
        o, i, t = conv.weight.shape
        conv_weight = torch.zeros(o // 2, i, t)
        conv_weight[range(o//2), range(i), :] = 1.0
        conv_weight = repeat(conv_weight, 'o ... -> (o 2) ...')

        conv.weight.data.copy_(conv_weight)
        nn.init.zeros_(conv.bias.data)

    def forward(self, x):
        x = rearrange(x, 'b c t h w -> b h w c t').contiguous()
        x, ps = pack([x], '* c t')

        out = self.conv(x)

        out =  rearrange(out, 'b (c p) t -> b c (t p)', p = 2).contiguous()
        out = unpack(out, ps, '* c t')[0]
        out = rearrange(out, 'b h w c t -> b c t h w').contiguous()
        if CLIP_FIRST_CHUNK:
            out = out[:, :, 1:, :, :]
        return out


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        x = self.norm(x)
        return self.fn(x, feat_cache=feat_cache, feat_idx=feat_idx)


class Block(nn.Module):
    def __init__(self, dim, dim_out, large_filter=False):
        super().__init__()
        self.block = nn.ModuleList([
            CausalConv3d(dim, dim_out, (3, 7, 7) if large_filter else (3, 3, 3),
                    padding=(1, 3, 3) if large_filter else (1, 1, 1)),
            LayerNorm(dim_out), nn.ReLU()]
        )

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        for layer in self.block:
            if isinstance(layer, CausalConv3d) and feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2).to(cache_x.device), cache_x], dim=2)
                x = layer(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = layer(x)
        return x



def module_forward(backbone, x, t=None, feat_cache=None, feat_idx=[0]):
    if t is None:
        if isinstance(backbone, CausalConv3d):
            if feat_cache is not None:
                idx = feat_idx[0]
                cache_x = x[:, :, -CACHE_T:, :, :].clone()
                if cache_x.shape[2] < 2 and feat_cache[idx] is not None:
                    # cache last frame of last two chunk
                    cache_x = torch.cat([feat_cache[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
                x = backbone(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
            else:
                x = backbone(x)
        elif type(backbone) in [Upsample, TimeUpsample, nn.Identity]:
            x  = backbone(x)
        else:
            x = backbone(x, feat_cache=feat_cache, feat_idx=feat_idx)
    else:
        x = backbone(x, t, feat_cache=feat_cache, feat_idx=feat_idx)
    return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, time_emb_dim=None, large_filter=False):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.LeakyReLU(0.2), nn.Linear(time_emb_dim, dim_out))
            if time_emb_dim is not None
            else None
        )

        self.block1 = Block(dim, dim_out, large_filter)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = CausalConv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, feat_cache=None, feat_idx=[0]):
        h = self.block1(x, feat_cache=feat_cache, feat_idx=feat_idx)

        if time_emb is not None:
            h = h + self.mlp(time_emb)[:, :, None, None, None]

        h = self.block2(h, feat_cache=feat_cache, feat_idx=feat_idx)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=None):
        super().__init__()
        if dim_head is None:
            dim_head = dim
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = CausalConv3d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = CausalConv3d(hidden_dim, dim, 1)

    def forward(self, x, feat_cache=None, feat_idx=[0]):
        b, c, t, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) t x y -> (b t) h c (x y)", h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "(b t) h c (x y) -> b (h c) t x y", h=self.heads, x=h, y=w, t=t)
        return self.to_out(out)


def count_conv3d(model):
    count = 0
    for m in model.modules():
        if isinstance(m, CausalConv3d):
            count += 1
    return count

def count_time_down_sample(model):
    count = 0
    for m in model.modules():
        if isinstance(m, TimeDownsample):
            count += 1
    return count


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def mode(self):
        return self.mean


class Compressor(nn.Module):
    def __init__(
        self,
        dim=64,
        dim_mults=(1, 2, 3, 4),
        reverse_dim_mults=(4, 3, 2, 1),
        space_down=(1, 1, 1, 1),
        time_down=(0, 0, 1, 1),
        new_upsample=False,
        channels=3,
        out_channels=None,
        latent_dim=64,
    ):
        super().__init__()
        self.channels = channels
        out_channels = out_channels or channels
        self.space_down = space_down
        self.new_upsample = new_upsample
        self.reversed_space_down = list(reversed(self.space_down))
        self.time_down = time_down
        self.reversed_time_down = list(reversed(self.time_down))
        self.dims = [channels, *map(lambda m: dim * m, dim_mults)]
        self.in_out = list(zip(self.dims[:-1], self.dims[1:]))
        self.reversed_dims = [*map(lambda m: dim * m, reverse_dim_mults), out_channels]
        self.reversed_in_out = list(zip(self.reversed_dims[:-1], self.reversed_dims[1:]))
        assert self.dims[-1] == self.reversed_dims[0]
        latent_dim = latent_dim or out_channels
        self.quant_conv = torch.nn.Conv3d(self.dims[-1], 2 * latent_dim, 1)
        self.post_quant_conv = torch.nn.Conv3d(latent_dim, self.dims[-1], 1)
        self.quant_conv_res = nn.ModuleList(
            [ResnetBlock(dim_in, dim_out) for dim_in, dim_out in self.reversed_in_out[:-1]]+
            [torch.nn.Conv3d(self.reversed_dims[-2], 2 * latent_dim, 1)])
        self.post_quant_conv_res = nn.ModuleList(
            [torch.nn.Conv3d(latent_dim, self.dims[1], 1)]+
            [ResnetBlock(dim_in, dim_out) for dim_in, dim_out in self.in_out[1:]])
        # build network
        self.build_network()

        # cache the last two frame of feature map
        self._conv_num = count_conv3d(self.post_quant_conv_res) + count_conv3d(self.dec)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num
        # cache encode
        self._enc_conv_num = count_conv3d(self.quant_conv_res) + count_conv3d(self.enc) + count_time_down_sample(self.enc)
        self._enc_conv_idx = [0]
        self._enc_feat_map = [None] * self._enc_conv_num


    @property
    def dtype(self):
        return self.enc[0][0].block1.block[0].weight.dtype

    def build_network(self):
        self.enc = nn.ModuleList([])
        self.dec = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(self.in_out):
            self.enc.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, None, True if ind == 0 else False),
                        Downsample(dim_out) if self.space_down[ind] else nn.Identity(),
                        TimeDownsample(dim_out) if self.time_down[ind] else nn.Identity(),
                    ]
                )
            )
            


        for ind, (dim_in, dim_out) in enumerate(self.reversed_in_out):
            is_last = ind >= (len(self.reversed_in_out) - 1)
            mapping_or_identity = ResnetBlock(dim_in, dim_out) if not self.reversed_space_down[ind] and is_last else nn.Identity()
            upsample_layer = Upsample(dim_out if not is_last else dim_in, dim_out, self.new_upsample) if self.reversed_space_down[ind] else mapping_or_identity
            self.dec.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out if not is_last else dim_in),
                        upsample_layer,
                        TimeUpsample(dim_out) if self.reversed_time_down[ind] else nn.Identity(),
                    ]
                )
            )

    def encode(self, input, deterministic=True):
        self._enc_conv_idx = [0]

        input = input.to(self.dtype)
        for i, (resnet, down, time_down) in enumerate(self.enc):
            input = resnet(input, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            input = down(input)
            if isinstance(time_down, TimeDownsample):
                input = time_down(input, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
            else:
                input = time_down(input)
        input = input.float()
        quant_conv = self.quant_conv.float()
        quant_conv_res = self.quant_conv_res.float()

        conv1_out = quant_conv(input)
        for layer in quant_conv_res:
            if not isinstance(layer, ResnetBlock):
                input = layer(input)
            else: # ResnetBlock
                input = layer(input, feat_cache=self._enc_feat_map, feat_idx=self._enc_conv_idx)
        input += conv1_out

        posterior = DiagonalGaussianDistribution(input)
        if deterministic:
            z = posterior.mode()
        else:
            z = posterior.sample()
        return z, input

    def decode(self, input):
        self._conv_idx = [0]

        input = input.float()
        post_quant_conv = self.post_quant_conv.float()
        post_quant_conv_res = self.post_quant_conv_res.float()

        post1 = post_quant_conv(input)
        for layer in post_quant_conv_res:
            if not isinstance(layer, ResnetBlock):
                input = layer(input)
            else: # ResnetBlock
                input = layer(input, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        input += post1

        input = input.to(self.dtype)
        output = []
        for i, (resnet, up, time_up) in enumerate(self.dec):
            input = resnet(input, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            input = up(input)
            input = time_up(input)
            output.append(input)
        return output[::-1]

    def clear_cache(self):
        self._feat_map = [None] * self._conv_num
        self._enc_feat_map = [None] * self._enc_conv_num
        self._conv_idx = [0]
        self._enc_conv_idx = [0]


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        context_dim=None,
        context_out_channels=None,
        context_dim_mults=(1, 2, 3, 3),
        space_down=(1, 1, 1, 1),
        time_down=(0, 0, 0, 1),
        channels=3,
        with_time_emb=True,
        new_upsample=False,
        embd_type="01",
        condition_times=4,
    ):
        super().__init__()
        self.channels = channels

        dims = [channels, *map(lambda m: dim * m, dim_mults)]
        context_dim = context_dim or dim
        context_out_channels = context_out_channels or context_dim
        context_dims = [context_out_channels, *map(lambda m: context_dim * m, context_dim_mults)]
        self.space_down = space_down + [1] * (len(dim_mults)-len(space_down)-1) + [0]
        self.reversed_space_down = list(reversed(self.space_down[:-1]))
        self.time_down = time_down + [0] * (len(dim_mults)-len(time_down))
        self.reversed_time_down = list(reversed(self.time_down[:-1]))
        in_out = list(zip(dims[:-1], dims[1:]))
        self.embd_type = embd_type
        self.condition_times = condition_times

        if with_time_emb:
            if embd_type == "01":
                time_dim = dim
                self.time_mlp = nn.Sequential(nn.Linear(1, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))
            else:
                raise NotImplementedError
        else:
            time_dim = None
            self.time_mlp = None

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(
                                dim_in + context_dims[ind]
                                if (not is_last) and (ind < self.condition_times)
                                else dim_in,
                                dim_out,
                                time_dim,
                                True if ind == 0 else False
                            ),
                        ResnetBlock(dim_out, dim_out, time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if self.space_down[ind] else nn.Identity(),
                        TimeDownsample(dim_out) if self.time_down[ind] else nn.Identity(),
                    ]
                )
            )


        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
                
            is_last = ind >= (num_resolutions - 1)
            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_dim),
                        ResnetBlock(dim_in, dim_in, time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in, new_upsample=new_upsample) if self.reversed_space_down[ind] else nn.Identity(),
                        TimeUpsample(dim_in) if self.reversed_time_down[ind] else nn.Identity(),
                    ]
                )
            )

        out_dim = out_dim or channels
        self.final_conv = nn.ModuleList([LayerNorm(dim), CausalConv3d(dim, out_dim, (3, 7, 7), padding=(1, 3, 3))])

        # cache the last two frame of feature map
        self._conv_num = count_conv3d(self.ups) + count_conv3d(self.downs) + \
            count_conv3d(self.mid_block1) + count_conv3d(self.mid_block2) + \
            count_conv3d(self.final_conv) + count_time_down_sample(self.downs)
        self._conv_idx = [0]
        self._feat_map = [None] * self._conv_num

    @property
    def dtype(self):
        return self.final_conv[1].weight.dtype

    def encode(self, x, t, context):
        h = []
        for idx, (backbone, backbone2, attn, downsample, time_downsample) in enumerate(self.downs):
            x = torch.cat([x, context[idx]], dim=1) if idx < self.condition_times else x
            x = backbone(x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            x = backbone2(x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            x = attn(x, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            h.append(x)
            x = downsample(x)
            if isinstance(time_downsample, TimeDownsample):
                x = time_downsample(x, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            else:
                x = time_downsample(x)

        x = self.mid_block1(x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        return x, h

    def decode(self, x, h, t):
        device = x.device
        dtype = x.dtype
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
        for backbone, backbone2, attn, upsample, time_upsample in self.ups:
            reference = h.pop()
            if x.shape[2:] != reference.shape[2:]:
                x = F.interpolate(
                    x.float(), size=reference.shape[2:], mode='nearest'
                ).type_as(x)
            x = torch.cat((x, reference), dim=1)
            x = module_forward(backbone, x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            x = module_forward(backbone2, x, t, feat_cache=self._feat_map, feat_idx=self._conv_idx)
            x = module_forward(attn, x)
            x = module_forward(upsample, x)
            x = module_forward(time_upsample, x)
        x = x.to(device).to(dtype)

        x = self.final_conv[0](x)
        if self._feat_map is not None:
            idx = self._conv_idx[0]
            cache_x = x[:, :, -CACHE_T:, :, :].clone()
            if cache_x.shape[2] < 2 and self._feat_map[idx] is not None:
                # cache last frame of last two chunk
                cache_x = torch.cat([self._feat_map[idx][:, :, -1, :, :].unsqueeze(2), cache_x], dim=2)
            x = self.final_conv[1](x, self._feat_map[idx])
            self._feat_map[idx] = cache_x
            self._conv_idx[0] += 1
        else:
            x = self.final_conv[1](x)
        return x

    def forward(self, x, time=None, context=None):
        self._conv_idx = [0]
        t = None
        if self.time_mlp is not None:
            time_mlp = self.time_mlp.float()
            t = time_mlp(time).to(self.dtype)

        x = x.to(self.dtype)
        x, h = self.encode(x, t, context)
        return self.decode(x, h, t)

    def clear_cache(self):
        self._feat_map = [None] * self._conv_num
        self._conv_idx = [0]


def extract(a, t, x_shape):
    a = a.to(t.device)
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn=None,
        context_fn=None,
        ae_fn=None,
        num_timesteps=8192,
        pred_mode="x",
        var_schedule="cosine",
    ):
        super().__init__()
        self.denoise_fn = denoise_fn
        self.context_fn = context_fn
        self.ae_fn = ae_fn
        self.otherlogs = {}
        self.var_schedule = var_schedule
        self.sample_steps = None
        assert pred_mode in ["noise", "x", "v"]
        self.pred_mode = pred_mode
        to_torch = partial(torch.tensor, dtype=torch.float32)

        train_betas = cosine_beta_schedule(num_timesteps)
        train_alphas = 1.0 - train_betas
        train_alphas_cumprod = np.cumprod(train_alphas, axis=0)
        (num_timesteps,) = train_betas.shape
        self.num_timesteps = int(num_timesteps)

        self.train_snr=to_torch(train_alphas_cumprod / (1 - train_alphas_cumprod))
        self.train_betas=to_torch(train_betas)
        self.train_alphas_cumprod=to_torch(train_alphas_cumprod)
        self.train_sqrt_alphas_cumprod=to_torch(np.sqrt(train_alphas_cumprod))
        self.train_sqrt_one_minus_alphas_cumprod=to_torch(np.sqrt(1.0 - train_alphas_cumprod))
        self.train_sqrt_recip_alphas_cumprod=to_torch(np.sqrt(1.0 / train_alphas_cumprod))
        self.train_sqrt_recipm1_alphas_cumprod=to_torch(np.sqrt(1.0 / train_alphas_cumprod - 1))

    def set_sample_schedule(self, sample_steps, device):
        self.sample_steps = sample_steps
        if sample_steps != 1:
            indice = torch.linspace(0, self.num_timesteps - 1, sample_steps, device=device).long()
        else:
            indice = torch.tensor([self.num_timesteps - 1], device=device).long()
        self.train_alphas_cumprod = self.train_alphas_cumprod.to(device)
        self.train_snr = self.train_snr.to(device)
        self.alphas_cumprod = self.train_alphas_cumprod[indice]
        self.snr = self.train_snr[indice]
        self.index = torch.arange(self.num_timesteps, device=device)[indice]
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_alphas_cumprod_prev = torch.sqrt(self.alphas_cumprod_prev)
        self.one_minus_alphas_cumprod = 1.0 - self.alphas_cumprod
        self.one_minus_alphas_cumprod_prev = 1.0 - self.alphas_cumprod_prev
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod_prev = torch.sqrt(1.0 - self.alphas_cumprod_prev)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod_prev = torch.sqrt(1.0 / self.alphas_cumprod_prev)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        self.sigma = self.sqrt_one_minus_alphas_cumprod_prev / self.sqrt_one_minus_alphas_cumprod * torch.sqrt(1.0 - self.alphas_cumprod / self.alphas_cumprod_prev)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        if self.training:
            return (
                extract(self.train_sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
            )
        else:
            return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
            )

    def predict_start_from_v(self, x_t, t, v):
        if self.training:
            return (
                extract(self.train_sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.train_sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
            )
        else:
            return (
                extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
            )

    def predict_start_from_noise(self, x_t, t, noise):
        if self.training:
            return (
                extract(self.train_sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.train_sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )

    def ddim(self, x, t, context, clip_denoised, eta=0):
        if self.denoise_fn.embd_type == "01":
            fx = self.denoise_fn(x, self.index[t].float().unsqueeze(-1) / self.num_timesteps, context=context)
        else:
            fx = self.denoise_fn(x, self.index[t], context=context)
        fx = fx.float()
        if self.pred_mode == "noise":
            x_recon = self.predict_start_from_noise(x, t=t, noise=fx)
        elif self.pred_mode == "x":
            x_recon = fx
        elif self.pred_mode == "v":
            x_recon = self.predict_start_from_v(x, t=t, v=fx)
        if clip_denoised:
            x_recon.clamp_(-1.0, 1.0)
        noise = fx if self.pred_mode == "noise" else self.predict_noise_from_start(x, t=t, x0=x_recon)
        x_next = (
            extract(self.sqrt_alphas_cumprod_prev, t, x.shape) * x_recon
            + torch.sqrt(
                (extract(self.one_minus_alphas_cumprod_prev, t, x.shape)
                - (eta * extract(self.sigma, t, x.shape)) ** 2).clamp(min=0)
            )
            * noise + eta * extract(self.sigma, t, x.shape) * torch.randn_like(noise)
        )
        return x_next

    def p_sample(self, x, t, context, clip_denoised, eta=0):
        return self.ddim(x=x, t=t, context=context, clip_denoised=clip_denoised, eta=eta)

    def p_sample_loop(self, shape, context, clip_denoised=False, init=None, eta=0):
        device = self.alphas_cumprod.device

        b = shape[0]
        img = torch.zeros(shape, device=device) if init is None else init
        for count, i in enumerate(reversed(range(0, self.sample_steps))):
            time = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(
                img,
                time,
                context=context,
                clip_denoised=clip_denoised,
                eta=eta,
            )
        return img

    @torch.no_grad()
    def compress(
        self,
        images,
        sample_steps=10,
        init=None,
        clip_denoised=True,
        eta=0,
    ):
        context_dict = self.context_fn(images, 'test')
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            context_dict["output"][0].device,
        )
        return self.p_sample_loop(
                    images.shape, context_dict["output"],
                    clip_denoised=clip_denoised, init=init, eta=eta
                )

    @torch.no_grad()
    def diffusion_decode(
        self,
        latent,
        sample_steps=30,
        init=None,
        time=None,
        clip_denoised=False,
        eta=0,
    ):
        context = self.context_fn.decode(latent)

        # breakpoint()
        self.set_sample_schedule(
            self.num_timesteps if (sample_steps is None) else sample_steps,
            context[0].device,
        )

        img = init
        img = self.p_sample(
            img,
            time,
            context=context,
            clip_denoised=clip_denoised,
            eta=eta,
        )
        return img


class ConditionedDiffusionTokenizer(nn.Module):

    def __init__(self,
            pretrained=None, # pretrained ckpt path
            enc_bs=1, # mini-batch to loop for both enc and dec
            enc_frames=13, # mini-batch frames to loop for both enc and dec
            dec_bs=1, # mini-batch to loop for dec
            dec_frames=4, # mini-batch frames to loop for dec
            z_overlap=1, # mini-batch inference overlap
            sample_steps=10, # decode diffusion steps
            sample_gamma=0.8,  # decode noise weight
            num_timesteps=8192,
            out_channels=3,
            context_dim=64,
            unet_dim=64,
            new_upsample=False,
            latent_dim=16,
            context_dim_mults = [1, 2, 3, 4],
            space_down = [1, 1, 1, 1],
            time_down = [0, 0, 1, 1],
            condition_times=4,
            **kwargs
            ):
        super().__init__()
        self.latent_scale = None
        self.enc_bs = enc_bs
        self.enc_frames = enc_frames
        self.dec_bs = dec_bs or enc_bs
        self.dec_frames = dec_frames or enc_frames
        self.space_factor = 2 ** sum(space_down)
        self.time_factor = 2 ** sum(time_down)
        self.sample_steps = sample_steps
        self.sample_gamma = sample_gamma
        self.z_overlap = z_overlap
        self.num_timesteps = num_timesteps
        self.condition_times = condition_times
        CausalConv3d.PAD_MODE = 'constant' if new_upsample else 'replicate'
        context_model = Compressor(
                            dim=context_dim,
                            latent_dim=latent_dim,
                            new_upsample=new_upsample,
                            channels=3,
                            out_channels=out_channels,
                            dim_mults=context_dim_mults,
                            reverse_dim_mults=reversed(context_dim_mults),
                            space_down=space_down,
                            time_down=time_down)
        denoise_model = Unet(
                            dim=unet_dim,
                            channels=3,
                            new_upsample=new_upsample,
                            context_dim=context_dim,
                            context_out_channels=out_channels,
                            dim_mults=[1, 2, 3, 4, 5, 6],
                            context_dim_mults=context_dim_mults,
                            space_down=space_down,
                            time_down=time_down,
                            condition_times=condition_times
                            )
        self.model = GaussianDiffusion(
                            context_fn=context_model,
                            denoise_fn=denoise_model,
                            num_timesteps=self.num_timesteps,
                            )
        self.z_dim = latent_dim
        if pretrained is not None:
            self.init_from_ckpt(pretrained)

    def init_from_ckpt(
        self,
        path: str,
    ) -> None:
        if path.endswith("safetensors"):
            sd = load_safetensors(path)
        else:
            # breakpoint()
            sd = torch.load(path, map_location="cpu", weights_only=False)
        sd = sd.get("state_dict", sd)
        sd = sd.get("module", sd)
        sd = {k:v.to(torch.float32) for k,v in sd.items()}
        missing, unexpected = self.load_state_dict(sd, strict=False)
        logging.info(
            f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys"
        )
        if len(missing) > 0:
            logging.info(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            logging.info(f"Unexpected Keys: {unexpected}")

    @torch.no_grad()
    def forward(self, x, **kwargs):
        latent = self.encode(x)
        x_recon = self.decode(latent)
        return x_recon

    @torch.no_grad()
    def forward_with_info(self, x, **kwargs):
        latent, posterior = self.encode(x, return_posterior=True)
        x_recon = self.decode(latent)
        mu = posterior.mean
        logvar = posterior.logvar
        z_std = mu.std()
        return x_recon, mu, logvar, z_std

    @torch.no_grad()
    def encode(self, x, scale_factor=1.0, return_posterior=False, **kwargs):
        N, C, T, H, W = x.shape
        mini_batch_size = min(self.enc_bs, N) if self.enc_bs else N
        enc_frames_ratio = (1024/H * 1024/W) ** 1.0
        enc_frames = int(self.enc_frames * enc_frames_ratio/4) * 4
        enc_frames = max(enc_frames, 4)
        logging.debug(f"enc: {x.shape}, {self.enc_frames}, {enc_frames_ratio}, {enc_frames}")
        z_overlap = 0
        frame_overlap = 0

        mini_frames = min(enc_frames, T) if enc_frames else T
        n_batches = int(math.ceil(N / mini_batch_size))
        n_frame_batches = math.ceil((T-frame_overlap) / max(1, mini_frames-frame_overlap))
        n_frame_batches = max(int(n_frame_batches), 1)
        remainder = T % mini_frames
        z = list()
        mean_std_list = list()
        for i_batch in range(n_batches):
            z_batch = []
            mean_std_batch = []
            x_batch = x[i_batch * mini_batch_size : (i_batch + 1) * mini_batch_size]
            frame_end = 0
            for i_frames in range(n_frame_batches):
                frame_start = frame_end
                if i_frames == 0 and remainder > 0:
                    frame_end = frame_start + remainder
                else:
                    frame_end = frame_start + mini_frames
                i_batch_input = x_batch[:, :, frame_start:frame_end, ...]
                logging.debug(f'enc: {i_frames}, {i_batch_input.shape}')
                i_batch_input = i_batch_input.to(next(self.parameters()).device)
                z_frames, mean_std = self.model.context_fn.encode(i_batch_input, deterministic=True)
                z_batch.append(z_frames[:, :, (z_overlap if i_frames>0 else 0):, ...])
                mean_std_batch.append(mean_std[:, :, (z_overlap if i_frames>0 else 0):, ...])
            z_batch = torch.cat(z_batch, dim=2)
            z.append(z_batch)
            mean_std_batch = torch.cat(mean_std_batch, dim=2)
            mean_std_list.append(mean_std_batch)
            self.model.context_fn.clear_cache()
        latent = torch.cat(z, 0)
        mean_std = torch.cat(mean_std_list, 0)
        posterior = DiagonalGaussianDistribution(mean_std)
        latent *= scale_factor
        if return_posterior:
            return latent, posterior
        else:
            return latent

    @torch.no_grad()
    def decode(self, latent, scale_factor=1.0, **kwargs):
        N, C, T, H, W = latent.shape
        latent = latent/scale_factor
        mini_batch_size = min(self.dec_bs, N) if self.dec_bs else N
        n_batches = int(math.ceil(N / mini_batch_size))
        dec_frames_ratio = (1024/self.space_factor/H * 1024/self.space_factor/W) ** 1.0
        dec_frames = int(self.dec_frames * dec_frames_ratio)
        dec_frames = max(1, int(dec_frames))
        logging.debug(f"dec: {latent.shape}, {self.dec_frames}, {dec_frames_ratio}, {dec_frames}")
        z_overlap = 0
        frame_overlap = 0

        assert dec_frames >= z_overlap + 1, f"dec_frames {dec_frames} too small"
        mini_frames = min(dec_frames, T) if dec_frames else T
        n_frame_batches = math.ceil((T-z_overlap) / max(1, (mini_frames-z_overlap)))
        n_frame_batches = max(int(n_frame_batches), 1)
        dec = list()
        target_shape = [N, 3, T*self.time_factor, H*self.space_factor, W*self.space_factor]
        init_noise = self.sample_gamma*torch.randn(target_shape, dtype=latent.dtype, device=latent.device)

        for i_batch in range(n_batches):
            x_batch = latent[i_batch * mini_batch_size : (i_batch + 1) * mini_batch_size]
            noise_batch = init_noise[i_batch * mini_batch_size : (i_batch + 1) * mini_batch_size]
            z_batch_list = [None] * n_frame_batches
            for count, t_idx in enumerate(reversed(range(0, self.sample_steps))):
                for i_frames in range(n_frame_batches):
                    global CLIP_FIRST_CHUNK
                    CLIP_FIRST_CHUNK = True if i_frames == 0 else False

                    latent_frame_start = i_frames * (mini_frames-z_overlap)
                    latent_frame_end = latent_frame_start+mini_frames
                    i_batch_input = x_batch[:, :, latent_frame_start:latent_frame_end, ...]
                    if count == 0:
                        frame_start = latent_frame_start*self.time_factor
                        frame_end = frame_start + (mini_frames)*self.time_factor
                        if CLIP_FIRST_CHUNK:
                            frame_start += 3
                        cur_noise = noise_batch[:, :, frame_start:frame_end, ...]
                    else:
                        cur_noise = z_batch_list[i_frames]
                    time = torch.full((cur_noise.shape[0],), t_idx, device=latent.device, dtype=torch.long)
                    logging.debug(f'dec: {i_frames}, {i_batch_input.shape}, {cur_noise.shape}')
                    x_rec = self.model.diffusion_decode(i_batch_input,
                                    sample_steps=self.sample_steps, init=cur_noise, time=time)
                    z_batch_list[i_frames] = x_rec[:, :, (0 if i_frames==0 else frame_overlap):, ...]
                # clear cache
                self.model.context_fn.clear_cache()
                self.model.denoise_fn.clear_cache()

            dec.append(torch.cat(z_batch_list, dim=2))
            del z_batch_list, x_batch, noise_batch

        dec = torch.cat(dec, 0)
        return dec


def load_cdt_base(
    ckpt = None,
    device='cpu',
    eval=True,
    sampling_step=1,
    **kwargs
):
    ckpt = ckpt or './pretrained/cdt_base.ckpt'
    print(f"Loading CDT-base from {ckpt}")


    model = ConditionedDiffusionTokenizer(pretrained=ckpt,
                                 enc_frames=kwargs.pop('enc_frames', 4),
                                 dec_frames=kwargs.pop('dec_frames', 1),
                                 space_down=[0,1,1,1],
                                 out_channels=16,
                                 latent_dim=16,
                                 context_dim=128,
                                 new_upsample=True,
                                 sample_steps=sampling_step,
                                 sample_gamma=kwargs.pop('sample_gamma', 0.8),
                                 **kwargs)
    model = model.to(device)
    if eval:
        model = model.eval()
    return model

def load_cdt_small(
    ckpt = None,
    device='cpu',
    eval=True,
    latent_dim=16,
    diffusion_step=8192,
    sampling_step=1,
    condition_times=4,
    **kwargs
):
    ckpt = ckpt or "./pretrained/cdt_small.ckpt"

    model = ConditionedDiffusionTokenizer(pretrained=ckpt,
                                 enc_frames=kwargs.pop('enc_frames', 4),
                                 dec_frames=kwargs.pop('dec_frames', 1),
                                 space_down=[1, 1, 1, 0],
                                 out_channels=3,
                                 latent_dim=latent_dim,
                                 context_dim=64,
                                 num_timesteps=diffusion_step,
                                 condition_times=condition_times,
                                 sample_steps=sampling_step,
                                 sample_gamma=kwargs.pop('sample_gamma', 0.8),
                                 **kwargs)
    model = model.to(device)
    if eval:
        model = model.eval()
    return model

def load_cdt(version='base', dtype=torch.float16, *args, **kwargs):
    VERSIONS = {
        'base': (load_cdt_base, 1.3),
        'small': (load_cdt_small, 1.3),
    }
    if version not in VERSIONS:
        print(f"ERROR: wrong version '{version}' of CDT, not in '{VERSIONS.keys()}'")
        return None
    model_func, latent_scale = VERSIONS[version]
    model = model_func(*args, **kwargs)
    model.latent_scale = latent_scale
    model = model.to(dtype)
    return model

