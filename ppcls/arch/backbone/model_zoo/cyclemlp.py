
import math
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.nn.initializer import TruncatedNormal, Constant

from paddle.vision.ops import deform_conv2d

from .vision_transformer import trunc_normal_, zeros_, ones_, to_2tuple, DropPath, Identity


class Mlp(nn.Layer):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CycleFC(nn.Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        assert in_channels % groups == 0, 'in_channels must be divisible by groups'
        assert out_channels % groups == 0, 'out_channels must be divisible by groups'
        assert stride == 1, "stride must be 1"
        assert padding == 0, "padding must be 0"
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups

        self.weight = self.create_parameter(shape=[out_channels, in_channels // groups, 1, 1], default_initializer=nn.initializer.KaimingUniform(math.sqrt(5))) 
        if bias:
            num_input_fmaps = in_channels // groups
            receptive_field_size = 1 * 1
            fan_in = num_input_fmaps * receptive_field_size
            bound = 1 / math.sqrt(fan_in)
            self.bias = self.create_parameter(shape=(out_channels, ), is_bias=True, default_initializer=nn.initializer.Uniform(-bound, bound))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

    def gen_offset(self):
        offset = paddle.zeros([1, self.in_channels * 2, 1, 1])
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, "kernel_size muse include 1"
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, x):
        B, C, H, W = x.shape
        return deform_conv2d(x, paddle.tile(self.offset, [B, 1, H, W]), self.weight, self.bias, stride=self.stride, padding=self.padding, dilation=self.dilation, deformable_groups=self.in_channels)


class CycleMLP(nn.Layer):
    def __init__(self, dim, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.mlp_c = nn.Linear(dim, dim, bias_attr=qkv_bias)
        self.sfc_h = CycleFC(dim, dim, (1, 3), 1, 0)
        self.sfc_w = CycleFC(dim, dim, (3, 1), 1, 0)

        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.sfc_h(x.transpose([0, 3, 1, 2])).transpose([0, 2, 3, 1])
        w = self.sfc_w(x.transpose([0, 3, 1, 2])).transpose([0, 2, 3, 1])
        c = self.mlp_c(x)

        a = (h + w + c).transpose([0, 3, 1, 2]).flatten(2).mean(2)
        a = self.reweight(a).reshape([B, C, 3]).transpose([2, 0, 1])
        a = F.softmax(a, axis=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CycleBlock(nn.Layer):
    def __init__(self, dim, mlp_ratio=4, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=CycleMLP):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


class PatchEmbedOverlapping(nn.Layer):
    def __init__(self, patch_size=16, stride=16, padding=0, in_channels=3, embed_dim=768, norm_layer=None, groups=1):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.patch_size = patch_size

        self.proj = nn.Conv2D(in_channels, embed_dim, kernel_size=patch_size, stride=stride, padding=padding, groups=groups)
        # self.norm = norm_layer(embed_dim) if norm_layer else Identity()

    def forward(self, x):
        x = self.proj(x)
        return x


class Downsample(nn.Layer):
    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        assert patch_size == 2, "patch_size must be 2"
        self.proj = nn.Conv2D(in_embed_dim, out_embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=1)

    def forward(self, x):
        x = x.transpose([0, 3, 1, 2])
        x = self.proj(x)
        x = x.transpose([0, 2, 3, 1])
        return x


def basic_blocks(dim, index, layers, mlp_ratio=3., qkv_bias=False, qk_scale=None, attn_drop=0., drop_path_rate=0., skip_lam=1.0, mlp_fn=CycleMLP, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(CycleBlock(dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)

    return blocks


class CycleNet(nn.Layer):
    def __init__(self, layers, img_size=224, patch_size=4, in_channels=3, class_num=1000, embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0, qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, mlp_fn=CycleMLP):
        super().__init__()
        self.class_num = class_num
        self.patch_embed = PatchEmbedOverlapping(patch_size=7, stride=4, padding=2, in_channels=3, embed_dim=embed_dims[0])
        
        self.network = nn.LayerList()
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
            self.network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i+1]:
                patch_size = 2 if transitions[i] else 1
                self.network.append(Downsample(embed_dims[i], embed_dims[i+1], patch_size))
            
        self.norm = norm_layer(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], class_num)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            zeros_(m.bias)
            ones_(m.weight)
        elif isinstance(m, CycleFC):
            trunc_normal_(m.weight)
            if isinstance(m, CycleFC) and m.bias is not None:
                zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.transpose([0, 2, 3, 1])
        # B, H, W, C -> B, N, C
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape([B, -1, C])
        x = self.norm(x)
        out = self.head(x.mean(1))
        return out


def CycleMLP_B1(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 2, 4, 2]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CycleNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, **kwargs)
    return model


def CycleMLP_B2(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [2, 3, 10, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CycleNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, **kwargs)
    return model


def CycleMLP_B3(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 18, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CycleNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, **kwargs)
    return model


def CycleMLP_B4(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 8, 27, 3]
    mlp_ratios = [8, 8, 4, 4]
    embed_dims = [64, 128, 320, 512]
    model = CycleNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, **kwargs)
    return model


def CycleMLP_B5(pretrained=False, **kwargs):
    transitions = [True, True, True, True]
    layers = [3, 4, 24, 3]
    mlp_ratios = [4, 4, 4, 4]
    embed_dims = [96, 192, 384, 768]
    model = CycleNet(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                     mlp_ratios=mlp_ratios, mlp_fn=CycleMLP, **kwargs)
    return model

























