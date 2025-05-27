import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import resnet50
from math import log
from .Res2Net import res2net50_v1b_26w_4s
import numpy as np
import math
from .base import BaseNetwork
from .transformer_block import FeedForward2D
class ConvBNR(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, stride=1, dilation=1, bias=False):
        super(ConvBNR, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size, stride=stride, padding=dilation, dilation=dilation, bias=bias),
            nn.BatchNorm2d(planes),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class Conv1x1(nn.Module):
    def __init__(self, inplanes, planes):
        super(Conv1x1, self).__init__()
        self.conv = nn.Conv2d(inplanes, planes, 1)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class EFM(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att + c
        x = self.conv2d(x)
        wei = self.avg_pool(x)
        wei = self.conv1d(wei.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        wei = self.sigmoid(wei)
        x = x * wei

        return x

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class DM(nn.Module):
    def __init__(self):
        super(DM, self).__init__()
        self.predict3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.PReLU(),
            nn.Conv2d(64, 1, kernel_size=1)
        )
        self.ra2_conv2 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64, 64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, xr, dualattention):
        crop_3 = F.interpolate(dualattention, xr.size()[2:], mode='bilinear', align_corners=False)
        re3_feat = self.predict3(torch.cat([xr, crop_3], dim=1))
        x = -1*(torch.sigmoid(crop_3)) + 1
        x = x.expand(-1, 64, -1, -1).mul(xr)
        x = F.relu(self.ra2_conv2(x))
        x = F.relu(self.ra2_conv3(x))
        ra3_feat = self.ra2_conv4(x)
        x = ra3_feat + crop_3 + re3_feat


        return x


class _DAHead(nn.Module):
    def __init__(self, in_channels, nclass, aux=True, norm_layer=nn.BatchNorm2d, norm_kwargs=None, **kwargs):
        super(_DAHead, self).__init__()
        self.aux = aux
        inter_channels = in_channels // 4
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.pam = _PositionAttentionModule(inter_channels, **kwargs)
        self.cam = _ChannelAttentionModule(**kwargs)
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.conv_c2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels, **({} if norm_kwargs is None else norm_kwargs)),
            nn.ReLU(True)
        )
        self.out = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, nclass, 1)
        )
        if aux:
            self.conv_p3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )
            self.conv_c3 = nn.Sequential(
                nn.Dropout(0.1),
                nn.Conv2d(inter_channels, nclass, 1)
            )

    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.pam(feat_p)
        feat_p = self.conv_p2(feat_p)

        feat_c = self.conv_c1(x)
        feat_c = self.cam(feat_c)
        feat_c = self.conv_c2(feat_c)

        feat_fusion = feat_p + feat_c

        outputs = []
        fusion_out = self.out(feat_fusion)
        outputs.append(fusion_out)
        if self.aux:
            p_out = self.conv_p3(feat_p)
            c_out = self.conv_c3(feat_c)
            outputs.append(p_out)
            outputs.append(c_out)

        return tuple(outputs)

def run_sobel(conv_x, conv_y, input):
    g_x = conv_x(input)
    g_y = conv_y(input)
    g = torch.sqrt(torch.pow(g_x, 2) + torch.pow(g_y, 2))
    assert not torch.isnan(g).any()
    return torch.sigmoid(g) * input

def get_sobel(in_chan, out_chan):
    '''
    filter_x = np.array([
        [3, 0, -3],
        [10, 0, -10],
        [3, 0, -3],
    ]).astype(np.float32)
    filter_y = np.array([
        [3, 10, 3],
        [0, 0, 0],
        [-3, -10, -3],
    ]).astype(np.float32)
    '''
    filter_x = np.array([
        [1, 0, -1],
        [2, 0, -2],
        [1, 0, -1],
    ]).astype(np.float32)
    filter_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1],
    ]).astype(np.float32)
    filter_x = filter_x.reshape((1, 1, 3, 3))
    filter_x = np.repeat(filter_x, in_chan, axis=1)
    filter_x = np.repeat(filter_x, out_chan, axis=0)

    filter_y = filter_y.reshape((1, 1, 3, 3))
    filter_y = np.repeat(filter_y, in_chan, axis=1)
    filter_y = np.repeat(filter_y, out_chan, axis=0)

    filter_x = torch.from_numpy(filter_x)
    filter_y = torch.from_numpy(filter_y)
    filter_x = nn.Parameter(filter_x, requires_grad=False)
    filter_y = nn.Parameter(filter_y, requires_grad=False)
    conv_x = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_x.weight = filter_x
    conv_y = nn.Conv2d(in_chan, out_chan, kernel_size=3, stride=1, padding=1, bias=False)
    conv_y.weight = filter_y
    sobel_x = nn.Sequential(conv_x, nn.BatchNorm2d(out_chan))
    sobel_y = nn.Sequential(conv_y, nn.BatchNorm2d(out_chan))
    return sobel_x, sobel_y

class GlobalFilter(nn.Module):
    def __init__(self, dim=32, h=64, w=33, fp32fft=True):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(h, w, dim, 2, dtype=torch.float32) * 0.02
        )
        self.w = w
        self.h = h
        self.fp32fft = fp32fft

    def forward(self, x):
        b, _, a, b = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()

        if self.fp32fft:
            dtype = x.dtype
            x = x.to(torch.float32)

        x = torch.fft.rfft2(x, dim=(1, 2), norm="ortho")
        #print(x.shape)
        weight = torch.view_as_complex(self.complex_weight)
       # print(x.shape)
        #print(weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(a, b), dim=(1, 2), norm="ortho")

        if self.fp32fft:
            x = x.to(dtype)

        x = x.permute(0, 3, 1, 2).contiguous()

        return x

class ERB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ERB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, relu=True):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        if relu:
            return self.relu(x + res)
        else:
            return x+res

class _PositionAttentionModule(nn.Module):
    """ Position attention module"""

    def __init__(self, in_channels, **kwargs):
        super(_PositionAttentionModule, self).__init__()
        self.conv_b = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_c = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv_d = nn.Conv2d(in_channels, in_channels, 1)
        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_b = self.conv_b(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        feat_c = self.conv_c(x).view(batch_size, -1, height * width)
        attention_s = self.softmax(torch.bmm(feat_b, feat_c))
        feat_d = self.conv_d(x).view(batch_size, -1, height * width)
        feat_e = torch.bmm(feat_d, attention_s.permute(0, 2, 1)).view(batch_size, -1, height, width)
        out = self.alpha * feat_e + x

        return out


class _ChannelAttentionModule(nn.Module):
    """Channel attention module"""

    def __init__(self, **kwargs):
        super(_ChannelAttentionModule, self).__init__()
        self.beta = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, _, height, width = x.size()
        feat_a = x.view(batch_size, -1, height * width)
        feat_a_transpose = x.view(batch_size, -1, height * width).permute(0, 2, 1)
        attention = torch.bmm(feat_a, feat_a_transpose)
        attention_new = torch.max(attention, dim=-1, keepdim=True)[0].expand_as(attention) - attention
        attention = self.softmax(attention_new)

        feat_e = torch.bmm(attention, feat_a).view(batch_size, -1, height, width)
        out = self.beta * feat_e + x

        return out
        
class EAM(nn.Module):
    def __init__(self):
        super(EAM, self).__init__()
        self.reduce1 = Conv1x1(256, 64)
        self.reduce4 = Conv1x1(2048, 256)
        self.block = nn.Sequential(
            ConvBNR(256 + 64, 256, 3),
            ConvBNR(256, 256, 3),
            nn.Conv2d(256, 1, 1))

    def forward(self, x4, x1):
        size = x1.size()[2:]
        x1 = self.reduce1(x1)
        x4 = self.reduce4(x4)
        x4 = F.interpolate(x4, size, mode='bilinear', align_corners=False)
        out = torch.cat((x4, x1), dim=1)
        out = self.block(out)

        return out

def attention(query, key, value):
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
        query.size(-1)
    )   # shape: (32,4,4)
    p_attn = F.softmax(scores, dim=-1)  
    p_val = torch.matmul(p_attn, value)
    return p_val, p_attn

class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, stride, d_model):
        super().__init__()
        self.stride = stride
        self.query_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.value_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.key_embedding = nn.Conv2d(
            d_model, d_model, kernel_size=1, padding=0
        )
        self.output_linear = nn.Sequential(
            nn.Conv2d(d_model, d_model, kernel_size=3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        b, c, h, w = x.size() #32,256,64,64
        d_k = c // len(self.stride)
        output = []
        _query = self.query_embedding(x) # 32,256,64,64
        _key = self.key_embedding(x) # 32,256,64,64
        _value = self.value_embedding(x)  # 32,256,64,64
        attentions = []
        for (ws, hs), query, key, value in zip(   # ws: width stride, hs: height stride
            self.stride,
            torch.chunk(_query, len(self.stride), dim=1),
            torch.chunk(_key, len(self.stride), dim=1),
            torch.chunk(_value, len(self.stride), dim=1),
        ):  
            
            #print('-----------query):',query.shape)  #8,64,64,64
            
            out_w, out_h = w // ws, h // hs   # out_w here is patch size
            ## 1) embedding and reshape
            query = torch.stack(
                [query[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )  # multi-head, shape = (32,64,2,32,2,32)
            
            query = query.reshape(b, ws*hs, d_k*out_w*out_h)

            key = torch.stack(
                [key[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )
            key = key.reshape(b, ws*hs, d_k*out_w*out_h)
            
            value = torch.stack(
                [value[:,:,i::hs,j::ws] for i, j in itertools.product(range(hs),range(ws))],dim=1
            )
            value = value.reshape(b, ws*hs, d_k*out_w*out_h)

            y, _ = attention(query, key, value)

            # 3) "Concat" using a view and apply a final linear.
            y = y.view(b, hs, ws, d_k, out_h, out_w)
            y = y.permute(0, 3, 1, 4, 2, 5).contiguous().view(b, d_k, h, w)  # (32,64,64,64)
            attentions.append(y)
            output.append(y)

        output = torch.cat(output, 1)   # stack together
        self_attention = self.output_linear(output)

        return self_attention


class TransformerBlock(nn.Module):
    """
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, stride, in_channel=256):
        super().__init__()
        self.attention = MultiHeadedAttention(stride, d_model=in_channel)
        self.feed_forward = FeedForward2D(
            in_channel=in_channel, out_channel=in_channel
        )

    def forward(self, rgb):
        self_attention = self.attention(rgb)
        output = rgb + self_attention
        output = output + self.feed_forward(output)
        return output


class PatchTrans(BaseNetwork):
    def __init__(self, in_channel, in_size): # 256, 64
        super(PatchTrans, self).__init__()
        self.in_size = in_size # 64

        stride = [
              (16,16),
              (8,8),
              (4,4),
              (2,2),
        ]

        self.t = TransformerBlock(stride, in_channel=in_channel)

    def forward(self, enc_feat):
        output = self.t(enc_feat)   
        return output


class multi(nn.Module):
    def __init__(self, channel):
        super(EFM, self).__init__()
        t = int(abs((log(channel, 2) + 1) / 2))
        k = t if t % 2 else t + 1
        self.conv2d = ConvBNR(channel, channel, 3)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, c, att):
        if c.size() != att.size():
            att = F.interpolate(att, c.size()[2:], mode='bilinear', align_corners=False)
        x = c * att
        return x


class CTO_stitchvit(nn.Module):
    def __init__(self,seg_classes):
        super(CTO_stitchvit, self).__init__()
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)       
        self.multi_trans = PatchTrans(in_channel=256,in_size=64)
               
        self.num_class = seg_classes
        self.eam = EAM()
        self.sobel_x1, self.sobel_y1 = get_sobel(256, 1)
        self.sobel_x4, self.sobel_y4 = get_sobel(2048, 1)
        
        
        self.head = _DAHead(2048+256, 2048, aux=False)

        self.reduce1 = Conv1x1(256, 64)
        self.reduce2 = Conv1x1(512, 64)
        self.reduce3 = Conv1x1(1024, 64)
        self.reduce4 = Conv1x1(2048, 64)

        self.dm1 = DM()
        self.dm2 = DM()
        self.dm3 = DM()

        self.predictor1 = nn.Conv2d(64, self.num_class, 1)
        self.predictor2 = nn.Conv2d(64, self.num_class, 1)
        self.predictor3 = nn.Conv2d(64, self.num_class, 1)


    def forward(self, x):
        x1, x2, x3 ,x4= self.resnet(x)#[16, 256, 64, 64]  [16, 512, 32, 32]   [16, 1024, 16, 16]   [16, 2048, 16, 16]
        
        trans = self.multi_trans(x1)#16,256,64,64
        
        s1 = run_sobel(self.sobel_x1, self.sobel_y1, x1)
        s4 = run_sobel(self.sobel_x4, self.sobel_y4, x4)
       
        edge = self.eam(s4, s1)
        edge_att = torch.sigmoid(edge)#[16, 1, 64, 64]
        
        trans = F.interpolate(trans,x4.size()[2:], mode='bilinear', align_corners=False)#256,8,8
        dual_attention = self.head(torch.cat([trans, x4], dim=1))[0]  #2048,16,16
        
        x1a = x1*edge_att
        edge_att2 = F.interpolate(edge_att, x2.size()[2:], mode='bilinear', align_corners=False)
        x2a = x2*edge_att2
        edge_att3 = F.interpolate(edge_att, x3.size()[2:], mode='bilinear', align_corners=False)
        x3a = x3*edge_att3
        
        x1r = self.reduce1(x1a)  
        x2r = self.reduce2(x2a)#128,32,32
        x3r = self.reduce3(x3a)#256,16,16
        
        dual_attention = self.reduce4(dual_attention)
       
        c3 = self.dm3(x3r, dual_attention) #256 16 16
        c2 = self.dm2(x2r, c3)  #128 32 32
        c1 = self.dm1(x1r, c2) #64 64 64

        o3 = self.predictor3(c3)
        o3 = F.interpolate(o3, scale_factor=16, mode='bilinear', align_corners=False)
        o2 = self.predictor2(c2)
        o2 = F.interpolate(o2, scale_factor=8, mode='bilinear', align_corners=False) 
        o1 = self.predictor1(c1)
        o1 = F.interpolate(o1, scale_factor=4, mode='bilinear', align_corners=False)
        oe = F.interpolate(edge_att, scale_factor=4, mode='bilinear', align_corners=False)

        return  o3, o2, o1, oe