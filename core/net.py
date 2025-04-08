import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.nn import init
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

# 用于构建神经网络中的基本块（或称为残差块）
class BasicBlock(nn.Module):

    expansion = 1
    
    def __init__(self, mastermodel, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),   
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        self.se = SEAttention(out_channels,16)# SAenet

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )
        self.mastermodel = mastermodel

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.se(self.residual_function(x)) + self.shortcut(x))

        # return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
# 用于构建神经网络中的残差块
class ResNet(nn.Module):
    def __init__(self, block=BasicBlock, num_block=[2, 2, 2, 2], avg_output=False, output_dim=-1, resprestride=1, res1ststride=1, res2ndstride=1, inchan=3):
        super().__init__()
        img_chan = inchan
        self.conv1 = nn.Sequential(
            nn.Conv2d(img_chan, 64, kernel_size=3, padding=1, bias=False, stride=resprestride),
            nn.BatchNorm2d(64),
            nn.LeakyReLU())
        self.in_channels = 64
        self.conv2_x = self._make_layer(block, 64, num_block[0], res1ststride)
        self.conv3_x = self._make_layer(block, 128, num_block[1], res2ndstride)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.conv6_x = nn.Identity() if output_dim <= 0 else self.conv_layer(512, output_dim, 1, 0)
        self.conv6_is_identity = output_dim <= 0
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        if output_dim > -1:
            self.output_dim = output_dim
        else:
            self.output_dim = 512 * block.expansion
        self.avg_output = avg_output

    def conv_layer(self, input_channel, output_channel, kernel_size=3, padding=1):
        print("conv layer input", input_channel, "output", output_channel)
        res = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size, 1, padding, bias=False),
            nn.BatchNorm2d(output_channel),
            nn.LeakyReLU(0.2))
        return res

    def _make_layer(self, block, out_channels, num_blocks, stride):
        print("Making resnet layer with channel", out_channels, "block", num_blocks, "stride", stride)

        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(None, self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.conv6_x(output)
        if self.avg_output:
            output = self.avg_pool(output)
            output = output.view(output.size(0), -1)
        return output

#  根据给定的参数构建网络骨干，返回骨干网络、输出维度和CAM尺寸。
def build_backbone(img_size, backbone_name, projection_dim, inchan = 3):
    if backbone_name == 'resnet18':
        backbone = ResNet(output_dim = projection_dim, inchan = inchan, resprestride=1, res1ststride = 1, res2ndstride = 2)
        cam_size = int(img_size / 8)
    elif backbone_name == 'resnet34':
        backbone = ResNet(output_dim = projection_dim, inchan = inchan, num_block = [3,4,6,3], resprestride=1, res1ststride = 2, res2ndstride = 2)
        cam_size = int(img_size / 32)
    else:
        valid_backbone = backbone_name
        raise Exception(f'Backbone \"{valid_backbone}\" is not defined.')
    
    return backbone, backbone.output_dim, cam_size    

class BaselineNet(nn.Module):
    def __init__(self, args):
        super(BaselineNet, self).__init__()
        backbone, feature_dim, _ = build_backbone(img_size=args['img_size'],
                                                  backbone_name=args['backbone'], 
                                                  pretrained=args['pretrained'], 
                                                  projection_dim=-1, 
                                                  inchan=3)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = conv1x1(feature_dim, args['num_known'])

    def forward(self, x, y=None):
        x = self.backbone(x)
        ft = self.classifier(x)
        logits = self.pool(ft)
        logits = logits.view(logits.size(0), -1)
        outputs = {'logits':[logits]}        
        return outputs

    def get_params(self, prefix = 'extractor'):
        extractor_params = list(self.backbone.parameters())
        extractor_params_ids = list(map(id, self.backbone.parameters()))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())
        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
        
class Classifier(nn.Module):
    def __init__(self, in_dim, out_dim, bias=False):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, input):
        logit = self.fc(input)
        if logit.dim() == 1:
            logit =logit.unsqueeze(0)
        return logit

class SEAttention(nn.Module):
    # 初始化SE模块，channel为通道数，reduction为降维比率
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化层，将特征图的空间维度压缩为1x1
        self.fc = nn.Sequential(  # 定义两个全连接层作为激励操作，通过降维和升维调整通道重要性
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数数量和计算量
            nn.ReLU(inplace=True),  # ReLU激活函数，引入非线性
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复到原始通道数
            nn.Sigmoid()  # Sigmoid激活函数，输出每个通道的重要性系数
        )

    # 权重初始化方法
    def init_weights(self):
        for m in self.modules():  # 遍历模块中的所有子模块
            if isinstance(m, nn.Conv2d):  # 对于卷积层
                init.kaiming_normal_(m.weight, mode='fan_out')  # 使用Kaiming初始化方法初始化权重
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 如果有偏置项，则初始化为0
            elif isinstance(m, nn.BatchNorm2d):  # 对于批归一化层
                init.constant_(m.weight, 1)  # 权重初始化为1
                init.constant_(m.bias, 0)  # 偏置初始化为0
            elif isinstance(m, nn.Linear):  # 对于全连接层
                init.normal_(m.weight, std=0.001)  # 权重使用正态分布初始化
                if m.bias is not None:
                    init.constant_(m.bias, 0)  # 偏置初始化为0

    # 前向传播方法
    def forward(self, x):
        b, c, _, _ = x.size()  # 获取输入x的批量大小b和通道数c
        y = self.avg_pool(x).view(b, c)  # 通过自适应平均池化层后，调整形状以匹配全连接层的输入
        y = self.fc(y).view(b, c, 1, 1)  # 通过全连接层计算通道重要性，调整形状以匹配原始特征图的形状
        return x * y.expand_as(x)  # 将通道重要性系数应用到原始特征图上，进行特征重新校准

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 自适应平均池化
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 自适应最大池化

        # 两个卷积层用于从池化后的特征中学习注意力权重
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)  # 第一个卷积层，降维
        self.relu1 = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)  # 第二个卷积层，升维
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))  # 对平均池化的特征进行处理
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))  # 对最大池化的特征进行处理
        out = avg_out + max_out  # 将两种池化的特征加权和作为输出
        return self.sigmoid(out)  # 使用sigmoid激活函数计算注意力权重

# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'  # 核心大小只能是3或7
        padding = 3 if kernel_size == 7 else 1  # 根据核心大小设置填充

        # 卷积层用于从连接的平均池化和最大池化特征图中学习空间注意力权重
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # Sigmoid函数生成最终的注意力权重

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 对输入特征图执行平均池化
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征图执行最大池化
        x = torch.cat([avg_out, max_out], dim=1)  # 将两种池化的特征图连接起来
        x = self.conv1(x)  # 通过卷积层处理连接后的特征图
        return self.sigmoid(x)  # 使用sigmoid激活函数计算注意力权重

# CBAM模块
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)  # 通道注意力实例
        self.sa = SpatialAttention(kernel_size)  # 空间注意力实例

    def forward(self, x):
        out = x * self.ca(x)  # 使用通道注意力加权输入特征图
        result = out * self.sa(out)  # 使用空间注意力进一步加权特征图
        return result  # 返回最终的特征图


class MultiBranchNet(nn.Module):
    def __init__(self, args=None):
        super(MultiBranchNet, self).__init__()
        backbone, feature_dim, self.cam_size = build_backbone(img_size=args['img_size'],
                                                              backbone_name=args['backbone'], 
                                                              projection_dim=-1, 
                                                              inchan=3)
        self.img_size  = args['img_size']
        self.gate_temp = args['gate_temp']
        self.num_known = args['num_known']
        self.avg_pool  = nn.AdaptiveAvgPool2d(1)        
        self.shared_l3 = nn.Sequential(*list(backbone.children())[:-6])
        
        self.branch1_l4  = nn.Sequential(*list(backbone.children())[-6:-3])
        self.branch1_l5  = nn.Sequential(*list(backbone.children())[-3])
        self.branch1_cls = conv1x1(feature_dim, self.num_known)

        self.branch2_l4  = copy.deepcopy(self.branch1_l4)
        self.branch2_l5  = copy.deepcopy(self.branch1_l5)
        self.branch2_cls = conv1x1(feature_dim, self.num_known)
        
        self.branch3_l4  = copy.deepcopy(self.branch1_l4)
        self.branch3_l5  = copy.deepcopy(self.branch1_l5)
        self.branch3_cls = conv1x1(feature_dim, self.num_known)
        
        self.gate_l3  = copy.deepcopy(self.shared_l3)
        self.gate_l4  = copy.deepcopy(self.branch1_l4)
        self.gate_l5  = copy.deepcopy(self.branch1_l5)
        self.gate_cls = nn.Sequential(Classifier(feature_dim, int(feature_dim/4), bias=True), Classifier(int(feature_dim/4), 3, bias=True))
        self.senet = SEAttention(channel = 3)
        self.cbam = CBAM(64)
            
    def forward(self, x, y=None, return_ft=False):
        # x = self.senet(x)
        # print(x.shape)

        b = x.size(0)

        ft_till_l3 = self.shared_l3(x)
        ft_till_l3 = self.cbam(ft_till_l3)
        # print(ft_till_l3.shape)


        branch1_l4 = self.branch1_l4(ft_till_l3.clone())
        branch1_l5 = self.branch1_l5(branch1_l4)
        b1_ft_cams = self.branch1_cls(branch1_l5)
        b1_logits  = self.avg_pool(b1_ft_cams).view(b, -1)
        
        branch2_l4 = self.branch2_l4(ft_till_l3.clone())
        branch2_l5 = self.branch2_l5(branch2_l4)
        b2_ft_cams = self.branch2_cls(branch2_l5)
        b2_logits  = self.avg_pool(b2_ft_cams).view(b, -1)
        
        branch3_l4 = self.branch3_l4(ft_till_l3.clone())
        branch3_l5 = self.branch3_l5(branch3_l4)
        b3_ft_cams = self.branch3_cls(branch3_l5)
        b3_logits  = self.avg_pool(b3_ft_cams).view(b, -1)
        
        if y is not None:
            y = y.long()
            # y = y.to(torch.int64) if not y.dtype == torch.int64 else y
            cams = torch.cat([
                b1_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b1_ft_cams.shape[-2], b1_ft_cams.shape[-1])),
                b2_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b2_ft_cams.shape[-2], b2_ft_cams.shape[-1])),
                b3_ft_cams.gather(dim=1, index=y[:,None,None,None].repeat(1, 1, b3_ft_cams.shape[-2], b3_ft_cams.shape[-1])),
            ], dim = 1)
        
        if return_ft:
            fts = b1_ft_cams.detach().clone() + b2_ft_cams.detach().clone() + b3_ft_cams.detach().clone()
    
        gate_l5   = self.gate_l5(self.gate_l4(self.gate_l3(x)))
        gate_pool = self.avg_pool(gate_l5).view(b, -1)
        gate_pred = F.softmax(self.gate_cls(gate_pool)/self.gate_temp, dim=1)

        gate_logits = torch.stack([b1_logits.detach(), b2_logits.detach(), b3_logits.detach()], dim=-1)
        gate_logits = gate_logits * gate_pred.view(gate_pred.size(0), 1, gate_pred.size(1))
        gate_logits = gate_logits.sum(-1)

        logits_list = [b1_logits, b2_logits, b3_logits, gate_logits]
        if return_ft and y is None:
            outputs = {'logits':logits_list, 'gate_pred': gate_pred, 'fts': fts}
        else:
            outputs = {'logits':logits_list, 'gate_pred': gate_pred, 'cams': cams}
        
        return outputs


    def get_params(self, prefix='extractor'):
        extractor_params = list(self.shared_l3.parameters()) +\
                           list(self.branch1_l4.parameters()) + list(self.branch1_l5.parameters()) +\
                           list(self.branch2_l4.parameters()) + list(self.branch2_l5.parameters()) +\
                           list(self.branch3_l4.parameters()) + list(self.branch3_l5.parameters()) +\
                           list(self.gate_l3.parameters()) + list(self.gate_l4.parameters()) + list(self.gate_l5.parameters())
        extractor_params_ids = list(map(id, extractor_params))
        classifier_params = filter(lambda p: id(p) not in extractor_params_ids, self.parameters())

        if prefix in ['extractor', 'extract']:
            return extractor_params
        elif prefix in ['classifier']:
            return classifier_params
        