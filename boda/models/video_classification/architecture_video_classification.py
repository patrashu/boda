import os
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor
from torch.autograd import Variable 
from torchinfo import summary
from typing import Tuple, List, Dict, Any, Union, Optional, Callable, Type

# from configuration_video_classification import VideoClassificationConfig
# from ...base_architecture import Model

PRETRAINED_ARCHIVES = {
    'resnet50': 'https://drive.google.com/file/d/18KktApSEhzTeaLfitWRyAaJkxGstrrp2/view?usp=sharing',
    'resnet101': 'https://drive.google.com/file/d/1Y6Dq1kIbaGPzC3Lu_wktYaWeHuhnAl4z/view?usp=sharing'
}


class Conv1x1x1(nn.Sequential):
    """1x1x1 convolution"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Tuple[int] = (1, 1, 1),
        stride: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, bias=False)
        )


class Conv3x3x3(nn.Sequential):
    """3x3x3 convolution with padding"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Tuple[int] = (3, 3, 3),
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, bias=False
            )
        )


class DepthWiseConv3x3x3(nn.Sequential):
    """3x3x3 depthwise convolution """
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: Tuple[int] = (3, 3, 3),
        stride: int = 1,
        padding: int = 1,
        groups: int = None,
    ) -> None:
        super().__init__(
            nn.Conv3d(
                in_planes, out_planes, kernel_size=kernel_size, 
                stride=stride, padding=padding, groups=groups, bias=False
            )
        )


class BasicStem(nn.Sequential):
    """Stem stage consist of conv-batchnorm-relu layer"""

    def __init__(self) -> None:
        super().__init__(
            nn.Conv3d(
                3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)
            ),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )


## 3D-ResNet
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        in_planes: int, 
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            Conv3x3x3(in_planes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Conv3x3x3(planes, planes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Sequential(
            Conv1x1x1(in_planes, planes),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            Conv3x3x3(planes, planes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)          
        )
        self.conv3 = nn.Sequential(
            Conv1x1x1(planes, planes*self.expansion),
            nn.BatchNorm3d(planes*self.expansion),
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class VideoResNet(nn.Module):
    """Generic video resnet generator

    Args:
        block (Type[Union[BasicBlock, Bottleneck]]): resnet building block
        layers (List[int]): number of blocks per stage
        stem (Callable[..., nn.Module]): ResNet stem
        num_classes (int, Optional): Dimension of the final FC layer.
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int] = None,
        stem: Callable[..., nn.Module] = BasicStem,
        num_classes: int = 400,
    ) -> None:

        self.inplanes = 64
        super(VideoResNet, self).__init__()

        self.stem = stem()

        self.stage1 = self._make_stage(block, 64, layers[0])
        self.stage2 = self._make_stage(block, 128, layers[1], stride=2)
        self.stage3 = self._make_stage(block, 256, layers[2], stride=2)
        self.stage4 = self._make_stage(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        self._init_weights()

    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.001)
                nn.init.constant_(m.bias, 0)


    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
        
    def _make_stage(
        self, 
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv1x1x1(
                    self.inplanes,
                    planes * block.expansion,
                    stride=stride,
                ),
                nn.BatchNorm3d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


class InvertedResidual(nn.Module):
    def __init__(
        self,
        in_planes: int,
        planes: int,
        stride: int,
        expand_ratio: int = 1,
    ) -> None:
        super(InvertedResidual, self).__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio

        hidden_dim = round(in_planes * expand_ratio)
        self.skip_connection = self.stride == (1,1,1) and in_planes == planes

        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                # Depthwise conv
                DepthWiseConv3x3x3(
                    hidden_dim, hidden_dim, kernel_size=(3, 3, 3), stride=stride,
                    padding=1, groups=hidden_dim
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),

                # Pointwise- linear conv
                nn.Conv3d(
                    hidden_dim, planes, kernel_size=1, stride=1,
                    padding=0, bias=False
                ),
                nn.BatchNorm3d(planes)
            )
        else:
            self.conv = nn.Sequential(
                # Pointwise conv
                Conv1x1x1(in_planes, hidden_dim),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),

                # Deptwise conv
                DepthWiseConv3x3x3(
                    hidden_dim, hidden_dim, kernel_size=(3, 3, 3), stride=stride,
                    padding=1, groups=hidden_dim
                ),
                nn.BatchNorm3d(hidden_dim),
                nn.ReLU6(inplace=True),

                # Pointwise-linear conv
                Conv1x1x1(hidden_dim, planes),
                nn.BatchNorm3d(planes)
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.skip_connection:
            return x + self.conv(x)
        else:
            return self.conv(x)

    
class VideoMobileNetV2(nn.Module):
    def __init__(
        self,
        block: Type[InvertedResidual],
        num_classes: int = 400,
        sample_size: int = 112,
        width_mult: float = 1.,
    ) -> None:
        super(VideoMobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        inverted_residual_setting = [
            # t, c, n, s
            [1,  16, 1, (1,1,1)],
            [6,  24, 2, (2,2,2)],
            [6,  32, 3, (2,2,2)],
            [6,  64, 4, (2,2,2)],
            [6,  96, 3, (1,1,1)],
            [6, 160, 3, (2,2,2)],
            [6, 320, 1, (1,1,1)],
        ]

        assert sample_size % 16 == 0
        input_channel = int(input_channel*width_mult)
        self.last_channel = int(last_channel*width_mult) if width_mult > 1.0 else last_channel
        self.features = [
            nn.Sequential(
                Conv3x3x3(3, input_channel, kernel_size=(1, 2, 2)),
                nn.BatchNorm3d(input_channel),
                nn.ReLU6(inplace=True)
            )
        ]
        
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c*width_mult)
            for i in range(n):
                stride = s if i == 0 else (1, 1, 1)
                self.features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel

        self.features.append(
            nn.Sequential(
                Conv1x1x1(input_channel, self.last_channel),
                nn.BatchNorm3d(self.last_channel),
                nn.ReLU6(inplace=True)
            )
        )
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = F.avg_pool3d(out, out.data.size()[-3:])
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

# class VideoClassificationModel(Model):
#     """
#     Args:
#         config (:class:`VideoClassificationConfig`):
#         backbone (:resnet50):
#         num_classes (:obj:`int`):


#     Examples::
#         >>> from boda import VideoClassificationConfig, VideoClassificationModel

#         >>> config = VideoClassificationConfig()
#         >>> model = VideoClassificationModel(config)
#     """

#     model_name = 'resnet50'

#     def __init__(
#         self,
#         config: VideoClassificationConfig,
#         num_classes: int = 400,
#         model_name: str = None,
#         pretrained_weights: os.PathLike = None,
#     ) -> None:
#         super().__init__(config)
#         self.num_classes = num_classes
#         self.update_config(config)


#         if model_name == 'resnet50':
#             model = resnet3d_50(weight=False).cuda()
#             print(model)
#             print(summary(model, input_size=(1, 3, 16, 224, 224), depth=1))
#             return

#         elif model_name == 'resnet101':
#             model = resnet3d_101(weight=False)
        
#         # elif model_name == 'mobilenetv2':
#         #     model = 

    
def resnet3d_50(weight = False, **kwargs):
    """Constructs a ResNet-50 model"""
    # if not weight:
    #     url = PRETRAINED_ARCHIVES('resnet50')
    #     transforms=partial(VideoClassificationModel)

    model = VideoResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    print(model)
    print(summary(model, input_size=(1, 3, 16, 224, 224), depth=1))
    # if weight is not None:
    #     model.load_state_dict(weight.get_state_dict(progress=True))

    return model

def resnet3d_101(**kwargs):
    """Constructs a ResNet-101 model"""

    model = VideoResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    print(model)
    print(summary(model, input_size=(1, 3, 16, 224, 224), depth=1))
    return model

def mobilenetv2_3d(**kwargs):
    model = VideoMobileNetV2(block=InvertedResidual, num_classes=600)
    print(summary(model, input_size=(1, 3, 16, 224, 224), depth=1))
    input_var = Variable(torch.randn(8, 3, 16, 112, 112))
    output = model(input_var)
    print(output.shape)
    return model

if __name__ == '__main__':
    # VideoClassificationModel(model_name='resnet50')
    mobilenetv2_3d()
    