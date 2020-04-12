import os, sys
sys.path.append('./')
import torch
import torch.nn as nn
from torchsummary import summary

import config

def conv_bn_relu(inp, oup, kernel_size, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )

class DepthWiseConv(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect=True, expand_ratio=1):
        super(DepthWiseConv, self).__init__()
        self.use_res_connect = use_res_connect
        self.conv = nn.Sequential(
                nn.Conv2d(
                    inp*expand_ratio,
                    inp*expand_ratio,
                    3,
                    stride,
                    1,
                    groups=inp*expand_ratio,
                    bias=False),
                nn.BatchNorm2d(inp*expand_ratio),
                nn.ReLU(inplace=True),
                nn.Conv2d(inp*expand_ratio, oup,  1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, use_res_connect, expand_ratio=6):
        super(InvertedResidual, self).__init__()
        self.stride=stride
        assert stride in [1, 2]
        if stride == 1 and use_res_connect:
            self.use_res_connect = use_res_connect
        else:
            self.use_res_connect = False

        self.conv = nn.Sequential(
            nn.Conv2d(inp, inp*expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp*expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                inp*expand_ratio, 
                inp*expand_ratio,
                3, 
                stride,
                1, 
                groups=inp*expand_ratio,
                bias=False),
            nn.BatchNorm2d(inp*expand_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp*expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class LNet(nn.Module):
    def __init__(self):
        super(LNet, self).__init__()
        self.conv1 = conv_bn_relu(3, 16, 3, 2) 
        self.conv2 = DepthWiseConv(16, 16, 1)
        
        self.block3_1 = InvertedResidual(16, 16, 2, False, 4) 
        self.block3_2 = InvertedResidual(16, 16, 1, True, 4)    
        self.block3_3 = InvertedResidual(16, 16, 1, True, 4)
                
        self.block4_1 = InvertedResidual(16, 32, 2, False, 2)
        self.block4_2 = InvertedResidual(32, 32, 1, True, 4)
        self.block4_3 = InvertedResidual(32, 32, 1, True, 4)
        self.block4_4 = InvertedResidual(32, 32, 1, True, 4)
                
        self.block5_1 = InvertedResidual(32, 64, 2, False, 2)
        self.block5_2 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_3 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_4 = InvertedResidual(64, 64, 1, True, 4)
        self.block5_5 = InvertedResidual(64, 64, 1, False, 2)
        
        self.conv5 = conv_bn_relu(64, 128, 3, 2)
        self.avg_pool1 = nn.AvgPool2d((20, 40))
        self.avg_pool2 = nn.AvgPool2d((10, 20))
        self.avg_pool3 = nn.AvgPool2d((5, 10))
        self.avg_pool4 = nn.AvgPool2d((3, 5))

        self.fc = nn.Linear(240, 64)
        self.prelu = nn.PReLU(64)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        
        x = self.block3_1(x)
        x = self.block3_2(x)
        x1 = self.block3_3(x)
        
        x = self.block4_1(x1)
        x = self.block4_2(x)
        x = self.block4_3(x)
        x2 = self.block4_4(x)
        
        x = self.block5_1(x2)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x3 = self.block5_5(x)

        x4 = self.conv5(x3)
        x4 = self.avg_pool4(x4)
        x4 = x4.view(x4.size(0), -1)

        x1 = self.avg_pool1(x1)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.avg_pool2(x2)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.avg_pool3(x3)
        x3 = x3.view(x3.size(0), -1)
        
        multi_scale = torch.cat([x1, x2, x3, x4], 1)
        x = self.prelu(self.fc(multi_scale)) 
        return x


if __name__ == "__main__":
    lnet = LNet()
    lnet.load_state_dict(torch.load('result/iris_lnet/check_point/{0}_landmarks_model_200.pth'.format(config.NUM_LANDMARKS)))
    summary(lnet.cuda(), (3, 80, 160))
    torch.onnx.export(lnet.cpu(), torch.randn(1, 3, 80, 160), 'convert2ncnn/iris_lnet.onnx',
            input_names=['data'], output_names=['prelu1'])


