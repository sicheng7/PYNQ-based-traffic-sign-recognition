import torch
import torch.nn as nn
import torchvision.models as models
import torchvision

from collections import OrderedDict

from .binarized_modules import BinaryConv2d

class ModelTrafic_32(nn.Module):
    def __init__(self, num_cls=58):
        super(ModelTrafic_32, self).__init__()
        self.features = nn.Sequential(
            #32-3+1 = 30
            #30 / 2 = 15
            nn.Conv2d(3, 128, kernel_size= 3, stride= 1, padding= 0, bias= False),
            nn.MaxPool2d(2, stride=2, return_indices=False),
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=2, return_indices=False),
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            nn.MaxPool2d(2, stride=1, return_indices=False),
        )

        self.classifier = nn.Sequential(
            BinaryConv2d(128, 128, 3, stride=1, padding=0),
            BinaryConv2d(128, 128, 1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, num_cls, 1, stride= 1, padding= 0, bias=False),
            nn.Softmax(dim=1)
        )

        # index of conv
        self.conv_layer_indices = [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 24]
        # feature maps
        self.feature_maps = OrderedDict()  # 会根据放入元素的先后顺序进行排序
        # switch
        self.pool_locs = OrderedDict()

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
            # print(x.size())
            # print(x)
        # print(x.size()) #***********************************
        # x = x.view(-1, 8*3)
        for layer in self.classifier:
            x = layer(x)
        return x


# if __name__ == '__main__':
#     model = models.vgg8(pretrained=True)
#     print(model)
