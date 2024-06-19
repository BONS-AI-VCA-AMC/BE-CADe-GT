"""IMPORT PACKAGES"""
import torch.nn as nn

# Import helper functions from other files
from models.UNet import UNet


"""""" """""" """""" """""" """""" """"""
"""" DEFINE CUSTOM CLS + SEG MODEL"""
"""""" """""" """""" """""" """""" """"""


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()

        # Define Backbone architecture
        if opt.backbone == 'ResNet-50-UNet':
            url = "https://download.pytorch.org/models/resnet50-11ad3fa6.pth"
            self.backbone = UNet(encoder_name='resnet50', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-101-UNet':
            url = "https://download.pytorch.org/models/resnet101-cd907fc2.pth"
            self.backbone = UNet(encoder_name='resnet101', url=url, num_classes=opt.num_classes)
        elif opt.backbone == 'ResNet-152-UNet':
            url = "https://download.pytorch.org/models/resnet152-f82ba261.pth"
            self.backbone = UNet(encoder_name='resnet152', url=url, num_classes=opt.num_classes)
        else:
            raise Exception('Unexpected Backbone {}'.format(opt.backbone))

        # Define segmentation branch architecture
        if opt.seg_branch is None and 'UNet' in opt.backbone:
            self.single_model = True
        else:
            raise Exception('Unexpected Segmentation Branch {}'.format(opt.seg_branch))

    def forward(self, img):
        if self.single_model:
            # Output of single model
            cls, seg = self.backbone(img)

        else:
            # Backbone output
            cls, low_level, high_level = self.backbone(img)

            # Segmentation output
            seg = self.seg_branch(img, low_level, high_level)

        return cls, seg
