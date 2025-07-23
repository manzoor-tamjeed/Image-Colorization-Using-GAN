import torch
import torch.nn as nn
from torchvision import models

class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features[:9]  # relu2_2
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg.to(device)
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).to(device).view(1, 3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).to(device).view(1, 3, 1, 1)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return nn.functional.l1_loss(self.vgg(x), self.vgg(y))