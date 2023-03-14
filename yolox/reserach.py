import torch, copy
from thop import profile
import torchvision

net = torchvision.models.efficientnet_v2_s