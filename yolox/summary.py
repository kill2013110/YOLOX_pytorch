
import torch, copy
from thop import profile
from models.yolo_fpn_TSCODE import YOLO_fpn_TSCODE
from models.yolo_pafpn import YOLOPAFPN
from models.yolo_head_TSCODE import YOLOXHead_TSCODE
from models.yolo_head import YOLOXHead
from models.yolox import YOLOX
from models.darknet import CSPDarknet
depth = 0.33
width = 0.5

backbone = YOLO_fpn_TSCODE(depth=depth, width=width)
'''YOLO_fpn_TSCODE
backbone flops:8.4909312 f
backbone params: 5.13504 M'''
yolofpn = YOLOPAFPN(depth=depth, width=width)
backbone.training = False
'''YOLOPAFPN
backbone flops:8.162304 f
backbone params: 7.04736 M'''

img = torch.zeros((1, 3, 640, 640), device=next(backbone.parameters()).device)
# flops, params = profile(copy.deepcopy(backbone), inputs=(img,), verbose=False)
# print(f'backbone flops:{flops/1e9} f')
# print(f'backbone params: {params/1e6} M')

head = YOLOXHead_TSCODE(width=width, reg_iou=False, num_classes=6)
yoloxhead = YOLOXHead(width=width, reg_iou=False, num_classes=6)
head.training = False
yoloxhead.training = False

yolox = YOLOX(backbone=yolofpn, head=yoloxhead)
net = YOLOX(backbone=backbone, head=head)
'''YOLO_fpn_TSCODE + YOLO_head_TSCODE
backbone flops:17.5017288 f
backbone params: 10.765076 M'''

'''YOLOPAFPN + YOLOHead
backbone flops:13.3219144 f
backbone params: 8.93923 M'''
yolox.training = False
net.training = False
out = net(img)

flops, params = profile(copy.deepcopy(net), inputs=(img,), verbose=False)
print(f'backbone flops:{flops/1e9} f')
print(f'backbone params: {params/1e6} M')
print()