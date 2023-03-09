#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os, socket
from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.seed = 0
        self.input_size = (512, 512)

        '''backone'''
        # self.backbone = 'yoloxpan'
        self.backbone = 'TSCODE'
        # self.spp_size = (3, 5, 7)
        # self.spp_size = (3, 7, 11)
        '''head'''
        self.reg_iou = 0
        # self.Assigner = 'TAL'
        self.Assigner = 'SimOTA'
        # self.head_type = 'var'
        # self.head_type = 'org'
        # self.head_type = 'points_branch_3'
        # self.var_config = 'star_early'
        # self.var_config = '8points_early'
        self.var_config = [None, None]
        # self.var_config = ['star_8points', 'last']
        self.vari_dconv_mask = False
        self.get_face_pionts = 0
        if self.get_face_pionts == 0: self.head_type = 'org'
        if self.backbone =='TSCODE': self.head_type = self.backbone
        # assert self.var_config in ['star', 'star_inter', None, '8points']

        '''loss'''
        self.cls_loss_weight = 1
        self.points_loss_weight = 0.05
        self.label_th = 0.9
        self.ada_pow = 0
        self.points_loss = 'Wing'

        '''lr, aug'''
        self.degrees = 0.
        self.aug_epochs = 100
        self.max_epoch = 120
        self.no_aug_epochs = self.max_epoch - self.aug_epochs
        self.min_lr_epochs = self.no_aug_epochs
        assert self.no_aug_epochs == self.max_epoch - self.aug_epochs

        self.exp_name = f'2{os.path.split(os.path.realpath(__file__))[1].split(".")[0]}_{self.head_type}_{self.var_config[0]}_{self.var_config[1]}'
        # if self.backbone!='yoloxpan':self.exp_name = self.backbone+'_'+self.exp_name
        if self.input_size[0] != 416: self.exp_name += f'_{self.input_size[0]}'
        if self.Assigner!='SimOTA': self.exp_name += f'_{self.Assigner}'
        if self.vari_dconv_mask: self.exp_name += f'_mask'
        self.exp_name += f'_{self.get_face_pionts}points_{self.aug_epochs}straug' \
                         f'_{self.max_epoch - self.min_lr_epochs}coslr'
        if self.get_face_pionts > 0:
            self.exp_name += f'_{self.points_loss_weight}_greater0.9_{self.points_loss}'

        if self.degrees != 10.:
            self.exp_name += f'_{self.degrees}deg' \
            # self.exp_name += f'_{self.degrees}nor0.3deg' \
        # self.arc_config = {'arc': False, 's': 5, 'm': 0.25}
        if self.spp_size != (5, 9, 13):
            self.exp_name += f'_spp{self.spp_size[0]}_{self.spp_size[1]}_{self.spp_size[2]}'
        if not self.reg_iou:
            self.exp_name += f'_IACS'
        if self.cls_loss_weight != 1:
            self.exp_name += f'_cls_loss{self.cls_loss_weight}'

        self.val_batch_size = 1


        # Define yourself dataset path
        self.name = ''
        print(f'{"-"*10} {socket.gethostname()} {"-"*10}')
        if socket.gethostname() == 'DESKTOP-OMJJ23Q':
            path_root = r'D:\liwenlong\Diverse_Masked_Faces_v2_m/'
        else:
            path_root = r'F:\datasets\Diverse_Masked_Faces_v2_m/'
        self.train_img_dir = path_root + 'new_img'
        self.val_img_dir =path_root + 'new_img'

        # self.train_ann =path_root + 'ann/train_v3.json'
        # self.val_ann =path_root + 'ann/val_v3.json'

        # self.train_ann = path_root + 'ann/val_v4_11points.json'
        self.train_ann =path_root + 'ann/train_v3_11points.json'
        self.val_ann = path_root + 'ann/val_v4_11points.json'
        # self.val_ann =path_root + 'ann/val_v3_11points.json'

        # self.train_ann = path_root + "ann/train_v3_small.json"
        # self.val_ann = path_root + "ann/val_v3_small.json"

        if self.input_size[0] != 416:
            self.multiscale_range = 5
        else:
            self.random_size = (10, 20)
        self.test_size = self.input_size
        self.cls_names = ('face', 'face_mask', 'nose_out', 'mouth_out', 'others', 'spoof')
        self.num_classes = len(self.cls_names)


        # self.l
        self.basic_lr_per_img = 0.00015625
        # self.basic_lr_per_img=0.00015625/10


        self.data_num_workers = 2
        self.print_interval = 100
        self.eval_interval = 5



