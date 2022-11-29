#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50

        self.seed = 0
        # self.head_type = 'org'
        self.head_type = 'points_branch_2'
        self.var_config = 'star'

        self.get_face_pionts = 6
        self.points_loss_weight = 0.1
        self.label_th = 0.9,
        self.ada_pow = 0,
        self.points_loss = 'Wing',

        # yolox_s_mask_org
        # self.exp_name = 'yolox_s_mask_org'
        self.exp_name = f'{os.path.split(os.path.realpath(__file__))[1].split(".")[0]}_{self.head_type}' \
                        f'_{self.get_face_pionts}points_{self.points_loss_weight}_strongaug_greater0.9'
                        # f'_{self.var_config}'
                        # f'test'
        # self.arc_config = {'arc': False, 's': 5, 'm': 0.25}


        self.val_batch_size = 1


        # Define yourself dataset path
        self.name = ''
        self.train_img_dir = r'F:\datasets\Diverse_Masked_Faces_v2_m\new_img'
        self.val_img_dir = r'F:\datasets\Diverse_Masked_Faces_v2_m\new_img'

        # self.train_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m/ann/train_v3.json"
        # self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m/ann/val_v3.json"

        self.train_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m/ann/train_v3_11points.json"
        # self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m/ann/val_v3_11points.json"
        self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m/ann/val_v4_11points.json"

        # self.train_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m\ann/train_v3_11points_small.json"
        # self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m\ann/val_v3_11points_small.json"
        # self.data_dir = "F:\datasets\Diverse_Masked_Faces"
        #
        # self.train_ann = "train.json"
        # # self.train_ann = "val.json"
        # self.name = 'all_img'
        # self.val_ann = "val.json"
        self.input_size = (416, 416)

        self.random_size = (10, 20)
        self.test_size = self.input_size
        self.cls_names = ('face', 'face_mask', 'nose_out', 'mouth_out', 'others', 'spoof')
        self.num_classes = len(self.cls_names)


        # self.l
        self.basic_lr_per_img = 0.00015625
        # self.basic_lr_per_img=0.00015625/10

        self.aug_epochs = 100
        self.max_epoch = 120
        self.no_aug_epochs = self.max_epoch - self.aug_epochs
        # self.min_lr_epochs = self.max_epoch - 80
        assert self.no_aug_epochs == self.max_epoch - self.aug_epochs
        self.data_num_workers = 3
        self.print_interval = 50
        self.eval_interval = 1

        # self.ckpt = r'E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\yolox_s_mask'