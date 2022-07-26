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
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.arc = True


        # Define yourself dataset path
        self.name = ''
        self.train_img_dir = r"F:\datasets\Mask_detection\train\pic"
        # self.train_ann = "face/train.json"
        self.train_ann = r"F:\datasets\Mask_detection\train.json"
        self.val_img_dir = r"F:\datasets\Mask_detection\val\pic"
        self.val_ann = r"F:\datasets\Mask_detection\val.json"


        # self.data_dir = "F:\datasets\Mask_detection"
        #
        # self.train_ann = "train.json"
        # # self.train_ann = "val.json"
        # self.name = 'all_img'
        # self.val_ann = "val.json"

        # self.mixup_prob = 0
        # self.mosaic_prob = 0
        # self.enable_mixup = False


        self.input_size = (416, 416)
        # self.multiscale_range = 5
        self.random_size = (10, 20)
        self.test_size = self.input_size
        self.num_classes = 5
        self.cls_names = ('face', 'face_mask', 'nose_out', 'mouth_out', 'others')

        # self.l
        self.basic_lr_per_img=0.00015625
        # self.basic_lr_per_img=0.00015625/10

        # self.no_aug_epochs = 15
        self.max_epoch = 55
        self.data_num_workers = 2
        self.print_interval = 50
        self.eval_interval = 1

        # self.ckpt = r'E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\yolox_s_mask'