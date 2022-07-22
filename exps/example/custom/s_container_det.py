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



        # Define yourself dataset path
        self.name = ''
        self.train_img_dir = r"E:\ocr\container_ocr\det_dataset\v3_img"
        # self.train_ann = "face/train.json"
        self.train_ann = r"E:\ocr\container_ocr\det_dataset\v3_train.json"
        self.val_img_dir = r"E:\ocr\container_ocr\det_dataset\v3_img"
        self.val_ann = r"E:\ocr\container_ocr\det_dataset\v3_val.json"
        self.input_size = (416, 416)
        self.random_size = (10, 20)
        self.test_size = self.input_size
        self.cls_names = ('a')
        self.num_classes = 1

        # self.l
        self.basic_lr_per_img=0.00015625
        # self.basic_lr_per_img=0.00015625/10

        # self.no_aug_epochs = 15
        self.max_epoch = 55
        self.data_num_workers = 2
        self.print_interval = 100
        self.eval_interval = 1

        # self.ckpt = r'E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\yolox_s_mask'