#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.67
        self.width = 0.75

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.arc_config = {'arc': False, 's': 5, 'm': 0.25}

        self.val_batch_size = 1

        # Define yourself dataset path
        # self.name = ''
        # self.train_img_dir = r"F:\datasets\Diverse_Masked_Faces_v1\train\pic"
        # self.train_ann = r"F:\datasets\Diverse_Masked_Faces_v1\annotations/train.json"
        # self.val_img_dir = r"F:\datasets\Diverse_Masked_Faces_v1\val\pic"
        # self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v1\annotations/val.json"

        self.name = ''
        self.train_img_dir = r'F:\datasets\Diverse_Masked_Faces_v2_m\new_img'
        self.train_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m\ann/train_v2.json"
        self.val_img_dir = r'F:\datasets\Diverse_Masked_Faces_v2_m\new_img'
        self.val_ann = r"F:\datasets\Diverse_Masked_Faces_v2_m\ann/val_v2.json"

        #
        # self.train_ann = "train.json"
        # # self.train_ann = "val.json"
        # self.name = 'all_img'
        # self.val_ann = "val.json"

        # self.mixup_prob = 0
        # self.mosaic_prob = 0
        # self.enable_mixup = False


        self.input_size = (640, 640)
        self.multiscale_range = 5
        # self.random_size = (10, 20)
        self.test_size = self.input_size
        self.cls_names = ('face', 'face_mask', 'nose_out', 'mouth_out', 'others', 'spoof')
        self.num_classes = len(self.cls_names)


        # self.l
        self.basic_lr_per_img=0.00015625
        # self.basic_lr_per_img=0.00015625/10

        self.no_aug_epochs = 15
        self.max_epoch = 45
        self.data_num_workers = 4
        self.print_interval = 50
        self.eval_interval = 1

        # self.ckpt = r'E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\yolox_s_mask'