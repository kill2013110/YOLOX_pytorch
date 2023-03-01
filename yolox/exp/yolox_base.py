#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn

from .base_exp import BaseExp


class Exp(BaseExp):
    def __init__(self):
        super().__init__()

        # ---------------- model config ---------------- #
        # detect classes number of model
        self.num_classes = 80
        # factor of model depth
        self.depth = 1.00
        # factor of model width
        self.width = 1.00
        # activation name. For example, if using "relu", then "silu" will be replaced to "relu".
        self.act = "silu"

        # ---------------- dataloader config ---------------- #
        # set worker to 4 for shorter dataloader init time
        # If your training process cost many memory, reduce this value.
        self.data_num_workers = 0
        self.input_size = (640, 640)  # (height, width)
        # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32].
        # To disable multiscale training, set the value to 0.
        self.multiscale_range = 5
        # You can uncomment this line to specify a multiscale range
        # self.random_size = (14, 26)
        # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.train_img_dir = None
        self.val_img_dir = None
        # name of annotation file for training
        self.train_ann = "instances_train2017.json"
        # name of annotation file for evaluation
        self.val_ann = "instances_val2017.json"
        # name of annotation file for testing
        self.test_ann = "instances_test2017.json"
        # --------------- transform config ----------------- #
        # prob of applying mosaic aug
        self.mosaic_prob = 1.0
        # prob of applying mixup aug
        self.mixup_prob = 1.0
        # prob of applying hsv aug
        self.hsv_prob = 1.0
        # prob of applying flip aug
        self.flip_prob = 0.5
        # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.degrees = 10.0
        # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
        # apply mixup aug or not
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
        # shear angle range, for example, if set to 2, the true range is (-2, 2)
        self.shear = 2.0

        # --------------  training config --------------------- #
        # epoch number used for warmup
        self.warmup_epochs = 5
        # max training epoch
        self.max_epoch = 300
        # minimum learning rate during warmup
        self.warmup_lr = 0
        self.min_lr_ratio = 0.05
        # learning rate for one image. During training, lr will multiply batchsize.
        self.basic_lr_per_img = 0.01 / 64.0
        # name of LRScheduler
        self.scheduler = "yoloxwarmcos"
        # last #epoch to close augmention like mosaic
        self.no_aug_epochs = 15
        self.min_lr_epochs = self.no_aug_epochs  # 默认情况下两者应该相等，但在加入关键点数据训练时不是这样
        # apply EMA during training
        self.ema = True

        # weight decay of optimizer
        self.weight_decay = 5e-4
        # momentum of optimizer
        self.momentum = 0.9
        # log period in iter, for example,
        # if set to 1, user could see log every iteration.
        self.print_interval = 10
        # eval period in epoch, for example,
        # if set to 1, model will be evaluate after every epoch.
        self.eval_interval = 10
        # save history checkpoint or not.
        # If set to False, yolox will only save latest and best ckpt.
        self.save_history_ckpt = True
        # name of experiment
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # -----------------  testing config ------------------ #
        # output image size during evaluation/test
        self.test_size = (640, 640)
        # confidence threshold during evaluation/test,
        # boxes whose scores are less than test_conf will be filtered
        self.test_conf = 0.01
        # nms threshold
        self.nmsthre = 0.65
        self.box_loss_weight = 5
        self.cls_loss_weight = 1
        self.reg_iou = True
        self.val_batch_size = None
        self.get_face_pionts = False
        self.label_th = 0.9
        self.ada_pow = 0
        # self.cls_loss = 'VF'
        self.points_loss = 'Wing'
        self.points_loss_weight = 0.
        self.head_type = 'org'
        self.arc_config = {'arc': None, 's': None, 'm': None}
        self.var_config = ''
        self.vari_dconv_mask = False
        self.spp_size = (5, 9, 13)
        self.Assigner = 'SimOTA'
    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead, \
            YOLOXHead_points_branch_3_dconv,\
            YOLOXHead_points_branch_1_dconv
            # YOLOXHead_points_branch_2, YOLOXHead_points_branch_3

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act, spp_size=self.spp_size)
            if self.head_type == 'org':
                head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                                get_face_pionts=self.get_face_pionts, points_loss_weight=self.points_loss_weight,
                                points_loss=self.points_loss, ada_pow=self.ada_pow, label_th=self.label_th,
                                var_config=self.var_config,
                                reg_iou=self.reg_iou, box_loss_weight=self.box_loss_weight,
                                cls_loss_weight=self.cls_loss_weight,
                                vari_dconv_mask=self.vari_dconv_mask,
                                 Assigner=self.Assigner,
                                 )
            # elif self.head_type == 'arc':
            #     head = YOLOXHeadArc(self.num_classes, self.width, in_channels=in_channels, act=self.act, get_face_pionts=self.get_face_pionts,
            #                         arc_config=self.arc_config)
            # elif self.head_type == 'var':
            #     head = YOLOXHeadVar(self.num_classes, self.width, in_channels=in_channels, act=self.act, get_face_pionts=self.get_face_pionts,
            #                         var_config=self.var_config)
            elif self.head_type == 'points_branch_1':
                head = YOLOXHead_points_branch_1_dconv(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                        get_face_pionts=self.get_face_pionts,
                        points_loss_weight=self.points_loss_weight,
                        points_loss=self.points_loss, ada_pow=self.ada_pow,
                        label_th=self.label_th, var_config=self.var_config,
                        reg_iou=self.reg_iou, box_loss_weight=self.box_loss_weight,
                        cls_loss_weight=self.cls_loss_weight,
                        vari_dconv_mask = self.vari_dconv_mask,
                                                 )
                # head = YOLOXHead_points_branch_1(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                #                                  get_face_pionts=self.get_face_pionts,points_loss_weight=self.points_loss_weight,
                #                                  points_loss=self.points_loss, ada_pow=self.ada_pow, label_th=self.label_th,
                #                                  )
            elif self.head_type == 'points_branch_2':
                head = YOLOXHead_points_branch_2(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                                                 get_face_pionts=self.get_face_pionts,points_loss_weight=self.points_loss_weight,
                                                 points_loss=self.points_loss, ada_pow=self.ada_pow, label_th=self.label_th,
                                                 )
            elif self.head_type == 'points_branch_3':
                head = YOLOXHead_points_branch_3_dconv(self.num_classes, self.width, in_channels=in_channels, act=self.act,
                        get_face_pionts=self.get_face_pionts,
                        points_loss_weight=self.points_loss_weight,
                        points_loss=self.points_loss, ada_pow=self.ada_pow,
                        label_th=self.label_th, var_config=self.var_config,
                        reg_iou=self.reg_iou, box_loss_weight=self.box_loss_weight,
                        cls_loss_weight=self.cls_loss_weight,
                        vari_dconv_mask=self.vari_dconv_mask,
                       Assigner=self.Assigner,
                                                       )
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)

        self.model.head.initialize_biases(1e-2)
        self.model.train()
        return self.model

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1:5:2] = targets[..., 1:5:2] * scale_x
            targets[..., 2:5:2] = targets[..., 2:5:2] * scale_y
            if self.get_face_pionts:  # outputs: [cls, xc,yc,w,h, [x,y]*6 ]
                targets[..., 5::3] = targets[..., 5::3] * scale_x
                targets[..., 6::3] = targets[..., 6::3] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2, pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                # name_l.append(k)
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay
                # else:
                #     if 'head'in k:
                #         print('fdsf')
                #     # print(k, v)

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            if self.vari_dconv_mask:
                pg3.append(self.model.head.dconv_mask)
                optimizer.add_param_group({"params": pg3, "lr": lr*10.})
            # if self.arc:
            #     # for v in self.model.head.cls_w:
            #     #     arc_w.append(v)
            #     # head.cls_w.0
            #     # head.cls_w.1
            #     # head.cls_w.2
            #     optimizer.add_param_group(
            #         {"params": self.model.head.cls_w, "weight_decay": self.weight_decay}
            #     )  # add arc_w with weight_decay

            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            # no_aug_epochs=self.no_aug_epochs,
            no_aug_epochs=self.min_lr_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_data_loader(
        self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import (
            COCODataset,
            TrainTransform,
            YoloBatchSampler,
            DataLoader,
            InfiniteSampler,
            MosaicDetection,
            worker_init_reset_seed,
        )
        from yolox.utils import (
            wait_for_the_master,
            get_local_rank,
        )

        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = COCODataset(
                get_face_pionts=self.get_face_pionts,
                img_dir=self.train_img_dir,
                name=self.name,
                json_file=self.train_ann,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=50,
                    get_face_pionts=self.get_face_pionts,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                cache=cache_img,
            )

        dataset = MosaicDetection(
            dataset,
            mosaic=not no_aug,
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=120,
                get_face_pionts=self.get_face_pionts,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob),
            degrees=self.degrees,
            translate=self.translate,
            mosaic_scale=self.mosaic_scale,
            mixup_scale=self.mixup_scale,
            shear=self.shear,
            enable_mixup=self.enable_mixup,
            mosaic_prob=self.mosaic_prob,
            mixup_prob=self.mixup_prob,
        )

        self.dataset = dataset

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0)

        batch_sampler = YoloBatchSampler(
            sampler=sampler,
            batch_size=batch_size,
            drop_last=False,
            mosaic=not no_aug,
        )

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method.
        # Check https://github.com/pytorch/pytorch/issues/63311 for more details.
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)

        return train_loader

    def get_eval_loader(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.data import COCODataset, ValTransform

        valdataset = COCODataset(
            get_face_pionts=self.get_face_pionts,
            img_dir=self.val_img_dir,
            json_file=self.val_ann if not testdev else self.test_ann,
            name=self.name if self.name!=None else "val2017" if not testdev else "test2017",
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
        )

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(
                valdataset, shuffle=False
            )
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {
            "num_workers": self.data_num_workers,
            "pin_memory": True,
            "sampler": sampler,
        }
        if self.val_batch_size!=None:
            batch_size = self.val_batch_size
        dataloader_kwargs["batch_size"] = batch_size
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)

        return val_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import COCOEvaluator

        val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
        evaluator = COCOEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            testdev=testdev,
            get_face_pionts=self.get_face_pionts
        )
        return evaluator

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
