#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import datetime
import os
import time
from loguru import logger
import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
import torch, copy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.data import DataPrefetcher
from yolox.utils import (
    MeterBuffer,
    ModelEMA,
    WandbLogger,
    adjust_status,
    all_reduce_norm,
    get_local_rank,
    get_model_info,
    get_rank,
    get_world_size,
    gpu_mem_usage,
    is_parallel,
    load_ckpt,
    occupy_mem,
    save_checkpoint,
    setup_logger,
    synchronize
)


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in
        # before_train methods.
        self.exp = exp
        self.args = args
        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.best_ap = 0
        self.best_epoch = 0
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(
            self.file_name,
            distributed_rank=self.rank,
            filename="train_log.txt",
            mode="a",
        )

    def train(self):
        self.before_train()
        try:
            self.train_in_epoch()
        except Exception:
            raise
        finally:
            self.after_train()

    def train_in_epoch(self):
        for self.epoch in range(self.start_epoch, self.max_epoch):
            self.before_epoch()
            self.train_in_iter()
            self.after_epoch()

    def train_in_iter(self):
        for self.iter in range(self.max_iter):
            self.before_iter()
            self.train_one_iter()
            self.after_iter()

    def train_one_iter(self):
        iter_start_time = time.time()

        inps, targets = self.prefetcher.next()
        inps = inps.to(self.data_type)
        targets = targets.to(self.data_type)
        targets.requires_grad = False
        inps, targets = self.exp.preprocess(inps, targets, self.input_size)
        '''
import cv2
j =1
coordinate = np.int16(targets[j].cpu().numpy()).copy()
points = np.int16(targets[j,:,5:].cpu().numpy()).copy()
x = np.uint8(inps[j].cpu().numpy().transpose(1,2,0)).copy()
for n in range(len(coordinate)):
    # cv2.polylines(x, [coordinate[n]], isClosed=True,
    #               color=[255, 255, 0])
    for i in range(6):
        cv2.circle(x, points[n, i * 3:i * 3 + 2], 2, color=(0, 40 * i, 0))
    cv2.rectangle(x, (coordinate[n][0+1]-int(coordinate[n][2+1]/2), 
                      coordinate[n][1+1]-int(coordinate[n][3+1]/2), \
                      coordinate[n][2+1] , coordinate[n][3+1]),
                  [255, 255, 0])
cv2.imshow('1', x)
cv2.waitKey()
        targets: [cls, cx, cy, w, h, x, y, score...]'''
        data_end_time = time.time()

        with torch.cuda.amp.autocast(enabled=self.amp_training):
            outputs = self.model(inps, targets)
            # import cv2
            # import numpy as np
            # a=inps
            # cv2.imshow('1', np.uint8(a.data.to('cpu').numpy()[7].transpose(1, 2, 0)))
            # cv2.waitKey()

        loss = outputs["total_loss"]

        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        if self.use_model_ema:
            self.ema_model.update(self.model)

        lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

        iter_end_time = time.time()
        self.meter.update(
            iter_time=iter_end_time - iter_start_time,
            data_time=data_end_time - iter_start_time,
            lr=lr,
            **outputs,
        )

    def before_train(self):
        logger.info("args: {}".format(self.args))
        logger.info("exp value:\n{}".format(self.exp))

        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        logger.info(
            "Model Summary: {}".format(get_model_info(model, self.exp.test_size))
        )
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        self.train_loader = self.exp.get_data_loader(
            batch_size=self.args.batch_size,
            is_distributed=self.is_distributed,
            no_aug=self.no_aug,
            cache_img=self.args.cache,
        )
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher = DataPrefetcher(self.train_loader)
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(
            self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter
        )
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model

        self.evaluator = self.exp.get_evaluator(
            batch_size=self.args.batch_size, is_distributed=self.is_distributed
        )
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        logger.info("\n{}".format(model))

    def after_train(self):
        logger.info(
            "Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100)
        )
        logger.info(f"best epoch:{self.best_epoch}, ap:{self.best_ap:4f}\n")
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()

    def before_epoch(self):
        logger.info("---> start train epoch{}".format(self.epoch + 1))
        if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
            logger.info("--->No mosaic aug now!")
            self.train_loader.close_mosaic()
            logger.info("--->Add additional L1 loss now!")
            if self.is_distributed:
                self.model.module.head.use_l1 = True
            else:
                self.model.head.use_l1 = True
            self.exp.eval_interval = 1
            if not self.no_aug:
                self.save_ckpt(ckpt_name="last_mosaic_epoch")

    def after_epoch(self):
        # self.save_ckpt(ckpt_name="latest")

        if (self.epoch + 1) % self.exp.eval_interval == 0:
            all_reduce_norm(self.model)
            self.evaluate_and_save_model()

    def before_iter(self):
        if self.args.logger == "tensorboard":
            self.tblogger.add_scalar("lr", self.optimizer.param_groups[0]['lr'],
                                     self.epoch * self.max_iter + self.iter + 1)
        # pass
    def after_iter(self):
        """
        `after_iter` contains two parts of logic:
            * log information
            * reset setting of resize
        """
        # log needed information
        if (self.iter + 1) % self.exp.print_interval == 0:
            # TODO check ETA logic
            left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
            eta_seconds = self.meter["iter_time"].global_avg * left_iters
            eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

            progress_str = "epoch: {}/{}, iter: {}/{}".format(
                self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter
            )
            loss_meter = self.meter.get_filtered_meter("loss")
            loss_str = ", ".join(
                ["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()]
            )

            time_meter = self.meter.get_filtered_meter("time")
            time_str = ", ".join(
                ["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()]
            )

            logger.info(
                "{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(
                    progress_str,
                    gpu_mem_usage(),
                    time_str,
                    loss_str,
                    self.meter["lr"].latest,
                )
                + (", size: {:d}, {}".format(self.input_size[0], eta_str))
            )

            if self.rank == 0:
                if self.args.logger == "tensorboard":

                    self.tblogger.add_scalar("train/total_loss", self.meter['total_loss'].latest, self.epoch *self.max_iter + self.iter + 1)
                    self.tblogger.add_scalar("train/iou_loss", self.meter['iou_loss'].latest, self.epoch *self.max_iter + self.iter + 1)
                    self.tblogger.add_scalar("train/conf_loss", self.meter['conf_loss'].latest, self.epoch *self.max_iter + self.iter + 1)
                    self.tblogger.add_scalar("train/cls_loss", self.meter['cls_loss'].latest, self.epoch *self.max_iter + self.iter + 1)
                    self.tblogger.add_scalar("train/l1_loss", self.meter['l1_loss'].latest, self.epoch *self.max_iter + self.iter + 1)
                    self.tblogger.add_scalar("train/points_loss", self.meter['points_loss'].latest, self.epoch *self.max_iter + self.iter + 1)

                    # for i in range(3):
                    #     self.tblogger.add_histogram(f"train/cls_convs/head_{i}",
                    #                                 self.model.head.cls_convs[i][0].conv.weight[0],
                    #                                 self.epoch * self.max_iter + self.iter + 1)
                if self.args.logger == "wandb":
                    self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                    self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

            self.meter.clear_meters()

        # random resizing
        if (self.progress_in_iter + 1) % 10 == 0:
            self.input_size = self.exp.random_resize(
                self.train_loader, self.epoch, self.rank, self.is_distributed
            )

    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt

            ckpt = torch.load(ckpt_file, map_location=self.device)
            # resume the model/optimizer state dict
            ckpt_weights_dict = ckpt["model"]
            p_weights_dict = {k: v for k, v in ckpt_weights_dict.items()
                                 if model.state_dict()[k].numel() == v.numel()}
            # print(model.load_state_dict(load_weights_dict, strict=False))
            model.load_state_dict(p_weights_dict, strict=False)
            # model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            self.best_epoch = ckpt.pop("best_epoch", 0)

            # resume the training states variables
            start_epoch = (
                self.args.start_epoch - 1
                if self.args.start_epoch is not None
                else ckpt["start_epoch"]
            )
            self.start_epoch = start_epoch
            logger.info(
                "loaded checkpoint '{}' (epoch {})".format(
                    self.args.resume, self.start_epoch
                )
            )  # noqa
        else:
            if self.args.ckpt is not None:
                if self.exp.only_backbone_pretrain:
                    logger.info("only loading backbone pretrain checkpoint for fine tuning")
                    ckpt_file = self.args.ckpt
                    ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                    temp = copy.deepcopy(model)
                    model.backbone.backbone = load_ckpt(temp, ckpt).backbone.backbone
                    '''
                    temp.backbone.state_dict()['C3_p3.m.0.conv2.bn.bias']==model.backbone.state_dict()['C3_p3.m.0.conv2.bn.bias']
                    model.backbone.backbone.state_dict()['dark5.2.m.0.conv1.bn.bias']==temp.backbone.backbone.state_dict()['dark5.2.m.0.conv1.bn.bias']
                    '''
                    del temp
                else:
                    logger.info("loading checkpoint for fine tuning")
                    ckpt_file = self.args.ckpt
                    ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                    model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        logger.info('dconv mask init state:')
        logger.info(self.model.head.dconv_mask)
        with adjust_status(evalmodel, training=False):
            cocoeval_stats, summary, res_50, res_75 = self.exp.eval(
                evalmodel, self.evaluator, self.is_distributed
            )
        ap50_95, ap50, ap75 = cocoeval_stats[:3]
        cls_AP, cls_AR = cocoeval_stats[12:12 + self.exp.num_classes], cocoeval_stats[-self.exp.num_classes:]

        update_best_ckpt = ap50_95 > self.best_ap
        if update_best_ckpt:
            self.best_ap =ap50_95
            self.best_epoch = self.epoch + 1
        '''save ckpt'''
        self.save_ckpt("last_epoch", update_best_ckpt=update_best_ckpt, ap=ap50_95)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}", ap=ap50_95)

        if self.rank == 0:
            if self.args.logger == "tensorboard":
                summary = summary + f"best epoch:{self.best_epoch}, ap:{self.best_ap:4f}\n"
                self.tblogger.add_text("val_COCO", summary.replace('\n', '  \n'), self.epoch + 1)
                self.tblogger.add_scalar("val_COCO/AP@50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val_COCO/AP@50:95", ap50_95, self.epoch + 1)
                self.tblogger.add_scalar("val_COCO/AP@75", ap75, self.epoch + 1)

                self.tblogger.add_scalar("val_cm/MacroF1@50", res_50[-1, -1], self.epoch + 1)
                self.tblogger.add_scalar("val_cm/MacroF1@75", res_75[-1, -1], self.epoch + 1)
                self.tblogger.add_scalar("val_cm//Accuracy@50", res_50[-1, -2], self.epoch + 1)
                self.tblogger.add_scalar("val_cm/Accuracy@75", res_75[-1, -2], self.epoch + 1)
                self.tblogger.add_text("val_cm/c_matrix@50", str(res_50).replace('\n', '').replace('  ', '&emsp;').replace('] [', '  \n')[2:-2], self.epoch + 1)
                self.tblogger.add_text("val_cm/c_matrix@75", str(res_75).replace('\n', '').replace('  ', '&emsp;').replace('] [', '  \n')[2:-2], self.epoch + 1)

                for i in range(self.exp.num_classes):
                    self.tblogger.add_scalar(f"val_cls_AP/{self.exp.cls_names[i]}", cls_AP[i], self.epoch + 1)
                    self.tblogger.add_scalar(f"val_cls_AR/{self.exp.cls_names[i]}", cls_AR[i], self.epoch + 1)

                    self.tblogger.add_scalar(f"val_cm_F1@50/{self.exp.cls_names[i]}", res_50[i, -1], self.epoch + 1)
                    self.tblogger.add_scalar(f"val_cm_F1@75/{self.exp.cls_names[i]}", res_75[i, -1], self.epoch + 1)

                    self.tblogger.add_scalar(f"val_cm_Precision@50/{self.exp.cls_names[i]}", res_50[-1, i], self.epoch + 1)
                    self.tblogger.add_scalar(f"val_cm_Precision@75/{self.exp.cls_names[i]}", res_75[-1, i], self.epoch + 1)

                    self.tblogger.add_scalar(f"val_cm_Recall@50/{self.exp.cls_names[i]}", res_50[i, -2], self.epoch + 1)
                    self.tblogger.add_scalar(f"val_cm_Recall@75/{self.exp.cls_names[i]}", res_75[i, -2], self.epoch + 1)


                # self.tblogger.add_scalar("val/Recall50", ap50_95, self.epoch + 1)
                # self.tblogger.add_scalar("val/Recall50", ap75, self.epoch + 1)
                # if self.exp.arc:
                #     for i in range(self.model.head.cls_w.state_dict()['0'].shape[0]):
                #         self.tblogger.add_histogram(f"train/arc_w/{i}", self.model.head.cls_w.state_dict()['0'][i], self.epoch + 1)
                # else:
                # for i in range(self.model.head.cls_preds[0].weight.shape[0]):
                #     self.tblogger.add_histogram(f"train/cls_preds/{i}", self.model.head.cls_preds[0].weight[i], self.epoch + 1)

            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({
                    "val/COCOAP50": ap50,
                    "val/COCOAP50_95": ap50_95,
                    "epoch": self.epoch + 1,
                })
            logger.info("\n" + summary)
        synchronize()
    def save_ckpt(self, ckpt_name, update_best_ckpt=False, ap=None):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info(f"Save weights to {self.file_name} {ckpt_name}")
            ckpt_state = {
                "start_epoch": self.epoch + 1,
                "cur_ap": ap,
                "model": save_model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "best_ap": self.best_ap,
                "best_epoch": self.best_epoch,
            }
            save_checkpoint(
                ckpt_state,
                update_best_ckpt,
                self.file_name,
                ckpt_name,
            )

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)
