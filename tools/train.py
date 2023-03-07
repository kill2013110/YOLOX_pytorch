#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import random
import warnings
from loguru import logger

import torch
import torch.backends.cudnn as cudnn

from yolox.core import Trainer, launch
from yolox.exp import get_exp
from yolox.utils import configure_nccl, configure_omp, get_num_devices
import os, socket
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'

# print(f'{"*" * 10} {socket.gethostname()} {"*" * 10}')
if socket.gethostname() == 'DESKTOP-OMJJ23Q':
    path_root = r'D:\liwenlong/'
else:
    path_root = 'E:\ocr\container_ocr/'

def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")
    parser.add_argument("-expn", "--experiment-name", type=str,
                        # default='g_s_mask_416_lr_arc_5_0.5_ota_arc_no_m',
                        # default = 'g_s_mask_416_lr_arc_5_0.25_resume40',
                        # default=None
    )
    # parser.add_argument("-describe", "--describe-info", type=str,
    #                     # default='yolo_archead cls_pred二维 ',
    #                     # default = 'g_s_mask_416_lr_arc_5_0.25_resume40',
    #                     default='yolox-s-v3'
    # )
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")

    # distributed

    parser.add_argument(
        "--dist-backend", default="nccl", type=str, help="distributed backend"
    )
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size")
    parser.add_argument(
        "-d", "--devices", default=1, type=int, help="device for training"
    )
    parser.add_argument(
        "-f",
        "--exp_file",
        # default=path_root + 'YOLOX\exps\example\custom/yolox_s_mask.py',
        default=path_root + 'YOLOX\exps\example\custom/s_test.py',
        type=str,
        help="plz input your experiment description file",
    )
    # 0.1987ashdlihasldh
    parser.add_argument(
        "--resume",
        default=True,
        # default=False,
        action="store_true", help="resume training"
    )
    temp_dir_name = \
        's_test_org_None_None_0points_100straug_100coslr_0.0deg_IACS'
    parser.add_argument("-c", "--ckpt",
                        # default= None,
                        # default=path_root + fr'YOLOX\tools\YOLOX_outputs\{temp_dir_name}\best_ckpt.pth',
                        default=path_root + fr'YOLOX\tools\YOLOX_outputs\{temp_dir_name}\last_epoch_ckpt.pth',
                        # default=path_root + 'YOLOX\weight\yolox_s.pth',
                        # default=path_root + 'YOLOX\tools\YOLOX_outputs\s_test_points_branch_1_8points_100straug_100coslr_0.1_greater0.9\best_ckpt.pth',
                        # default=path_root + 'YOLOX\tools\YOLOX_outputs\s_test_points_branch_1_6points_45.0deg_100straug_100coslr_0.1_greater0.9\epoch_99_ckpt.pth',
                        type=str, help="checkpoint file")
    parser.add_argument(
        "-e",
        "--start_epoch",
        default=71,
        type=int,
        help="resume training start epoch",
    )
    parser.add_argument(
        "--num_machines", default=1, type=int, help="num of node for training"
    )
    parser.add_argument(
        "--machine_rank", default=0, type=int, help="node rank for multi-node training"
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision training.",
    )
    parser.add_argument(
        "--cache",
        dest="cache",
        default=False,
        action="store_true",
        help="Caching imgs to RAM for fast training.",
    )
    parser.add_argument(
        "-o",
        "--occupy",
        dest="occupy",
        default=False,
        action="store_true",
        help="occupy GPU memory first for training.",
    )
    parser.add_argument(
        "-l",
        "--logger",
        type=str,
        help="Logger to be used for metrics",
        default="tensorboard"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser


@logger.catch
def main(exp, args):
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! You may see unexpected behavior "
            "when restarting from checkpoints."
        )

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    trainer.train()


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    assert num_gpu <= get_num_devices()

    dist_url = "auto" if args.dist_url is None else args.dist_url
    launch(
        main,
        num_gpu,
        args.num_machines,
        args.machine_rank,
        backend=args.dist_backend,
        dist_url=dist_url,
        args=(exp, args),
    )
