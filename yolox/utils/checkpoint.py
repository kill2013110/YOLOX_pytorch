#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
import os
import shutil
from loguru import logger

import torch


def load_ckpt(model, ckpt):
    model_state_dict = model.state_dict()
    matched_w = {}
    unmatched_w = {}
    for key_model, v in model_state_dict.items():
        if key_model not in ckpt:
            unmatched_w[key_model] = v
            continue
        v_ckpt = ckpt[key_model]
        if v.shape != v_ckpt.shape:
            unmatched_w[key_model] = v
            continue
        matched_w[key_model] = v_ckpt
    logger.warning("These weight keys in the model, but not match with the ckpt:")
    logger.warning(unmatched_w.keys())

    logger.warning("These weight keys in the model, and matched with the ckpt:")
    logger.warning(matched_w.keys())

    model.load_state_dict(matched_w, strict=False)
    return model


def save_checkpoint(state, is_best, save_dir, model_name=""):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, model_name + "_ckpt.pth")
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, "best_ckpt.pth")
        shutil.copyfile(filename, best_filename)
