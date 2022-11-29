#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import random
import torch
import cv2
import numpy as np

from yolox.utils import adjust_box_anns, get_local_rank

from ..data_augment import random_affine
from .datasets_wrapper import Dataset


def get_mosaic_coordinate(mosaic_image, mosaic_index, xc, yc, w, h, input_h, input_w):
    # TODO update doc
    # index0 to top left part of image
    if mosaic_index == 0:
        x1, y1, x2, y2 = max(xc - w, 0), max(yc - h, 0), xc, yc
        small_coord = w - (x2 - x1), h - (y2 - y1), w, h # 可以看出右下角与中心点（xc, yc）重合
    # index1 to top right part of image
    elif mosaic_index == 1:
        x1, y1, x2, y2 = xc, max(yc - h, 0), min(xc + w, input_w * 2), yc
        small_coord = 0, h - (y2 - y1), min(w, x2 - x1), h
    # index2 to bottom left part of image
    elif mosaic_index == 2:
        x1, y1, x2, y2 = max(xc - w, 0), yc, xc, min(input_h * 2, yc + h)
        small_coord = w - (x2 - x1), 0, w, min(y2 - y1, h)
    # index2 to bottom right part of image
    elif mosaic_index == 3:
        x1, y1, x2, y2 = xc, yc, min(xc + w, input_w * 2), min(input_h * 2, yc + h)  # noqa
        small_coord = 0, 0, min(w, x2 - x1), min(y2 - y1, h)
    return (x1, y1, x2, y2), small_coord


class MosaicDetection(Dataset):
    """Detection dataset wrapper that performs mixup for normal dataset."""

    def __init__(
        self, dataset, img_size, mosaic=True, preproc=None,
        degrees=10.0, translate=0.1, mosaic_scale=(0.5, 1.5),
        mixup_scale=(0.5, 1.5), shear=2.0, enable_mixup=True,
        mosaic_prob=1.0, mixup_prob=1.0, *args
    ):
        """

        Args:
            dataset(Dataset) : Pytorch dataset object.
            img_size (tuple):
            mosaic (bool): enable mosaic augmentation or not.
            preproc (func):
            degrees (float):
            translate (float):
            mosaic_scale (tuple):
            mixup_scale (tuple):
            shear (float):
            enable_mixup (bool):
            *args(tuple) : Additional arguments for mixup random sampler.
        """
        super().__init__(img_size, mosaic=mosaic)
        self._dataset = dataset
        self.preproc = preproc
        self.degrees = degrees
        self.translate = translate
        self.scale = mosaic_scale
        self.shear = shear
        self.mixup_scale = mixup_scale
        self.enable_mosaic = mosaic
        self.enable_mixup = enable_mixup
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.local_rank = get_local_rank()
        self.get_face_pionts = self.preproc.get_face_pionts
    def __len__(self):
        return len(self._dataset)

    @Dataset.mosaic_getitem
    def __getitem__(self, idx):
        # with torch.no_grad():
        #     assert not (self.enable_mosaic and self.preproc.get_face_pionts)
            if self.enable_mosaic and random.random() < self.mosaic_prob:
                mosaic_labels, mosaic_points = [], []
                input_dim = self._dataset.input_dim
                input_h, input_w = input_dim[0], input_dim[1]

                # yc, xc = s, s  # mosaic center x, y
                yc = int(random.uniform(0.5 * input_h, 1.5 * input_h))
                xc = int(random.uniform(0.5 * input_w, 1.5 * input_w))

                # 3 additional image indices
                indices = [idx] + [random.randint(0, len(self._dataset) - 1) for _ in range(3)]

                for i_mosaic, index in enumerate(indices):
                    img, _labels_points, _, img_id = self._dataset.pull_item(index)
                    _labels = _labels_points[:, :5]
                    if self.get_face_pionts:
                        _points = _labels_points[:, 5:]
                    h0, w0 = img.shape[:2]  # orig hw
                    scale = min(1. * input_h / h0, 1. * input_w / w0)
                    img = cv2.resize(
                        img, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_LINEAR
                    )
                    # generate output mosaic image
                    (h, w, c) = img.shape[:3]
                    if i_mosaic == 0:
                        mosaic_img = np.full((input_h * 2, input_w * 2, c), 114, dtype=np.uint8)

                    # suffix l means large image, while s means small image in mosaic aug.
                    (l_x1, l_y1, l_x2, l_y2), (s_x1, s_y1, s_x2, s_y2) = get_mosaic_coordinate(
                        mosaic_img, i_mosaic, xc, yc, w, h, input_h, input_w
                    )

                    mosaic_img[l_y1:l_y2, l_x1:l_x2] = img[s_y1:s_y2, s_x1:s_x2]
                    padw, padh = l_x1 - s_x1, l_y1 - s_y1

                    labels = _labels.copy()
                    if self.get_face_pionts:
                        points = _points.copy()
                        if _labels.size > 0:
                            points[:, 0::3] = scale * _points[:, 0::3] + padw
                            points[:, 1::3] = scale * _points[:, 1::3] + padh
                        mosaic_points.append(points)
                    # Normalized xywh to pixel xyxy format
                    if _labels.size > 0:
                        labels[:, 0] = scale * _labels[:, 0] + padw
                        labels[:, 1] = scale * _labels[:, 1] + padh
                        labels[:, 2] = scale * _labels[:, 2] + padw
                        labels[:, 3] = scale * _labels[:, 3] + padh
                    mosaic_labels.append(labels)


                if len(mosaic_labels): #仅有截断
                    mosaic_labels = np.concatenate(mosaic_labels, 0)
                    np.clip(mosaic_labels[:, 0], 0, 2 * input_w, out=mosaic_labels[:, 0])
                    np.clip(mosaic_labels[:, 1], 0, 2 * input_h, out=mosaic_labels[:, 1])
                    np.clip(mosaic_labels[:, 2], 0, 2 * input_w, out=mosaic_labels[:, 2])
                    np.clip(mosaic_labels[:, 3], 0, 2 * input_h, out=mosaic_labels[:, 3])
                    if self.get_face_pionts:
                        mosaic_points = np.concatenate(mosaic_points, 0)
                '''
coordinate = np.int16(mosaic_labels.copy())
points = np.int16(mosaic_points.copy())
x = np.uint8(mosaic_img.copy())
for n in range(len(coordinate)):
    for i in range(6):
        cv2.circle(x, points[n, i * 3:i * 3 + 2], 2, color=(0, 40 * i, 0))
    cv2.rectangle(x, (coordinate[n][0], coordinate[n][1], \
                      coordinate[n][2] - coordinate[n][0], coordinate[n][3] - coordinate[n][1]),
                  [255, 255, 0])
cv2.imshow('1', x)
cv2.waitKey()
                '''
                mosaic_img, mosaic_labels, mosaic_points = random_affine(
                    mosaic_img,
                    mosaic_labels,
                    target_size=(input_w, input_h),
                    degrees=self.degrees,
                    translate=self.translate,
                    scales=self.scale,
                    shear=self.shear,
                    points=mosaic_points,
                    get_face_pionts=self.get_face_pionts,
                )
                if self.get_face_pionts:
                    # 将不在框内的point的score设置为0
                    assert self.get_face_pionts == int(mosaic_points.shape[1]/3)
                    pre_gt_x1 = mosaic_labels[:, 0].reshape(-1,1)
                    pre_gt_y1 = mosaic_labels[:, 1].reshape(-1,1)
                    pre_gt_x2 = mosaic_labels[:, 2].reshape(-1,1)
                    pre_gt_y2 = mosaic_labels[:, 3].reshape(-1,1)
                    mosaic_points[:, 2::3] = np.where( np.logical_and(
                        np.logical_and(np.tile(pre_gt_x1, self.get_face_pionts)<=mosaic_points[:, 0::3], mosaic_points[:, 0::3]<= np.tile(pre_gt_x2, self.get_face_pionts)), #横坐标在gt框内
                        np.logical_and(np.tile(pre_gt_y1, self.get_face_pionts)<=mosaic_points[:, 1::3], mosaic_points[:, 1::3]<= np.tile(pre_gt_y2, self.get_face_pionts))), #纵坐标在gt框内
                        mosaic_points[:, 2::3], 0) # 若在gt框内，置信度不变，否则置为0
                if self.get_face_pionts:
                    mosaic_labels = np.hstack((mosaic_labels, mosaic_points))

                # -----------------------------------------------------------------
                # CopyPaste: https://arxiv.org/abs/2012.07177
                # -----------------------------------------------------------------
                if (
                    self.enable_mixup
                    and not len(mosaic_labels) == 0
                    and random.random() < self.mixup_prob
                ):
                    mosaic_img, mosaic_labels = self.mixup(mosaic_img, mosaic_labels, self.input_dim,
                        get_face_pionts=self.get_face_pionts,)
                mix_img, padded_labels = self.preproc(mosaic_img, mosaic_labels, self.input_dim,)
                img_info = (mix_img.shape[1], mix_img.shape[0])
                '''
coordinate = np.int16(padded_labels[:, :5].copy())
points = np.float32(padded_labels[:, 5:].copy())
x = np.uint8(mix_img[0].copy())
for n in range(len(coordinate)):
    for i in range(6):
        if points[n, i * 3 + 2] > 0:
            cv2.circle(x, np.int16(points[n, i * 3:i * 3 + 2]), 2, color=(0, 40 * i, 0))
    # cv2.rectangle(x, (coordinate[n][0], coordinate[n][1], \
    #                   coordinate[n][2] - coordinate[n][0], coordinate[n][3] - coordinate[n][1]),
    #               [255, 255, 0])
    cv2.rectangle(x, (int(coordinate[n][0+1]-coordinate[n][2+1]/2), int(coordinate[n][1+1]-coordinate[n][3+1]/2), \
    coordinate[n][2+1], coordinate[n][3+1]),
[255, 255, 0])

cv2.imshow('1', x)
cv2.waitKey()
                '''
                # -----------------------------------------------------------------
                # img_info and img_id are not used for training.
                # They are also hard to be specified on a mosaic image.
                # -----------------------------------------------------------------
                return mix_img, padded_labels, img_info, img_id

            else:
                self._dataset._input_dim = self.input_dim
                img, label, img_info, img_id = self._dataset.pull_item(idx)
                img, label = self.preproc(img, label, self.input_dim)
                return img, label, img_info, img_id

    def mixup(self, origin_img, origin_labels, input_dim,
                    get_face_pionts=False,):
        jit_factor = random.uniform(*self.mixup_scale)
        FLIP = random.uniform(0, 1) > 0.5
        cp_labels = []
        while len(cp_labels) == 0:
            cp_index = random.randint(0, self.__len__() - 1) # 一个完整的数据读取流程
            cp_labels = self._dataset.load_anno(cp_index)
        img, cp_labels, _, _ = self._dataset.pull_item(cp_index)

        if len(img.shape) == 3:
            cp_img = np.ones((input_dim[0], input_dim[1], 3), dtype=np.uint8) * 114
        else:
            cp_img = np.ones(input_dim, dtype=np.uint8) * 114

        cp_scale_ratio = min(input_dim[0] / img.shape[0], input_dim[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * cp_scale_ratio), int(img.shape[0] * cp_scale_ratio)),
            interpolation=cv2.INTER_LINEAR,
        )

        cp_img[
            : int(img.shape[0] * cp_scale_ratio), : int(img.shape[1] * cp_scale_ratio)
        ] = resized_img

        cp_img = cv2.resize(
            cp_img,
            (int(cp_img.shape[1] * jit_factor), int(cp_img.shape[0] * jit_factor)),
        )
        cp_scale_ratio *= jit_factor

        if FLIP:
            cp_img = cp_img[:, ::-1, :]

        origin_h, origin_w = cp_img.shape[:2]
        target_h, target_w = origin_img.shape[:2]
        padded_img = np.zeros(
            (max(origin_h, target_h), max(origin_w, target_w), 3), dtype=np.uint8
        )
        padded_img[:origin_h, :origin_w] = cp_img

        x_offset, y_offset = 0, 0
        if padded_img.shape[0] > target_h:
            y_offset = random.randint(0, padded_img.shape[0] - target_h - 1)
        if padded_img.shape[1] > target_w:
            x_offset = random.randint(0, padded_img.shape[1] - target_w - 1)
        padded_cropped_img = padded_img[
            y_offset: y_offset + target_h, x_offset: x_offset + target_w
        ]

        cp_bboxes_origin_np = adjust_box_anns(
            cp_labels[:, :4].copy(), cp_scale_ratio, 0, 0, origin_w, origin_h
        )
        if FLIP:
            cp_bboxes_origin_np[:, 0::2] = (
                origin_w - cp_bboxes_origin_np[:, 0::2][:, ::-1]
            )
        cp_bboxes_transformed_np = cp_bboxes_origin_np.copy()
        cp_bboxes_transformed_np[:, 0::2] = np.clip(
            cp_bboxes_transformed_np[:, 0::2] - x_offset, 0, target_w
        )
        cp_bboxes_transformed_np[:, 1::2] = np.clip(
            cp_bboxes_transformed_np[:, 1::2] - y_offset, 0, target_h
        )

        if get_face_pionts: #加入关键点检测的
            cp_points = cp_labels[:, 5:].copy()
            cp_points[:, 0::3] = cp_points[:, 0::3] * cp_scale_ratio + 0
            cp_points[:, 1::3] = cp_points[:, 1::3] * cp_scale_ratio + 0
            if FLIP:
                cp_points[:, 0::3] = origin_w - cp_points[:, 0::3]
                if get_face_pionts == 5:
                    '''nose_l, 'nose_r, 'mouth_l, 'mouth_r, 'mouth_t, 'nose'''
                    cp_points[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] = cp_points[:,
                                                                             [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
                if get_face_pionts==6:
                    # '''nose_l, 'nose_r, 'mouth_l, 'mouth_r, 'mouth_t, 'mouth_b''' 6点的左右交换策略
                    cp_points[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]] = cp_points[:,
                    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]
                if get_face_pionts == 11:
                    '''
                    'nose_l', 'nose_r', 
                   'mouth_l', 'mouth_r',
                   'brow_l', 'brow_r',
                   'eye_l', 'eye_r',
                   'mouth_t', 'mouth_b', 'nose',
                    '''
                    cp_points[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]] = cp_points[:,
                    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8, 15, 16, 17, 12, 13, 14, 21, 22, 23, 18, 19, 20]]
            cp_points_transformed_np = cp_points.copy()
            cp_points_transformed_np[:, 0::3] = cp_points_transformed_np[:, 0::3] - x_offset
            cp_points_transformed_np[:, 1::3] = cp_points_transformed_np[:, 1::3] - y_offset

            # 将不在框内的point的score设置为0
            assert get_face_pionts == int(cp_points_transformed_np.shape[1] / 3)
            pre_gt_x1 = cp_bboxes_transformed_np[:, 0].reshape(-1, 1)
            pre_gt_y1 = cp_bboxes_transformed_np[:, 1].reshape(-1, 1)
            pre_gt_x2 = cp_bboxes_transformed_np[:, 2].reshape(-1, 1)
            pre_gt_y2 = cp_bboxes_transformed_np[:, 3].reshape(-1, 1)
            cp_points_transformed_np[:, 2::3] = np.where(np.logical_and(
                np.logical_and(np.tile(pre_gt_x1, get_face_pionts) <= cp_points_transformed_np[:, 0::3],
                               cp_points_transformed_np[:, 0::3] <= np.tile(pre_gt_x2, get_face_pionts)),  # 横坐标在gt框内
                np.logical_and(np.tile(pre_gt_y1, get_face_pionts) <= cp_points_transformed_np[:, 1::3],
                               cp_points_transformed_np[:, 1::3] <= np.tile(pre_gt_y2, get_face_pionts))),  # 纵坐标在gt框内
                cp_points_transformed_np[:, 2::3], 0)  # 若在gt框内，置信度不变，否则置为0

        cls_labels = cp_labels[:, 4:5].copy()
        box_labels = cp_bboxes_transformed_np
        if get_face_pionts:
            labels = np.hstack((box_labels, cls_labels, cp_points_transformed_np))
        else:
            labels = np.hstack((box_labels, cls_labels))
        origin_labels = np.vstack((origin_labels, labels))
        origin_img = origin_img.astype(np.float32)
        origin_img = 0.5 * origin_img + 0.5 * padded_cropped_img.astype(np.float32)

        return origin_img.astype(np.uint8), origin_labels
