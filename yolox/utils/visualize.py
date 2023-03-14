#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import cv2
import numpy as np

__all__ = ["vis"]

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, label=False, points=None, vis_pos=None):
    t_size, t_thick = 0.5, 1
    # t_size, t_thick = 0.3, 2
    mask = []
    res_boxes = []
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[class_names[cls_id]] * 255).astype(np.uint8).tolist()

        # if txt_size[0]>(x1-y1)*1.5:
        # t_size = t_size/(txt_size[0]/200)
        # txt_size = cv2.getTextSize(text, font, t_size, t_thick)[0]
        # if txt_size[0]<150:t_thick=1
        if label:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 1)
            if points!=None:
                p_per_box = points[i]
                for j in range(int(len(p_per_box)/2)):
                    x = int(p_per_box[j * 2])
                    y = int(p_per_box[j * 2 + 1])
                    cv2.circle(img, [x,y], 1, color=(255,255,0))
            continue
        else:
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            if points!=None:
                p_per_box = points[i]
                for j in range(int(len(p_per_box)/2)):
                    x = int(p_per_box[j * 2])
                    y = int(p_per_box[j * 2 + 1])
                    cv2.circle(img, [x,y], 2, color=color)

        text = f'{class_names[cls_id]}:{score:.2f}'
        txt_color = (0, 0, 0) if np.mean(_COLORS[class_names[cls_id]]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, t_size, t_thick)[0]
        txt_bk_color = (_COLORS[class_names[cls_id]] * 255 * 0.7).astype(np.uint8).tolist()
        if vis_pos == 'up':
            v_x0, v_y0 = x0, y0-int(1.5*txt_size[1])
        else:
            v_x0, v_y0 = x0, y0
        cv2.rectangle(
            img,
            (v_x0, v_y0 + 1),
            (v_x0 + txt_size[0] + 1, v_y0 + int(1.5*txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (v_x0, v_y0 + txt_size[1]), font, t_size, txt_color, thickness=t_thick)
        mask.append(cls_id)
        res_boxes.append([x0, y0, x1, y1, class_names[cls_id]])
    return img, mask, res_boxes

_COLORS = {
        'face': np.array([1., 0., 0.]),
        'face_mask': np.array([0., 1., 0.]),
        'nose_out': np.array([1., 1., 0.]),
        'mouth_out': np.array([0., 0., 1.]),
        'others': np.array([0., 1., 1.]),
        'spoof': np.array([1., 0., 1.])
}
# def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None, show_conf=True):
#     t_size, t_thick = 0.7, 2
#     for i in range(len(boxes)):
#         box = boxes[i]
#         cls_id = int(cls_ids[i])
#         score = scores[i]
#         if score < conf:
#             continue
#         x0 = int(box[0])
#         y0 = int(box[1])
#         x1 = int(box[2])
#         y1 = int(box[3])
#
#         color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
#         text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
#         txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
#         font = cv2.FONT_HERSHEY_SIMPLEX
#
#         txt_size = cv2.getTextSize(text, font, t_size, t_thick)[0]
#         # if txt_size[0]>(x1-y1)*1.5:
#         #     t_size = t_size/(txt_size[0]/(x1-x0))
#         #     txt_size = cv2.getTextSize(text, font, t_size, t_thick)[0]
#         # if txt_size[0]<150:t_thick=1
#         cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
#
#         if show_conf:
#             txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
#             cv2.rectangle(
#                 img,
#                 (x0, y0 + 1),
#                 (x0 + txt_size[0] + 1, y0 + int(1.5*txt_size[1])),
#                 txt_bk_color,
#                 -1
#             )
#             cv2.putText(img, text, (x0, y0 + txt_size[1]), font, t_size, txt_color, thickness=t_thick)
#
#     return img
#
# _COLORS = np.array(
#     [
#         1.,0.,0.,
#         0.,1.,0.,
#         1.,1.,0.,
#         0.,0.,1.,
#         0.,1.,1.
#     ]
# ).astype(np.float32).reshape(-1, 3)

_COLORS_org = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)
