#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import time
from loguru import logger

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess, vis

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]


def make_parser():
    parser = argparse.ArgumentParser("YOLOX Demo!")
    parser.add_argument(
        "--demo", default="shot", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='iou')
    parser.add_argument("-n", "--name", type=str, default='yolox-s', help="model name")

    parser.add_argument(
        "--path", default=r"F:\liwenlong\kapao\2.mp4", help="path to images or video"
    )
    # parser.add_argument(
    #     "--path", default="F:\datasets\Mask_detection/test_pic/", help="path to images or video"
    # )
    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=False,
        action="store_true",
        help="whether to save the inference result of image/video",
    )

    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='E:\ocr\container_ocr\YOLOX\exps\example\custom/yolox_s_face.py',
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt",
                        default=r"E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\s_face_alpha_ciou\best_ckpt.pth",
                        type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.3, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=416, type=int, help="test img size")
    parser.add_argument(
        "--fp16",
        dest="fp16",
        default=False,
        action="store_true",
        help="Adopting mix precision evaluating.",
    )
    parser.add_argument(
        "--legacy",
        dest="legacy",
        default=False,
        action="store_true",
        help="To be compatible with older versions",
    )
    parser.add_argument(
        "--fuse",
        dest="fuse",
        default=False,
        action="store_true",
        help="Fuse conv and bn for testing.",
    )
    parser.add_argument(
        "--trt",
        dest="trt",
        default=False,
        action="store_true",
        help="Using TensorRT model for testing.",
    )
    return parser


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if exp.cls_names != None:
            self.cls_names = exp.cls_names
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img, info_vis=True):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width, _ = img.shape
        # img_min_side = 1000
        # if width <= height:
        #     f = float(img_min_side) / width
        #     resized_height = int(f * height)
        #     resized_width = int(img_min_side)
        # else:
        #     f = float(img_min_side) / height
        #     resized_width = int(f * width)
        #     resized_height = int(img_min_side)
        # img = cv2.resize(img,(resized_width, resized_height))


        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            if info_vis:
                logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35, show_conf=True):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names,show_conf=show_conf)
        return vis_res

def cs_shot(predictor, args):
    import numpy as np
    # from mss import mss
    from pynput import mouse, keyboard
    import win32api, win32gui,win32con, pyautogui

    def switch(key):
        key2 = keyboard.Key.alt.alt_l
        key1 = keyboard.Key.enter
        if key == key2:
            if off_on:
                off_on = False
            else: off_on = True

    # 320 256 192 128 64
    off_on = True
    mouse_con = mouse.Controller()
    half_scale = 128
    t0 = time.time()
    with keyboard.Listener(on_press=switch) as listener:
        hwnd = win32gui.FindWindow(None, 'Counter-Strike: Global Offensive - Direct3D 9')
        rect = win32gui.GetWindowRect(hwnd)
        # rect = get_window_rect(hwnd)

        while 1:
            stat_xy = (0, 0)
            region = rect[0], rect[1], rect[2] - rect[0], rect[3] - rect[1]

            scale = 1920/region[2], 1080/region[3],

            frame = np.array(pyautogui.screenshot(region=region))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # cv2.imshow('1', frame)
            # cv2.waitKey()

            ih, iw = frame.shape[:2]
            t, l = int(ih/2-half_scale), int(iw/2-half_scale)
            img = frame[t:t+half_scale*2, l:l+half_scale*2].copy()

            cv2.rectangle(frame, (l, t), (l+half_scale*2, t+half_scale*2), (255, 255, 255), 2)

            outputs, img_info = predictor.inference(img, info_vis=False)
            # result_frame = predictor.visual(outputs[0], img_info, predictor.confthre,show_conf=False)
            # frame[t:half_scale*2+t, l:half_scale*2+l] = cv2.resize(result_frame, (half_scale*2, half_scale*2))

            if outputs[0] != None:
                boxes = np.array(outputs[0].cpu())
                boxes[:, :4] = np.int16(boxes[:, :4]/img_info['ratio'])
                scores = boxes[:, 4] * boxes[:, 5]

                w_h = boxes[:, 2:4] - boxes[:, :2]
                # idx = np.argmax(w_h[:, 0] * w_h[:, 1])
                idx = np.argmax(scores)
                shot_x = int((boxes[idx, 0] + boxes[idx, 2])/2) + l + rect[0]
                shot_y = int((boxes[idx, 1] + boxes[idx, 3])/2 + t + rect[1])
                # shot_y = int((boxes[idx, 1] + boxes[idx, 3])/2 + t + rect[1] + w_h[idx, 1]*2)
                stat_xy = (shot_x, shot_y)

                if off_on:
                    org_xy = win32api.GetCursorPos()
                    # org_xy = (1920, 1080)

                    move_xy = (stat_xy[0] - org_xy[0], stat_xy[1] - org_xy[1])
                    # move_xy = int(move_xy[0]*scale[0]), int(move_xy[1]*scale[1])
                    move_d = (move_xy[0] ** 2 + move_xy[1] ** 2) ** 0.5
                    if stat_xy != (0, 0) and move_d < half_scale:
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_xy[0], move_xy[1], 0, 0)
                        # time.sleep(0.001)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

                        # print(f'move:', org_xy, stat_xy, move_xy, move_d, '\n')
                    # else:
                    #     print(f'not move:', org_xy, stat_xy, move_xy, move_d, '\n')

                cv2.circle(frame, (shot_x, shot_y), 2, (0, 255, 0), 3)

                for i, box in enumerate(boxes):
                    if scores[i]>predictor.confthre:
                        cv2.rectangle(frame,
                        (int(box[0] + l), int(box[1] + t)),
                        (int(box[2] + l), int(box[3] + t)), (0, 255, 0), 2)

            t = time.time()
            # print('{:.1f} FPS'.format(1 / (t - t0)))
            cv2.putText(frame, '{:.1f} FPS'.format(1 / (t - t0)), (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=4)
            cv2.imshow("shot", cv2.resize(frame, (960, 540)))
            cv2.waitKey(1)
            t0 = time.time()


def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, "vis_res")
        os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"

    logger.info("Args: {}".format(args))

    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    model = exp.get_model()
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))

    if args.device == "gpu":
        model.cuda()
        if args.fp16:
            model.half()  # to FP16
    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, "best_ckpt.pth")
        else:
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")

    if args.fuse:
        logger.info("\tFusing model...")
        model = fuse_model(model)

    if args.trt:
        assert not args.fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
        logger.info("Using TensorRT to inference")
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(
        model, exp, COCO_CLASSES, trt_file, decoder,
        args.device, args.fp16, args.legacy,
    )
    current_time = time.localtime()
    if args.demo == "shot":
        cs_shot(predictor, args)

if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
