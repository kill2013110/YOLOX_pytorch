#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os, json
import time
import os, shutil
from tqdm import tqdm
import numpy as np
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
        "--demo", default="val", help="demo type, eg. image, video and webcam"
    )
    parser.add_argument("-expn", "--experiment-name", type=str, default='iou')
    parser.add_argument("-n", "--name", type=str, default='s_container_det_ciou', help="model name")

    parser.add_argument(
        "--path",
        default=r"F:\datasets\Diverse_Masked_Faces_v2_m\new_img",
        # default=r'F:\liwenlong\reptile\img',
        # default=[r'F:\datasets\Diverse_Masked_Faces_v2_m\reptile_img',
        #          r'F:\datasets\Diverse_Masked_Faces_v2_m\RWMFD_img'],
        # default=[r"F:\datasets\aizoo_face_mask",
        #          r'F:\datasets\123',
        #          r'F:\datasets\archive',
        #          r'F:\liwenlong\reptile\baidu_rec1'
        #          'F:\datasets\MAFA'],
        help="path to images or video"
    )

    parser.add_argument("--camid", type=int, default=0, help="webcam demo camera id")
    parser.add_argument(
        "--save_result",
        default=True,
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument(
        "--vis_folder",
        default=r'E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\s_mask_v2\vis',
        type=str,
        help="where to save the inference result of image/video",
    )
    # exp file
    parser.add_argument(
        "-f",
        "--exp_file",
        default='E:\ocr\container_ocr\YOLOX\exps\example\custom/s_test.py',
        # default='E:\ocr\container_ocr\YOLOX\exps\example\custom/yolox_s_mask.py',
        type=str,
        help="pls input your experiment description file",
    )
    parser.add_argument("-c", "--ckpt",
                        # default=r"E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\s_test_org_landmark_test_points_0.1_strongaug_greater0.9\best_ckpt.pth",
                        default=r"E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\s_test_points_branch_1_landmark_test_6points_0.1_strongaug_greater0.9\best_ckpt.pth",
                        # default=r"E:\ocr\container_ocr\YOLOX\tools\YOLOX_outputs\yolox_s_mask_org\best_ckpt.pth",
                        type=str, help="ckpt for eval")
    parser.add_argument(
        "--device",
        default="gpu",
        type=str,
        help="device to run our model, can either be cpu or gpu",
    )
    parser.add_argument("--conf", default=0.3, type=float, help="test conf")
    parser.add_argument("--nms", default=0.65, type=float, help="test nms threshold")
    parser.add_argument("--tsize", default=640, type=int, help="test img size")
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


def get_image_list(paths):
    image_names = []

    if os.path.basename(paths).split('.')[-1]=='json':
        with open(paths, "r") as in_file:
            json_dict = json.load(in_file)
        for img_info in json_dict['images']:
            image_names.append(os.path.join(r'F:\datasets\Diverse_Masked_Faces_v2_m/new_img', img_info['file_name']))
    else:
        if not isinstance(paths, list):
            paths = [paths]
        for path in paths:
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

    def inference(self, img, vis=512):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
            org_img = img.copy()
        else:
            img_info["file_name"] = None

        img_info["f"] =None
        if vis:
            height, width, _ = img.shape
            img_min_side = 720
            if width <= height:
                f = float(img_min_side) / width
                resized_height = int(f * height)
                resized_width = int(img_min_side)
            else:
                f = float(img_min_side) / height
                resized_width = int(f * width)
                resized_height = int(img_min_side)
            img_info["f"] = f
            img = cv2.resize(img, (resized_width, resized_height))


        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img


        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(org_img, None, self.test_size) # resize也会影响检测结果
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
            output = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=False
            )[0]
            # logger.info(f"Infer time: {time.time() - t0:.4f}s")
        return output, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        img = img_info["raw_img"]
        if output is None:
            return img, [], []
        output = output.cpu()
        output[:, 0:4] = output[:, 0:4]/img_info['ratio']
        bboxes = output[:, 0:4]
        points = output[:, 7:]/img_info['ratio']

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res, id_list, res_boxes = vis(img, bboxes, scores, cls, cls_conf, self.cls_names, points=points)
        return vis_res, id_list, res_boxes


def image_demo(predictor, vis_folder, path, current_time, save_result):
    if save_result:
        save_folder = vis_folder
        os.makedirs(save_folder, exist_ok=True)
    files = get_image_list(path)
    files.sort()

    for image_name in files:
        output, img_info = predictor.inference(image_name)
        result_image, id_list, res_boxes = predictor.visual(output, img_info, predictor.confthre)

        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            # logger.info("Saving detection result in {}".format(save_file_name))
            cv2.imwrite(save_file_name, result_image)
        # ch = cv2.waitKey(0)
        # if ch == 27 or ch == ord("q") or ch == ord("Q"):
        #     break


def val_demo(predictor, vis_folder, path, current_time, save_result):
    if save_result:
        save_folder = vis_folder
        os.makedirs(save_folder, exist_ok=True)
    with open(r"F:\datasets\Diverse_Masked_Faces_v2_m\ann\val_v3_11points.json", "r") as f:
    # with open(r"F:\datasets\Diverse_Masked_Faces_v2_m\ann\val_v3_small.json", "r") as f:
        all_json = json.load(f)

    files = []
    for i in all_json['images']:files.append(os.path.join(r'F:\datasets\Diverse_Masked_Faces_v2_m\new_img', i['file_name']))
    for image_name in tqdm(files):
        output, img_info = predictor.inference(image_name)
        result_image, id_list, res_boxes = predictor.visual(output, img_info, predictor.confthre)

        json_file = os.path.join(r'F:\datasets\Diverse_Masked_Faces_v2_m\\new_json_yolov5_face_PIP_l_score',
                                 os.path.split(image_name)[1][:-4] + '.json')
        with open(json_file, "r") as f:
            json_dict = json.load(f)
        gt_bboxes, gt_scores, gt_cls = [], [], []
        for shapes in json_dict["shapes"]:
            if shapes['shape_type'] == 'rectangle':
                gt_cls.append(predictor.cls_names.index(shapes["label"]))
                gt_bboxes.append(np.array(shapes['points']).flatten() * img_info["f"])
                gt_scores.append(1)
                # rects.append(rect)

        result_image, id_list, res_boxes = predictor.visual(output, img_info, predictor.confthre)
        result_image, id_list, res_boxes = vis(img_info["raw_img"], gt_bboxes, gt_scores, gt_cls, 0.5, predictor.cls_names, label=True)

        if save_result:
            save_file_name = os.path.join(save_folder, os.path.basename(image_name))
            cv2.imwrite(save_file_name, result_image)



def count_cat(a):
    b = np.zeros(5)
    for i in range(5):
        b[i] = a.count(i)
    return b

def image_det2json(predictor, vis_folder, path, current_time, save_result):
    # path =r"F:\datasets\Diverse_Masked_Faces_v2_m\new_img"
    path =r"F:\liwenlong\reptile\black"
    files = get_image_list(path)
    files.sort()
    json_root = r'F:\liwenlong\reptile\json_yolox_s'
    img_sub = r"F:\liwenlong\reptile\img_sub"
    print(path, json_root, img_sub)
    all_info = []
    # json_root = r'F:\datasets\Diverse_Masked_Faces_v2_m\new_json_yolov5_face_PIP_l_score'
    for image_name in tqdm(files[:5]):
        new_name = os.path.basename(image_name)[:-4]
        json_file = os.path.join(json_root, new_name.split('.')[0] + '.json')
        # if 3000<=int(new_name)<=5999 or 82000<=int(new_name):
        # if 81100<=int(new_name)<82000:
        if 1:
            img = cv2.imread(image_name)
            if img.__sizeof__() < 20:
                os.remove(image_name)
                continue
            output, img_info = predictor.inference(img, vis=False)
            if output == None:
                shutil.move(image_name, os.path.join(img_sub, os.path.split(image_name)[1]))
            else:
                h0, w0 = img_info["height"], img_info["width"]
                output = output.cpu()
                output[:, 0:4] = output[:, 0:4] / img_info['ratio']
                bboxes = output[:, 0:4]
                cls = output[:, 6]
                scores = output[:, 4] * output[:, 5]

                with open(r"F:\datasets\Diverse_Masked_Faces_v2_m\00000.json", "r") as in_file:
                    json_dict = json.load(in_file)
                rectangle = []
                for i, b in enumerate(bboxes):
                    if scores[i] > predictor.confthre:
                        xyxy = list(map(int, b[:4]))
                        shapes = {}
                        points = np.array(xyxy, 'i').reshape((2, 2))
                        points.sort(0)
                        points[:, 0] = points[:, 0].clip(0, w0)
                        points[:, 1] = points[:, 1].clip(0, h0)

                        shapes['label'] = predictor.cls_names[int(cls[i])]
                        shapes['shape_type'] = 'rectangle'
                        shapes['points'] = points.tolist()
                        rectangle.append(shapes)
                        all_info.extend([int(cls[i])])

                json_dict['shapes'] = rectangle
                json_dict['imageHeight'], json_dict['imageWidth'] = h0, w0
                json_dict['imagePath'] = new_name.split('.')[0] + '.jpg'
                with open(json_file, 'w', encoding='utf-8') as fw:
                    json.dump(json_dict, fw, indent=4)
    print('face', all_info.count(0))
    print('face_mask', all_info.count(1))
    print('nose_out', all_info.count(2))
    print('mouth_out', all_info.count(3))
    print('others', all_info.count(4))
    print('spoof', all_info.count(5))
    print('')

def imageflow_demo(predictor, vis_folder, current_time, args):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps_list = []
    if args.save_result:
        save_folder = os.path.join(
            vis_folder, time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
        )
        os.makedirs(save_folder, exist_ok=True)
        if args.demo == "video":
            save_path = os.path.join(save_folder, os.path.basename(args.path))
        else:
            save_path = os.path.join(save_folder, "camera.mp4")
        logger.info(f"video save_path is {save_path}")
        vid_writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
        )
    while True:
        t0 = time.time()
        ret_val, frame = cap.read()
        if ret_val:
            outputs, img_info = predictor.inference(frame)
            result_frame = predictor.visual(outputs[0], img_info, predictor.confthre)
            if args.save_result:
                vid_writer.write(result_frame)
            else:
                # cv2.namedWindow("yolox", cv2.WINDOW_NORMAL)
                t = time.time()
                fps_list.append(1 / (t - t0))
                # print('{:.1f} FPS'.format(1 / (t - t0)))
                cv2.putText(result_frame, '{:.1f} FPS'.format(1 / (t - t0)), (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), thickness=4)
                cv2.imshow("yolox", result_frame)
            ch = cv2.waitKey(1)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        else:
            print(np.array(fps_list).mean())
            break

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok=True)




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
            if args.save_result:
                vis_folder = os.path.splitext(args.ckpt)[0] + f'_{exp.test_size[0]}_vis'
                os.makedirs(vis_folder, exist_ok=True)
            ckpt_file = args.ckpt
        logger.info("loading checkpoint")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"], 0)
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
    if args.demo == "image":
        image_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    if args.demo == "val":
        val_demo(predictor, vis_folder, args.path, current_time, args.save_result)
    if args.demo == "json": ##write json
        image_det2json(predictor, vis_folder, args.path, current_time, args.save_result)
    elif args.demo == "video" or args.demo == "webcam":
        imageflow_demo(predictor, vis_folder, current_time, args)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)
