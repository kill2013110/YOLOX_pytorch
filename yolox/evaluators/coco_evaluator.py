#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io, sys, os
import itertools
import json
import shutil
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm
# import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import numpy as np

import torch

from yolox.data.datasets import COCO_CLASSES
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh,
    vis,
)

# from .evaluators import confusion_matrix
from .confusion_matrix import *

def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table, result_pair[1::2]


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table, result_pair[1::2]


class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = True,
        per_class_AR: bool = True,
        get_face_pionts=False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.get_face_pionts = get_face_pionts

    def evaluate(
        self,
        model,
        distributed=False,
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
        ckpt_path=None,
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """
        res_boxes = []
        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt

        for cur_iter, (imgs, _, info_imgs, ids) in enumerate(
            progress_bar(self.dataloader)
        ):
            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)  # cv2.imwrite('1.jpg', np.uint8(imgs.cpu().numpy()[0, 0]*255))
                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start

                output = postprocess(
                    outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic=False, get_face_pionts=self.get_face_pionts
                )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                 # '''pkl'''
                # pre_img_boxes = []
                # boxes = []
                # for box in outputs:
                #     box = box.cpu().numpy()
                #     boxes.append(box)
                # boxes = np.array(boxes).reshape((-1,7))
                # for i in range(self.num_classes):
                #     cls_box = boxes[boxes[:,-1]==i][:,:-2]
                #     pre_img_boxes.append(cls_box)
                # res_boxes.append(pre_img_boxes)
            coco_data, x1y1x2y2 = self.convert_to_coco_format(output, info_imgs, ids)
            data_list.extend(coco_data)
            res_boxes.extend(x1y1x2y2)

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        c_m_50 = c_m(
            self.dataloader.dataset,
            res_boxes,
            score_thr=0.3,
            tp_iou_thr=0.5)
        c_m_75 = c_m(
            self.dataloader.dataset,
            res_boxes,
            score_thr=0.3,
            tp_iou_thr=0.75)
        res_50, res_75 = f1(c_m_50), f1(c_m_75)

        # if os.path.split(sys.argv[0])[-1] == 'eval.py':
        #     self.eval_vis(self.dataloader.dataset, res_boxes, ckpt_path, vis_th=0.3)

        coco_results, info = self.evaluate_prediction(data_list, statistics, ckpt_path)
        synchronize()
        return coco_results, info, res_50, res_75

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        all_img_boxes = []
        # all_xyxy_boxes = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            pre_img_boxes = []
            xyxy_boxes = []
            if output is None:
                for i in range(self.num_classes):
                    pre_img_boxes.append(np.zeros(shape=(0,5)))
                all_img_boxes.append(pre_img_boxes)
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            xyxy = bboxes/scale
            bboxes = xyxy2xywh(bboxes/scale)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            ''''''

            # for box in outputs:
            #     box = box.cpu().numpy()
            #     boxes.append(box)
            # boxes = np.array(boxes).reshape((-1,7))
            # for i in range(self.num_classes):
            #     cls_box = boxes[boxes[:,-1]==i][:,:-2]
            #     pre_img_boxes.append(cls_box)
            # res_boxes.append(pre_img_boxes)
            ''''''

            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

                '''x1y1x2y2'''
                box = xyxy[ind].numpy()
                # box[2:] = box[0:2] + box[2:]
                xyxy_boxes.append(np.hstack((box, scores[ind].numpy(), label-1)))
            xyxy_boxes = np.array(xyxy_boxes).reshape((-1, 6))
            for i in range(self.num_classes):
                pre_cls_box = xyxy_boxes[xyxy_boxes[:, -1] == i][:, :-1]
                pre_img_boxes.append(pre_cls_box)
            all_img_boxes.append(pre_img_boxes)
            # all_xyxy_boxes = []
        # return data_list, []
        return data_list, all_img_boxes

    def evaluate_prediction(self, data_dict, statistics, ckpt_path=None):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                    ["forward", "NMS", "inference"],
                    [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
                )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
                _, tmp = tempfile.mkstemp()
                json.dump(data_dict, open(tmp, "w"))
                cocoDt = cocoGt.loadRes(tmp)
            #想设置为训练时调用cpp更高效eval执行
            # if False:
            if os.path.split(sys.argv[0])[-1] == 'eval.py':
                from pycocotools.cocoeval import COCOeval
                logger.warning("Use standard COCOeval.")
            else:
                try:
                    from yolox.layers import COCOeval_opt as COCOeval
                except ImportError:
                    from pycocotools.cocoeval import COCOeval
                    logger.warning("Use standard COCOeval.")


            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            cls_AP, cls_AR = [], []
            if self.per_class_AP:
                AP_table, cls_AP= per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table, cls_AR = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            stats_50_95 = cocoEval.summarize_50_95(vis=False)
            info += f"coco summary: {cocoEval.stats} \n"
            info += f"stats_50_95: {stats_50_95} \n"
            '''所有iou阈值下'''
            # AP_iou_th = []
            # iou_th = np.array([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8,0.85,0.9,0.95])
            # # iou_th = cocoEval.params.iouThrs.copy()
            # for i in iou_th:
            #     cocoEval.params.iouThrs = np.array([i])
            #     cocoEval.summarize(vis=False)
            #     AP_iou_th.append(cocoEval.stats[0])
            # print('*'*20+'\n')
            # print(str(np.array(AP_iou_th).round(4)))
            # print('*' * 20 + '\n')
            if os.path.split(sys.argv[0])[-1] == 'eval.py' and self.confthre>0.1:
                '''对于定位，针对每张图分析漏检和误检'''
                # th_list = [0.5, 0.75]
                th_list = [0.5,]
                vised_path = os.path.splitext(ckpt_path)[0] + f'_{self.img_size[0]}_vis'
                for th in th_list:
                    img_p, img_r = {}, {}
                    vised_p_r_img_path = os.path.splitext(ckpt_path)[0]+f'_p_r_{self.img_size[0]}_{th}'
                    os.makedirs(vised_p_r_img_path, exist_ok=True)
                    f =  open(os.path.splitext(ckpt_path)[0]+f'eval_res_{self.img_size[0]}_{th}.txt', 'w')

                    for per_img_res in cocoEval.pre_img_info:
                        for per_cls_res in per_img_res:
                            if per_cls_res!=None:
                                image_id = per_cls_res['image_id']
                                img_name = cocoEval.cocoGt.loadImgs(int(image_id))[0]["file_name"]
                                img_r[image_id]=[[]for i in range(6)]
                                img_p[image_id]=[[]for i in range(6)]
                                category_id = per_cls_res['category_id']
                                # keep_mask = np.array(per_cls_res['dtScores'])>=0.3
                                gtMatches = per_cls_res['gtMatches']
                                dtMatches = per_cls_res['dtMatches']
                                gt_r = (gtMatches!=0).sum(0) # 匹配到的阈值个数
                                gt_p = (dtMatches!=0).sum(0)
                                img_r[image_id][category_id-1] = gt_r
                                img_p[image_id][category_id-1] = gt_p
                                #                                              匹配到的阈值个数有等于零的则True
                                r_cond = len(per_cls_res['gtIds'])*(True if gt_r.shape[0]==0 else gt_r.min()==0)
                                p_cond = len(per_cls_res['dtIds'])*(True if gt_p.shape[0]==0 else gt_p.min()==0)

                                # print(cocoEval.cocoGt.loadImgs(int(image_id))[0]["file_name"])
                                # print(cocoEval.cocoGt.cats[int(category_id)]["name"])
                                if r_cond:
                                    r_info = f'漏检 FN: {img_name} class {cocoEval.cocoGt.cats[int(category_id)]["name"]}'
                                    print(r_info)
                                    f.write(r_info+'\n')
                                    shutil.copy(os.path.join(vised_path,img_name), os.path.join(vised_p_r_img_path,img_name))
                                if p_cond:
                                    p_info = f'  误检 FP: {img_name} class {cocoEval.cocoGt.cats[int(category_id)]["name"]}'
                                    print(p_info)
                                    f.write(p_info+'\n')
                                    shutil.copy(os.path.join(vised_path,img_name), os.path.join(vised_p_r_img_path,img_name))
                    f.close()
            return list(cocoEval.stats) + cls_AP + cls_AR, info
        else:
            return 0, info

    def eval_vis(self):
        pass



