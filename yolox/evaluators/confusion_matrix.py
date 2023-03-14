# from mmcv.ops import nms
# from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
# import mmcv
import numpy as np


def compute_iou(groundtruth_box, detection_box):
    '''兼容xyxy格式'''
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def c_m(
    dataset,
    results,
    score_thr=0,
    nms_iou_thr=None,
    tp_iou_thr=0.5):
    '''
    https://github.com/svpino/tf_object_detection_cm/blob/master/confusion_matrix.py
    https://github.com/kaanakan/object_detection_confusion_matrix/blob/master/confusion_matrix.py
    '''

    num_classes = len(dataset._classes)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)

    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res
        if len(res_bboxes) == 0: continue
        pred_bboxes, pred_scores = np.hsplit(np.vstack((res_bboxes)), [4])
        pred_scores=pred_scores[:,0]

        pred_classes = []
        for i, cls_boxes in enumerate(res_bboxes):
            if len(cls_boxes)>0:
                pred_classes.extend([i]*len(cls_boxes))
        # pred_classes = np.array(pred_classes).reshape(len(pred_scores), 1)
        pred_classes = np.array(pred_classes)

        CONFIDENCE_THRESHOLD, IOU_THRESHOLD = score_thr, tp_iou_thr
        detection_scores = pred_scores
        # detection_classes = pred_classes
        # detection_boxes = pred_bboxes
        detection_classes = pred_classes[detection_scores >= CONFIDENCE_THRESHOLD]
        detection_boxes = pred_bboxes[detection_scores >= CONFIDENCE_THRESHOLD]

        img_bboxes, (ow, oh), (iw, ih), img_name = dataset.annotations[idx]
        gt_bboxes = np.zeros_like(img_bboxes[:,:4])
        gt_bboxes[:, [0,2]] = img_bboxes[:,[0,2]]*(ow/iw)
        gt_bboxes[:, [1,3]] = img_bboxes[:,[1,3]]*(oh/ih)
        labels = img_bboxes[:,4]
        groundtruth_boxes, groundtruth_classes = gt_bboxes, labels

        matches = []

        # if idx % 100 == 0:
        #     print("Processed %d images" % (idx))

        for i in range(len(groundtruth_boxes)):
            for j in range(len(detection_boxes)):
                # iou = compute_iou(groundtruth_boxes[i,[1,0,3,2]], detection_boxes[j,[1,0,3,2]])
                iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                if iou > IOU_THRESHOLD:
                    matches.append([i, j, iou])

        matches = np.array(matches)
        if matches.shape[0] > 0:
            # Sort list of matches by descending IOU so we can remove duplicate detections
            # while keeping the highest IOU entry.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate detections from the list.
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]

            # Sort the list again by descending IOU. Removing duplicates doesn't preserve
            # our previous sort.
            matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

            # Remove duplicate ground truths from the list.
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]

        for i in range(len(groundtruth_boxes)):
            if matches.shape[0] > 0 and matches[matches[:, 0] == i].shape[0] == 1: # TP and error classes FP
                detection_class = int(detection_classes[int(matches[matches[:, 0] == i, 1][0])])
                confusion_matrix[int(groundtruth_classes[i])][detection_class] += 1
                # confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:, 0] == i, 1][0])] - 1] += 1
            else: # error classes FN
                confusion_matrix[int(groundtruth_classes[i])][-1] += 1

        for i in range(len(detection_boxes)): # background FN
            if matches.shape[0] > 0 and matches[matches[:, 1] == i].shape[0] == 0:
                confusion_matrix[-1][int(detection_classes[i])] += 1
        # else:
        #     print("Skipped image %d" % (idx))

    return confusion_matrix

def f1(c_m):
    ep = 1e-8
    assert c_m.shape[0] == c_m.shape[1]
    cls = c_m.shape[0]
    res = np.zeros((cls + 2, cls + 3))
    res[:cls, :cls] = c_m

    res[:cls, -3] = c_m.sum(1)  # gt num
    res[-2, :cls] = c_m.sum(0)  # pred num
    res[-2, -3] = res[:cls - 1, -3].sum()  # T
    cls_TP = np.diagonal(res)[:cls-1]  # every cls TP (exclude background)
    res[-2, -2] = cls_TP.sum()  # total cls TP

    res[-1, -2] = res[-2, -2]/(res[-2, -3] + ep)  # accuracy
    res[-1, :cls-1] = cls_TP / (res[-2, :cls-1] + ep)  # precision
    res[:cls - 1, -2] = cls_TP / (res[:cls-1, -3] + ep)  # recall

    res[:cls - 1, -1] = 2*res[-1, :cls-1]*res[:cls - 1, -2]/ (res[-1, :cls-1]+res[:cls - 1, -2]+ep)  # cls F1
    res[-1, -1] = res[:cls - 1, -1].mean()

    return np.around(res, 5)






'''
基于mmdet的代码所做
'''
'''
def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset. <class 'pycocotools.coco.COCO'>
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    info = np.zeros([1, 5])
    # img_id_ann = {}
    # for i in dataset.dataset['annotations']:
    #     if img_id_ann[i['image_id']] == None:
    #         img_id_ann[i['image_id']] = []
    #     img_id_ann[i['image_id']].append(i['bbox'] + i['category_id'])
    num_classes = len(dataset._classes)
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    assert len(dataset) == len(results)
    # prog_bar = mmcv.ProgressBar(len(results))
    for idx, per_img_res in enumerate(results):
        if isinstance(per_img_res, tuple):
            res_bboxes, _ = per_img_res
        else:
            res_bboxes = per_img_res

        img_bboxes, (ow, oh), (iw, ih), img_name = dataset.annotations[idx]
        gt_bboxes = np.zeros_like(img_bboxes[:,:4])
        gt_bboxes[:, [0,2]] = img_bboxes[:,[0,2]]*(ow/iw)
        gt_bboxes[:, [1,3]] = img_bboxes[:,[1,3]]*(oh/ih)

        labels = img_bboxes[:,4]
        analyze_per_img_dets(confusion_matrix, gt_bboxes, labels, res_bboxes,
                             score_thr, tp_iou_thr, nms_iou_thr)
        # prog_bar.update()
    return confusion_matrix



def analyze_per_img_dets(confusion_matrix,
                         gt_bboxes,
                         gt_labels,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gt_bboxes (ndarray): Ground truth bboxes, has shape (num_gt, 4).
        gt_labels (ndarray): Ground truth labels, has shape (num_gt).
        result (ndarray): Detection results, has shape
            (num_classes, num_bboxes, 5).
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    true_positives = np.zeros_like(gt_labels)
    for det_label, det_bboxes in enumerate(result):
        # if nms_iou_thr:
        #     det_bboxes, _ = nms(
        #         det_bboxes[:, :4],
        #         det_bboxes[:, -1],
        #         nms_iou_thr,
        #         score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, det_bbox in enumerate(det_bboxes):
            score = det_bbox[4]
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[int(gt_label), int(det_label)] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[int(gt_label), -1] += 1
'''