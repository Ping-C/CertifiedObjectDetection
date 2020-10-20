from __future__ import division
import math
import time
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pdb
import scipy.stats as stats
import utils.debug

def to_cpu(tensor):
    return tensor.detach().cpu()

class DetectionsAcc:
    OBJECT_SORT=0
    CENTER_SORT=1
    CORNER_SORT=2
    SINGLE_BIN=0
    LABEL_BIN=1
    LOCATION_BIN=2
    LOCATION_LABEL_BIN=3

    def __init__(self, bin=SINGLE_BIN, sort=OBJECT_SORT, loc_bin_count=None):
        self.detections_list = []
        self.max_num_detections = 0
        #count the number of classes in each class bin
        self.bin_counts = {}
        self.detections_tensor = None
        self.id_index_map = {}

        self.sort = sort
        self.bin = bin
        self.loc_bin_count = loc_bin_count
    def track(self, detections):
        #dim of detections (# of simulations, tensor((#of detections, 7)))
        self.detections_list.extend(detections)
        for detection in detections:
            if detection is not None:

                temp_count = {}
                if self.bin == DetectionsAcc.SINGLE_BIN:
                    box_count = detection.size(0)
                    if box_count > self.max_num_detections:
                        self.max_num_detections = box_count
                elif (self.bin ==  DetectionsAcc.LABEL_BIN
                      or self.bin == DetectionsAcc.LOCATION_BIN
                      or self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        # for label binning
                        ids = detection[:, -1].tolist()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        # for location binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx/416*self.loc_bin_count).floor()
                        yids = (midy/416*self.loc_bin_count).floor()
                        ids = (xids+yids*10).tolist()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        # for location+label binning
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = (xids + yids * 10 + labels * 100).tolist()

                    for id in ids:
                        if id not in temp_count:
                            temp_count[id] = 1
                        else:
                            temp_count[id] += 1
                    for id, count in temp_count.items():
                        if id not in self.bin_counts:
                            self.bin_counts[id] = count
                        elif self.bin_counts[id] < count:
                            self.bin_counts[id] = count

    def tensorize(self):
        if self.bin == DetectionsAcc.SINGLE_BIN:
            self.detection_len = self.max_num_detections
        elif (self.bin == DetectionsAcc.LABEL_BIN or
                self.bin == DetectionsAcc.LOCATION_BIN or
                self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
            self.detection_len = 0
            for id, count in self.bin_counts.items():
                self.id_index_map[id] = self.detection_len
                self.detection_len += count
        else:
            raise ValueError("Invalid bin parameter")


        self.detections_tensor = torch.ones(
            (len(self.detections_list), self.detection_len, 7)
        )*float('inf')
        # self.detections_tensor[0:len(self.detections_list)//2] *= -1
        for i, detection in enumerate(self.detections_list):
            if detection is not None:
                if self.sort == DetectionsAcc.OBJECT_SORT:
                    detection_count = detection.size(0)
                elif self.sort == DetectionsAcc.CENTER_SORT:
                    detection_count = detection.size(0)
                    midy = (detection[:, 1]+detection[:, 3])/2
                    _, sort_idx = midy.sort(dim=0)
                    detection = detection[sort_idx]
                    midx = (detection[:, 0]+detection[:, 2])/2
                    _, sort_idx = midx.sort(dim=0)
                    detection = detection[sort_idx]

                if self.bin == DetectionsAcc.SINGLE_BIN:
                    self.detections_tensor[i, 0:detection_count] = detection
                elif (self.bin == DetectionsAcc.LABEL_BIN or
                        self.bin == DetectionsAcc.LOCATION_BIN or
                        self.bin == DetectionsAcc.LOCATION_LABEL_BIN):
                    if self.bin == DetectionsAcc.LABEL_BIN:
                        ids = detection[:, -1]
                        unique_ids = detection[:, -1].unique()
                    elif self.bin == DetectionsAcc.LOCATION_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        ids = xids + yids * 10
                        unique_ids = ids.unique()
                    elif self.bin == DetectionsAcc.LOCATION_LABEL_BIN:
                        midx = (detection[:, 0] + detection[:, 2]) / 2
                        midy = (detection[:, 1] + detection[:, 3]) / 2
                        xids = (midx / 416 * self.loc_bin_count).floor()
                        yids = (midy / 416 * self.loc_bin_count).floor()
                        labels = detection[:, -1]
                        ids = xids + yids * 10 + labels * 100
                        unique_ids = ids.unique()

                    for id in unique_ids:
                        filtered_detection = detection[ids == id]
                        filtered_len = filtered_detection.size(0)
                        idx_st = self.id_index_map[id.cpu().item()]
                        self.detections_tensor[i, idx_st:idx_st+filtered_len]= filtered_detection





        self.detections_tensor, _ = self.detections_tensor.sort(dim=0)
    def median(self):
        result = self.detections_tensor[len(self.detections_list) // 2]
        return result
    def upper(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(alpha))]
        return result
    def lower(self, alpha=.05):
        result = self.detections_tensor[int(len(self.detections_list)*(1-alpha))]
        return result
    def k(self, q):
        result = self.detections_tensor[q]
        return result
    def clear(self):
        self.detections_list = []
        self.max_num_detections = 0
        self.detections_tensor = None




def load_classes(path):
    """
    Loads class labels at 'path'
    """
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

def estimated_alpha(eps, sample_count, sigma, conf_thres = .99999):
    theo_perc_u = stats.norm.cdf(eps/sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    alpha_u_u = 1
    alpha_u_l = 0
    alpha_l_u = 1
    alpha_l_l = 0
    for i in range(30):
        alpha_u_m = (alpha_u_u + alpha_u_l)/2
        theo_test_perc_u, theo_test_perc_l = emp_perc_to_theo_perc(alpha_u_m, sample_count, conf_thres=conf_thres)
        if theo_perc_u <= theo_test_perc_l:
            alpha_u_u = alpha_u_m
        else:
            alpha_u_l = alpha_u_m

        alpha_l_m = (alpha_l_u + alpha_l_l) / 2
        theo_test_perc_u, theo_test_perc_l = emp_perc_to_theo_perc(alpha_l_m, sample_count, conf_thres=conf_thres)
        if theo_perc_l >= theo_test_perc_u:
            alpha_l_l = alpha_l_m
        else:
            alpha_l_u = alpha_l_m
    return alpha_u_u, alpha_l_l

import math

def estimated_qu_ql(eps, sample_count, sigma, conf_thres = .99999):
    theo_perc_u = stats.norm.cdf(eps/sigma)
    theo_perc_l = stats.norm.cdf(-eps / sigma)

    q_u_u = sample_count + 1
    q_u_l = math.ceil(theo_perc_u*sample_count)
    q_l_u = math.floor(theo_perc_l*sample_count)
    q_l_l = 0
    q_u_final = q_u_u
    for q_u in range(q_u_l, q_u_u):
        conf = stats.binom.cdf(q_u-1, sample_count, theo_perc_u)
        if conf > conf_thres:
            q_u_final = q_u
            break

    q_l_final = q_l_l
    for q_l in range(q_l_u, q_l_l, -1):
        conf = 1-stats.binom.cdf(q_l-1, sample_count, theo_perc_l)
        if conf > conf_thres:
            q_l_final = q_l
            break

    return q_u_final, q_l_final

def emp_perc_to_theo_perc(emp_perc, sample_count, conf_thres=.9999):
    theo_perc_u_l = 0
    theo_perc_u_u = 1
    for i in range(30):
        perc_test = (theo_perc_u_u+theo_perc_u_l)/2
        conf = 1-stats.binom.cdf(int(sample_count*emp_perc), sample_count, perc_test)
        if conf > conf_thres:
            theo_perc_u_u = perc_test
        else:
            theo_perc_u_l = perc_test

    theo_perc_l_l = 0
    theo_perc_l_u = 1
    for i in range(30):
        perc_test = (theo_perc_l_u+theo_perc_l_l)/2
        conf = stats.binom.cdf(int(sample_count*emp_perc), sample_count, perc_test)
        if conf > conf_thres:
            theo_perc_l_l = perc_test
        else:
            theo_perc_l_u = perc_test

    return theo_perc_u_u, theo_perc_l_l


def rescale_boxes(boxes, current_dim, original_shape):
    """ Rescales bounding boxes to the original shape """
    orig_h, orig_w = original_shape
    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (current_dim / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (current_dim / max(original_shape))
    # Image height and width after padding is removed
    unpad_h = current_dim - pad_y
    unpad_w = current_dim - pad_x
    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h
    return boxes


def xywh2xyxy(x):
    y = x.new(x.shape)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y


def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in tqdm.tqdm(unique_classes, desc="Computing AP"):
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

def pr_overall(tp, target_cls):
    # overall precision/recall
    total_gt = len(target_cls)
    total_pred = len(tp)
    total_tp = tp.sum()
    total_recall = total_tp / total_gt
    total_precision = total_tp / total_pred
    total_f1 = 2 * total_precision * total_recall / (total_precision + total_recall + 1e-16)
    return total_precision, total_recall, total_f1


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def get_batch_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs)):
        if outputs[sample_i] is None:
            continue
        output = outputs[sample_i]
        filter = ((abs(output) == float('inf')).sum(dim=1)==0)
        pred_boxes = output[filter, :4]
        pred_scores = output[filter, 4]
        pred_labels = output[filter, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue
                iou = bbox_iou(pred_box.unsqueeze(0), target_boxes)
                iou_filtered = torch.where(target_labels==pred_label, iou, -torch.ones_like(iou))
                iou, box_index = iou_filtered.max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives, pred_scores, pred_labels])
    return batch_metrics

def get_batch_statistics_worst(outputs, outputs_u, outputs_l, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    batch_metrics = []
    for sample_i in range(len(outputs_u)):

        if outputs_u[sample_i] is None:
            continue
        output = outputs[sample_i]
        output_u = outputs_u[sample_i]
        output_l = outputs_l[sample_i]




        #output.shape = (# predictions, 7)
        max_pred_filter = (
                (
                        1
                        -((output_u == float('inf')) & (output_l == float('inf'))).float()
                        -((output_u == -float('inf')) & (output_l == -float('inf'))).float()
                ).sum(dim=1) == 7
        )

        pred_boxes = output[max_pred_filter, :4]
        pred_scores = output[max_pred_filter, 4]
        pred_labels = output[max_pred_filter, -1]
        pred_boxes_u = output_u[max_pred_filter, :4]
        pred_scores_u = output_u[max_pred_filter, 4]
        pred_labels_u = output_u[max_pred_filter, -1]
        pred_boxes_l = output_l[max_pred_filter, :4]
        pred_scores_l = output_l[max_pred_filter, 4]
        pred_labels_l = output_l[max_pred_filter, -1]

        true_positives_worst = np.zeros(pred_boxes_u.shape[0])

        annotations = targets[targets[:, 0] == sample_i][:, 1:]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box_u, pred_label_u,
                         pred_box_l, pred_label_l) in enumerate(zip(pred_boxes_u, pred_labels_u,
                                                                pred_boxes_l, pred_labels_l)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore prediction if the prediction if even the label is not certified
                if pred_label_u != pred_label_l or pred_label_u not in target_labels:
                    continue
                # Ignore prediction if the bounding boxes are unbounded
                if (abs(pred_box_u) == float('inf')).sum()>0 or (abs(pred_box_l) == float('inf')).sum()>0:
                    continue

                #worst case iou
                #worst case label
                valid_labels = (target_labels == pred_label_u)
                iou_worst = bbox_iou_worst(pred_box_u.unsqueeze(0),
                                           pred_box_l.unsqueeze(0),
                                           target_boxes)
                iou_worst_flip =  bbox_iou_worst(pred_box_l.unsqueeze(0),
                                           pred_box_u.unsqueeze(0),
                                           target_boxes)
                if (iou_worst_flip-iou_worst).abs().sum()>0:
                    pdb.set_trace()
                iou_filtered_worst = torch.where(valid_labels, iou_worst, -torch.ones_like(iou_worst))
                iou, box_index = iou_filtered_worst.max(0)
                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives_worst[pred_i] = 1
                    detected_boxes += [box_index]
        batch_metrics.append([true_positives_worst, pred_labels])
    return batch_metrics


def bbox_wh_iou(wh1, wh2):
    wh2 = wh2.t()
    w1, h1 = wh1[0], wh1[1]
    w2, h2 = wh2[0], wh2[1]
    inter_area = torch.min(w1, w2) * torch.min(h1, h2)
    union_area = (w1 * h1 + 1e-16) + w2 * h2 - inter_area
    return inter_area / union_area


def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def bbox_iou_worst(box1_u, box1_l, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        raise NotImplementedError()
    else:
        if len(box1_u) != 1 or len(box1_l) != 1:
            raise NotImplementedError("It can only handle single box input for box1_u & box1_l")
        # Get the coordinates of bounding boxes
        b1_u_x1, b1_u_y1, b1_u_x2, b1_u_y2 = box1_u[:, 0], box1_u[:, 1], box1_u[:, 2], box1_u[:, 3]
        b1_l_x1, b1_l_y1, b1_l_x2, b1_l_y2 = box1_l[:, 0], box1_l[:, 1], box1_l[:, 2], box1_l[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    iou_worst = torch.ones((box2.size(0))).to(box2.device)*float('inf')
    for b1_x1 in (b1_u_x1, b1_l_x1):
        for b1_x2 in (b1_u_x2, b1_l_x2):
            for b1_y1 in (b1_u_y1, b1_l_y1):
                for b1_y2 in (b1_u_y2, b1_l_y2):
                    # b1_x1_worst = b2_x1.clamp(min=b1_l_x1[0], max=b1_u_x1[0])
                    # b1_y1_worst = b2_y1.clamp(min=b1_l_y1[0], max=b1_u_y1[0])
                    # b1_x2_worst = b2_x2.clamp(min=b1_l_x2[0], max=b1_u_x2[0])
                    # b1_y2_worst = b2_y2.clamp(min=b1_l_y2[0], max=b1_u_y2[0])
                    # get the corrdinates of the intersection rectangle
                    inter_rect_x1 = torch.max(b1_x1, b2_x1)
                    inter_rect_y1 = torch.max(b1_y1, b2_y1)
                    inter_rect_x2 = torch.min(b1_x2, b2_x2)
                    inter_rect_y2 = torch.min(b1_y2, b2_y2)
                    # Intersection area
                    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
                        inter_rect_y2 - inter_rect_y1 + 1, min=0
                    )
                    # Union Area
                    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
                    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)
                    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)
                    iou_worst = torch.where(iou_worst < iou, iou_worst, iou)
    return iou_worst


def non_max_suppression(prediction, conf_thres=0.5, nms_thres=0.4, others=None, x1y1x2y2=False):
    """
    Removes detections with lower object confidence score than 'conf_thres' and performs
    Non-Maximum Suppression to further filter detections.
    Returns detections with shape:
        (x1, y1, x2, y2, object_conf, class_score, class_pred)
    """

    # From (center x, center y, width, height) to (x1, y1, x2, y2)
    if not x1y1x2y2:
        prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [None for _ in range(len(prediction))]
    if others is not None:
        others_output = [[None for _ in range(len(prediction))] for _ in range(len(others))]
        temp = [[None for _ in range(len(prediction))] for _ in range(len(others))]
    else:
        others_output = None


    for image_i, image_pred in enumerate(prediction):
        # Filter out confidence scores below threshold
        conf_filter = (image_pred[:, 4] >= conf_thres)
        image_pred = image_pred[conf_filter]
        if others is not None:
            for k, other in enumerate(others):
                temp[k][image_i] = other[image_i][conf_filter]

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue

        # Object confidence times class confidence
        score = image_pred[:, 4] * image_pred[:, 5:].max(1)[0]

        # Sort by it
        image_pred = image_pred[(-score).argsort()]
        if others is not None:
            for k in range(len(temp)):
                temp[k][image_i] = temp[k][image_i][(-score).argsort()]

        class_confs, class_preds = image_pred[:, 5:].max(1, keepdim=True)
        detections = torch.cat((image_pred[:, :5], class_confs.float(), class_preds.float()), 1)
        if others is not None:
            for k in range(len(temp)):
                temp[k][image_i] = torch.cat((temp[k][image_i][:, :5], class_confs.float(), class_preds.float()), 1)

        # Perform non-maximum suppression
        keep_boxes = []
        others_keep_boxes = []
        if others is not None:
            for k, other in enumerate(others):
                others_keep_boxes.append([])
        while detections.size(0):
            large_overlap = bbox_iou(detections[0, :4].unsqueeze(0), detections[:, :4]) > nms_thres
            label_match = detections[0, -1] == detections[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            # weights = detections[invalid, 4:5]
            # # Merge overlapping bboxes by order of confidence
            # detections[0, :4] = (weights * detections[invalid, :4]).sum(0) / weights.sum()
            keep_boxes += [detections[0]]
            detections = detections[~invalid]
            if others is not None:
                for k in range(len(temp)):
                    others_keep_boxes[k] += [temp[k][image_i][0]]
                    temp[k][image_i] = temp[k][image_i][~invalid]
        if keep_boxes:
            output[image_i] = torch.stack(keep_boxes)
            if others is not None:
                for k in range(len(temp)):
                    others_output[k][image_i] = torch.stack(others_keep_boxes[k])

    return output, others_output


def build_targets(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)

    # Output tensors
    obj_mask = ByteTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = ByteTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    tcls[b, best_n, gj, gi, target_labels] = 1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf

def build_targets_adv(pred_boxes, pred_cls, target, anchors, ignore_thres):

    ByteTensor = torch.cuda.ByteTensor if pred_boxes.is_cuda else torch.ByteTensor
    BoolTensor = torch.cuda.BoolTensor if pred_boxes.is_cuda else torch.BoolTensor
    FloatTensor = torch.cuda.FloatTensor if pred_boxes.is_cuda else torch.FloatTensor

    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nC = pred_cls.size(-1)
    nG = pred_boxes.size(2)
    # Output tensors
    obj_mask = BoolTensor(nB, nA, nG, nG).fill_(0)
    noobj_mask = BoolTensor(nB, nA, nG, nG).fill_(1)
    class_mask = FloatTensor(nB, nA, nG, nG).fill_(0)
    iou_scores = FloatTensor(nB, nA, nG, nG).fill_(0)
    tx = FloatTensor(nB, nA, nG, nG).fill_(0)
    ty = FloatTensor(nB, nA, nG, nG).fill_(0)
    tw = FloatTensor(nB, nA, nG, nG).fill_(0)
    th = FloatTensor(nB, nA, nG, nG).fill_(0)
    tcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)
    advcls = FloatTensor(nB, nA, nG, nG, nC).fill_(0)

    # Convert to position relative to box
    target_boxes = target[:, 2:6] * nG
    gxy = target_boxes[:, :2]
    gwh = target_boxes[:, 2:]
    # Get anchors with best iou
    ious = torch.stack([bbox_wh_iou(anchor, gwh) for anchor in anchors])
    best_ious, best_n = ious.max(0)
    # Separate target values
    b, target_labels = target[:, :2].long().t()
    gx, gy = gxy.t()
    gw, gh = gwh.t()
    gi, gj = gxy.long().t()
    # Set masks
    obj_mask[b, best_n, gj, gi] = 1
    noobj_mask[b, best_n, gj, gi] = 0

    # Set noobj mask to zero where iou exceeds ignore threshold
    for i, anchor_ious in enumerate(ious.t()):
        noobj_mask[b[i], anchor_ious > ignore_thres, gj[i], gi[i]] = 0

    # Coordinates
    tx[b, best_n, gj, gi] = gx - gx.floor()
    ty[b, best_n, gj, gi] = gy - gy.floor()
    # Width and height
    tw[b, best_n, gj, gi] = torch.log(gw / anchors[best_n][:, 0] + 1e-16)
    th[b, best_n, gj, gi] = torch.log(gh / anchors[best_n][:, 1] + 1e-16)
    # One-hot encoding of label
    weight = torch.ones((target_labels.shape[0], 80))
    weight[torch.arange(target_labels.shape[0]), target_labels] = 0
    adv_labels = torch.multinomial(weight, num_samples=1).squeeze()
    # advcls[b, best_n, gj, gi, adv_labels] = 1
    tcls[b, best_n, gj, gi, target_labels] = 1
    advcls = (tcls-1)*-1
    # Compute label correctness and iou at best anchor
    class_mask[b, best_n, gj, gi] = (pred_cls[b, best_n, gj, gi].argmax(-1) == target_labels).float()
    iou_scores[b, best_n, gj, gi] = bbox_iou(pred_boxes[b, best_n, gj, gi], target_boxes, x1y1x2y2=False)

    tconf = obj_mask.float()
    return iou_scores, class_mask, obj_mask, noobj_mask, tx, ty, tw, th, tcls, tconf, advcls

