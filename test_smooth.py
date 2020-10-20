from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from utils import debug

from code.architectures import get_architecture, IMAGENET_CLASSIFIERS

import os
import sys
import time
import datetime
import argparse
import tqdm
import json
import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import pdb

def evaluate(model, path, iou_thres, img_size, batch_size, test_count, start_count,
             smooth, smooth_count, smooth_batch_size, sigma, q_u, q_l, bin, sort, loc_bin_count=None, attack=False):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    if smooth:
        if bin == "single":
            bin = DetectionsAcc.SINGLE_BIN
        elif bin == "label":
            bin = DetectionsAcc.LABEL_BIN
        elif bin == "location":
            bin = DetectionsAcc.LOCATION_BIN
        elif bin == "location+label":
            bin = DetectionsAcc.LOCATION_LABEL_BIN
        else:
            raise ValueError("invalid binning option")

        if sort == "object":
            sort = DetectionsAcc.OBJECT_SORT
        elif sort == "center":
            sort = DetectionsAcc.CENTER_SORT
        else:
            raise ValueError("invalid sort option")

        accumulator = DetectionsAcc(bin=bin, sort=sort, loc_bin_count=loc_bin_count)
        smoothed_model = SmoothMedianNMS(model, sigma, accumulator)
        sample_metrics_smooth = []  # List of tuples (TP, pred)
    total_count = 0
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects", total=test_count+start_count)):
        if total_count >= test_count + start_count:
            break
        if total_count < start_count:
            total_count += len(imgs)
            continue
        # Extract labels
        labels += targets[:, 1].tolist()


        imgs = Variable(imgs.type(Tensor), requires_grad=False)
        if attack:
            ori_img = Variable(imgs.type(Tensor), requires_grad=False)
            adv_img = Variable(imgs.clone().detach().type(Tensor), requires_grad=True)
            targets_clone = Variable(targets.clone().to(device), requires_grad=False)

            attack_sample = 5
            first_idx = torch.arange(attack_sample).repeat_interleave(targets_clone.shape[0])
            targets_clone = targets_clone.repeat(attack_sample, 1)
            targets_clone[:, 0] = first_idx

            opt = optim.Adam([adv_img], lr=.001)
            radius = 0.36
            for i in range(20):
                noise = torch.randn_like(adv_img.repeat(attack_sample,1,1,1), requires_grad=False) * sigma
                # adv_loss = model[0].adv_loss(adv_img+noise, targets_clone)#/5
                adv_loss = model[1][0].adv_loss(model[0](adv_img+noise), targets_clone)
                # adv_loss = model[0].adv_loss(adv_img, targets_clone)
                opt.zero_grad()
                adv_loss.backward()
                adv_img.data -= adv_img.grad/adv_img.grad.view(adv_img.shape[0], -1).norm(dim=1)*.2*radius
                # opt.step()
                with torch.no_grad():
                    diff_ori = (adv_img-ori_img)
                    diff = diff_ori.view(diff_ori.shape[0], -1)
                    norm = diff.norm(dim=1)
                    div = torch.where(norm>radius, norm/radius, torch.ones_like(norm))
                    adv_img.data = diff_ori/div[:, None, None, None] + ori_img
            imgs = adv_img.clone().detach().requires_grad_(False)
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size
        with torch.no_grad():
            if smooth:
                outputs, outputs_l, outputs_u = smoothed_model.predict_range(
                    imgs, n=smooth_count, batch_size=smooth_batch_size, q_u=q_u, q_l=q_l)
                #outputs.dim (# of images per batch, # of detections, 7)
                #outputs sometimes would contain infinite predictions, that means that even though one of the entries
                # would be used at some percentile of the distribution, but the # of predictions in the base classifier
                # may not be enough to make it into the median/upper bound/lower bound
                sample_metrics_smooth += get_batch_statistics_worst(outputs, outputs_u, outputs_l, targets, iou_threshold=iou_thres)

            else:
                outputs = model(imgs)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)
        total_count += len(imgs)


    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    if smooth:
        true_positives_worst, pred_labels_worst = [np.concatenate(x, 0) for x in list(zip(*sample_metrics_smooth))]
        precision_all_worst, recall_all_worst, f1_all_worst = pr_overall(true_positives_worst, labels)

        print("min correct", sum(true_positives_worst))
        print("max predict", len(true_positives_worst))
        print("total correct", sum(true_positives))
        print("total predict", len(true_positives))
        print("total ground truth", len(labels))
    else:
        precision_all_worst, recall_all_worst, f1_all_worst = None, None, None
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
    precision_all, recall_all, f1_all = pr_overall(true_positives, labels)
    return precision, recall, AP, f1, ap_class, \
           precision_all, recall_all, f1_all, \
           precision_all_worst, recall_all_worst, f1_all_worst


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="size of each image batch")

    parser.add_argument("--model_type", type=str, default="yolo", choices=["yolo", "faster_rcnn", "mask_rcnn"], help="types of model")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--data_config", type=str, default="config/coco.data", help="path to data config file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--iou_thres", type=float, default=0.5, help="iou threshold required to qualify as detected")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    parser.add_argument("--test_count", type=int, default=5000, help="sample used for evaluation max is 5000")
    parser.add_argument("--start_count", type=int, default=0, help="start count for evaluation")
    parser.add_argument("--smooth", action='store_true', help="use smoothing classifier")
    parser.add_argument("--smooth_count", type=int, default=2000, help="number of samples used to estimate the smooth classifier")
    parser.add_argument("--smooth_batch_size", type=int, default=20, help="batchsize when estimating smooth classifer")
    parser.add_argument("--cert_conf", type=float, default=.99999, help="confidence of certificate")
    parser.add_argument("--sigma", type=float, default=.25, help="sigma for the normal noise")
    parser.add_argument("--eps", type=float, default=.36, help="radius that we try to certify")
    parser.add_argument("--denoise", action='store_true', help="denoise image after smoothing")
    parser.add_argument("--bin", default="single", help="binning method")
    parser.add_argument("--loc_bin_count", type=int, default=3, help="binning count for location binning")
    parser.add_argument("--sort", default="object", help="sorting method")
    parser.add_argument("--attack", action='store_true', help="generate attack against the object detector")

    parser.add_argument("--seed", type=int, default=0, help="random seed")
    opt = parser.parse_args()
    print(json.dumps(vars(opt), indent=4))

    torch.manual_seed(opt.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_config = parse_data_config(opt.data_config)
    valid_path = data_config["valid"]
    class_names = load_classes(data_config["names"])

    # Get the empirical order statistics that should be used
    if opt.smooth:
        cert_conf = opt.cert_conf
        q_u, q_l = estimated_qu_ql(opt.eps, opt.smooth_count, opt.sigma, conf_thres=cert_conf)
        print(f"Certified Eps (with {cert_conf:6.6%} confidence): {opt.eps: 0.2f}")
        print(f"q_u:{q_u}, q_l:{q_l}")
    else:
        cert_conf = None
        q_u = None
        q_l = None


    # Initialize models
    if opt.model_type == "yolo":
        model = Darknet(opt.model_def).to(device)
        if opt.weights_path.endswith(".weights"):
            # Load darknet weights
            model.load_darknet_weights(opt.weights_path)
        else:
            # Load checkpoint weights
            model.load_state_dict(torch.load(opt.weights_path))
        model = torch.nn.Sequential(model, NMSModule(opt.conf_thres, opt.nms_thres))
    elif opt.model_type == "faster_rcnn":
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).to(device)
        model.roi_heads.score_thresh = opt.conf_thres
        model.roi_heads.nms_thresh = opt.nms_thres
        model = torch.nn.Sequential(model, Concat())
    elif opt.model_type == "mask_rcnn":
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).to(device)
        model.roi_heads.score_thresh = opt.conf_thres
        model.roi_heads.nms_thresh = opt.nms_thres
        model = torch.nn.Sequential(model, Concat())

    if opt.denoise:
        checkpoint = torch.load("pretrained_models/trained_denoisers/imagenet/mse_obj/dncnn_5epoch_lr1e-4/noise_0.25/checkpoint.pth.tar")
        denoiser = get_architecture("imagenet_dncnn", "imagenet")
        denoiser.load_state_dict(checkpoint['state_dict'])
        model = torch.nn.Sequential(denoiser, model)

    print("Compute mAP...")

    precision, recall, AP, f1, ap_class, \
    precision_all, recall_all, f1_all, \
    precision_all_worst, recall_all_worst, f1_all_worst = evaluate(
        model,
        path=valid_path,
        iou_thres=opt.iou_thres,
        img_size=opt.img_size,
        batch_size=opt.batch_size,
        test_count=opt.test_count,
        start_count=opt.start_count,
        smooth=opt.smooth,
        smooth_count=opt.smooth_count,
        smooth_batch_size=opt.smooth_batch_size,
        sigma=opt.sigma,
        q_u=q_u,
        q_l=q_l,
        sort=opt.sort,
        bin=opt.bin,
        loc_bin_count=opt.loc_bin_count,
        attack=opt.attack
    )

    print("Average Precisions:")
    for i, c in enumerate(ap_class):
        print(f"+ Class '{c}' ({class_names[c]}) - AP: {AP[i]} Precision: {precision[i]} Recall: {recall[i]} f1: {f1[i]}")

    print(f"mAP: {AP.mean()}")
    print(f"mean Precision: {precision.mean()}")
    print(f"mean Recall: {recall.mean()}")
    print(f"mean F1: {f1.mean()}")
    print(f"overall Precision: {precision_all} / {precision_all_worst}")
    print(f"overall Recall: {recall_all} / {recall_all_worst}")
    print(f"overall F1: {f1_all} / {f1_all_worst}")
