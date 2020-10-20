from __future__ import division

from models import *
from utils.utils import *
from utils import debug
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from code.architectures import get_architecture, IMAGENET_CLASSIFIERS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")

    #smoothing parameters
    parser.add_argument("--denoise", action='store_true', help="denoise image after smoothing")
    parser.add_argument("--smooth_count", type=int, default=2000, help="number of samples for estimating the smoothed classifier")
    parser.add_argument("--bin", default="location+label", choices=["location", "label", "location+label"],help="binning method")
    parser.add_argument("--loc_bin_count", type=int, default=3, help="binning count for location binning")
    parser.add_argument("--sort", default="center", choices=["object", "center"], help="sorting method")
    parser.add_argument("--cert_conf", type=float, default=.99999, help="confidence of certificate")
    parser.add_argument("--eps", type=float, default=.36, help="radius for certificate")
    parser.add_argument("--sigma", type=float, default=.25, help="noise added to images")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

    if opt.weights_path.endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights(opt.weights_path)
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load(opt.weights_path))

    if opt.denoise:
        checkpoint = torch.load("pretrained_models/trained_denoisers/imagenet/mse_obj/dncnn_5epoch_lr1e-4/noise_0.25/checkpoint.pth.tar")
        denoiser = get_architecture("imagenet_dncnn", "imagenet")
        denoiser.load_state_dict(checkpoint['state_dict'])
        model = torch.nn.Sequential(denoiser, model)

    model = torch.nn.Sequential(model, NMSModule(opt.conf_thres, opt.nms_thres))

    q_u, q_l = estimated_qu_ql(opt.eps, opt.smooth_count, opt.sigma, conf_thres=opt.cert_conf)

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index
    img_detections_l = []
    img_detections_u = []
    print("\nPerforming object detection:")
    prev_time = time.time()
    if opt.bin == "single":
        bin = DetectionsAcc.SINGLE_BIN
    elif opt.bin == "label":
        bin = DetectionsAcc.LABEL_BIN
    elif opt.bin == "location":
        bin = DetectionsAcc.LOCATION_BIN
    elif opt.bin == "location+label":
        bin = DetectionsAcc.LOCATION_LABEL_BIN
    else:
        raise ValueError("invalid binning option")

    if opt.sort == "object":
        sort = DetectionsAcc.OBJECT_SORT
    elif opt.sort == "center":
        sort = DetectionsAcc.CENTER_SORT
    else:
        raise ValueError("invalid sort option")

    accumulator = DetectionsAcc(bin=bin, sort = sort, loc_bin_count=opt.loc_bin_count)
    smoothed_model = SmoothMedianNMS(model, opt.sigma, accumulator)
    for batch_i, (img_paths, input_imgs) in enumerate(dataloader):
        if input_imgs.shape[0] != 1:
            raise ValueError("input_imgs but have size 1")
        detections, detections_l, detections_u = smoothed_model.predict_range(
            input_imgs.type(Tensor), n=opt.smooth_count, batch_size=20, q_u=q_u, q_l=q_l)
        # Log progress
        current_time = time.time()
        inference_time = datetime.timedelta(seconds=current_time - prev_time)
        prev_time = current_time
        print("\t+ Batch %d, Inference Time: %s" % (batch_i, inference_time))

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)
        img_detections_l.extend(detections_l)
        img_detections_u.extend(detections_u)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 200)]

    print("\nSaving images:")
    # Iterate through images and save plot of detections
    for img_i, (path, detections, detections_l, detections_u) in enumerate(zip(imgs, img_detections, img_detections_l, img_detections_u)):

        print("(%d) Image: '%s'" % (img_i, path))

        # Create plot
        img = np.array(Image.open(path))
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Draw bounding boxes and labels of detections
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, img.shape[:2])
            detections_l = rescale_boxes(detections_l, opt.img_size, img.shape[:2])
            detections_u = rescale_boxes(detections_u, opt.img_size, img.shape[:2])
            unique_labels = torch.cat(
                (detections[:, -1].cpu().unique(),
                 detections_l[:, -1].cpu().unique(),
                 detections_u[:, -1].cpu().unique()), dim=0).unique()
            n_cls_preds = len(unique_labels)
            bbox_colors = random.sample(colors, n_cls_preds)
            for detect_i, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(detections):
                x1_l, y1_l, x2_l, y2_l, conf_l, cls_conf_l, cls_pred_l = detections_l[detect_i]
                x1_u, y1_u, x2_u, y2_u, conf_u, cls_conf_u, cls_pred_u = detections_u[detect_i]
                med_flag = (
                        abs(x1) != float('inf') and abs(y1) != float('inf') and abs(x2) != float('inf') and abs(y2) != float('inf')
                        and abs(conf) != float('inf') and abs(cls_conf) != float('inf') and abs(cls_pred) != float('inf')
                )
                low_flag = (
                        abs(x1_l) != float('inf') and abs(y1_l) != float('inf') and abs(x2_l) != float('inf') and abs(y2_l) != float('inf')
                        and abs(conf_l) != float('inf') and abs(cls_conf_l) != float('inf') and abs(cls_pred_l) != float('inf')
                )
                up_flag = (
                        abs(x1_u) != float('inf') and abs(y1_u) != float('inf') and abs(x2_u) != float('inf') and abs(y2_u) != float('inf')
                        and abs(conf_u) != float('inf') and abs(cls_conf_u) != float('inf') and abs(cls_pred_u) != float('inf')
                )
                bound_overlap_flag = (x1_u>x2_l) or (y1_u>y2_l)
                if med_flag:
                    print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
                    box_w = x2 - x1
                    box_h = y2 - y1
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor=color, facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(
                        x1,
                        y1,
                        s=classes[int(cls_pred)],
                        color="white",
                        verticalalignment="top",
                        bbox={"color": color, "pad": 0},
                    )
                if low_flag and up_flag and not bound_overlap_flag:
                    # Create a Outer Rectangle
                    box_w = x2_u - x1_l
                    box_h = y2_u - y1_l
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred_l))[0])]
                    bbox = patches.Rectangle((x1_l, y1_l), box_w, box_h, linewidth=2, edgecolor=color,
                                             facecolor="none", linestyle="--")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)

                    box_w = x2_l - x1_u
                    box_h = y2_l - y1_u
                    color = bbox_colors[int(np.where(unique_labels == int(cls_pred_l))[0])]
                    # Create an Inner Rectangle patch
                    bbox = patches.Rectangle((x1_u, y1_u), box_w, box_h, linewidth=2, edgecolor=color,
                                             facecolor="none", linestyle="--")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)

                #draw arrows
                if med_flag:
                    if up_flag and low_flag and not bound_overlap_flag:
                        pass
                        # plt.arrow(x1, y1, (x1_u-x1), (y1_u-y1), width=3, head_width=15, head_length=10, zorder=10)
                        # plt.arrow(x1, y1, (x1_l-x1), (y1_l-y1), width=3, head_width=15, head_length=10, zorder=10)
                        # plt.arrow(x2, y2, (x2_u-x2), (y2_u-y2), width=3,head_width=15, head_length=10,zorder=10)
                        # plt.arrow(x2, y2, (x2_l-x2), (y2_l-y2), width=3, head_width=15, head_length=10, zorder=10)
                    else:
                        plt.scatter(x1, y2, s=300, c='red', marker='X', zorder=10)
                        plt.scatter(x2, y1, s=300, c='red', marker='X', zorder=10)



        # Save generated image with detections
        plt.axis("off")
        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        filename = path.split("/")[-1].split(".")[0]
        plt.savefig(f"output/{filename}_smooth_{opt.bin}_{opt.sort}.png", bbox_inches="tight", pad_inches=0.0, dpi = 1200)
        plt.close()