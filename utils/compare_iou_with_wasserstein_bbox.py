import cv2
import math
import torch
import pickle
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def wasserstein_loss(pred, target, eps=1e-7, mode='exp', gamma=1, constant=12.8):
    r"""`Implementation of paper `Enhancing Geometric Factors into
    Model Learning and Inference for Object Detection and Instance
    Segmentation <https://arxiv.org/abs/2005.03572>`_.

    Code is modified from https://github.com/Zzh-tju/CIoU.

    Args:
        pred (Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (Tensor): Corresponding gt bboxes, shape (n, 4).
        eps (float): Eps to avoid log(0).
    Return:
        Tensor: Loss tensor.
    """
    center1 = (pred[:, :2] + pred[:, 2:]) / 2
    center2 = (target[:, :2] + target[:, 2:]) / 2

    whs = center1[:, :2] - center2[:, :2]

    center_distance = whs[:, 0] * whs[:, 0] + whs[:, 1] * whs[:, 1] + eps #

    w1 = pred[:, 2] - pred[:, 0]  + eps
    h1 = pred[:, 3] - pred[:, 1]  + eps
    w2 = target[:, 2] - target[:, 0]  + eps
    h2 = target[:, 3] - target[:, 1]  + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wasserstein_2 = center_distance + wh_distance

    if mode == 'exp':
        normalized_wasserstein = torch.exp(-torch.sqrt(wasserstein_2)/constant)
        wloss = 1 - normalized_wasserstein
    
    if mode == 'sqrt':
        wloss = torch.sqrt(wasserstein_2)
    
    if mode == 'log':
        wloss = torch.log(wasserstein_2 + 1)

    if mode == 'norm_sqrt':
        wloss = 1 - 1 / (gamma + torch.sqrt(wasserstein_2))

    if mode == 'w2':
        wloss = wasserstein_2

    return wloss

def bbox_iou(box1, box2, xywh=True, GIoU=False, DIoU=False, CIoU=False, MDPIoU=False, feat_h=640, feat_w=640, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,4) to box2(n,4)

    # Get the coordinates of bounding boxes
    if xywh:  # transform from xywh to xyxy
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_x2, b1_y1, b1_y2 = x1 - w1_, x1 + w1_, y1 - h1_, y1 + h1_
        b2_x1, b2_x2, b2_y1, b2_y2 = x2 - w2_, x2 + w2_, y2 - h2_, y2 + h2_
    else:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

    # Union Area
    union = w1 * h1 + w2 * h2 - inter + eps

    # IoU
    iou = inter / union
    if CIoU or DIoU or GIoU:
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex (smallest enclosing box) width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center dist ** 2
            if CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + eps))
                return iou - (rho2 / c2 + v * alpha)  # CIoU
            return iou - rho2 / c2  # DIoU
        c_area = cw * ch + eps  # convex area
        return iou - (c_area - union) / c_area  # GIoU https://arxiv.org/pdf/1902.09630.pdf
    elif MDPIoU:
        d1 = (b2_x1 - b1_x1) ** 2 + (b2_y1 - b1_y1) ** 2
        d2 = (b2_x2 - b1_x2) ** 2 + (b2_y2 - b1_y2) ** 2
        mpdiou_hw_pow = feat_h ** 2 + feat_w ** 2
        return iou - d1 / mpdiou_hw_pow - d2 / mpdiou_hw_pow  # MPDIoU
    return iou  # IoU

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y



if True:
    px1, py1, px2, py2 = 20, 20, 30, 30
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")

    px1, py1, px2, py2 = 20, 20, 25, 25
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")

    px1, py1, px2, py2 = 20, 20, 21, 21
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")

    px1, py1, px2, py2 = 20, 20, 20, 20
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")

    px1, py1, px2, py2 = 40, 40, 40, 40
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")

    px1, py1, px2, py2 = 100, 100, 100, 100
    tx1, ty1, tx2, ty2 = 20, 20, 30, 30
    b1 = torch.tensor([[px1, py1, px2, py2]])
    b2 = torch.tensor([[tx1, ty1, tx2, ty2]])
    iou = 1.0 - bbox_iou(b1, b2, xywh=False, CIoU=True)
    w2 = wasserstein_loss(b1, b2, mode='exp').view([-1, 1])
    print(f"pred = ({px1, py1, px2, py2}, tar = {tx1, ty1, tx2, ty2}, iou = {iou}, w2 = {w2}")


if False:
    with open("/home/vladimir/Work/Projects/yolo9/debug/pred_tar.pkl", "rb") as f:
        pred_bboxes_pos, target_bboxes_pos = pickle.load(f)

    print(pred_bboxes_pos.shape)
    print(target_bboxes_pos.shape)

    iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
    # loss_iou = 1.0 - iou

    w2 = wasserstein_loss(pred_bboxes_pos, target_bboxes_pos, mode='exp').view([-1, 1])

    # print(iou)
    # print(w2)

    preds = pred_bboxes_pos.detach().cpu().numpy().astype(int)
    tars = target_bboxes_pos.detach().cpu().numpy().astype(int)

    tars2 = xyxy2xywh(tars)

    cat = np.concatenate((preds, tars), 0)
    w = int(cat[:, 2].max(0) + cat[:, 0].min(0) * 2)
    h = int(cat[:, 3].max(0) + cat[:, 1].min(0) * 2)

    print(w, h)

    imgpred = np.zeros(shape=(h, w, 3), dtype=np.uint8)
    imgtar = np.zeros(shape=(h, w, 3), dtype=np.uint8)

    for i in range(tars.shape[0]):
        if tars2[i][2:].min() < 4:
            continue
        imgpred = cv2.rectangle(imgpred, tars[i][:2], tars[i][2:], (0, 255, 0), thickness=1)
        imgtar = cv2.rectangle(imgtar, preds[i][:2], preds[i][2:], (255, 0, 0), thickness=1)

    plt.figure()
    plt.imshow(imgpred)
    plt.figure()
    plt.imshow(imgtar)
    plt.show()
