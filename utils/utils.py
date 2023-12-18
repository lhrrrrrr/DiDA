import math
import torch
import numpy as np
from scipy.spatial.distance import cdist
import torchvision
from torchvision import transforms
from PIL import Image
import random



def image_train(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def image_target(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_shift(resize_size=256, crop_size=224):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize
    ])


def image_test(resize_size=256, crop_size=224):
    return transforms.Compose([
        transforms.Resize((resize_size, resize_size)),
        transforms.CenterCrop(crop_size),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
    ])


def rgb_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def l_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')


def get_features(loader, model):
    start_test = True
    with torch.no_grad():
        iter_test = iter(loader)
        for i in range(len(loader)):
            data = next(iter_test)
            inputs = data[0]
            gt_labels = data[1]
            inputs = inputs.cuda()

            outputs = model(inputs, eval_only=True)
            if len(outputs.shape) == 1:
                outputs = outputs.reshape(1, -1)
            if start_test:
                all_output = outputs.cpu().detach().numpy()
                all_label = gt_labels.detach().numpy()
                start_test = False
            else:
                try:
                    all_output = np.concatenate((all_output, outputs.cpu().detach().numpy()), 0)
                    all_label = np.concatenate((all_label, gt_labels.detach().numpy()), 0)
                except:
                    print()
    return all_output, all_label


def eval_AP_inner(inst_id, scores, gt_labels, top=None):
    pos_flag = gt_labels == inst_id
    tot = scores.shape[0]
    tot_pos = np.sum(pos_flag)

    sort_idx = np.argsort(-scores)
    tp = pos_flag[sort_idx]
    fp = np.logical_not(tp)

    if top is not None:
        top = min(top, tot)
        tp = tp[:top]
        fp = fp[:top]
        tot_pos = min(top, tot_pos)

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    try:
        rec = tp / tot_pos
        prec = tp / (tp + fp)
    except:
        print(inst_id, tot_pos)
        return np.nan

    ap = VOCap(rec, prec)
    return ap


def VOCap(rec, prec):
    mrec = np.append(0, rec)
    mrec = np.append(mrec, 1)

    mpre = np.append(0, prec)
    mpre = np.append(mpre, 0)

    for ii in range(len(mpre) - 2, -1, -1):
        mpre[ii] = max(mpre[ii], mpre[ii + 1])

    msk = [i != j for i, j in zip(mrec[1:], mrec[0:-1])]
    ap = np.sum((mrec[1:][msk] - mrec[0:-1][msk]) * mpre[1:][msk])
    return ap


def cal_map_sda(features_query, gt_labels_query, features_gallary, gt_labels_gallery):
    mAP_ls = [[] for _ in range(len(np.unique(gt_labels_query)))]
    scores = - cdist(features_query, features_gallary)
    for fi in range(features_query.shape[0]):
        mapi = eval_AP_inner(gt_labels_query[fi], scores[fi], gt_labels_gallery)
        try:
            mAP_ls[gt_labels_query[fi]].append(mapi)
        except:
            print()
    mAP = np.array([np.nanmean(maps) for maps in mAP_ls]).mean()
    # print('mAP - real value: {:.3f}'.format(mAP))
    return mAP


def adjust_learning_rate(args, optimizer, epoch, lr):
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 2)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epoch)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr




