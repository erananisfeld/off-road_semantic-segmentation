import torch
import argparse
import yaml
import math
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
from torch.utils.data import DataLoader
from torch.nn import functional as F
from semseg.models import *
from semseg.datasets import *
from semseg.augmentations import get_val_augmentation
from semseg.metrics import Metrics
from semseg.utils.utils import setup_cudnn
from torchvision.utils import make_grid
from torchvision import transforms as T
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import classification_report

def convert_label(self, label, inverse=False):
    temp = label.copy()
    if inverse:
        for v, k in self.label_mapping.items():
            label[temp == k] = v
    else:
        for k, v in self.label_mapping.items():
            label[temp == k] = v

    return label

@torch.no_grad()
def evaluate(model, dataloader, device, flag_save = False, loss_fn=None):
    PALETTE = torch.tensor(
        [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
         [250, 170, 30], [220, 220, 0], [107, 142, 35],
         [152, 251, 152], [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
         [0, 80, 100]])

    print('Evaluating...')
    model.eval()
    metrics = Metrics(dataloader.dataset.n_classes, dataloader.dataset.ignore_label, device)
    val_loss = 0.0


    for k,(images, labels) in enumerate(tqdm(dataloader)):
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).softmax(dim=1)

        if loss_fn:
            loss = loss_fn(preds, labels)
            val_loss += loss.item()
        if flag_save:
            preds1=preds.argmax(dim=1)
            #print(f"Image Shape\t: {images.shape}")
            #print(f"Label Shape\t: {preds1.shape}")
            #print(f"Classes\t\t: {preds1.unique().tolist()}")

            preds1[preds1 == -1] = 0
            preds1[preds1 == 255] = 0
            #
            # new_preds = []
            # for lbl in preds:
            #     lbl = lbl.cpu()
            #     new_lbl = np.array(PALETTE[lbl]).transpose(2, 0, 1)
            #     new_preds.append(new_lbl)
            #
            # new_preds = np.stack(new_preds)
            preds1 = preds1.cpu()
            preds1 = [PALETTE[lbl.to(int)].permute(2, 0, 1) for lbl in preds1]
            preds1 = torch.stack(preds1)

            inv_normalize = T.Normalize(
                mean=(-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225),
                std=(1 / 0.229, 1 / 0.224, 1 / 0.225)
            )
            image = inv_normalize(images)
            image *= 255
            image=image.cpu()
            images = torch.vstack([image, preds1])

            plt.imshow(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
            plt.show()
            save_img = Image.fromarray(make_grid(images, nrow=4).to(torch.uint8).numpy().transpose((1, 2, 0)))
            save_img.save(os.path.join(str(cfg['SAVE_DIR']) + r'\test_results', f'{k}' + '.png'))

        metrics.update(preds, labels)

    val_loss /= k + 1
    ious, miou = metrics.compute_iou()
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()

    return val_loss, acc, macc, f1, mf1, ious, miou


@torch.no_grad()
def evaluate_msf(model, dataloader, device, scales, flip):
    model.eval()

    n_classes = dataloader.dataset.n_classes
    metrics = Metrics(n_classes, dataloader.dataset.ignore_label, device)

    for images, labels in tqdm(dataloader):
        labels = labels.to(device)
        B, H, W = labels.shape
        scaled_logits = torch.zeros(B, n_classes, H, W).to(device)

        for scale in scales:
            new_H, new_W = int(scale * H), int(scale * W)
            new_H, new_W = int(math.ceil(new_H / 32)) * 32, int(math.ceil(new_W / 32)) * 32
            scaled_images = F.interpolate(images, size=(new_H, new_W), mode='bilinear', align_corners=True)
            scaled_images = scaled_images.to(device)
            logits = model(scaled_images)
            logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
            scaled_logits += logits.softmax(dim=1)

            if flip:
                scaled_images = torch.flip(scaled_images, dims=(3,))
                logits = model(scaled_images)
                logits = torch.flip(logits, dims=(3,))
                logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=True)
                scaled_logits += logits.softmax(dim=1)

        metrics.update(scaled_logits, labels)
    
    acc, macc = metrics.compute_pixel_acc()
    f1, mf1 = metrics.compute_f1()
    ious, miou = metrics.compute_iou()
    return acc, macc, f1, mf1, ious, miou


def main(cfg):
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], r'val', transform)
    #train_dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], 'train', transform)
    dataloader = DataLoader(dataset, batch_size=4, pin_memory=True)
    #train_dataloader = DataLoader(train_dataset, batch_size=1, pin_memory=True)

    model_path = Path(eval_cfg['MODEL_PATH'])
    if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
    print(f"Evaluating {model_path}...")

    model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], dataset.n_classes)
    model_cfg = cfg['MODEL']
    model.init_pretrained(model_cfg['PRETRAINED'])
    model.load_state_dict(torch.load(str(model_path), map_location='cpu'), strict=True)
    model = model.to(device)

    if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(model, dataloader, device, eval_cfg['MSF']['SCALES'], eval_cfg['MSF']['FLIP'])
    else:
        val_loss, acc, macc, f1, mf1, ious, miou = evaluate(model, dataloader, device, flag_save=False)

    histogram = np.array(
        [1.09747967e+01, 1.86997069e+01, 3.60069005e-03, 2.79958570e+00, 1.72549386e+01, 1.08879793e+00,
         8.77118600e+00, 1.23928635e+01, 2.60774922e-01, 1.99168844e+01, 7.01678179e-01, 7.94230630e-01,
         2.39875710e+00, 3.05476475e+00, 5.87071062e-01, 2.99557435e-01, 8.05509965e-04])
    avg_iou = sum((np.array(ious) * histogram)) / 100
    print('avg_iou = ', avg_iou)

    val_table = {
        'Class': list(dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }
    print('validation:\n' + tabulate(val_table, headers='keys'))

    """if eval_cfg['MSF']['ENABLE']:
        acc, macc, f1, mf1, ious, miou = evaluate_msf(model, train_dataloader, device, eval_cfg['MSF']['SCALES'],
                                                      eval_cfg['MSF']['FLIP'])
    else:
        val_loss, acc, macc, f1, mf1, ious, miou = evaluate(model, train_dataloader, device, flag_save=False)

    train_table = {
        'Class': list(train_dataset.CLASSES) + ['Mean'],
        'IoU': ious + [miou],
        'F1': f1 + [mf1],
        'Acc': acc + [macc]
    }

    print('train:\n' + tabulate(train_table, headers='keys'))"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'C:\final_project_eran_nadav\semantic-segmentation-main\configs\elbit.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)
