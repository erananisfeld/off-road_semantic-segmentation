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
import cv2




def main(cfg):
    device = torch.device(cfg['DEVICE'])
    eval_cfg = cfg['EVAL']
    transform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])
    dataset = eval(cfg['DATASET']['NAME'])(cfg['DATASET']['ROOT'], r'train', transform)
    dataloader = DataLoader(dataset, batch_size=1, pin_memory=True)

    #alllabels = []
    #values = np.zeros(17)
    #total = 0
    keys = Elbit.CLASSES
    #max_iter = len(dataloader)
    counter = np.zeros(17)
    #dict = {}
    #for key in keys:
        #dict[key] = 0
    image_path = dataset.img_path
    histograms = np.zeros((17, 1))
    for k, labels in enumerate(tqdm(dataset.masks)):
        path = os.path.join(image_path, labels)
        labels = cv2.imread(path)
        gray = cv2.cvtColor(labels, cv2.COLOR_BGR2GRAY)
        histogram = cv2.calcHist([gray], [0], None, [17], [0, 17])
        if histogram[2] != 0:
            print(path)
        histograms += histogram

        #labels = labels[labels != 255]
        #alllabels.extend(labels.numpy())
        #labels = np.squeeze(labels.numpy()).flatten()
        #for label in labels:
            #if label != 255:
                #counter[label] += 1
            #for i, key in enumerate(keys):
                #dict[key] += counter[i]
            #values += np.array(local_values)
            #total += len(alllabels)
            #alllabels = []

        #if (k % 800 == 0 and k > 0) or k == max_iter-1:
            #y_true = (np.squeeze(np.array(alllabels))).flatten()
            #y_true = alllabels
            #y_pred = y_true
            #local_report = classification_report(alllabels, alllabels, target_names=Elbit.CLASSES, digits=4, output_dict=True)
            #local_values = []
    total = sum(histograms)
    #values = np.array(dict.values())
    histograms = histograms * 100/total
    plt.figure(figsize=(16, 8))
    plt.bar(keys, np.squeeze(histograms))
    plt.xticks(rotation=45, fontsize='small')
    plt.xlabel('labels')
    plt.ylabel('pixels (%)')
    plt.title('Histogram of classes in pixel percentage for train set')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'C:\final_project_eran_nadav\semantic-segmentation-main\configs\elbit.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    setup_cudnn()
    main(cfg)


