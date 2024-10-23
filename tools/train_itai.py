import sys
sys.path.insert(0,'C:/final_project_eran_nadav/semantic-segmentation-main')
import matplotlib.pyplot as plt
import torch
import argparse
import yaml
import time
import multiprocessing as mp
from tabulate import tabulate
from tqdm import tqdm
from torch.utils.data import DataLoader
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler
from torch import distributed as dist
from semseg.models import *
from semseg.datasets import * 
from semseg.augmentations import get_train_augmentation, get_val_augmentation
from semseg.losses import get_loss
from semseg.schedulers import get_scheduler
from semseg.optimizers import get_optimizer
from semseg.utils.utils import fix_seeds, setup_cudnn, cleanup_ddp, setup_ddp
from val import evaluate
from semseg.metrics import Metrics
import numpy as np

def main(cfg, gpu, save_dir, save_checkpoint=False, load_checkpoint=False):
    start = time.time()
    best_mIoU = 55.0
    #num_workers = mp.cpu_count()
    num_workers = 1
    device = torch.device(cfg['DEVICE'])
    train_cfg, eval_cfg = cfg['TRAIN'], cfg['EVAL']
    dataset_cfg, model_cfg = cfg['DATASET'], cfg['MODEL']
    loss_cfg, optim_cfg, sched_cfg = cfg['LOSS'], cfg['OPTIMIZER'], cfg['SCHEDULER']
    epochs, lr = train_cfg['EPOCHS'], optim_cfg['LR']
    
    traintransform = get_train_augmentation(train_cfg['IMAGE_SIZE'], seg_fill=dataset_cfg['IGNORE_LABEL'])

    valtransform = get_val_augmentation(eval_cfg['IMAGE_SIZE'])

    trainset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'train', traintransform)
    valset = eval(dataset_cfg['NAME'])(dataset_cfg['ROOT'], 'val', valtransform)
    
    model = eval(model_cfg['NAME'])(model_cfg['BACKBONE'], trainset.n_classes)
    model.init_pretrained(model_cfg['PRETRAINED'])
    np.save(model_cfg['CHECKPOINT'] + r'best_mIoU.npy', np.array(best_mIoU))
    if load_checkpoint:
        model_path = Path(model_cfg['CHECKPOINT'] + '_checkpoint_model_65.36.pth')
        #if not model_path.exists(): model_path = Path(cfg['SAVE_DIR']) / f"{cfg['MODEL']['NAME']}_{cfg['MODEL']['BACKBONE']}_{cfg['DATASET']['NAME']}.pth"
        #model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], val_dataset.n_classes)
        model.load_state_dict(torch.load(str(model_path), map_location='cpu'))
        best_mIoU = np.load(model_cfg['CHECKPOINT'] + r'best_mIoU.npy')
        lr = 0.0001
    model = model.to(device)


    if train_cfg['DDP']: 
        sampler = DistributedSampler(trainset, dist.get_world_size(), dist.get_rank(), shuffle=True)
        model = DDP(model, device_ids=[gpu])
    else:
        sampler = RandomSampler(trainset)
    
    trainloader = DataLoader(trainset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, drop_last=True, pin_memory=True, sampler=sampler)
    valloader = DataLoader(valset, batch_size=train_cfg['BATCH_SIZE'], num_workers=num_workers, pin_memory=True)

    iters_per_epoch = len(trainset) // train_cfg['BATCH_SIZE']
    # class_weights = trainset.class_weights.to(device)
    loss_fn = get_loss(loss_cfg['NAME'], trainset.ignore_label, None)
    optimizer = get_optimizer(model, optim_cfg['NAME'], lr, optim_cfg['WEIGHT_DECAY'])
    scheduler = get_scheduler(sched_cfg['NAME'], optimizer, epochs, sched_cfg['POWER'], iters_per_epoch * sched_cfg['WARMUP'], sched_cfg['WARMUP_RATIO'], 0.15)
    scaler = GradScaler(enabled=train_cfg['AMP'])
    writer = SummaryWriter(str(save_dir / 'logs'))

    train_miou_list = []
    val_miou_list = []
    train_loss_list = []
    val_loss_list = []
    lr_list = []


    #init_val_miou = evaluate(model, valloader, device)[-1]
    # print('init before val miou:' + str(init_val_miou))

    epoch_end = time.time()
    for epoch in range(epochs):
        metrics = Metrics(trainloader.dataset.n_classes, trainloader.dataset.ignore_label, device)
        model.train()

        if train_cfg['DDP']: sampler.set_epoch(epoch)

        train_loss = 0.0

        pbar = tqdm(enumerate(trainloader), total=iters_per_epoch, desc=f"Epoch: [{epoch+1}/{epochs}] Iter: [{0}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss:.8f}")
        epoch_start = time.time()
        print(f'time_between_epochos: {epoch_start - epoch_end} sec')
        for iter, (img, lbl) in pbar:

            optimizer.zero_grad(set_to_none=True)

            #plt.imshow(img[0].permute(1,2,0))
            #plt.show()

            img = img.to(device)
            lbl = lbl.to(device)


            with autocast(enabled=train_cfg['AMP']):
                logits = model(img)
                loss = loss_fn(logits, lbl)

            scaler.scale(loss).backward()
            scaler.step(optimizer)

            scaler.update()

            torch.cuda.synchronize()

            train_loss += loss.item()
            metrics.update(logits, lbl)

            #pbar.set_description(f"Epoch: [{epoch+1}/{epochs}] Iter: [{iter+1}/{iters_per_epoch}] LR: {lr:.8f} Loss: {train_loss / (iter+1):.8f}")
        scheduler.step()
        lr = scheduler.get_lr()
        lr = sum(lr) / len(lr)

        ious, train_miou = metrics.compute_iou()
        train_miou_list.append(train_miou)
        train_loss /= iter+1
        train_loss_list.append(train_loss)
        lr_list.append(lr)
        writer.add_scalar('train/loss', train_loss, epoch)
        torch.cuda.empty_cache()
        print(f"Current train_mIoU: {train_miou}")

        if (epoch+1) % train_cfg['EVAL_INTERVAL'] == 0 or (epoch+1) == epochs:
            #train_miou = evaluate(model, trainloader, device)[-1]
            val = evaluate(model, valloader, device, loss_fn=loss_fn)
            val_loss, val_miou = val[0], val[-1]
            train_miou_list.append(train_miou)
            val_miou_list.append(val_miou)
            val_loss_list.append(val_loss)
            writer.add_scalar('train/mIoU', train_miou, epoch)
            writer.add_scalar('val/mIoU', val_miou, epoch)

            if val_miou > best_mIoU:
                best_mIoU = val_miou
                if save_checkpoint:
                    torch.save(model.state_dict(), f"{model_cfg['CHECKPOINT']}_checkpoint_model_{best_mIoU}.pth")
                    np.save(model_cfg['CHECKPOINT'] + 'best_mIoU.npy', np.array(best_mIoU))
                torch.save(model.module.state_dict() if train_cfg['DDP'] else model.state_dict(), save_dir / f"{model_cfg['NAME']}_{model_cfg['BACKBONE']}_{dataset_cfg['NAME']}.pth")
            print(f"Current train_mIoU: {train_miou} Current val_mIoU: {val_miou} Best mIoU: {best_mIoU}")

        epoch_end = time.time()
    writer.close()
    pbar.close()
    end = time.gmtime(time.time() - start)

    table = [
        ['Best mIoU', f"{best_mIoU:.2f}"],
        ['Total Training Time', time.strftime("%H:%M:%S", end)]
    ]
    print(tabulate(table, numalign='right'))


    epochs_in_intervals = list(range(1, epochs + 1, train_cfg['EVAL_INTERVAL']))

    plt.plot(list(range(1, len(train_miou_list)+1)), train_miou_list, label='Training mIoU')
    plt.plot(epochs_in_intervals, val_miou_list, label='Validation mIoU')

    plt.title('Mean Intersection over Union (mIoU) vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('mIoU')
    plt.legend()
    plt.show()

    #epochs_plot = range(1, epochs + 1)
    plt.plot(list(range(1, len(train_loss_list)+1)), train_loss_list)

    plt.title('train loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    #plt.legend()
    plt.show()

    plt.plot(epochs_in_intervals, val_loss_list)
    plt.title('val loss vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    #plt.legend()
    plt.show()


    plt.plot(list(range(1, len(lr_list)+1)), lr_list)
    plt.title('learning rate vs. Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('learning rate')
    # plt.legend()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'C:\final_project_eran_nadav\semantic-segmentation-main\configs\elbit.yaml', help='Configuration file to use')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    fix_seeds(3407)
    setup_cudnn()
    gpu = setup_ddp()
    save_dir = Path(cfg['SAVE_DIR'])
    save_dir.mkdir(exist_ok=True)
    main(cfg, gpu, save_dir, save_checkpoint=True, load_checkpoint=False)
    cleanup_ddp()