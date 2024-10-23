import torch
print(torch.__version__)
import argparse
import yaml
import math
from torch import Tensor
from torch.nn import functional as F
from pathlib import Path
from torchvision import io
from torchvision import transforms as T
import sys
sys.path.insert(0,'C:/final_project_eran_nadav/semantic-segmentation-main')
from semseg.models import *
from semseg.datasets import *
from semseg.utils.utils import timer
from semseg.utils.visualize import draw_text
import numpy as np
from rich.console import Console
console = Console()

from PIL import Image

class SemSeg:
    def __init__(self, cfg) -> None:
        # inference device cuda or cpu
        self.device = torch.device(cfg['DEVICE'])

        # get dataset classes' colors and labels
        self.palette = eval(cfg['DATASET']['NAME']).PALETTE
        self.labels = eval(cfg['DATASET']['NAME']).CLASSES

        # initialize the model and load weights and send to device
        self.model = eval(cfg['MODEL']['NAME'])(cfg['MODEL']['BACKBONE'], len(self.palette))
        self.model.load_state_dict(torch.load(cfg['TEST']['MODEL_PATH'], map_location='cpu'))
        self.model = self.model.to(self.device)
        self.model.eval()

        # preprocess parameters and transformation pipeline
        self.size = cfg['TEST']['IMAGE_SIZE']
        self.tf_pipeline = T.Compose([
            T.Lambda(lambda x: x / 255),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            T.Lambda(lambda x: x.unsqueeze(0))
        ])

    def preprocess(self, image: Tensor) -> Tensor:
        H, W = image.shape[1:]
        console.print(f"Original Image Size > [red]{H}x{W}[/red]")
        # scale the short side of image to target size
        scale_factor = self.size[0] / min(H, W)
        nH, nW = round(H*scale_factor), round(W*scale_factor)
        # make it divisible by model stride
        nH, nW = int(math.ceil(nH / 32)) * 32, int(math.ceil(nW / 32)) * 32
        console.print(f"Inference Image Size > [red]{nH}x{nW}[/red]")
        # resize the image
        image = T.Resize((nH, nW))(image)
        # divide by 255, norm and add batch dim
        image = self.tf_pipeline(image).to(self.device)
        return image

    def postprocess(self, orig_img: Tensor, seg_map: Tensor, overlay: bool) -> Tensor:
        # resize to original image size
        seg_map = F.interpolate(seg_map, size=orig_img.shape[-2:], mode='bilinear', align_corners=True)
        # get segmentation map (value being 0 to num_classes)
        seg_map = seg_map.softmax(dim=1).argmax(dim=1).cpu().to(int)

        # convert segmentation map to color map
        seg_image = self.palette[seg_map].squeeze()
        seg_image_map = seg_image.to(torch.uint8)
        seg_image_map = Image.fromarray(seg_image_map.numpy())

        seg_image = (orig_img.permute(1, 2, 0) * 0.4) + (seg_image * 0.6)
        image_overlay = draw_text(seg_image, seg_map, self.labels)

        return seg_image_map, image_overlay

    @torch.inference_mode()
    @timer
    def model_forward(self, img: Tensor) -> Tensor:
        return self.model(img)
        
    def predict(self, img_fname: str, overlay: bool) -> Tensor:
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        seg_map = self.model_forward(img)
        seg_map, overlay_map = self.postprocess(image, seg_map, overlay)
        return seg_map, overlay_map

    def orig_processing(self, img_fname: str):
        image = io.read_image(img_fname)
        img = self.preprocess(image)
        new_img = img.softmax(dim=1).argmax(dim=1).cpu().to(int).permute(1, 2, 0)
        return new_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default=r'C:\final_project_eran_nadav\semantic-segmentation-main\configs\elbit_harddrive.yaml')
    args = parser.parse_args()

    with open(args.cfg) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    test_file = Path(cfg['DATASET']['ROOT'] + r'\PNG')
    test_file = Path(cfg['DATASET']['ROOT'] + r'\png')
    if not test_file.exists():
        raise FileNotFoundError(test_file)

    console.print(f"Model > [red]{cfg['MODEL']['NAME']} {cfg['MODEL']['BACKBONE']}[/red]")
    console.print(f"Model > [red]{cfg['DATASET']['NAME']}[/red]")

    save_dir_seg = Path(cfg['SAVE_DIR']) / 'seg_maps'
    save_dir_overlay = Path(cfg['SAVE_DIR']) / 'overlays'
    save_dir_seg.mkdir(exist_ok=True)
    save_dir_overlay.mkdir(exist_ok=True)
    
    semseg = SemSeg(cfg)

    with console.status("[bright_green]Processing..."):
        if test_file.is_file():
            console.rule(f'[green]{test_file}')
            #segmap = semseg.predict(str(test_file), cfg['TEST']['OVERLAY'])
            #segmap.save(save_dir / f"{str(test_file.stem)}.png")
            segmap_image, overlay_image = semseg.predict(str(file), True)
            segmap_image.save(save_dir_seg / f"{str(file.stem)}_colored.png")
            overlay_image.save(save_dir_overlay / f"{str(file.stem)}_overlay.png")
        else:
            files = test_file.glob('*.*')
            for file in files:
                if 'label' not in str(file):
                    console.rule(f'[green]{file}')
                    segmap_image, overlay_image = semseg.predict(str(file), True)
                    #segmap_no_overlay = semseg.predict(str(file), False)
                    #orig_image = io.read_image(file).permute(1, 2, 0)
                    #orig_image = semseg.preprocess(orig_image).squeeze().permute(1, 2, 0).cpu()
                    #orig_image = semseg.orig_processing(str(file))
                    #overlay_plus_original = torch.vstack([Tensor(segmap_overlay), Tensor(orig_image)])
                    segmap_image.save(save_dir_seg / f"{str(file.stem)}_colored.png")
                    overlay_image.save(save_dir_overlay / f"{str(file.stem)}_overlay.png")
                    #overlay_plus_original.save(save_dir / f"{str(file.stem)}_overlay_plus_original.png")
    console.rule(f"[cyan]Segmentation results are saved in `{save_dir_seg}`")