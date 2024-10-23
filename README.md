## <div align="center">Description</div>

<details open>
This project contains a trained model that is able to receive an off-road image and perform semantic segmentation of the image pixels and output a segmented image.
the classes are:
 0 = Terrain
 1 = Unpaved Route
 2 = Paved Road
 3 = Tree Trunk
 4 = Tree Foliage
 5 = Rocks
 6 = Large Shrubs
 7 = Low Vegetation
 8 = Wire Fence
 9 = Sky
 10 = Person
 11 = Vehicle
 12 = Building
 13 = ignore
 14 = Misc
 15 = Water
 16 = Animal
 
## <div align="center">Usage</div>

<details open>
  <summary><strong>Installation</strong></summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Then, clone the repo and install the project with:

```bash
$ git clone https://github.com/erananisfeld/semantic-segmentation_off-road
$ cd semantic-segmentation_off-road
$ pip install -e .
```

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

Create a configuration file in `configs`. Sample configuration for ADE20K dataset can be found [here](configs/ade20k.yaml). Then edit the fields you think if it is needed. This configuration file is needed for all of training, evaluation and prediction scripts.

</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

To train with a single GPU:

```bash
$ python tools/train.py --cfg configs/CONFIG_FILE.yaml
```

To train with multiple gpus, set `DDP` field in config file to `true` and run as follows:

```bash
$ python -m torch.distributed.launch --nproc_per_node=2 --use_env tools/train.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>

Make sure to set `MODEL_PATH` of the configuration file to your trained model directory.

```bash
$ python tools/val.py --cfg configs/<CONFIG_FILE_NAME>.yaml
```

To evaluate with multi-scale and flip, change `ENABLE` field in `MSF` to `true` and run the same command as above.

</details>

<br>
<details open>
  <summary><strong>Inference</strong></summary>

To make an inference, edit the parameters of the config file from below.
* Change `MODEL` >> `NAME` and `BACKBONE` to your desired pretrained model.
* Change `DATASET` >> `NAME` to the dataset name depending on the pretrained model.
* Set `TEST` >> `MODEL_PATH` to pretrained weights of the testing model.
* Change `TEST` >> `FILE` to the file or image folder path you want to test.
* Testing results will be saved in `SAVE_DIR`.

```bash
## example using ade20k pretrained models
$ python tools/infer.py --cfg configs/ade20k.yaml
```

Example test results (SegFormer-B2):

![test_result](./output/elbit_results/results_test_segformer.mp4)

</details>


<br>
<details>
  <summary><strong>Inference</strong> </summary>

```bash
## Inference
$ python tools/infer.py --model <MODEL_PATH> --img-path <TEST_IMAGE_PATH>

```

</details>

<br>
<details>
  <summary><strong>References</strong> (click to expand)</summary>

* https://github.com/sithu31296/semantic-segmentation.git

</details>

<br>
<details>
  <summary><strong>Citations</strong> (click to expand)</summary>

```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}

@misc{xiao2018unified,
  title={Unified Perceptual Parsing for Scene Understanding}, 
  author={Tete Xiao and Yingcheng Liu and Bolei Zhou and Yuning Jiang and Jian Sun},
  year={2018},
  eprint={1807.10221},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{hong2021deep,
  title={Deep Dual-resolution Networks for Real-time and Accurate Semantic Segmentation of Road Scenes},
  author={Hong, Yuanduo and Pan, Huihui and Sun, Weichao and Jia, Yisong},
  journal={arXiv preprint arXiv:2101.06085},
  year={2021}
}

@misc{zhang2021rest,
  title={ResT: An Efficient Transformer for Visual Recognition}, 
  author={Qinglong Zhang and Yubin Yang},
  year={2021},
  eprint={2105.13677},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{huang2021fapn,
  title={FaPN: Feature-aligned Pyramid Network for Dense Image Prediction}, 
  author={Shihua Huang and Zhichao Lu and Ran Cheng and Cheng He},
  year={2021},
  eprint={2108.07058},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wang2021pvtv2,
  title={PVTv2: Improved Baselines with Pyramid Vision Transformer}, 
  author={Wenhai Wang and Enze Xie and Xiang Li and Deng-Ping Fan and Kaitao Song and Ding Liang and Tong Lu and Ping Luo and Ling Shao},
  year={2021},
  eprint={2106.13797},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@article{Liu2021PSA,
  title={Polarized Self-Attention: Towards High-quality Pixel-wise Regression},
  author={Huajun Liu and Fuqiang Liu and Xinyi Fan and Dong Huang},
  journal={Arxiv Pre-Print arXiv:2107.00782 },
  year={2021}
}

@misc{chao2019hardnet,
  title={HarDNet: A Low Memory Traffic Network}, 
  author={Ping Chao and Chao-Yang Kao and Yu-Shan Ruan and Chien-Hsiang Huang and Youn-Long Lin},
  year={2019},
  eprint={1909.00948},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@inproceedings{sfnet,
  title={Semantic Flow for Fast and Accurate Scene Parsing},
  author={Li, Xiangtai and You, Ansheng and Zhu, Zhen and Zhao, Houlong and Yang, Maoke and Yang, Kuiyuan and Tong, Yunhai},
  booktitle={ECCV},
  year={2020}
}

@article{Li2020SRNet,
  title={Towards Efficient Scene Understanding via Squeeze Reasoning},
  author={Xiangtai Li and Xia Li and Ansheng You and Li Zhang and Guang-Liang Cheng and Kuiyuan Yang and Y. Tong and Zhouchen Lin},
  journal={ArXiv},
  year={2020},
  volume={abs/2011.03308}
}

@ARTICLE{Yucondnet21,
  author={Yu, Changqian and Shao, Yuanjie and Gao, Changxin and Sang, Nong},
  journal={IEEE Signal Processing Letters}, 
  title={CondNet: Conditional Classifier for Scene Segmentation}, 
  year={2021},
  volume={28},
  number={},
  pages={758-762},
  doi={10.1109/LSP.2021.3070472}
}

@misc{yan2022lawin,
  title={Lawin Transformer: Improving Semantic Segmentation Transformer with Multi-Scale Representations via Large Window Attention}, 
  author={Haotian Yan and Chuang Zhang and Ming Wu},
  year={2022},
  eprint={2201.01615},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{yu2021metaformer,
  title={MetaFormer is Actually What You Need for Vision}, 
  author={Weihao Yu and Mi Luo and Pan Zhou and Chenyang Si and Yichen Zhou and Xinchao Wang and Jiashi Feng and Shuicheng Yan},
  year={2021},
  eprint={2111.11418},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{wightman2021resnet,
  title={ResNet strikes back: An improved training procedure in timm}, 
  author={Ross Wightman and Hugo Touvron and Hervé Jégou},
  year={2021},
  eprint={2110.00476},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{liu2022convnet,
  title={A ConvNet for the 2020s}, 
  author={Zhuang Liu and Hanzi Mao and Chao-Yuan Wu and Christoph Feichtenhofer and Trevor Darrell and Saining Xie},
  year={2022},
  eprint={2201.03545},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

@misc{li2022uniformer,
  title={UniFormer: Unifying Convolution and Self-attention for Visual Recognition}, 
  author={Kunchang Li and Yali Wang and Junhao Zhang and Peng Gao and Guanglu Song and Yu Liu and Hongsheng Li and Yu Qiao},
  year={2022},
  eprint={2201.09450},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}

```

</details>
