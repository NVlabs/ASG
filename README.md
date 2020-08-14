![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
 
# ASG
 
<!-- ### [Project](https://) | [Paper](https://arxiv.org/abs/2007.06965) -->
[Paper](https://arxiv.org/abs/2007.06965)
 
Automated Synthetic-to-Real Generalization.<br>
[Wuyang Chen](https://chenwydj.github.io/),  [Zhiding Yu](https://chrisding.github.io/), [Zhangyang Wang](https://www.atlaswang.com/), [Anima Anandkumar](http://tensorlab.cms.caltech.edu/users/anima/).<br>
In ICML 2020.

* Visda-17 to COCO
- [x] train resnet101 with only proxy guidance
- [x] train resnet101 with both proxy guidance and L2O policy
- [x] evaluation
* GTA5 to Cityscapes

## Usage

### Visda-17
* Download [Visda-17 Dataset](http://ai.bu.edu/visda-2017/#download)

#### Evaluation
* Download [pretrained ResNet101 on Visda17](https://drive.google.com/file/d/1jjihDIxU1HIRtJEZyd7eTpYfO21OrY36/view?usp=sharing)
* Put the checkpoint under `./ASG/pretrained/`
* Put the code below in `train.sh`
```bash
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--lwf 0.1 \
--resume pretrained/res101_vista17_best.pth.tar \
--evaluate
```
* Run `CUDA_VISIBLE_DEVICES=0 bash train.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.

#### Train with SGD
* Put the code below in `train.sh`
```bash
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--lwf 0.1
```
* Run `CUDA_VISIBLE_DEVICES=0 bash train.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.

#### Train with L2O
* Download [pretrained L2O Policy on Visda17](https://drive.google.com/file/d/1Rc2Ey-FspUagFPTjnEozeSEIdA4ir7b1/view?usp=sharing)
* Put the checkpoint under `./ASG/pretrained/`
* Put the code below in `l2o_train.sh`
```bash
python train.py \
--epochs 30 \
--batch-size 32 \
--lr 1e-4 \
--lwf 0.1
```
* Run `CUDA_VISIBLE_DEVICES=0 bash l2o_train.sh`
  - Please update the GPU index via `CUDA_VISIBLE_DEVICES` based on your need.
 
## Citation
 
If you use this code for your research, please cite:
 
```BibTeX
@incollection{chen2020automated,
 author = {Chen, Wuyang and Yu, Zhiding and Wang, Zhangyang and Anandkumar, Anima},
 booktitle = {Proceedings of Machine Learning and Systems 2020},
 pages = {8272--8282},
 title = {Automated Synthetic-to-Real Generalization},
 year = {2020}
}
```
