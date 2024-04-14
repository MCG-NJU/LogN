# Logit Normalization for Long-Tail Object Detection

This repo is an official implementation of our IJCV paper: **Logit Normalization for Long-Tail Object Detection**, which was published in 08 January 2024.

Please refer to [springer](https://link.springer.com/article/10.1007/s11263-023-01971-y) and [arxiv](https://arxiv.org/abs/2203.17020) for more details about our paper!


## Key Idea

Real-world data with skewed distributions poses a serious challenge to existing object detectors.
Via adding this `LogitNormHead` implemented below on the predicted logit vector for calibration, the long-tail bias will get greatly alleviated!

In general, our LogN is training- and tuning-free (i.e. require no extra training and tuning process), model- and label distribution-agnostic (i.e. generalization to different kinds of detectors and datasets), and also plug-and-play (i.e. direct application without any bells and whistles).

```python
class LogitNormHead(Shared2FCBBoxHead):

    def __init__(self, momentum=1e-4, *args, **kwargs):
        super(LogitNormHead, self).__init__(*args, **kwargs)
        self.bn = nn.BatchNorm1d(self.num_classes + 1, eps=1e-05, momentum=momentum, affine=False)
    
    def get_statistics(self):
        mean_val = self.bn.running_mean
        mean_val[-1] = 0
        std_val = torch.sqrt(torch.clamp(self.bn.running_var, min=1e-11))
        std_val[-1] = 1
        beta = torch.zeros_like(mean_val)
        beta[:-1] = mean_val[:-1].min()

        return mean_val.view(1, -1), std_val.view(1, -1), beta.view(1, -1)
    
    def forward(self, x):
        cls_score, bbox_pred = super(LogitNormHead, self).forward(x)

        if self.training:
            cls_score = self.bn(cls_score)
            return cls_score, bbox_pred
        else:
            mean_val, std_val, beta = self.get_statistics()
            cls_score = (cls_score - (mean_val - beta)) / std_val
            return cls_score, bbox_pred
```
Please refer to `models/roi_heads/bbox_heads/logit_norm_head.py` for more details!

## Installation

Please refer to [get_started.md](docs/get_started.md) for installation.

## Prepare LVIS Dataset

***for images***

LVIS uses same images as COCO's, so you need to donwload COCO dataset at folder ($COCO), and link those `train`, `val` under folder `lvis`($LVIS).

```
mkdir -p data/lvis
ln -s $COCO/train $LVIS
ln -s $COCO/val $LVIS
ln -s $COCO/test $LVIS
```
***for annotations***

Download the annotations from [lvis webset](https://lvisdataset.org/)

```
cd $LVIS
mkdir annotations
```
then places the annotations at folder ($LVIS/annotations)

Finally you will have the file structure like below:

    data
      ├── lvis
      |   ├── annotations
      │   │   │   ├── lvis_v1_val.json
      │   │   │   ├── lvis_v1_train.json
      │   ├── train2017
      │   │   ├── 000000004134.png
      │   │   ├── 000000031817.png
      │   │   ├── ......
      │   ├── val2017
      │   ├── test2017

***for API***

The official lvis-api and mmlvis can lead to some bugs of multiprocess. See [issue](https://github.com/open-mmlab/mmdetection/issues/4112)

So you can install this LVIS API from my modified repo.
```
pip install git+https://github.com/tztztztztz/lvis-api.git
```

## Pipeline

1. Download the pretrained models from the below urls. You can also train the baselines from scrath, please refer to `scripts/baseline.sh`; 
2. Please refer to `scripts/logn.sh`. Specifically, for logit-normalized calibration, we are now using an online approach, which involves finetuning the pretrained model for 1 epoch with the `LogitNormHead` attached for aggregating statistics and then directly perform calibrated evaluation. 

## Pretrained Models on LVIS

[] To be released.

## Citation

If you use the equalization losses, please cite our papers.

```
@article{zhao2024logit,
  title={Logit Normalization for Long-tail Object Detection},
  author={Zhao, Liang and Teng, Yao and Wang, Limin},
  journal={International Journal of Computer Vision},
  year={2024},
  publisher={Springer}
}
```