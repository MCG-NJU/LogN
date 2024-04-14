#!/usr/bin/env bash


CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/mask_rcnn_r50_fpn_random_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/mask_rcnn_r50_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/mask_rcnn_r101_fpn_random_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
--nproc_per_node=4 --master_port=10020 tools/train.py ./configs/logn/cascade_mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune.py \
--launcher pytorch