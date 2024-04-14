# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
from mmcv import Config
import torch
from mmdet.datasets import build_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('result_path', help='evaluation result file path')
    parser.add_argument('config_path', help='test config file path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config_path)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True

    dataset = build_dataset(cfg.data.test)
    dataset.format_results(mmcv.load(args.result_path), jsonfile_prefix=args.result_path[:-4])


if __name__ == '__main__':
    main()
