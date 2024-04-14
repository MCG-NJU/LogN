from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core.visualization import imshow_det_bboxes
import sys
import os.path
import glob
import mmcv
import json
import numpy as np

config_file = './configs/logn/cascade_mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune1.py'
checkpoint_file = './work_dirs/cascade_mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1_finetune1/epoch_1.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:2')
with open(f"/data/zhaoliang/ltod/eqlv2/data/lvis/annotations/lvis_v1_train.json", 'r') as f:
    data = json.load(f)
# import torch
# model = torch.load('./work_dirs/mask_rcnn_r101_fpn_sample1e-3_logn_normed_mask_mstrain_2x_lvis_v1/epoch_24.pth')['state_dict']
# # print(model.keys())
# label_sum = model['roi_head.bbox_head.label_sum']
# moving_mean = model['roi_head.bbox_head.moving_mean']
# moving_var = model['roi_head.bbox_head.moving_var']
# print(label_sum)
# print(moving_mean)


# for dataset in ["train", "val"]:
#     with open(f"/data/zhaoliang/ltod/eqlv2/data/lvis/annotations/lvis_v1_{dataset}.json", 'r') as f:
#         print(f"{dataset}: -------------------------------------------")
#         data = json.load(f)
#         print(f"{dataset} set: {len(data['annotations']) / len(data['images'])}")

#         catedict, freqdict, cateannodict, cateimgdict, cateimgsetdict = {}, {}, {}, {}, {}
#         for c in data['categories']:
#             catedict[c['id']] = c['name']
#             freqdict[c['id']] = c['frequency']
#         assert len(catedict) == len(freqdict) == 1203, print(len(catedict))

#         for anno in data['annotations']:
#             if anno['category_id'] not in cateannodict.keys():
#                 cateannodict[anno['category_id']] = 0
#             cateannodict[anno['category_id']] += 1

#             if anno['category_id'] not in cateimgsetdict.keys():
#                 cateimgsetdict[anno['category_id']] = set()
#             cateimgsetdict[anno['category_id']].add(anno['image_id'])

#         for k, v in cateimgsetdict.items():
#             cateimgdict[k] = len(v)

#         freqratio = {'f': 0, 'c': 0, 'r': 0}
#         freqcnt = {'f': 0, 'c': 0, 'r': 0}
#         freqanno = {'f': 0, 'c': 0, 'r': 0}
#         freqimg = {'f': 0, 'c': 0, 'r': 0}
#         for i in range(1204):
#             if i not in cateannodict.keys():
#                 continue
#             freqratio[freqdict[i]] += cateannodict[i] / cateimgdict[i]
#             freqcnt[freqdict[i]] += 1
#             freqanno[freqdict[i]] += cateannodict[i]
#             freqimg[freqdict[i]] += cateimgdict[i]

#         for group in ['f', 'c', 'r']:
#             print(f"version pre: {group}: {freqratio[group] / freqcnt[group]}")
#             print(f"version post: {group}: {freqanno[group] / freqimg[group]}")


catedict = {}
newfreqdict = {}
freqdict = {}
for c in data['categories']:
    catedict[c['id']] = c['name']
    if c['name'] == 'speaker_(stero_equipment)':
        newfreqdict['speaker_(stereo_equipment)'] = c['frequency']
    else:
        newfreqdict[c['name']] = c['frequency']
    freqdict[c['id']] = c['frequency']
# catelist = [0] * 1204
# for id, name in catedict.items():
#     catelist[id] = name

with open(f"/data/zhaoliang/ltod/eqlv2/data/lvis/annotations/lvis_v1_val.json", 'r') as f:
    data = json.load(f)
img_path = '/data/zhaoliang/ltod/eqlv2/data/lvis/val2017/'
cnt = 0
for idx, name in enumerate(glob.glob(img_path + '*.jpg')):
    # print(name)

    img_id = -1
    for img in data['images']:
        if name.split('/')[-1] == img['coco_url'].split('/')[-1]:
            img_id = img['id']
            break

    has_rare = False
    gt_bboxes, gt_labels, gt_masks = [], [], []
    for anno in data['annotations']:
        if anno['image_id'] == img_id:
            gt_bboxes.append(anno['bbox'])
            gt_labels.append(anno['category_id'])
            if freqdict[anno['category_id']] == 'r':
                has_rare = True
            gt_masks.append(anno['segmentation'])
            # class_names.append(catedict[anno['category_id']])

    if has_rare == False:
        continue
    else:
        cnt += 1
        if cnt >= 100:
            break

    print(name)

    # for sth in [0.1, 0.2, 0.3]:
    # show_result_pyplot(model, name, result=inference_detector(model, name), score_thr=0.003, out_file=f'/data/zhaoliang/ltod/56mmdet/vis/{idx}_logn_final.jpg', freqdict=newfreqdict)
    show_result_pyplot(model, name, result=inference_detector(model, name), score_thr=0.3, out_file=f'/data/zhaoliang/ltod/56mmdet/vis/{idx}_baseline_final.jpg', freqdict=newfreqdict)

    # if len(gt_bboxes) == 0:
    #     continue

    # gt_bboxes = np.array(gt_bboxes)
    # gt_labels = np.array(gt_labels)
    # gt_masks = np.array(gt_masks)
    # # class_names = class_names

    # gt_bboxes[:, 2] += gt_bboxes[:, 0]
    # gt_bboxes[:, 3] += gt_bboxes[:, 1]

    # img = imshow_det_bboxes(
    #     mmcv.imread(name),
    #     gt_bboxes,
    #     gt_labels,
    #     # gt_masks,
    #     None,
    #     class_names=catelist,
    #     bbox_color=(255, 102, 61),
    #     text_color=(255, 102, 61),
    #     mask_color=(255, 102, 61),
    #     thickness=2,
    #     font_size=13,
    #     out_file=f'/data/zhaoliang/ltod/56mmdet/vis/{idx}_gt.jpg',
    #     show=False,
    #     freqdict=freqdict)