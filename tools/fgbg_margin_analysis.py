from PIL import Image
from mmdet.apis import init_detector, inference_detector, show_result_pyplot
from mmdet.core.visualization import imshow_det_bboxes
import sys
import os.path
import glob
import mmcv
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import seaborn as sns

plt.rc('font',family='Times New Roman')
sns.set_style()

with open(f"/data/zhaoliang/ltod/eqlv2/data/lvis/annotations/lvis_v1_train.json", 'r') as f:
    data = json.load(f)['categories']

catelist = [0] * 1204
for c in data:
    catelist[c['id'] - 1] = c['instance_count']
catelist[-1] = 1e8
catelist = np.array(catelist)
idx = np.argsort(catelist)[::-1].copy()

model = torch.load('./work_dirs/mask_rcnn_r50_fpn_random_logn_normed_mask_mstrain_2x_lvis_v1/epoch_24.pth')['state_dict']
# print(model.keys())
label_sum = np.log(model['roi_head.bbox_head.label_sum'])
moving_mean = model['roi_head.bbox_head.moving_mean']
moving_var = model['roi_head.bbox_head.moving_var']

label_sum = label_sum[idx]
moving_mean = moving_mean[idx]
moving_var = moving_var[idx]
print(label_sum)
print(moving_mean)
print(moving_var)

# moving_mean += label_sum[0] - moving_mean[0]
moving_mean = moving_mean + 13
# normed_label_sum = label_sum.clone()
# lognormed_label_sum = label_sum.clone()
# normed_label_sum[1:] = label_sum[1:] - moving_mean[1:]
# lognormed_label_sum[1:] = label_sum[1:] - moving_mean[1:] + moving_mean[1:].min()
# label_sum[1:] = label_sum[1:] - moving_mean[1:]

time = np.arange(1203 + 1)
fig = plt.figure(figsize=(7, 4))
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# plt.tick_params(labelsize=10)

ax2 = fig.add_subplot(111)
# ax2.set_title('Internal Sampler', y=0.88, fontsize=20)

ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(True)

lns4 = ax2.plot(time, moving_mean, '-', alpha=0.8, label='Logit Mean', color='limegreen', linewidth=1.5)
lns4 = ax2.plot(time, label_sum, '-', alpha=0.8, label='Training Label Distribution', color='pink', linewidth=1.5)
# lns4s = ax2.plot(time, normed_label_sum, '-', alpha=0.8, label = 'Normed TLD', color='green', linewidth=1.3) #violet
# lns4s = ax2.plot(time, lognormed_label_sum, '-', alpha=0.8, label = 'LogNormed TLD', color='violet', linewidth=1.3) #violet
a = ax2.hlines(moving_mean[0], 1, 150, colors='black', linestyles='dashdot')
c = ax2.arrow(150-3, moving_mean[0], 4, 0, color='black',head_width=0.5,head_length=8)
b = ax2.text(168, moving_mean[0] - 0.3, 'Background Logit', fontsize=13)
idx = np.argmax(moving_mean[1:])
# print(idx, moving_mean[idx])
a = ax2.hlines(moving_mean[idx+1], idx+1, 150, colors='black', linestyles='dashdot')
c = ax2.arrow(150-3, moving_mean[idx+1], 4, 0, color='black',head_width=0.5,head_length=8)
b = ax2.text(168, moving_mean[idx+1] - 0.3, 'Maximum Foreground Logit', fontsize=13)
a = ax2.vlines(150-10, moving_mean[idx+1] + 1, moving_mean[0] - 1, colors='black')
c = ax2.arrow(150-10, moving_mean[idx+1] + 2, 0, -1, color='black',head_width=8,head_length=0.5)
c = ax2.arrow(150-10, moving_mean[0] - 2, 0, 1, color='black',head_width=8,head_length=0.5)
b = ax2.text(150, (moving_mean[idx+1] + moving_mean[0]) / 2 - 1.8, 'Margin', rotation=270, fontsize=13)


ax2.plot(0, moving_mean[0], marker="o", markersize=3, markeredgecolor="black", markerfacecolor="black")
ax2.plot(idx+1, moving_mean[idx+1], marker="o", markersize=3, markeredgecolor="black", markerfacecolor="black")

# lns3 = ax2.plot(time, freq_val, '-', alpha=1.0, label = 'Prior Label', color='black', linewidth=2.5, linestyle='--') # 'navy' 'royalblue'


ax2.set_xlabel("Class Index", fontsize=17)

ax2.legend(ncol=1, loc=1, fontsize=14)  # loc=2 #, fontsize=20
ax2.grid(color='gainsboro', linestyle='--', axis='y')
# ax2.set_ylabel(r"Difference of Logarithm", fontsize=17)

ax_min, ax_max = -12, 9
ax2_min, ax2_max = -17, 4
# ax2_yticks = np.arange(ax2_min, ax2_max, 2) + 0.0 + 0.5

# ax2.set_ylim(ax2_min, ax2_max)
# ax2.set_yticks(ax2_yticks)

ax2.axes.yaxis.set_ticklabels([])

plt.tight_layout()

plt.savefig('fgbg_margin_fig.pdf')
# plt.show()
