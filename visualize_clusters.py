import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
from functools import partial
import torch
from typing import Optional
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from log import make_logger
from glob import glob
from natsort import natsorted
import seaborn as sns


torch.cuda.set_device(1)


from conditional_evaluation import ConditionalEvaluation

from datasets.TextImageFilesDataset import TextImageFilesDataset
from utils import load_and_concatenate_all


# Define file paths
image_folders = [

]

captions_files = [

]

image_feats_paths = [

]

text_feats_paths = [

]

top_text = 6
top_image = 5

m, max_id = text_eigvals.topk(top_text)


for i in range(top_text):
    top_eig_text = text_eigvecs[:, max_id[i]]

    # Text cluster
    if top_eig_text.sum() < 0:
        top_eig_text = -top_eig_text
    topk_id_text = top_eig_text.argsort(descending=True)[:100]

    prompts = []
    for k, idx in enumerate(topk_id_text.cpu()):
        # text_clusters[i].append(int(idx.cpu()))
        prompts.append(dataset[idx][1])

    file1 = open(f'{folder_name}/text-cluster={i}-prompts.txt', 'w')
    file1.writelines("\n".join(prompts))
    file1.close()

    top_eig_text = top_eig_text.reshape((-1, 1)) # [feature_dim, 1]  TODO: check if it's ok
    K_ut = top_eig_text @ top_eig_text.T
    KoU = K_i * (K_ut)
    KoU = KoU / KoU.trace()

    cond, mutual, joint, x, y = EvalModel.conditional_entropy(K_i, K_ut, order=order, compute_kernel=False)
    logger.info(f'order= {order}, text-cluster={i}: cond: {torch.exp(cond)}, mutual: {torch.exp(mutual)}, images: {torch.exp(x)}, text-cluster: {torch.exp(y)}')

    img_eigvals, img_eigvecs = torch.linalg.eigh(KoU)
    img_eigvals = img_eigvals.real
    img_eigvecs = img_eigvecs.real


    _, max_id_img = img_eigvals.topk(top_image)

    for j in range(top_image):
        top_eig_img = img_eigvecs[:, max_id_img[j]]
        if top_eig_img.sum() < 0:
            top_eig_img = -top_eig_img
        topk_id = top_eig_img.argsort(descending=True)[:36]


        # save_folder_name = os.path.join(args.path_save_visual, 'backbone_{}/{}_{}/'.format(args.backbone, args.visual_name, now_time), 'top{}'.format(i+1))
        save_folder_name = f'{folder_name}/text-cluster={i}/{j}'
        os.makedirs(save_folder_name, exist_ok=True)

        summary = []

        for k, idx in enumerate(topk_id.cpu()):
            print(idx)
            top_imgs = dataset[idx][0]
            summary.append(top_imgs)
            # save_image(top_imgs, os.path.join(save_folder_name, '{}.png'.format(k)), nrow=1)

        # save_image(summary, os.path.join(save_folder_name, 'summary.jpg'), nrow=6)
        save_image(summary[:9], os.path.join(folder_name, f'text={i}_image={j}.jpg'), nrow=3)
