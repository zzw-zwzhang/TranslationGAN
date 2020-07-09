# Written by Zhiwei Zhang

import os
import time
from datetime import timedelta

import torch
import torchvision

from ..utils.file_utils import mkdir_if_missing


@torch.no_grad()
def test_translation(
        cfg,
        model,
        dataset
    ):

    start_time = time.monotonic()
    print("\n***************************** Start Translating Images *****************************\n")

    root_path = cfg.work_dir
    train_path = cfg.save_train_path
    model = model.cuda()
    i = 0
    for data in dataset:
        model.eval()
        img = data['img'].cuda()
        path = data['path']
        adapted_img = model(img)

        batch_size = len(path)
        for j in range(batch_size):
            dataset_name = path[j].split("/")[-4]
            img_name = '%s.jpg' % path[j].split("/")[-1][:-4]
            save_path = os.path.join(root_path, dataset_name, train_path, img_name)
            mkdir_if_missing(os.path.join(root_path, dataset_name, train_path))

            torchvision.utils.save_image((adapted_img.data[j] + 1) / 2.0, save_path, padding=0)
        i = i + batch_size
        if i % 200 == 0: print('processing (%05d)-th image...' % i)

    end_time = time.monotonic()
    print('Translating time: ', timedelta(seconds=end_time - start_time))
    print("\n*************************** Finished Translating Images ****************************\n")