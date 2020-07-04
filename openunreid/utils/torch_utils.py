from __future__ import absolute_import
import os
import sys
import json
import time
import errno
import numpy as np
import random
import os.path as osp
import warnings
import requests
import shutil
from PIL import Image

import torch
from torch.nn import Parameter

from .dist_utils import get_dist_info, synchronize
from .file_utils import mkdir_if_missing
from . import bcolors


def to_numpy(tensor):
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def read_image(path):
    """Reads image from path using ``PIL.Image``.
    Args:
        path (str): path to an image.
    Returns:
        PIL image
    """
    got_img = False
    if not osp.exists(path):
        raise IOError('"{}" does not exist'.format(path))
    while not got_img:
        try:
            img = Image.open(path).convert('RGB')
            got_img = True
        except IOError:
            print(
                'IOError incurred when reading "{}". Will redo. Don\'t worry. Just chill.'
                .format(path)
            )
    return img


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def save_adapted_images(cfg, path, input):
    root_path = cfg.work_dir
    train_path = cfg.save_train_path
    dataset_name = path[0].split("/")[-4]
    img_name = '%s.jpg' % path[0].split("/")[-1][:-4]

    save_path = os.path.join(root_path, dataset_name, train_path, img_name)
    mkdir_if_missing(os.path.join(root_path, dataset_name, train_path))

    img = tensor2im(input)
    save_image(img, save_path)


def save_image(image_numpy, image_path, aspect_ratio=1.0):
    """Save a numpy image to the disk

    Parameters:
        image_numpy (numpy array) -- input numpy array
        image_path (str)          -- the path of the image
    """

    image_pil = Image.fromarray(image_numpy)
    h, w, _ = image_numpy.shape

    if aspect_ratio > 1.0:
        image_pil = image_pil.resize((h, int(w * aspect_ratio)), Image.BICUBIC)
    if aspect_ratio < 1.0:
        image_pil = image_pil.resize((int(h / aspect_ratio), w), Image.BICUBIC)
    image_pil.save(image_path)


def save_checkpoint(state, is_best, fpath='checkpoint.pth.tar'):
    mkdir_if_missing(osp.dirname(fpath))
    torch.save(state, fpath)
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), 'model_best.pth'))


def load_checkpoint(fpath):
    if osp.isfile(fpath):
        # map to CPU to avoid extra GPU cost
        checkpoint = torch.load(fpath, map_location=torch.device('cpu'))
        print("=> Loaded checkpoint '{}'".format(fpath))
        return checkpoint
    else:
        raise ValueError("=> No checkpoint found at '{}'".format(fpath))


def copy_state_dict(state_dict, model, strip=None):
    tgt_state = model.state_dict()
    copied_names = set()
    unexpected_keys = set()
    for name, param in state_dict.items():
        if (strip is not None and name.startswith(strip)):
            name = name[len(strip):]
        if (name not in tgt_state):
            unexpected_keys.add(name)
            continue
        if isinstance(param, Parameter):
            param = param.data
        if (param.size() != tgt_state[name].size()):
            warnings.warn('mismatch: {} {} {}'.format(name, param.size(), tgt_state[name].size()))
            continue
        tgt_state[name].copy_(param)
        copied_names.add(name)

    missing = set(tgt_state.keys()) - copied_names
    missing = set([m for m in missing if not m.endswith('num_batches_tracked')])
    if len(missing) > 0:
        warnings.warn("missing keys in state_dict: {}".format(missing))
    if len(unexpected_keys)>0:
        warnings.warn("unexpected keys in checkpoint: {}".format(unexpected_keys))

    return model
