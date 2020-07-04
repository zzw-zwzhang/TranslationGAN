import os
import os.path as osp
import sys
import copy
import argparse
import time
import shutil
import itertools
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn

from openunreid.apis import TranslationBaseRunner
from openunreid.models import build_adaption_model
from openunreid.models.losses import build_loss
from openunreid.data import build_train_dataloader
from openunreid.data.transformers import build_test_transformer
from openunreid.core.solvers import build_optimizer, build_lr_scheduler
from openunreid.utils.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.logger import Logger
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.torch_utils import read_image, save_adapted_images


def parge_config():
    parser = argparse.ArgumentParser(description='Pretraining on source-domain datasets')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models', default='')
    parser.add_argument('--save-train-path', default='bounding_box_train',
                        help='relative save path of adapted images: bounding_box_train | train')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    parser.add_argument('--set', dest='set_cfgs', default=None,
                        nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    if not args.work_dir:
        args.work_dir = Path(args.config).stem
    cfg.work_dir = cfg.LOGS_ROOT / args.work_dir
    mkdir_if_missing(cfg.work_dir)
    cfg.save_train_path = args.save_train_path
    # mkdir_if_missing(cfg.save_path)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    shutil.copy(args.config, cfg.work_dir / 'config.yaml')

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    synchronize()

    # init logging file
    logger = Logger(cfg.work_dir / 'log.txt', debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build train loader
    train_loader, train_sets = build_train_dataloader(cfg, joint=False)

    # build model
    Generator, Discriminator, Metric_Net = build_adaption_model(cfg)
    Ga = Generator.cuda(); Gb = Generator.cuda()
    Da = Discriminator.cuda(); Db = Discriminator.cuda()
    MeNet = Metric_Net.cuda()

    if dist:
        ddp_cfg = {'device_ids': [cfg.gpu], 'output_device': cfg.gpu, 'find_unused_parameters': True}
        Ga = nn.parallel.DistributedDataParallel(Ga, **ddp_cfg)
        Gb = nn.parallel.DistributedDataParallel(Gb, **ddp_cfg)
        Da = nn.parallel.DistributedDataParallel(Da, **ddp_cfg)
        Db = nn.parallel.DistributedDataParallel(Db, **ddp_cfg)
        MeNet = nn.parallel.DistributedDataParallel(MeNet, **ddp_cfg)
    elif (cfg.total_gpus>1):
        Ga = nn.DataParallel(Ga); Gb = nn.DataParallel(Gb)
        Da = nn.DataParallel(Da); Db = nn.DataParallel(Db)
        MeNet = nn.DataParallel(MeNet)

    Gs = {'Ga': Ga, 'Gb': Gb}
    Ds = {'Da': Da, 'Db': Db}

    # build optimizer
    optimizer_G = build_optimizer(itertools.chain(Ga.parameters(), Gb.parameters()), **cfg.TRAIN.OPTIM)
    optimizer_D = build_optimizer(itertools.chain(Da.parameters(), Db.parameters()), **cfg.TRAIN.OPTIM)
    optimizer_MeNet = build_optimizer(MeNet.parameters(), **cfg.TRAIN.OPTIM)
    optimizers = {'G': optimizer_G, 'D': optimizer_D, 'MeNet': optimizer_MeNet,}

    # build lr_scheduler
    if cfg.TRAIN.SCHEDULER.lr_scheduler is not None:
        lr_scheduler_G = build_lr_scheduler(optimizer_G, **cfg.TRAIN.SCHEDULER)
        lr_scheduler_D = build_lr_scheduler(optimizer_D, **cfg.TRAIN.SCHEDULER)
        lr_scheduler_MeNet = build_lr_scheduler(optimizer_MeNet, **cfg.TRAIN.SCHEDULER)
    else:
        lr_scheduler_G = None; lr_scheduler_D = None; lr_scheduler_MeNet = None
    lr_schedulers = {'G': lr_scheduler_G, 'D': lr_scheduler_D, 'MeNet': lr_scheduler_MeNet}


    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, cuda=True)

    # build runner
    runner = TranslationBaseRunner(
        cfg,
        [Gs, Ds, MeNet],
        optimizers,
        criterions,
        train_loader,
        lr_schedulers=None
    )

    # resume
    if args.resume_from:
        runner.resume(args.resume_from)

    # start training
    runner.run()

    # save models
    '''
    runner.save_model({'Da': Da.state_dict(),
                     'Db': Db.state_dict(),
                     'Ga': Ga.state_dict(),
                     'Gb': Gb.state_dict(),
                     'MeNet': MeNet.state_dict()}, cfg.work_dir)
    '''

    # load the best model
    if args.resume_from:
        ckpt = runner.resume(args.resume_from)
        Da.load_state_dict(ckpt['Da'])
        Db.load_state_dict(ckpt['Db'])
        Ga.load_state_dict(ckpt['Ga'])
        Gb.load_state_dict(ckpt['Gb'])
        MeNet.load_state_dict(ckpt['Me'])

    # final testing
    print('^---v---^---v---^--- Begin Testing ---^---v---^---v---^')
    test_loader, test_sets = build_train_dataloader(cfg, joint=False)
    data_source = test_sets[0].data
    data_target = test_sets[1].data
    test_transformer = build_test_transformer(cfg)

    print('============== Adapted Source Images ==============')
    for i, path1 in enumerate(data_source):
        input1 = read_image(path1[0])
        input1 = test_transformer(input1)
        input1 = torch.unsqueeze(input1, 0)
        adapted_source = Ga(input1)

        if i % 200 == 0:
            print('processing (%05d)-th source image...' % i)
        save_adapted_images(cfg, path1, adapted_source)

    print('============== Adapted Target Images ==============')
    for j, path2 in enumerate(data_target):
        input2 = read_image(path2[0])
        input2 = test_transformer(input2)
        input2 = torch.unsqueeze(input2, 0)
        adapted_target = Ga(input2)

        if j % 200 == 0:
            print('processing (%05d)-th source image...' % j)
        save_adapted_images(cfg, path2, adapted_target)

    # print time
    end_time = time.monotonic()
    print('Total running time: ',
          timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()
