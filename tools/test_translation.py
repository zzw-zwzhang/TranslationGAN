import os
import sys
import time
import argparse
from pathlib import Path
from datetime import timedelta

import torch.nn as nn

from openunreid.apis.test import test_translation
from openunreid.models import build_adaption_model
from openunreid.data import build_val_dataloader
from openunreid.utils.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.logger import Logger
from openunreid.utils.torch_utils import load_checkpoint_translation


def parge_config():
    parser = argparse.ArgumentParser(description='Pretraining on source-domain datasets')
    parser.add_argument('resume', help='the checkpoint file to resume from')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--save-train-path', default='bounding_box_train',
                        help='relative save path of adapted images: bounding_box_train | train')
    parser.add_argument('--launcher', type=str,
                        choices=['none', 'pytorch', 'slurm'],
                        default='none', help='job launcher')
    parser.add_argument('--tcp-port', type=str, default='5017')
    parser.add_argument('--set', dest='set_cfgs', default=None,
                        nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')
    args = parser.parse_args()

    cfg.resume = args.resume
    args.resume = Path(args.resume)
    cfg.work_dir = args.resume
    if not args.config:
        args.config = cfg.work_dir / 'config.yaml'
    cfg_from_yaml_file(args.config, cfg)
    cfg.launcher = args.launcher
    cfg.tcp_port = args.tcp_port
    cfg.MODEL.sync_bn = False  # not required for inference
    cfg.save_train_path = args.save_train_path
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    start_time = time.monotonic()

    # init distributed training
    args, cfg = parge_config()
    dist = init_dist(cfg)
    synchronize()

    # init logging file
    logger = Logger(os.path.join(cfg.resume, 'log_test.txt'), debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build model
    Generator, Discriminator, Metric_Net = build_adaption_model(cfg)
    Ga = Generator.cuda()
    Gb = Generator.cuda()

    if dist:
        ddp_cfg = {'device_ids': [cfg.gpu], 'output_device': cfg.gpu, 'find_unused_parameters': True}
        Ga = nn.parallel.DistributedDataParallel(Ga, **ddp_cfg)
        Gb = nn.parallel.DistributedDataParallel(Gb, **ddp_cfg)
    elif (cfg.total_gpus > 1):
        Ga = nn.DataParallel(Ga)
        Gb = nn.DataParallel(Gb)

    # load checkpoint
    Ga, Gb = load_checkpoint_translation(Ga, Gb, cfg)

    # load test data_loader
    test_loader, test_sets = build_val_dataloader(cfg, for_clustering=True, all_datasets=True)

    test_translation(cfg, Gb, test_loader[0])
    test_translation(cfg, Ga, test_loader[1])

    # print time
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()