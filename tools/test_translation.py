import sys
import argparse
import time
import shutil
from datetime import timedelta
from pathlib import Path

import torch
import torch.nn as nn

from openunreid.apis import test_translation
from openunreid.models import build_adaption_model
from openunreid.data import build_train_dataloader
from openunreid.utils.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.logger import Logger
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils.torch_utils import copy_state_dict, load_checkpoint


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
    logger = Logger(cfg.work_dir / 'log_test.txt', debug=False)
    sys.stdout = logger
    print("==========\nArgs:{}\n==========".format(args))
    log_config_to_file(cfg)

    # build model
    Generator, Discriminator, Metric_Net = build_adaption_model(cfg)
    Generator.cuda()

    if dist:
        Generator = torch.nn.parallel.DistributedDataParallel(
                    Generator,
                    device_ids=[cfg.gpu],
                    output_device=cfg.gpu,
                    find_unused_parameters=True,
                )
    elif (cfg.total_gpus>1):
        Generator = torch.nn.DataParallel(Generator)

    # load checkpoint
    state_dict = load_checkpoint(args.resume)

    # load test data_loader
    test_loader, test_sets = build_train_dataloader(cfg, joint=False)

    for key in state_dict:
        if not key.startswith('state_dict'):
            continue

        print ("==> Test with {}".format(key))
        copy_state_dict(state_dict[key], Generator)

        # start translating
        test_translation(cfg, Generator, test_sets[0])

    # print time
    end_time = time.monotonic()
    print('Total running time: ', timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()