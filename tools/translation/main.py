import os
import sys
import time
import shutil
import argparse
import itertools
from pathlib import Path
from datetime import timedelta

import torch
import torchvision
import torch.nn as nn

from openunreid.apis.train import batch_processor
from openunreid.apis import TranslationBaseRunner, test_translation
from openunreid.models import build_adaption_model
from openunreid.models.losses import build_loss
from openunreid.data import build_train_dataloader, build_val_dataloader
from openunreid.core.solvers import build_optimizer, build_lr_scheduler
from openunreid.utils.config import cfg, cfg_from_yaml_file, cfg_from_list, log_config_to_file
from openunreid.utils.dist_utils import init_dist, synchronize
from openunreid.utils.logger import Logger
from openunreid.utils.file_utils import mkdir_if_missing
from openunreid.utils import bcolors

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


class SPGANRunner(TranslationBaseRunner):

    def train_step(self, batch_source, batch_target):
        data_source = batch_processor(batch_source, self.cfg.MODEL.dsbn)
        data_target = batch_processor(batch_target, self.cfg.MODEL.dsbn)
        self.real_A = data_source['img'][0].cuda()
        self.real_B = data_target['img'][0].cuda()

        # Forward
        self.fake_B = self.Gb(self.real_A)  # G_B(A)
        self.rec_A = self.Ga(self.fake_B)   # G_A(G_B(A))
        self.fake_A = self.Gb(self.real_B)  # G_A(B)
        self.rec_B = self.Gb(self.fake_A)   # G_B(G_A(B))

        # save translated images
        pictures = (torch.cat([self.real_A, self.fake_B, self.rec_A,
                               self.real_B, self.fake_A, self.rec_B], dim=0).data + 1) / 2.0
        torchvision.utils.save_image(pictures, '%s/epoch_%d.jpg'
                                     % (self.save_dir, self._epoch + 1), nrow=4)

        # G_A and G_B
        if self._iter % 2 == 0:
            self.optimizers['G'].zero_grad()
            self.backward_G()
            self.optimizers['G'].step()

        if self._epoch > 0:
            self.optimizers['MeNet'].zero_grad()
            self.backward_MeNet()
            self.optimizers['MeNet'].step()

        # D_A and D_B
        self.optimizers['D'].zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizers['D'].step()

        if self._epoch == 0:
            # print('Epoch: [0]--------MeNet does not update--------')
            meters = {'adversarial': self.loss_adv_G + self.loss_D_A + self.loss_D_B,
                      'cycle_consistent': self.loss_cycle,
                      'identity': self.loss_idt,
                      'contrastive': 0.000}
            self.train_progress.update(meters)
        else:
            meters = {'adversarial': self.loss_adv_G + self.loss_D_A + self.loss_D_B,
                      'cycle_consistent': self.loss_cycle,
                      'identity': self.loss_idt,
                      'contrastive': self.loss_MeNet}
            self.train_progress.update(meters)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # Adversarial loss D_A(G_A(B))
        loss_G_A = self.criterions['adversarial'](self.Da(self.fake_A), True)
        # Adversarial loss D_B(G_B(A))
        loss_G_B = self.criterions['adversarial'](self.Db(self.fake_B), True)
        loss_adv_G = loss_G_A + loss_G_B
        self.loss_adv_G = loss_adv_G.item()

        # Forward cycle loss || G_A(G_B(A)) - A||
        loss_cycle_A = self.criterions['cycle_consistent'](self.rec_A, self.real_A)
        # Backward cycle loss || G_B(G_A(B)) - B||
        loss_cycle_B = self.criterions['cycle_consistent'](self.rec_B, self.real_B)
        loss_cycle = (loss_cycle_A + loss_cycle_B) \
                     * self.cfg.TRAIN.LOSS.losses['cycle_consistent']
        self.loss_cycle = loss_cycle.item()

        # G_A should be identity if real_B is fed: ||G_B(B) - B||
        self.idt_A = self.Gb(self.real_B)
        loss_idt_A = self.criterions['identity'](self.idt_A, self.real_B)
        # G_B should be identity if real_A is fed: ||G_A(A) - A||
        self.idt_B = self.Ga(self.real_A)
        loss_idt_B = self.criterions['identity'](self.idt_B, self.real_A)
        loss_idt = (loss_idt_A + loss_idt_B) * self.cfg.TRAIN.LOSS.losses['identity']
        self.loss_idt = loss_idt.item()

        # Contrastive loss for G
        self.con_A_G  = self.MeNet(self.real_A)  # x_S
        self.con_B_G  = self.MeNet(self.real_B)  # x_T
        self.conA2B_G = self.MeNet(self.fake_B)  # G(x_S)
        self.conB2A_G = self.MeNet(self.fake_A)  # F(x_T)
        # positive pairs
        loss_pos0_G = self.criterions['contrastive'](self.con_A_G, self.conA2B_G, 1)  # X_S and G(X_S)
        loss_pos1_G = self.criterions['contrastive'](self.con_B_G, self.conB2A_G, 1)  # X_T and F(X_T)
        # negative pairs
        loss_neg0_G = self.criterions['contrastive'](self.con_A_G, self.conB2A_G, 0)  # x_S and F(x_T)
        loss_neg1_G = self.criterions['contrastive'](self.con_B_G, self.conA2B_G, 0)  # x_T and G(x_S)
        # contrastive loss
        loss_MeNet_G = (loss_pos0_G + loss_pos1_G + 0.5 * (loss_neg0_G + loss_neg1_G)) / 4.0

        # combined loss and calculate gradients
        if self._epoch > 0:
            loss_G = loss_adv_G + loss_cycle + loss_idt + loss_MeNet_G * self.cfg.TRAIN.LOSS.losses['contrastive']
        else:
            loss_G = loss_adv_G + loss_cycle + loss_idt
        loss_G.backward()

    def backward_MeNet(self):
        """Calculate contrastive loss for MeNet

        Contrastive loss (reference to Sec 3.2.2 of SPGAN paper)
        positive pairs: x_S and G(x_S), x_T and F(x_T)
        negative pairs: x_S and F(x_T), x_T and G(x_S)
        """

        self.con_A = self.MeNet(self.real_A)  # x_S
        self.con_B = self.MeNet(self.real_B)  # x_T
        self.conA2B = self.MeNet(self.fake_B.detach())  # G(x_S)
        self.conB2A = self.MeNet(self.fake_A.detach())  # F(x_T)

        # positive pairs
        loss_pos0 = self.criterions['contrastive'](self.con_A, self.conA2B, 1)  # X_S and G(X_S)
        loss_pos1 = self.criterions['contrastive'](self.con_B, self.conB2A, 1)  # X_T and F(X_T)
        # negative pairs
        loss_neg = self.criterions['contrastive'](self.con_A, self.con_B, 0)  # # X_S and X_T

        # contrastive loss for G
        loss_MeNet = (loss_pos0 + loss_pos1 + 2 * loss_neg) / 3.0
        self.loss_MeNet = loss_MeNet.item()
        loss_MeNet.backward()

    def save_model(self):
        print(bcolors.OKGREEN + '\n * Finished epoch {:2d}'.format(self._epoch) + bcolors.ENDC)
        print(" Saving models...")
        save_path = self.cfg.work_dir
        if (self._rank == 0):
            torch.save(self.Ga.state_dict(), '%s/Ga.pth' % save_path)
            torch.save(self.Gb.state_dict(), '%s/Gb.pth' % save_path)
            torch.save(self.Da.state_dict(), '%s/Da.pth' % save_path)
            torch.save(self.Db.state_dict(), '%s/Db.pth' % save_path)
            torch.save(self.MeNet.state_dict(), '%s/MeNet.pth' % save_path)
        print("\tDone.\n")

    def resume(self):
        resume_path = self.cfg.resume_from
        print("\nLoading pre-trained models.")
        self.Ga.load_state_dict(torch.load(os.path.join(resume_path, 'Ga.pth')))
        self.Gb.load_state_dict(torch.load(os.path.join(resume_path, 'Gb.pth')))
        self.Da.load_state_dict(torch.load(os.path.join(resume_path, 'Da.pth')))
        self.Db.load_state_dict(torch.load(os.path.join(resume_path, 'Db.pth')))
        self.MeNet.load_state_dict(torch.load(os.path.join(resume_path, 'MeNet.pth')))
        print("\tDone.\n")


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
    cfg.resume_from = args.resume_from
    cfg.save_train_path = args.save_train_path
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
        lr_schedulers = {'G': lr_scheduler_G, 'D': lr_scheduler_D, 'MeNet': lr_scheduler_MeNet}
    else:
        lr_schedulers = None

    # build loss functions
    criterions = build_loss(cfg.TRAIN.LOSS, cuda=True)

    # build runner
    if cfg.MODEL.metric_net:
        runner = SPGANRunner(
            cfg,
            [Gs, Ds, MeNet],
            optimizers,
            criterions,
            train_loader,
            lr_schedulers=lr_schedulers)
    else:
        runner = TranslationBaseRunner(
            cfg,
            [Gs, Ds, MeNet],
            optimizers,
            criterions,
            train_loader,
            lr_schedulers=lr_schedulers)

    # resume
    if args.resume_from:
        runner.resume()

    # start training
    runner.run()

    # final testing
    print('^---v---^---v---^--- Begin Testing ---^---v---^---v---^')
    test_loader, test_sets = build_val_dataloader(cfg, for_clustering=True, all_datasets=True)

    test_translation(cfg, Gb, test_loader[0])
    test_translation(cfg, Ga, test_loader[1])

    # print time
    end_time = time.monotonic()
    print('Total running time: ',
          timedelta(seconds=end_time - start_time))


if __name__ == '__main__':
    main()