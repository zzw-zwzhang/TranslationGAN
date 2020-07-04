# Written by Yixiao Ge

import os
import os.path as osp
import shutil
import time
import torch
import warnings
import collections
import numpy as np

from .train import batch_processor, set_random_seed
from .test import val_reid
from ..data import build_train_dataloader, build_val_dataloader
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint
from ..utils.meters import Meters
from ..utils.image_pool import ImagePool
from ..core.label_generators import LabelGenerator
from ..core.metrics.accuracy import accuracy
from ..core.solvers import build_optimizer, build_lr_scheduler
from ..utils import bcolors


class BaseRunner(object):
    """
    Base Runner
    """

    def __init__(
            self,
            cfg,
            model,
            optimizer,
            criterions,
            train_loader,
            train_sets = None,
            lr_scheduler = None,
            meter_formats = {'Time': ':.3f',
                            # 'Data': ':.3f',
                            'Acc@1': ':.2%'},
            print_freq = 10,
            reset_optim = True,
            label_generator = None,
        ):
        super(BaseRunner, self).__init__()
        set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.criterions = criterions
        self.lr_scheduler = lr_scheduler
        self.print_freq = print_freq
        self.reset_optim = reset_optim
        self.label_generator = label_generator

        self.is_pseudo = 'PSEUDO_LABELS' in self.cfg.TRAIN and \
                        self.cfg.TRAIN.unsup_dataset_indexes is not None
        if self.is_pseudo:
            if (self.label_generator is None):
                self.label_generator = LabelGenerator(self.cfg, self.model)

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0
        self._best_mAP = 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets
        self.val_loader, self.val_set = build_val_dataloader(cfg)

        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ':.3f'
        self.train_progress = Meters(meter_formats, self.cfg.TRAIN.iters, prefix='Train: ')


    def run(self):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # generate pseudo labels
            if self.is_pseudo:
                if (ep % self.cfg.TRAIN.PSEUDO_LABELS.freq == 0 or ep == self._start_epoch):
                    self.update_labels()
                    synchronize()

            # train
            self.train()
            synchronize()

            # validate
            if ((ep+1) % self.cfg.TRAIN.val_freq == 0 or (ep+1) == self.cfg.TRAIN.epochs):
                mAP = self.val()
                self.save(mAP)

            # update learning rate
            if (self.lr_scheduler is not None):
                self.lr_scheduler.step()

            # synchronize distributed processes
            synchronize()


    def update_labels(self):
        print ("\n************************* Start updating pseudo labels on epoch {} *************************\n"
                    .format(self._epoch))

        # generate pseudo labels
        pseudo_labels, label_centers = self.label_generator(
                                            self._epoch,
                                            print_freq=self.print_freq
                                        )

        # update train loader
        self.train_loader, self.train_sets = build_train_dataloader(
                                                    self.cfg,
                                                    pseudo_labels,
                                                    self.train_sets,
                                                    self._epoch,
                                                )

        # update criterions
        if ('cross_entropy' in self.criterions.keys()):
            self.criterions['cross_entropy'].num_classes = self.train_loader.loader.dataset.num_pids

        # reset optim (optional)
        if self.reset_optim:
            self.optimizer.state = collections.defaultdict(dict)

        # update classifier centers
        start_cls_id = 0
        for idx, dataset in enumerate(self.cfg.TRAIN.datasets.keys()):
            if (idx in self.cfg.TRAIN.unsup_dataset_indexes):
                labels = torch.arange(start_cls_id, start_cls_id+self.train_sets[idx].num_pids)
                centers = label_centers[self.cfg.TRAIN.unsup_dataset_indexes.index(idx)]
                if isinstance(self.model, list):
                    for model in self.model:
                        model.module.initialize_centers(centers, labels)
                else:
                    self.model.module.initialize_centers(centers, labels)
            start_cls_id += self.train_sets[idx].num_pids

        print ("\n****************************** Finished updating pseudo label ******************************\n")


    def train(self):
        # one loop for training
        if isinstance(self.model, list):
            for model in self.model:
                model.train()
        else:
            self.model.train()

        self.train_progress.reset(prefix='Epoch: [{}]'.format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)

        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):

            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()
            # self.train_progress.update({'Data': time.time()-end})

            loss = self.train_step(iter, batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.train_progress.update({'Time': time.time()-end})
            end = time.time()

            if (iter % self.print_freq == 0):
                self.train_progress.display(iter)


    def train_step(self, iter, batch):
        # need to be re-written case by case
        assert not isinstance(self.model, list), "please re-write 'train_step()' to support list of models"

        data = batch_processor(batch, self.cfg.MODEL.dsbn)
        if (len(data['img'])>1):
            warnings.warn("please re-write the 'runner.train_step()' function to make use of mutual transformer.")

        inputs = data['img'][0].cuda()
        targets = data['id'].cuda()

        results = self.model(inputs)
        if ('prob' in results.keys()):
            results['prob'] = results['prob'][:,:self.train_loader.loader.dataset.num_pids]

        total_loss = 0
        meters = {}
        for key in self.criterions.keys():
            loss = self.criterions[key](results, targets)
            total_loss += loss * float(self.cfg.TRAIN.LOSS.losses[key])
            meters[key] = loss.item()

        if ('prob' in results.keys()):
            acc = accuracy(results['prob'].data, targets.data)
            meters['Acc@1'] = acc[0]
            
        self.train_progress.update(meters)

        return total_loss


    def val(self):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        better_mAP = 0
        for idx in range(len(model_list)):
            if (len(model_list)>1):
                print ("==> Val on the no.{} model".format(idx))
            cmc, mAP = val_reid(
                            self.cfg,
                            model_list[idx],
                            self.val_loader[0],
                            self.val_set[0],
                            self._epoch,
                            self.cfg.TRAIN.val_dataset,
                            self._rank,
                            print_freq = self.print_freq
                        )
            better_mAP = max(better_mAP, mAP)

        return better_mAP


    def save(self, mAP):
        is_best = mAP > self._best_mAP
        self._best_mAP = max(self._best_mAP, mAP)
        print(bcolors.OKGREEN+'\n * Finished epoch {:3d}  mAP: {:5.1%}  best: {:5.1%}{}\n'.
                  format(self._epoch, mAP, self._best_mAP, ' *' if is_best else '')+bcolors.ENDC)

        fpath = osp.join(self.cfg.work_dir, 'checkpoint.pth')
        if (self._rank == 0):
            # only on cuda:0
            self.save_model(is_best, fpath)


    def save_model(self, is_best, fpath):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        state_dict = {}
        for idx, model in enumerate(model_list):
            state_dict['state_dict_'+str(idx+1)] = model.state_dict()
        state_dict['epoch'] = self._epoch + 1
        state_dict['best_mAP'] = self._best_mAP

        save_checkpoint(state_dict, is_best, fpath=fpath)


    def resume(self, path):
        # resume from a training checkpoint (not source pretrain)
        state_dict = load_checkpoint(path)
        self.load_model(state_dict)
        synchronize()


    def load_model(self, state_dict):
        if not isinstance(self.model, list):
            model_list = [self.model]
        else:
            model_list = self.model

        for idx, model in enumerate(model_list):
            copy_state_dict(state_dict['state_dict_'+str(idx+1)], model)

        self._start_epoch = state_dict['epoch']
        self._best_mAP = state_dict['best_mAP']


    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size


class TranslationBaseRunner(object):
    """
    Adaption Base Runner
    """

    def __init__(
            self,
            cfg,
            models,
            optimizers,
            criterions,
            train_loader,
            train_sets=None,
            lr_schedulers=None,
            meter_formats={'Time': ':.3f'},
            print_freq=10,
            reset_optim=True,
    ):
        super(TranslationBaseRunner, self).__init__()
        set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        self.cfg = cfg
        self.models = models
        self.optimizers = optimizers
        self.criterions = criterions
        self.lr_schedulers = lr_schedulers
        self.print_freq = print_freq
        self.reset_optim = reset_optim

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets

        self.fake_A_pool = ImagePool(50)
        self.fake_B_pool = ImagePool(50)

        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ':.3f'
        self.train_progress = Meters(meter_formats, self.cfg.TRAIN.iters, prefix='Train: ')

    def run(self):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # train
            self.train()
            synchronize()

            # update learning rate
            if (self.lr_schedulers is not None):
                if self._epoch > 0:
                    self.lr_schedulers['G'].step()
                    self.lr_schedulers['D'].step()
                    self.lr_schedulers['MeNet'].step()

            # synchronize distributed processes
            synchronize()

    def train(self):
        self.models[0]['Ga'].train()
        self.models[0]['Gb'].train()
        self.models[1]['Da'].train()
        self.models[1]['Db'].train()
        self.models[2].train() # MeNet

        self.train_progress.reset(prefix='Epoch: [{}]'.format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader_source in self.train_loader:
                loader_source.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)


        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):
            self._iter = iter

            batch_source = self.train_loader[0].next()
            batch_target = self.train_loader[1].next()

            # forward
            self.train_step(batch_source, batch_target)

            self.train_progress.update({'Time': time.time() - end})
            end = time.time()

            if (iter % self.print_freq == 0) and self._epoch > 0:
                self.train_progress.display(iter)

    def train_step(self, batch_source, batch_target):
        assert isinstance(self.models, list)
        data_source = batch_processor(batch_source, self.cfg.MODEL.dsbn)
        data_target = batch_processor(batch_target, self.cfg.MODEL.dsbn)
        self.real_A = data_source['img'][0].cuda()
        self.real_B = data_target['img'][0].cuda()

        # Forward
        self.fake_B = self.models[0]['Ga'](self.real_A)    # G_A(A)
        self.rec_A =  self.models[0]['Gb'](self.fake_B)    # G_B(G_A(A))
        self.fake_A = self.models[0]['Gb'](self.real_B)    # G_B(B)
        self.rec_B =  self.models[0]['Ga'](self.fake_A)    # G_A(G_B(B))


        # G_A and G_B
        # self.set_requires_grad([self.models[1]['Da'], self.models[1]['Db']], False)
        if self._iter % 2 == 0:
            self.optimizers['G'].zero_grad()
            self.backward_G()
            self.optimizers['G'].step()

        if self._epoch > 0:
            self.optimizers['MeNet'].zero_grad()
            self.backward_MeNet()
            self.optimizers['MeNet'].step()

        # D_A and D_B
        # self.set_requires_grad([self.models[1]['Da'], self.models[1]['Db']], True)
        self.optimizers['D'].zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizers['D'].step()

        if self._epoch == 0:
            meters = {'adversarial': self.loss_adv_G + self.loss_D_A + self.loss_D_B,
                      'cycle_consistent': self.loss_cycle,
                      'identity': self.loss_idt,
                      'contrastive': 0.0}
            self.train_progress.update(meters)
        else:
            meters = {'adversarial': self.loss_adv_G + self.loss_D_A + self.loss_D_B,
                      'cycle_consistent': self.loss_cycle,
                      'identity': self.loss_idt,
                      'contrastive': self.loss_MeNet}
            self.train_progress.update(meters)


    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""

        # Adversarial loss D_A(G_A(A))
        self.loss_G_A = self.criterions['adversarial'](self.models[1]['Da'](self.fake_B), True)
        # Adversarial loss D_B(G_B(B))
        self.loss_G_B = self.criterions['adversarial'](self.models[1]['Db'](self.fake_A), True)
        self.loss_adv_G = self.loss_G_A + self.loss_G_B

        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterions['cycle_consistent'](self.rec_A, self.real_A)
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterions['cycle_consistent'](self.rec_B, self.real_B)
        self.loss_cycle = (self.loss_cycle_A + self.loss_cycle_B) \
                          * self.cfg.TRAIN.LOSS.losses['cycle_consistent']

        # G_A should be identity if real_B is fed: ||G_A(B) - B||
        self.idt_A = self.models[0]['Ga'](self.real_B)
        self.loss_idt_A = self.criterions['identity'](self.idt_A, self.real_B)
        # G_B should be identity if real_A is fed: ||G_B(A) - A||
        self.idt_B = self.models[0]['Gb'](self.real_A)
        self.loss_idt_B = self.criterions['identity'](self.idt_B, self.real_A)
        self.loss_idt = (self.loss_idt_A + self.loss_idt_B) * self.cfg.TRAIN.LOSS.losses['identity']

        # Contrastive loss for G
        self.con_A_G = self.models[2](self.real_A)   # x_S
        self.con_B_G = self.models[2](self.real_B)   # x_T
        self.conA2B_G = self.models[2](self.fake_B)  # G(x_S)
        self.conB2A_G = self.models[2](self.fake_A)  # F(x_T)
        # positive pairs
        self.loss_pos0_G = self.criterions['contrastive'](self.con_A_G, self.conA2B_G, 1)  # X_S and G(X_S)
        self.loss_pos1_G = self.criterions['contrastive'](self.con_B_G, self.conB2A_G, 1)  # X_T and F(X_T)
        # negative pairs
        self.loss_neg0_G = self.criterions['contrastive'](self.con_A_G, self.conB2A_G, 0)  # x_S and F(x_T)
        self.loss_neg1_G = self.criterions['contrastive'](self.con_B_G, self.conA2B_G, 0)  # x_T and G(x_S)
        self.loss_neg_G = self.criterions['contrastive'](self.con_A_G, self.con_B_G, 0)    # X_S and X_T
        # contrastive loss
        self.loss_MeNet_G = (self.loss_pos0_G + self.loss_pos1_G + 0.5 * (self.loss_neg0_G + self.loss_neg1_G)) / 4.0

        # combined loss and calculate gradients
        if self._epoch > 0:
            self.loss_G = self.loss_adv_G + self.loss_cycle + self.loss_idt + self.loss_MeNet_G
        else:
            self.loss_G = self.loss_adv_G + self.loss_cycle + self.loss_idt

        self.loss_G.backward()

    def backward_MeNet(self):
        """Calculate contrastive loss for MeNet

        Contrastive loss (reference to Sec 3.2.2 of SPGAN paper)
        positive pairs: x_S and G(x_S), x_T and F(x_T)
        negative pairs: x_S and F(x_T), x_T and G(x_S)
        """

        self.con_A = self.models[2](self.real_A)               # x_S
        self.con_B = self.models[2](self.real_B)               # x_T
        self.conA2B = self.models[2](self.fake_B.detach())     # G(x_S)
        self.conB2A = self.models[2](self.fake_A.detach())     # F(x_T)

        # positive pairs
        self.loss_pos0 = self.criterions['contrastive'](self.con_A, self.conA2B, 1)    # X_S and G(X_S)
        self.loss_pos1 = self.criterions['contrastive'](self.con_B, self.conB2A, 1)    # X_T and F(X_T)
        # negative pairs
        # self.loss_neg0 = self.criterions['contrastive'](self.con_A, self.conB2A, 0)  # x_S and F(x_T)
        # self.loss_neg1 = self.criterions['contrastive'](self.con_B, self.conA2B, 0)  # x_T and G(x_S)
        self.loss_neg = self.criterions['contrastive'](self.con_A, self.con_B, 0)      # # X_S and X_T

        # contrastive loss for G
        self.loss_MeNet = (self.loss_pos0 + self.loss_pos1 + 2*self.loss_neg) / 3.0
        self.loss_MeNet.backward()

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """

        # Real
        pred_real = netD(real)
        loss_D_real = self.criterions['adversarial'](pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterions['adversarial'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.models[1]['Da'], self.real_B, fake_B)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.models[1]['Db'], self.real_A, fake_A)

    def save_model(state, save_path, is_best=False, max_keep=None):
        # save checkpoint
        torch.save(state, save_path)

        # deal with max_keep
        save_dir = os.path.dirname(save_path)
        list_path = os.path.join(save_dir, 'latest_checkpoint')

        save_path = os.path.basename(save_path)
        if os.path.exists(list_path):
            with open(list_path) as f:
                ckpt_list = f.readlines()
                ckpt_list = [save_path + '\n'] + ckpt_list
        else:
            ckpt_list = [save_path + '\n']

        if max_keep is not None:
            for ckpt in ckpt_list[max_keep:]:
                ckpt = os.path.join(save_dir, ckpt[:-1])
                if os.path.exists(ckpt):
                    os.remove(ckpt)
            ckpt_list[max_keep:] = []

        with open(list_path, 'w') as f:
            f.writelines(ckpt_list)

        # copy best
        if is_best:
            shutil.copyfile(save_path, os.path.join(save_dir, 'model_best.ckpt'))

    def resume(self, ckpt_dir_or_file, map_location=None, load_best=False):
        if os.path.isdir(ckpt_dir_or_file):
            if load_best:
                ckpt_path = os.path.join(ckpt_dir_or_file, 'model_best.ckpt')
            else:
                with open(os.path.join(ckpt_dir_or_file, 'latest_checkpoint')) as f:
                    ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
        else:
            ckpt_path = ckpt_dir_or_file
        ckpt = torch.load(ckpt_path, map_location=map_location)
        print(' [*] Loading checkpoint from %s succeed!' % ckpt_path)
        return ckpt

    def load_model(self, state_dict):
        if not isinstance(self.models, list):
            model_list = [self.models]
        else:
            model_list = self.models

        for idx, model in enumerate(model_list):
            copy_state_dict(state_dict['state_dict_' + str(idx + 1)], model)

        self._start_epoch = state_dict['epoch']

    @property
    def epoch(self):
        """int: Current epoch."""
        return self._epoch

    @property
    def rank(self):
        """int: Rank of current process. (distributed training)"""
        return self._rank

    @property
    def world_size(self):
        """int: Number of processes participating in the job.
        (distributed training)"""
        return self._world_size