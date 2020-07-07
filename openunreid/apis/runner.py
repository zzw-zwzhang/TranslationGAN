# Written by Yixiao Ge

import os
import time
import torch
import warnings
import collections
import os.path as osp

from .test import val_reid
from .train import batch_processor, set_random_seed
from ..data import build_train_dataloader, build_val_dataloader
from ..data.utils.dataset_wrapper import IterLoader
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.torch_utils import copy_state_dict, load_checkpoint, save_checkpoint
from ..utils.meters import Meters
from ..utils.image_pool import ImagePool
from ..core.label_generators import LabelGenerator
from ..core.metrics.accuracy import accuracy
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
            print_freq=100,
            reset_optim=True,
    ):
        super(TranslationBaseRunner, self).__init__()
        set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        self.cfg = cfg
        self.optimizers = optimizers
        self.criterions = criterions
        self.lr_schedulers = lr_schedulers
        self.print_freq = print_freq
        self.reset_optim = reset_optim

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets

        # models
        self.Ga = models[0]['Ga']
        self.Gb = models[0]['Gb']
        self.Da = models[1]['Da']
        self.Db = models[1]['Db']
        if cfg.MODEL.metric_net:
            self.MeNet = models[2]

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

        # save training variables
        for key in criterions.keys():
            meter_formats[key] = ':.3f'
        self.train_progress = Meters(meter_formats, self.cfg.TRAIN.iters, prefix='Train: ')

    def run(self, cfg):
        # the whole process for training
        for ep in range(self._start_epoch, self.cfg.TRAIN.epochs):
            self._epoch = ep

            # train
            self.train(cfg)
            synchronize()

            # validate
            if ((ep + 1) % self.cfg.TRAIN.val_freq == 0 or (ep + 1) == self.cfg.TRAIN.epochs):
                self.save_model(cfg)

            # update learning rate
            if (self.lr_schedulers is not None):
                self.lr_schedulers['G'].step()
                self.lr_schedulers['D'].step()
                if cfg.MODEL.metric_net:
                    self.lr_schedulers['MeNet'].step()

            # synchronize distributed processes
            synchronize()

    def train(self, cfg):
        self.Ga.train()
        self.Gb.train()
        if cfg.MODEL.metric_net:
            self.MeNet.train() # MeNet

        self.train_progress.reset(prefix='Epoch: [{}]'.format(self._epoch))

        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)

        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):
            self._iter = iter

            if isinstance(self.train_loader, list):
                batch = [loader.next() for loader in self.train_loader]
            else:
                batch = self.train_loader.next()

            batch_A = batch[0]
            batch_B = batch[1]

            # forward
            self.train_step(batch_A, batch_B)

            self.train_progress.update({'Time': time.time() - end})
            end = time.time()

            if (iter % self.print_freq == 0):
                self.train_progress.display(iter)

    def train_step(self, batch_source, batch_target):
        data_source = batch_processor(batch_source, self.cfg.MODEL.dsbn)
        data_target = batch_processor(batch_target, self.cfg.MODEL.dsbn)
        self.real_A = data_source['img'][0].cuda()
        self.real_B = data_target['img'][0].cuda()

        # Forward
        self.fake_B = self.Gb(self.real_A)     # G_B(A)
        self.rec_A  =  self.Ga(self.fake_B)    # G_A(G_B(A))
        self.fake_A = self.Ga(self.real_B)     # G_A(B)
        self.rec_B  =  self.Gb(self.fake_A)    # G_B(G_A(B))

        # G_A and G_B
        self.optimizers['G'].zero_grad()
        self.backward_G()
        self.optimizers['G'].step()

        # D_A and D_B
        self.optimizers['D'].zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizers['D'].step()

        meters = {'adversarial': self.loss_adv_G + self.loss_D_A + self.loss_D_B,
                  'cycle_consistent': self.loss_cycle,
                  'identity': self.loss_idt}
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

        # combined loss and calculate gradients
        loss_G = loss_adv_G + loss_cycle + loss_idt
        loss_G.backward()

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
        pred_fake = netD(fake)
        loss_D_fake = self.criterions['adversarial'](pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self.backward_D_basic(self.Da, self.real_A, fake_A)
        self.loss_D_A = loss_D_A.item()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_B = self.backward_D_basic(self.Db, self.real_B, fake_B)
        self.loss_D_B = loss_D_B.item()


    def save_model(self, cfg):
        print(bcolors.OKGREEN + '\n * Finished epoch {:2d}'.format(self._epoch) + bcolors.ENDC)
        print(" Saving models...")
        save_path = cfg.work_dir
        if (self._rank == 0):
            torch.save(self.Ga.state_dict(), '%s/Ga.pth' % save_path)
            torch.save(self.Gb.state_dict(), '%s/Gb.pth' % save_path)
            torch.save(self.Da.state_dict(), '%s/Da.pth' % save_path)
            torch.save(self.Db.state_dict(), '%s/Db.pth' % save_path)
        print("\tDone.\n")

    def resume(self, cfg):
        resume_path = cfg.resume_from
        print("\nLoading pre-trained models.")
        self.Ga.load_state_dict(torch.load(os.path.join(resume_path, 'Ga.pth')))
        self.Gb.load_state_dict(torch.load(os.path.join(resume_path, 'Gb.pth')))
        self.Da.load_state_dict(torch.load(os.path.join(resume_path, 'Da.pth')))
        self.Db.load_state_dict(torch.load(os.path.join(resume_path, 'Db.pth')))
        print("\tDone.\n")

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