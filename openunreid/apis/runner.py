import os
import time
import os.path as osp

import torch
import torchvision

from .train import batch_processor, set_random_seed
from ..utils.dist_utils import get_dist_info, synchronize
from ..utils.file_utils import mkdir_if_missing
from ..utils.meters import Meters
from ..utils.image_pool import ImagePool
from ..utils import bcolors


class TranslationBaseRunner(object):
    """
    Adaption Base Runner
    """

    def __init__(
            self,
            cfg,
            models,
            optimizer,
            criterions,
            train_loader,
            train_sets=None,
            lr_schedulers=None,
            meter_formats={'Time': ':.3f'},
            print_freq=100,
            reset_optim=True,
    ):
        super(TranslationBaseRunner, self).__init__()
        # set_random_seed(cfg.TRAIN.seed, cfg.TRAIN.deterministic)

        self.cfg = cfg
        self.models = models
        self.optimizers = optimizer
        self.criterions = criterions
        self.lr_schedulers = lr_schedulers
        self.print_freq = print_freq
        self.reset_optim = reset_optim
        self.save_dir = osp.join(self.cfg.work_dir, 'sample_images_every_iter')
        mkdir_if_missing(self.save_dir)

        self._rank, self._world_size, self._is_dist = get_dist_info()
        self._epoch, self._start_epoch = 0, 0

        # build data loaders
        self.train_loader, self.train_sets = train_loader, train_sets

        # models
        self.Ga = self.models[0]['Ga']
        self.Gb = self.models[0]['Gb']
        self.Da = self.models[1]['Da']
        self.Db = self.models[1]['Db']
        if self.cfg.MODEL.metric_net:
            self.MeNet = self.models[2]

        self.fake_A_pool = ImagePool()
        self.fake_B_pool = ImagePool()

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

            # validate
            if ((ep + 1) % self.cfg.TRAIN.val_freq == 0 or (ep + 1) == self.cfg.TRAIN.epochs):
                self.save_model()

            # update learning rate
            if (self.lr_schedulers is not None):
                self.lr_schedulers['G'].step()
                self.lr_schedulers['D'].step()
                if self.cfg.MODEL.metric_net:
                    self.lr_schedulers['MeNet'].step()

            # synchronize distributed processes
            synchronize()

    def train(self):
        '''
        if isinstance(self.train_loader, list):
            for loader in self.train_loader:
                loader.new_epoch(self._epoch)
        else:
            self.train_loader.new_epoch(self._epoch)
        '''
        self.a_loader = self.train_loader[0]
        self.b_loader = self.train_loader[1]

        self.train_progress.reset(prefix='Epoch: [{}]'.format(self._epoch))

        end = time.time()
        for iter in range(self.cfg.TRAIN.iters):
            self._iter = iter

            self.Ga.train()
            self.Gb.train()

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
        self.fake_A = self.Ga(self.real_B)     # G_A(B)
        self.fake_B = self.Gb(self.real_A)     # G_B(A)
        self.rec_A  =  self.Ga(self.fake_B)    # G_A(G_B(A))
        self.rec_B  =  self.Gb(self.fake_A)    # G_B(G_A(B))

        # save translated images
        pictures = (torch.cat([self.real_A, self.fake_B, self.rec_A,
                               self.real_B, self.fake_A, self.rec_B], dim=0).data + 1) / 2.0
        torchvision.utils.save_image(pictures, '%s/epoch_%d.jpg'
                                     % (self.save_dir, self._epoch + 1), nrow=4)

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
        # import pdb; pdb.set_trace()
        self.fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_A = self.backward_D_basic(self.Da, self.real_A, self.fake_A)
        self.loss_D_A = loss_D_A.item()

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_B = self.backward_D_basic(self.Db, self.real_B, self.fake_B)
        self.loss_D_B = loss_D_B.item()


    def save_model(self):
        print(bcolors.OKGREEN + '\n * Finished epoch {:2d}'.format(self._epoch) + bcolors.ENDC)
        print(" Saving models...")
        save_path = self.cfg.work_dir
        if (self._rank == 0):
            torch.save(self.Ga.state_dict(), '%s/Ga.pth' % save_path)
            torch.save(self.Gb.state_dict(), '%s/Gb.pth' % save_path)
            torch.save(self.Da.state_dict(), '%s/Da.pth' % save_path)
            torch.save(self.Db.state_dict(), '%s/Db.pth' % save_path)
        print("\tDone.\n")

    def resume(self):
        resume_path = self.cfg.resume_from
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