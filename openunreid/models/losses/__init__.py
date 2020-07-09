# Written by Zhiwei Zhang

import torch.nn as nn
from .adaption import ContrastiveLoss, AdversarialLoss


def build_loss(
    cfg,
    cuda=False
):

    criterions = {}
    for loss_name in cfg.losses.keys():

        if (loss_name=='adversarial'):
            criterions['adversarial'] = AdversarialLoss()

        elif (loss_name=='cycle_consistent'):
            criterions['cycle_consistent'] = nn.L1Loss()

        elif (loss_name=='identity'):
            criterions['identity'] = nn.L1Loss()

        elif (loss_name=='contrastive'):
            if ('margin' not in cfg): cfg.margin = 2
            criterions['contrastive'] = ContrastiveLoss(margin=cfg.margin)

        else:
            raise KeyError("Unknown loss:", loss_name)

    if cuda:
        for loss in criterions.values():
            loss.cuda()

    return criterions