from __future__ import absolute_import

from .adapnet import *


__all__ = ['build_bakcbone', 'names']

__factory = {
    'generator': generator,
    'discriminator': discriminator,
    'metric_net': metric_net
}

def names():
    return sorted(__factory.keys())

def build_bakcbone(name, pretrained=True, *args, **kwargs):
    """
    Create a backbone model.
    Parameters
    ----------
    name : str
        The backbone name.
    pretrained : str
        ImageNet pretrained.
    """
    if name not in __factory:
        raise KeyError("Unknown backbone:", name)
    return __factory[name](pretrained=pretrained, *args, **kwargs)
