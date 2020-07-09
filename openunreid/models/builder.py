from .backbones import build_bakcbone

__all__ = ['build_adaption_model']


def build_adaption_model(cfg):
        Generator = build_bakcbone('generator', pretrained=False)
        Discriminator = build_bakcbone('discriminator', pretrained=False)
        Metric_Net = build_bakcbone('metric_net', pretrained=False)

        return Generator, Discriminator, Metric_Net