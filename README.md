# OpenUnReID_Translation

## Introduction
`OpenUnReID_Translation` is a PyTorch-based implementation of translation-based 
unsupervised domain adaption (**UDA**) on re-ID tasks. 


### Supported method
- [x] [CycleGAN](https://arxiv.org/pdf/1703.10593.pdf)
- [x] [SPGAN](https://arxiv.org/pdf/1711.09020.pdf)
- [ ] [SSG](https://arxiv.org/abs/1811.10144v2)
- [ ] [SDA](https://arxiv.org/pdf/2003.06650.pdf)


### Supported datasets
- [x] Market-1501-v15.09.15
- [x] DukeMTMC-reID
- [ ] MSMT17_V1
- [ ] PersonX_v1
- [ ] VehicleID_V1.0
- [ ] AIC20_ReID_Simulation
- [ ] VeRi_with_plate


### Results

- `Direct infer` models are trained on the source-domain datasets ([source_pretrain](../tools/source_pretrain)) and directly tested on the target-domain datasets.

#### DukeMTMC-reID -> Market-1501

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | DukeMTMC | 27.2 | 58.9 | 75.7 | 81.4 | ~1h | [[config]](https://drive.google.com/file/d/1_gnPfjwf9uTOJyg1VsBzbMNQ-SGuhohP/view?usp=sharing) [[model]](https://drive.google.com/file/d/1MH-eIuWICkkQ8Ka3stXbiTq889yUZjBV/view?usp=sharing) |
| [CycleGAN]() | ResNet50 | DukeMTMC |  |  |  |  |  | [[config]]() [[model]]() |
| [SPGAN]()    | ResNet50 | DukeMTMC |  |  |  |  |  | [[config]]() [[model]]() |
| [SSG]()      | ResNet50 | DukeMTMC |  |  |  |  |  | [[config]]() [[model]]() |
| [SDA]()      | ResNet50 | DukeMTMC |  |  |  |  |  | [[config]]() [[model]]() |

#### Market-1501 -> DukeMTMC-reID

| Method | Backbone | Pre-trained | mAP(%) | top-1(%) | top-5(%) | top-10(%) | Train time | Download |
| ----- | :------: | :---------: | :----: | :------: | :------: | :-------: | :------: | :------: |
| Direct infer | ResNet50 | Market | 28.1 | 49.3 | 64.3 | 69.7 | ~1h | [[config]](https://drive.google.com/file/d/1FOuW_Hwl2ASPx0iXeDNxZ1R9MwFBr3gx/view?usp=sharing) [[model]](https://drive.google.com/file/d/13dkhrjz-VIH3jCjIep185MLZxFSD_F7R/view?usp=sharing) |
| [CycleGAN]() | ResNet50 | Market |  |  |  |  |  | [[config]]() [[model]]() |
| [SPGAN]()    | ResNet50 | Market |  |  |  |  |  | [[config]]() [[model]]() |
| [SSG]()      | ResNet50 | Market |  |  |  |  |  | [[config]]() [[model]]() |
| [SDA]()      | ResNet50 | Market |  |  |  |  |  | [[config]]() [[model]]() |


## Updates

+ [2020-07-04] `OpenUnReID_Translation` is released.

## Installation & Get Started

Please refer to [OpenUnReID](https://github.com/open-mmlab/OpenUnReID) for 
[installation](https://github.com/open-mmlab/OpenUnReID/blob/master/docs/INSTALL.md) and 
[get started](https://github.com/open-mmlab/OpenUnReID/blob/master/docs/GETTING_STARTED.md).

### Train

#### Training commands

+ Distributed training with multiple GPUs:
```shell
bash dist_train.sh ${METHOD} ${WORK_DIR} [optional arguments]
```
+ Non-distributed training with a single GPU:
```shell
python ${METHOD}/main.py ${METHOD}/config.yaml --work-dir ${WORK_DIR} --launcher "none" --set [optional arguments]
```

#### Arguments

+ `${METHOD}`: method for training, e.g. `translation`.
+ `${WORK_DIR}`: folder for saving logs and checkpoints, e.g. `translation_spgan`, the absolute path will be `LOGS_ROOT/${WORK_DIR}` (`LOGS_ROOT` is defined in config files).
+ `[optional arguments]`: modify some key values from the loaded config file, e.g. `TRAIN.val_freq 10`. (it's also ok to make the modification directly in the config file)

#### Examples
```
bash dist_train.sh translation translation_spgan

python translation/main.py translation/config.yaml --work-dir translation_spgan --launcher "none"
```

#### Configs

+ The differences of CycleGAN & SPGAN
```shell
MODEL:
  metric_net: True or False

TRAIN:
  epochs: 20
  iters: 4130    # max images / batchsize

  LOSS:
    losses: {'adversarial': 1., 'cycle_consistent': 10., 'identity': 5., 'contrastive': 2.}   # SPGAN
    # losses: {'adversarial': 1., 'cycle_consistent': 10., 'identity': 0.5}                   # CycleGAN
```

## Acknowledgement

Some parts of `openunreid_translation` are learned from 
[OpenUnReID](https://github.com/open-mmlab/OpenUnReID),
[eSPAGN](https://github.com/Simon4Yan/eSPGAN),
[Learning-via-Translation](https://github.com/Simon4Yan/Learning-via-Translation) and
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

