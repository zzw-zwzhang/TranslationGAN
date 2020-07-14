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

### Benchmark and model zoo
Results and models are available in the [model zoo](docs/MODEL_ZOO.md).

## Updates

+ [2020-07-14] `OpenUnReID_Translation` is released.

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
+ `${WORK_DIR}`: folder for saving logs, checkpoints and translated images, e.g. `translation_spgan`, the absolute path will be `LOGS_ROOT/${WORK_DIR}` (`LOGS_ROOT` is defined in config files).
+ `[optional arguments]`: modify some key values from the loaded config file, e.g. `TRAIN.val_freq 10`. (it's also ok to make the modification directly in the config file)

#### Examples
```
bash dist_train.sh translation translation_spgan

python translation/main.py translation/config.yaml --work-dir translation_spgan --launcher "none"
```


### Test

#### Testing commands

+ Distributed testing with multiple GPUs:
```shell
sh dist_test.sh ${RESUME} ${CONFIG} [optional arguments]
```
+ Distributed testing with multiple machines:
```shell
sh slurm_test.sh ${PARTITION} ${RESUME} ${CONFIG} [optional arguments]
```
+ Non-distributed testing with a single GPU:
```shell
python test_translation.py ${RESUME} --config ${CONFIG} --launcher "none" --set [optional arguments]
```

#### Arguments

+ `${RESUME}`: model for testing, e.g. `../logs/translation_spgan`.
+ `${CONFIG}`: config file for the model, e.g. `translation/config.yaml`. **Note** the config is required to match the model.
+ `[optional arguments]`: modify some key values from the loaded config file, e.g. `TEST.rerank True`. (it's also ok to make the modification directly in the config file)



#### Configs

+ The differences of CycleGAN & SPGAN
```shell
MODEL:
  metric_net: True or False

TRAIN:

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

