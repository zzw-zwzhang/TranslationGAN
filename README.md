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


## Acknowledgement

Some parts of `openunreid_translation` are learned from 
[OpenUnReID](https://github.com/open-mmlab/OpenUnReID),
[eSPAGN](https://github.com/Simon4Yan/eSPGAN),
[Learning-via-Translation](https://github.com/Simon4Yan/Learning-via-Translation) and
[pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

