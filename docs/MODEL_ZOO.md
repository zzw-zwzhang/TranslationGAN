## Benchmark and model zoo

- `Direct infer` models are trained on the source-domain datasets ([source_pretrain](https://github.com/open-mmlab/OpenUnReID/tree/master/tools/source_pretrain)) and directly tested on the target-domain datasets.

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