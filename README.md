# PAOD: Rethinking Prediction Alignment in One-stage Object Detection

The official implementation of the paper Rethinking Prediction Alignment in One-stage Object Detection.

## News

- **2022.08.09:** We release the code and models of PAOD.

## Model Zoo

#### COCO

| Model |    Backbone    | Lr Schd | mAP  | AP50 | AP75 | Config                                                       | Model |
| :---: | :------------: | :-----: | :--: | ---- | ---- | ------------------------------------------------------------ | ----- |
| PAOD  |   ResNeXt101   |   2x    | 48.8 | 67.3 | 53.3 | [Config](https://github.com/JunruiXiao/PAOD/tree/main/configs/paod/paod_x101_64x4d_fpn_mstrain_2x_coco.py) |    [Google Drive](https://drive.google.com/file/d/1JDGwjNXnLCe5EPxtsB3a4DKXlJjDeMYn/view?usp=sharing)   |
| PAOD  | ResNeXt101-DCN |   2x    | 50.4 | 68.9 | 55.0 | [Config](https://github.com/JunruiXiao/PAOD/tree/main/configs/paod/paod_x101_64x4d_fpn_dcn_mstrain_2x_coco.py) |   [Google Drive](https://drive.google.com/file/d/1MRMzi0AqGZh_qS9rr7ZOHcEzJNF2fVKM/view?usp=sharing)    |
| PAOD  |  Res2Net-DCN   |   2x    | 51.1 | 69.6 | 55.8 | [Config](https://github.com/JunruiXiao/PAOD/tree/main/configs/paod/paod_r2101_fpn_dcn_mstrain_2x_coco.py) |  [Google Drive](https://drive.google.com/file/d/1dOOpMAcboLNhqAS7nUiaSeUvbFSnIz2p/view?usp=sharing)     |

#### Pascal VOC

| Model | Backbone | Lr Schd | mAP  | AP50 | AP75 | Config                                                       | Model |
| :---: | :------: | :-----: | :--: | ---- | ---- | ------------------------------------------------------------ | ----- |
| PAOD  | ResNet50 |   1x    | 65.0 | 85.6 | 71.2 | [Config](https://github.com/JunruiXiao/PAOD/tree/main/configs/paod/paod_r50_fpn_1x_voc.py) |   [Google Drive](https://drive.google.com/file/d/1MNeAX9jY0CWo40suPApyfwXzXFb8g-JC/view?usp=sharing)    |

#### CrowdHuman

| Detector | Backbone | AP ↑ | MR ↓ | JI ↑ |                            Config                            | Model |
| :------: | :------: | :--: | :--: | :--: | :----------------------------------------------------------: | ----- |
|   PAOD   | ResNet50 | 89.2 | 46.5 | 77.7 | [Config](https://github.com/JunruiXiao/PAOD/tree/main/configs/paod/paod_r50_fpn_1x_crowd.py) |  [Google Drive](https://drive.google.com/file/d/1K5O1iOXkVfl6zdFHkKbiND-tI8MEPBOv/view?usp=sharing)     |

## Requirements

- Please check [installation](https://github.com/JunruiXiao/PAOD/tree/main/blob/installation.md) for installation.

## Training,  Evaluation and Visualization

To train PAOD with 8 GPUs, run:
```bash
bash tools/dist_train.sh $CONFIG 8
```

or you can run the .sh file in script:

```bash
bash train_on_coco.sh
bash train_on_voc.sh
bash train_on_crow.sh
```

To evaluate PAOD with 8 GPU, run:

```bash
bash tools/dist_test.sh $YOUR_CONFIG $YOUR_CKPT 8 --eval=bbox
```

To visualize the predictions, run:
```bash
python tools/test.py $YOUR_CONFIG $YOUR_CKPT --eval=bbox --show
```

## Acknowledgement 

This project is mainly based on the following open-sourced projects: [open-mmlab](https://github.com/open-mmlab), and we thank [DDOD](https://github.com/zehuichen123/DDOD) for their code on CrowdHuman.

