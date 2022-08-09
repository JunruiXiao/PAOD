# PAOD: Rethinking Prediction Alignment in One-stage Object Detection

The official implementation of the paper Rethinking Prediction Alignment in One-stage Object Detection.

## News

- **2022.08.09:** We release the code and models of PAOD.

## Model Zoo

#### COCO

| Model |    Backbone    | Lr Schd | mAP  | AP50 | AP75 | Config | Model |
| :---: | :------------: | :-----: | :--: | ---- | ---- | ------ | ----- |
| PAOD  |   ResNeXt101   |   2x    | 48.8 | 67.3 | 53.3 |        |       |
| PAOD  | ResNeXt101-DCN |   2x    | 50.4 | 68.9 | 55.0 |        |       |
| PAOD  |  Res2Net-DCN   |   2x    | 51.1 | 69.6 | 55.8 |        |       |

#### Pascal VOC

| Model | Backbone | Lr Schd | mAP  | AP50 | AP75 | Config | Model |
| :---: | :------: | :-----: | :--: | ---- | ---- | ------ | ----- |
| PAOD  | ResNet50 |   1x    | 65.0 | 85.6 | 71.2 |        |       |

#### CrowdHuman

| Detector | Backbone | AP ↑ | MR ↓ | JI ↑ | Config | Model |
| :------: | :------: | :--: | :--: | :--: | :----: | ----- |
|   PAOD   | ResNet50 | 89.2 | 46.5 | 77.7 |        |       |

## Requirements

- Please check [installation](https://github.com/JunruiXiao/PAOD/blob/main/docs/installation.md) for installation and [data_preparation](https://github.com/JunruiXiao/PAOD/blob/main/docs/data_preparation.md) for preparing the dataset.

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

