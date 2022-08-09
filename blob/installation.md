Modified from the official mmdet [getting_started.md](https://github.com/open-mmlab/mmdetection/blob/master/docs/en/getting_started.md).

# Prerequisites

PAOD is developed with the following version of modules.

- Linux or macOS (Windows is not currently officially supported)
- Python 3.7
- PyTorch 1.9.0
- CUDA 11.1
- GCC 9.4.0
- MMCV==1.3.17
- MMDetection==2.20.0

# Installation

**a. Create a conda virtual environment and activate it.**

```
conda create -n paod python=3.7 -y
conda activate paod
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch
```

**c. Install [MMCV](https://mmcv.readthedocs.io/en/latest/), [MMDetection](https://github.com/open-mmlab/mmdetection), and other requirements.**

```
pip install -r requirements.txt
```

**f. Clone the BEVerse repository.**

```
git clone https://github.com/JunruiXiao/PAOD
cd PAOD
```

**g.Install build requirements and then install BEVerse.**

```
python setup.py develop
```

