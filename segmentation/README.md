# Reproduce DUSA for Segmentation Tasks

<!-- TOC -->

- [Environment Setup](#environment-setup)
- [Pre-trained Models](#pre-trained-models)
- [Prepare Datasets](#prepare-datasets)
  - [ADE20K](#ade20k)
  - [ADE20K-C](#ade20k-c)
- [Run Experiments](#run-experiments)
  - [Fully TTA of SegFormer-B5](#fully-tta-of-segformer-b5)
- [Acknowledgments](#acknowledgments)

<!-- /TOC -->

## Environment Setup

We recommend following the instructions below to setup the environment:

```shell
# create dusa_seg environment
conda create -n dusa_seg -y python=3.9.5
conda activate dusa_seg
# install torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# install open-mmlab
pip install openmim==0.3.9
mim install "mmcv==2.1.0"
mim install "mmengine==0.10.4"
# install other requirements
conda env update -n dusa_seg -f env.yml
```

## Pre-trained Models

> [!IMPORTANT]
> The final tree of files should be:
>
> ```text
> DUSA/segmentation/pretrained_models/
> └── segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth
> ```

Access pre-trained weights from the following resources:

- [`SegFormer-B5`](https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth)
- [`Stable Diffusion v1.5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - for reproduce, we switch to a [fork](https://huggingface.co/KiwiXR/stable-diffusion-v1-5)
- [`Controlnet v1.1`](https://huggingface.co/lllyasviel/control_v11p_sd15_seg)

We assume the weights are placed in the `./pretrained_models` directory. To prepare in cli, refer to the following:

```shell
# create folder if not exists
mkdir -p pretrained_models && cd pretrained_models
# download weights for
# SegFormer-B5
wget https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_mit-b5_512x512_160k_ade20k/segformer_mit-b5_512x512_160k_ade20k_20210726_145235-94cedf59.pth
```

Stable Diffusion v1.5 and Controlnet v1.1 should be automatically downloaded while running experiments.

> [!WARNING]
> Due to the removal of `runwayml/stable-diffusion-v1-5` from Huggingface, we are using a fork of the model (i.e., `KiwiXR/stable-diffusion-v1-5`) to ensure reproducibility.
>
> Users are also welcome to switch to other sources of their choice by simply modifying the `model_path` value in `tta_configs/ade/sdtta/sd_controlnet_slide_topk.py`.

***Alternatively***, we could download them manually (note that these commands download full repos and can result in larger cache than that downloaded in running):

```shell
huggingface-cli download --resume-download KiwiXR/stable-diffusion-v1-5
huggingface-cli download --resume-download lllyasviel/control_v11p_sd15_seg
```

> [!TIP]
> In case of network issues, give <https://hf-mirror.com/> a try:
>
> ```shell
> HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download KiwiXR/stable-diffusion-v1-5
> HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download lllyasviel/control_v11p_sd15_seg
> ```

## Prepare Datasets

> [!IMPORTANT]
> The final tree of files should be:
>
> ```text
> DUSA/segmentation/data
> ├── ADE20K_val-c
> └── ADEChallengeData2016 -> /path/to/ade20k
> ```

### ADE20K

The ADE20K dataset can be downloaded from [here](https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip).

Or download with cli:

```shell
wget https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
```

Extract the dataset to your selected `/path/to/ade20k` and symlink the dataset:

```shell
mkdir -p data
ln -s /path/to/ade20k data/ADEChallengeData2016
```

> [!IMPORTANT]
> Make sure the validation set annotation files are in `data/ADEChallengeData2016/annotations/validation/`.

### ADE20K-C

There is randomness in the corruption generation process of [imagecorruptions](https://github.com/bethgelab/imagecorruptions), so we recommend to directly download our version [[google drive](https://drive.google.com/file/d/1vTYoksyYHdpARqDZxu1LRJny9__tf8xT/view)] of corrupted validation set for the sake of reproducibility:

```shell
cd data
gdown --fuzzy https://drive.google.com/file/d/1vTYoksyYHdpARqDZxu1LRJny9__tf8xT/view
md5sum ADE20K_val-c.tar.gz # 1ca0ed04dc27f61b5d3932e67f25444f
tar -xzvf ADE20K_val-c.tar.gz -C ./
```

However, if it is preferred to re-generate the set, please run the following command and the corrupted set will be saved to `data/ADE20K_val-c/`:

```shell
bash build_ADE20k_c.sh
```

## Run Experiments

The scripts are available in the `./sh` directory. Change `CUDA_VISIBLE_DEVICES` if needed.

### Fully TTA of SegFormer-B5

- DUSA-seg

  ```shell
  bash sh/segformer-b5/dusa-seg_segformer_ade-c.sh
  ```

- Baselines

  ```shell
  bash sh/segformer-b5/baselines_segformer_ade-c.sh
  ```

## Acknowledgments

This implementation is based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and inspired by [Diffusion-TTA](https://github.com/mihirp1998/Diffusion-TTA). The baseline code are borrowed from their official implementations in [Tent](https://github.com/DequanWang/tent) and [CoTTA](https://github.com/qinenergy/cotta). We thank their authors for making the source code publicly available.
