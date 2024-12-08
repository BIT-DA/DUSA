# Reproduce DUSA for Classification Tasks

<!-- TOC -->

- [Environment Setup](#environment-setup)
- [Pre-trained Models](#pre-trained-models)
- [Prepare Datasets](#prepare-datasets)
  - [ImageNet-C](#imagenet-c)
  - [ImageNet val](#imagenet-val)
- [Run Experiments](#run-experiments)
  - [Fully TTA of ConvNeXt-Large](#fully-tta-of-convnext-large)
  - [Continual TTA of ConvNeXt-Large](#continual-tta-of-convnext-large)
  - [Fully TTA of ViT-B/16](#fully-tta-of-vit-b16)
  - [Fully TTA of ResNet50-GN](#fully-tta-of-resnet50-gn)
- [Acknowledgments](#acknowledgments)

<!-- /TOC -->

## Environment Setup

We recommend following the instructions below to setup the environment:

```shell
# create dusa environment
conda create -n dusa -y python=3.9.18
conda activate dusa
# install torch
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
# install open-mmlab
pip install openmim==0.3.9
mim install "mmcv==2.1.0"
mim install "mmengine==0.10.2"
# install other requirements
conda env update -n dusa -f env.yml
```

## Pre-trained Models

> [!IMPORTANT]
> The final tree of files should be:
>
> ```text
> DUSA/classification/pretrained_models/
> ├── B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz
> ├── convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth
> ├── DiT-XL-2-256x256.pt
> └── resnet50_gn_a1h2-8fe6c4d0.pth
> ```

We use pre-trained weights from the following sources:

- [`ResNet50-GN`](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth)
- [`ViT-B/16`](https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz)
- [`ConvNeXt-Large`](https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth)
- [`DiT-XL/2`](https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt)
  - [`stabilityai/sd-vae-ft-ema`](https://huggingface.co/stabilityai/sd-vae-ft-ema) is the VAE used in DiT-XL/2

We assume the weights are placed in the `./pretrained_models` directory. To prepare in cli, refer to the following:

```shell
# create folder if not exists
mkdir -p pretrained_models && cd pretrained_models
# download weights for
# ResNet50-GN
wget https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_gn_a1h2-8fe6c4d0.pth
# Vit-B/16
wget https://storage.googleapis.com/vit_models/augreg/B_16-i1k-300ep-lr_0.001-aug_strong2-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz
# ConvNeXt-Large
wget https://download.openmmlab.com/mmclassification/v0/convnext/convnext-large_3rdparty_64xb64_in1k_20220124-f8a0ded0.pth
# DiT-XL/2
wget https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt
```

The VAE used in DiT-XL/2 should be automatically downloaded while running experiments.

***Alternatively***, we could download it manually:

```shell
huggingface-cli download --resume-download stabilityai/sd-vae-ft-ema config.json diffusion_pytorch_model.safetensors
```

> [!TIP]
> In case of network issues, give <https://hf-mirror.com/> a try:
>
> ```shell
> HF_ENDPOINT=https://hf-mirror.com huggingface-cli download --resume-download stabilityai/sd-vae-ft-ema config.json diffusion_pytorch_model.safetensors
> ```

## Prepare Datasets

> [!IMPORTANT]
> The final tree of files should be:
>
> ```text
> DUSA/classification/data
> ├── ImageNet -> /path/to/imagenet
> └── imagenet-c -> /path/to/imagenet-c
> ```

### ImageNet-C

The ImageNet-C dataset can be downloaded from [here](https://zenodo.org/record/2235448). Refer to the commands below:

```shell
wget https://zenodo.org/records/2235448/files/blur.tar?download=1 -c -O blur.tar
wget https://zenodo.org/records/2235448/files/digital.tar?download=1 -c -O digital.tar
wget https://zenodo.org/records/2235448/files/extra.tar?download=1 -c -O extra.tar
wget https://zenodo.org/records/2235448/files/noise.tar?download=1 -c -O noise.tar
wget https://zenodo.org/records/2235448/files/weather.tar?download=1 -c -O weather.tar
```

Extract the dataset to your selected `/path/to/imagenet-c` and symlink the dataset:

```shell
mkdir -p data
ln -s /path/to/imagenet-c data/imagenet-c
```

### ImageNet val

> [!NOTE]
> Only ImageNet-C is required for reproducing DUSA.
>
> To reproduce the EATA baseline, the validation set of ImageNet is also required.

ImageNet validation set can be officially accessed [here](https://image-net.org/download.php).

Or download with cli:

```shell
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar --no-check-certificate
```

Extract the val split to your selected `/path/to/imagenet` and symlink the dataset:

```shell
mkdir -p data
ln -s /path/to/imagenet data/ImageNet
```

> [!IMPORTANT]
> Make sure the validation set files are in `data/ImageNet/val/`.

## Run Experiments

The scripts are available in the `./sh` directory. Change `CUDA_VISIBLE_DEVICES` if needed.

### Fully TTA of ConvNeXt-Large

- DUSA
  ```shell
  bash sh/convnext-l/dusa_convnext_in-c.sh
  ```
- DUSA-U
  ```shell
  bash sh/convnext-l/dusa-u_convnext_in-c.sh
  ```
- Ablation on Noise
  ```shell
  bash sh/convnext-l/dusa_ablation_convnext.sh
  ```
- Baselines
  ```shell
  bash sh/convnext-l/baselines_convnext_in-c.sh
  ```

### Continual TTA of ConvNeXt-Large

- DUSA
  ```shell
  bash sh/convnext-l/dusa_continual_convnext_in-c.sh
  ```
- Baselines
  ```shell
  bash sh/convnext-l/baselines_continual_convnext_in-c.sh
  ```

### Fully TTA of ViT-B/16

- DUSA
  ```shell
  bash sh/vit-b/dusa_vit_in-c.sh
  ```
- DUSA-U
  ```shell
  bash sh/vit-b/dusa-u_vit_in-c.sh
  ```
- Baselines
  ```shell
  bash sh/vit-b/baselines_vit_in-c.sh
  ```

### Fully TTA of ResNet50-GN

- DUSA
  ```shell
  bash sh/res50-gn/dusa_res50_in-c.sh
  ```
- DUSA-U
  ```shell
  bash sh/res50-gn/dusa-u_res50_in-c.sh
  ```
- Ablation on Noise
  ```shell
  bash sh/res50-gn/dusa_ablation_res50.sh
  ```
- Baselines
  ```shell
  bash sh/res50-gn/baselines_res50_in-c.sh
  ```

## Acknowledgments

This implementation is based on [MMPreTrain](https://github.com/open-mmlab/mmpretrain) and inspired by [Diffusion-TTA](https://github.com/mihirp1998/Diffusion-TTA). The baseline code are borrowed from their official implementations in [Tent](https://github.com/DequanWang/tent), [CoTTA](https://github.com/qinenergy/cotta), [EATA](https://github.com/mr-eggplant/EATA), [SAR](https://github.com/mr-eggplant/SAR), and [RoTTA](https://github.com/BIT-DA/RoTTA). We thank their authors for making the source code publicly available.
