# Exploring Structured Semantic Priors Underlying Diffusion Score for Test-time adaptation

**by [Mingjia Li](https://kiwixr.github.io), [Shuang Li](https://shuangli.xyz), [Tongrui Su](https://molarsu.github.io), [Longhui Yuan](https://yuanlonghui.github.io), [Jian Liang](https://scholar.google.com/citations?user=mrunnpoAAAAJ), and [Wei Li](https://scholar.google.com/citations?user=i8jP6q8AAAAJ)**

[![Project Page](https://img.shields.io/badge/Project%20Page-%23D80082?logo=&style=flat-square)](https://kiwixr.github.io/projects/dusa)&nbsp;&nbsp;
[![Paper](https://img.shields.io/badge/Paper-%23B31B1B?style=flat-square)](https://openreview.net/forum?id=c7m1HahBNf)&nbsp;&nbsp;
[![Video](https://img.shields.io/badge/Video-%2350A3A4?style=flat-square)](https://neurips.cc/virtual/2024/poster/94444)&nbsp;&nbsp;

> [!NOTE]
> **DUSA** is a method that:
>
> - Leverages diffusion models for robust performance.
> - Draws inspiration from the structure of score functions.
> - Incorporates semantic priors to enhance adaptability.
> - Operates efficiently with only a single timestep.
>
> It is highly competitive in fully, continual, and potentially other test-time adaptation tasks.

> TODO:
>
> - [ ] Release code for test-time segmentation

## Overview

Capitalizing on the complementary advantages of generative and discriminative models has always been a compelling vision in machine learning, backed by a growing body of research. This work discloses the hidden semantic structure within score-based generative models, unveiling their potential as effective discriminative priors. Inspired by our theoretical findings, we propose DUSA to exploit the structured semantic priors underlying diffusion score to facilitate the test-time adaptation of image classifiers or dense predictors. Notably, DUSA extracts knowledge from a single timestep of denoising diffusion, lifting the curse of Monte Carlo-based likelihood estimation over timesteps.

For more details please refer to our [project page](https://kiwixr.github.io/projects/dusa) or [paper](https://openreview.net/pdf?id=c7m1HahBNf).

## Reproduce

To replicate the results for classification tasks, please refer to instructions [here](./classification/README.md).

## License

This project is released under the [Apache 2.0 License](./LICENSE).

Portions of the code are based on [MMPreTrain](https://github.com/open-mmlab/mmpretrain), which is also licensed under the [Apache 2.0 License](https://github.com/open-mmlab/mmpretrain/blob/main/LICENSE).

Portions of the code are from [DiT](https://github.com/facebookresearch/DiT), which is licensed under [CC BY-NC 4.0 License](https://github.com/facebookresearch/DiT/blob/main/LICENSE.txt).

Please ensure to review the licenses coming with the official implementations of baselines as they may differ.

## Citation

```text
@inproceedings{li2024exploring,
  title={Exploring Structured Semantic Priors Underlying Diffusion Score for Test-time Adaptation},
  author={Mingjia Li and Shuang Li and Tongrui Su and Longhui Yuan and Jian Liang and Wei Li},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
  url={https://openreview.net/forum?id=c7m1HahBNf}
}
```
