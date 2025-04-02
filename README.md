<h1 align="center"><b><span style="color:#800000;">C</span>onditioned <span style="color:#800000;">D</span>iffusion-based Video <span style="color:#800000;">T</span>okenizer (CDT)</b></h1>
<p align="center">
    <a href="https://arxiv.org/abs/2503.03708"><img alt="Publication" src="https://img.shields.io/static/v1?label=Pub&message=arXiv%2725&color=%238B0000"></a>
    <a href="https://github.com/ali-vilab/CDT/pulls"><img src="https://img.shields.io/badge/PRs-Welcome-blue" alt="PRs"></a>
    <a href="https://github.com/ali-vilab/CDT/blob/master/LICENSE"><img alt="License" src="https://img.shields.io/github/license/ali-vilab/CDT?color=green"></a>
    <a href="https://github.com/ali-vilab/CDT/stargazers"><img src="https://img.shields.io/github/stars/ali-vilab/CDT?color=purple&label=Star" alt="Stars"></a>
</p>

Official implementation for our paper:

**Rethinking Video Tokenization: A Conditioned Diffusion-based Approach**

Author List: Nianzu Yang<sup>†</sup>, Pandeng Li<sup>†</sup>, Liming Zhao, Yang Li, Chen-Wei Xie, Yehui Tang, Xudong Lu, Zhihang Liu, Yun Zheng, Yu Liu, Junchi Yan<sup>*</sup>

<sup>†</sup> Equal contribution; <sup>*</sup> Corresponding author


# Content

- [Folder Specification](#folder-specification)
- [Preparation](#preparation)
  - [Environment Setup](#environment-setup)
  - [Download Pre-trained Models](#download-pre-trained-models)
  - [Prepare Data](#prepare-data)
- [Evaluation](#evaluation)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)


## Folder Specification
```bash
├── evaluate.py  # script for evaluating the performance of CDT on reconstruction task
├── model # directory of CDT model
│   └── cdt.py # definition of CDT model
├── opensora_evaluate # scripts for metrics calculation
│   ├── cal_lpips.py # calculate LPIPS
│   ├── cal_psnr.py # calculate PSNR
│   └── cal_ssim.py # calculate SSIM
├── pretrained # directory of pretrained models, which you should create by yourself
│   ├── cdt_base.ckpt # CDT-base
│   └── cdt_small.ckpt # CDT-small
├── README.md # README
├── rec_image_eval.py # script for evaluating the performance of CDT on image reconstruction task
├── rec_video_eval.py # script for evaluating the performance of CDT on video reconstruction task
├── requirements.txt # dependencies
└── utils.py # utility functions
```

## Preparation

### Environment Setup

You can create a new environment and install the dependencies by running the following command:
```shell
conda create -n cdt python=3.10
conda activate cdt
pip install -r requirements.txt
```

### Download Pre-trained Models

We provide the pre-trained models, i.e., CDT-base and CDT-small, on [Hugging Face](https://huggingface.co/yangnianzu/CDT). You can download them and put them in the `pretrained` folder.

### Prepare Data

In our paper, we use two datasets for benchmarking the reconstruction performance:

- `COCO2017-val` for image reconstruction
- `Webvid-val` for video reconstruction

You can download these two datasets and put them in the 'data' folder. Next, you need to specify the path of these two datasets in the 'evaluate.py' file.

## Evaluation

Evaluate the performance of CDT-base on image reconstruction:
```shell
python evaluate.py --method CDT-base --dataset coco17  --mode image
```

Evaluate the performance of CDT-base on video reconstruction at the 256x256 resolution:

```shell
python evaluate.py --method CDT-base --dataset webvid --mode video
```

Evaluate the performance of CDT-base on video reconstruction at the 720x720 resolution:

```shell
python evaluate.py --method CDT-base --dataset webvid --mode video --size 720
```

---

Evaluate the performance of CDT-small on image reconstruction:
```shell
python evaluate.py --method CDT-small --dataset coco17  --mode image
```

Evaluate the performance of CDT-small on video reconstruction at the 256x256 resolution:
```shell
python evaluate.py --method CDT-small --dataset webvid --mode video
```

Evaluate the performance of CDT-small on video reconstruction at the 720x720 resolution:
```shell
python evaluate.py --method CDT-small --dataset webvid --mode video --size 720
```

The reconstructed images or videos and the evaluation results will be saved in the `reconstructed_results` folder.

## Citation
If you find this work useful in your research, please consider citing:

```bibtex
@article{yang2025rethinking,
  title={Rethinking Video Tokenization: A Conditioned Diffusion-based Approach},
  author={Yang, Nianzu and Li, Pandeng and Zhao, Liming and Li, Yang and Xie, Chen-Wei and Tang, Yehui and Lu, Xudong and Liu, Zhihang and Zheng, Yun and Liu, Yu and Yan, Junchi},
  journal={arXiv preprint arXiv:2503.03708},
  year={2025}
}
```

## Acknowledgement

We would like to thank the authors of [Open-Sora-Plan](https://github.com/PKU-YuanGroup/Open-Sora-Plan) for their excellent work, which provides the code for the evaluation metrics.

## Contact
Welcome to contact us [yangnianzu@sjtu.edu.cn](mailto:yangnianzu@sjtu.edu.cn) for any question.