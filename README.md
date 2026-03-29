<h1 align="center">
[MedIA 2026] Ultrasound Localization Microscopy Learned From Power Doppler by Uncertainty Frequency Density Estimation and Semantic Consistency Awareness
</h1>

<p align="center">
⭐ Qinghua Lin†, Xuan Ren†, Boqian Zhou, Junyi Wang, Xin Liu*
</p>
<p align="center">
<a href="https://www.sciencedirect.com/science/article/abs/pii/S1361841526001234">📄 Paper</a>
</p>

## :fire: News

- [2026/03/26] We have released part of the code, and the full version will be available soon.
- [2026/03/24] Our paper has been accepted to MedIA.

## :rocket: Overview

Here's a framework overview of our **PDSR** method:

<p align="center">
  <img src="./Image/PDSR.png" width="600"/>
</p>


## 🛠️ Prepare your own dataset

To get started with PDSR, follow the instructions below.

1. Enter the data directory

```
cd ./PDSR_for_Training/datasets
```

2. Organize the data in the following format
```
Your Dataset/
├── trainA/
│ ├── xxx.png
│ ├── xxx.png
│ └── ...
├── trainB/
│ ├── xxx.png
│ └── ...
├── testA/
│ ├── xxx.png
│ └── ...
└── testB/
├── xxx.png
└── ...
```

### :blue_book: Example Usage

1. Training
```
cd ./PDSR_for_Training
```

```sh
CUDA_VISIBLE_DEVICES=0 python3 train.py --dataroot datasets/dataset'name
```

2. Testing

```
cd ./PDSR_for_Inferencing
```

Place the trained checkpoints in the current directory

```sh
CUDA_VISIBLE_DEVICES=0 python3 test.py --dataroot datasets/dataset'name
```

## 📁 Datasets

The test cases in PDSR can be obtained at the following link:

🔗 [Baidu Cloud](https://pan.baidu.com/s/1UnBEJ6IlojNqJHA2XVdNVw?pwd=4c4f ) 

🔗 [Google Cloud](https://drive.google.com/file/d/1wZDU2vc1SuSisKimzROM_xh-nxHwI8aY/view)

## :pushpin: Citation
If you use PDSR in your work, please cite it using the following BibTeX:

```bibtex
@article{lin2026ultrasound,
  title={Ultrasound Localization Microscopy Learned From Power Doppler by Uncertainty Frequency Density Estimation and Semantic Consistency Awareness},
  author={Lin, Qinghua and Ren, Xuan and Zhou, Boqian and Wang, Junyi and Liu, Xin},
  journal={Medical Image Analysis},
  pages={104055},
  year={2026},
  publisher={Elsevier}
}
```

## 🙏 Acknowledgement
We would like to express our sincere gratitude to the developers of the following projects for their valuable contributions: [Decent](https://github.com/Mid-Push/Decent), [EnCo](https://github.com/XiudingCai/EnCo-pytorch) and [CUT](https://github.com/taesungp/contrastive-unpaired-translation). This work builds upon and is inspired by these outstanding efforts.
