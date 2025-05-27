# Convolution, Transformer, and Operator (CTO)
This is the official code for our IPMI 2023 and Medical Image Analysis papers:

> [Rethinking Boundary Detection in Deep Learning Models for Medical Image Segmentation](https://arxiv.org/abs/2305.00678) <br>
> Yi Lin*, Dong Zhang*, Xiao Fang, Yufan Chen, Kwang-Ting Cheng, Hao Chen

> [Rethinking boundary detection in deep learning-based medical image segmentation](https://arxiv.org/pdf/2505.04652) <br>
> Yi Lin*, Dong Zhang*, Xiao Fang*, Yufan Chen, Kwang-Ting Cheng, Hao Chen

## Highlights
<p align="justify">
We introduce a network architecture, referred to as Convolution, Transformer, and Operator (CTO), for medical image segmentation. CTO employs a combination of CNNs, ViT, and an explicit boundary detection operator to achieve high recognition accuracy while maintaining an optimal balance between accuracy and efficiency. The proposed CTO follows the standard encoder-decoder segmentation paradigm, where the encoder network incorporates a popular CNN backbone for capturing local semantic information, and a lightweight ViT assistant for integrating long-range dependencies. To enhance the learning capacity on boundary, a boundary-guided decoder network is proposed that uses a boundary mask obtained from a dedicated boundary detection operator as explicit supervision to guide the decoding learning process. 

| Methods | mDice $\uparrow$ | HD $\uparrow$| Aorta | Gallb. | Kid (L) | Kid (R) | Liver | Panc. | Spleen | Stom. |
|---------|--------|--------|------|------|------|------|------|------|------|------|
| CLIP | 68.81 | - | 75.34 | 51.87 | 77.10 | 80.75 | 87.84 | 40.05 | 80.56 | 56.98 |
| DARR | 69.77 | - | 74.74 | 53.77 | 72.31 | 73.24 | 94.08 | 54.18 | 89.90 | 45.96
| U-Net | 76.85 | 39.70 | 89.07 | 69.72 | 77.77 | 68.60 | 93.43 | 53.98 | 86.67 | 75.58
| R50-UNet | 74.68 | 36.87 | 84.18 | 62.84 | 79.19 | 71.29 | 93.35 | 48.23 | 84.41 | 73.92
| Att-UNet  | 77.77 | 36.02 | 89.55 | 68.88 | 77.98 | 71.11 | 93.57 | 58.04 | 87.30 | 75.75
| R50-AttUNet  | 75.57 | 36.97 | 55.92 | 63.91 | 79.20 | 72.71 | 93.56 | 49.37 | 87.19 | 74.95
| R50-ViT  | 71.29 | 32.87 | 73.73 | 55.13 | 75.80 | 72.20 | 91.51 | 45.99 | 81.99 | 73.95
| TransUNet  | 77.48 | 31.69 | 87.23 | 63.13 | 81.87 | 77.02 | 94.08 | 55.86 | 85.08 | 75.62
| SwinUNet  | 79.12 | 21.55 | 85.47 | 66.53 | 83.28 | 79.61 | 94.29 | 56.58 | 90.66 | 76.60
| CTO(Ours) | 81.10 | 18.75 | 87.72 | 66.44 | 84.49 | 81.77 | 94.88 | 62.74 | 90.60 | 80.20

## 🔥 Update Log
- [2025/05/27] Release the code of CTO that integrates StitchViT.

## Usage
The code is largely built using [MedISeg](https://github.com/hust-linyi/MedISeg) framework.

### Using the code
Please clone the following repositories:
```
git clone https://github.com/xiaofang007/CTO
cd CTO  
git clone https://github.com/hust-linyi/MedISeg
```
### Requirement
```
pip install -r MedISeg/requirements.txt
```

### Data preparation
ISIC-2018: please follow the data preparation procedure in [MedISeg](https://github.com/hust-linyi/MedISeg) framework.

### Training & Evaluation
We provide the shell scripts for training and evaluation.  
ISIC-2018:  
Run the following command for training
```
sh config/isic_baseline.sh
```  
Run the following command for testing
```
sh config/isic_test_baseline.sh
```

## Citation
Please cite the paper if you use the code.
```bibtex
@article{lin2025rethinking,
  title={Rethinking boundary detection in deep learning-based medical image segmentation},
  author={Lin, Yi and Zhang, Dong and Fang, Xiao and Chen, Yufan and Cheng, Kwang-Ting and Chen, Hao},
  journal={Medical Image Analysis},
  pages={103615},
  year={2025},
  publisher={Elsevier}}

@inproceedings{lin2023rethinking,
    title={Rethinking Boundary Detection in Deep Learning Models for Medical Image Segmentation},
    author={Yi Lin, Dong Zhang, Xiao Fang, Yufan Chen, Kwang-Ting Cheng, Hao Chen},
    booktitle={Information Processing in Medical Imaging (IPMI)},
    year={2023}}

@article{zhang2022deep,
  title={Deep learning for medical image segmentation: tricks, challenges and future directions},
  author={Zhang, Dong and Lin, Yi and Chen, Hao and Tian, Zhuotao and Yang, Xin and Tang, Jinhui and Cheng, Kwang Ting},
  journal={arXiv},
  year={2022}}
```

## Acknowledgment 
Our code is based on [MedISeg](https://github.com/hust-linyi/MedISeg). 
