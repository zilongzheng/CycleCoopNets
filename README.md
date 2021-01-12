# CycleCoopNets
Official Implementation for AAAI 2021 accepted paper "Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation"

## Prerequisites

- Linux 16.04+
- Python 3.5+
- NVIDIA GPU + CUDA CuDNN

## Getting Started
### Data
- Download dataset from [CycleGAN official website](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh) and data in `input` folder.
### Train/Test
- Train a model (e.g. summer2winter_yosemite) by 
```
python train.py --dataroot input --category summer2winter_yosemite --output_dir output
```
- Visualize training progress
```
tensorboard --logdir output/summer2winter_yosemite/log
```

## Acknowledgements
This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)