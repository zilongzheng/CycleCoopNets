# Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation
This repository contains our Tensorflow implementation for "Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation".

[Project](http://www.stat.ucla.edu/~jxie/CycleCoopNets/) | [Paper] (https://arxiv.org/pdf/2103.04285.pdf)

## Prerequisites

- Linux 16.04+
- Python 3.5+
- NVIDIA GPU + CUDA CuDNN

## Getting Started
### Data
Download dataset from [CycleGAN dataset](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/datasets/download_cyclegan_dataset.sh) and data in `input` folder.

Or you can do (*e.g.* summer2winter_yosemite)
```bash
bash ./input/download_cyclegan_dataset.sh summer2winter_yosemite
```

### Train/Test
- Train a model (*e.g.* summer2winter_yosemite) by 
```
python train.py --dataroot input --category summer2winter_yosemite --output_dir output
```
- Visualize training progress
```
tensorboard --logdir output/summer2winter_yosemite/log
```

## Citation
If you this code for your research, please cite our paper.

```bibtex
@article{xie2021cyclecoopnets,
    title={Learning Cycle-Consistent Cooperative Networks via Alternating MCMC Teaching for Unsupervised Cross-Domain Translation},
    author={Xie, Jianwen and Zheng, Zilong and Fang, Xiaolin and Zhu, Song-Chun and Wu, Ying Nian},
    journal={The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI)},
    year={2021}
}
```

## Acknowledgements
This code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)