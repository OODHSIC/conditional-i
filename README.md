# Out-of-Distribution Detection via Conditional Kernel Independence Model
This repository is the official [PyTorch](http://pytorch.org/) implementation of **Conditional-i** method.

## 0 Requirements

- Python 3.8
- [PyTorch](http://pytorch.org) install = 1.8.0
- torchvision install = 0.9.0
- CUDA 10.2
- Other dependencies: numpy, sklearn, six, pickle, lmdb

## 1 Experiments on IN1K (inliers) and IN22K (outliers)
### 1.1 Training
We release a demo for the proposed Conditional-i method. The model is built based on ResNet-18 architecture.

To train Conditional-i for 100 epochs on ImageNet1K and ImageNet21K, run:

```shell
DATASET='in1k'
MODEL='r18_bank'
DIRNAME=${DATASET}_${MODEL}_conditional_i

python train.py \
    ${DATASET} \
    --model ${MODEL} \
    --hsic-sigma 4 \
    --cond-i-weight 0.06 \
    --shuffle-ood 1 \
    --sample-cls 1 \
    --save ./outputs/${DIRNAME}
```

### 1.2 Evaluation

We present a demo for our novel evaluation metric.

```shell
DIRNAME=dirname_demo

python test.py \
    --method_name ${DIRNAME} \
    --save dirname_demo \
    --load dirname_demo/checkpoints/ckp-99.pth \
    --num_to_avg 10
```

## 2 Experiments on CIFAR-100 (inliers) and 300K Random Images (outliers)
The 80 Million Tiny Images dataset seems to be suspended recently. We therefore will supplement the results of Table 1 by training Conditional-i on CIFAR-100 and [300K Random Images](https://people.eecs.berkeley.edu/~hendrycks/300K_random_images.npy) (A cleaned subset of the original 80 Million Tiny Images) instead. The results will come soon.

## 3 Citation
```
@inproceedings{wangout,
  title={Out-of-Distribution Detection via Conditional Kernel Independence Model},
  author={Wang, Yu and Zou, Jingjing and Lin, Jingyang and Ling, Qing and Pan, Yingwei and Yao, Ting and Mei, Tao},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}
```
