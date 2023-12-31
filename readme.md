# Class-Balanced Loss Based on Effective Number of Samples

Paper link: [https://arxiv.org/abs/1901.05555](https://arxiv.org/abs/1901.05555) (CVPR2019)

## Environment and Prepare
- torch (2.0.1)
- torchvision (0.15.2)
- os
- numpy (1.26.0)

## Datasets

- Long-Tailed CIFAR10/CIFAR100 with imbalance ratio {1, 10, 20, 50, 100, 200}. 

## Training and Evaluation

- ```--outf``` set folder to output images and model best checkpoints.
- ```--imbalance_ratio``` set an imbalance ratio ```({1, 10, 20, 50, 100, 200})``` from CIFAR10/CIFAR100 (default: 'cifar10').
- ```--loss_type``` set a loss type ```('sigmoid'/'softmax'/'focal')``` with class-balance loss (default: 'softmax'). 
- ```--dataset``` set a dataset name ```('cifar10'/'cifar100')``` (default: 'cifar10').

```python
python train.py --imbalance_ratio 1
python train.py --imbalance_ratio 10
python train.py --imbalance_ratio 20
python train.py --imbalance_ratio 50
python train.py --imbalance_ratio 100
python train.py --imbalance_ratio 200

python train.py --loss_type "focal" --imbalance_ratio 1
python train.py --loss_type "focal" --imbalance_ratio 10
python train.py --loss_type "focal" --imbalance_ratio 20
python train.py --loss_type "focal" --imbalance_ratio 50
python train.py --loss_type "focal" --imbalance_ratio 100
python train.py --loss_type "focal" --imbalance_ratio 200

python train.py --loss_type "sigmoid" --imbalance_ratio 1
python train.py --loss_type "sigmoid" --imbalance_ratio 10
python train.py --loss_type "sigmoid" --imbalance_ratio 20
python train.py --loss_type "sigmoid" --imbalance_ratio 50
python train.py --loss_type "sigmoid" --imbalance_ratio 100
python train.py --loss_type "sigmoid" --imbalance_ratio 200

python train.py --loss_type "focal" --imbalance_ratio 1
python train.py --loss_type "focal" --imbalance_ratio 10
python train.py --loss_type "focal" --imbalance_ratio 20
python train.py --loss_type "focal" --imbalance_ratio 50
python train.py --loss_type "focal" --imbalance_ratio 100
python train.py --loss_type "focal" --imbalance_ratio 200


python train.py --loss_type "focal" --imbalance_ratio 1
python train.py --loss_type "focal" --imbalance_ratio 10
python train.py --loss_type "focal" --imbalance_ratio 20
python train.py --loss_type "focal" --imbalance_ratio 50
python train.py --loss_type "focal" --imbalance_ratio 100
python train.py --loss_type "focal" --imbalance_ratio 200
```


- Loss Type: Sigmoid Loss

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR10 |   Sigmoid Loss     |      95.26      |
|       10        | 	ResNet-18    |  CIFAR10 |   Sigmoid Loss	 |      90.26      |
|       20        | 	ResNet-18    |  CIFAR10 |   Sigmoid Loss	 |      87.35      |
|       50        | 	ResNet-18    |  CIFAR10 |   Sigmoid Loss	 |      81.85      |
|       100       | 	ResNet-18    |  CIFAR10 |   Sigmoid Loss	 |      75.47      |
|       200       | 	ResNet-18    |  CIFAR10 |   Sigmoid Loss	 |      65.75      |
|       1         | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      94.92      |
|       10        | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      89.90      |
|       20        | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      86.91      |
|       50        | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      81.99      |
|       100       | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      77.13      |
|       200       | 	ResNet-32    |  CIFAR10 |   Sigmoid Loss	 |      71.54      |

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR100 |   Sigmoid Loss    |      72.61      |
|       10        | 	ResNet-18    |  CIFAR100 |   Sigmoid Loss	 |      45.31      |
|       20        | 	ResNet-18    |  CIFAR100 |   Sigmoid Loss	 |      40.53      |
|       50        | 	ResNet-18    |  CIFAR100 |   Sigmoid Loss	 |      35.69      |
|       100       | 	ResNet-18    |  CIFAR100 |   Sigmoid Loss	 |      29.16      |
|       200       | 	ResNet-18    |  CIFAR100 |   Sigmoid Loss	 |      23.86      |
|       1         | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      74.07      |
|       10        | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      50.03      |
|       20        | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      44.11      |
|       50        | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      37.44      |
|       100       | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      32.30      |
|       200       | 	ResNet-32    |  CIFAR100 |   Sigmoid Loss	 |      27.20      |

- Loss Type: Softmax Loss

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR10 |   Softmax Loss     |      95.10      |
|       10        | 	ResNet-18    |  CIFAR10 |   Softmax Loss	 |      90.08      |
|       20        | 	ResNet-18    |  CIFAR10 |   Softmax Loss	 |      86.84      |
|       50        | 	ResNet-18    |  CIFAR10 |   Softmax Loss	 |      81.24      |
|       100       | 	ResNet-18    |  CIFAR10 |   Softmax Loss	 |      76.87      |
|       200       | 	ResNet-18    |  CIFAR10 |   Softmax Loss	 |      66.52      |
|       1         | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      95.36      |
|       10        | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      90.02      |
|       20        | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      87.16      |
|       50        | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      81.54      |
|       100       | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      77.21      |
|       200       | 	ResNet-32    |  CIFAR10 |   Softmax Loss	 |      71.79      |

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR100 |   Softmax Loss    |     	72.35      |
|       10        | 	ResNet-18    |  CIFAR100 |   Softmax Loss	 |      45.24      |
|       20        | 	ResNet-18    |  CIFAR100 |   Softmax Loss	 |      39.52      |
|       50        | 	ResNet-18    |  CIFAR100 |   Softmax Loss	 |      34.04      |
|       100       | 	ResNet-18    |  CIFAR100 |   Softmax Loss	 |      29.08      |
|       200       | 	ResNet-18    |  CIFAR100 |   Softmax Loss	 |      23.35      |
|       1         | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      75.18      |
|       10        | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      48.09      |
|       20        | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      43.69      |
|       50        | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      38.65      |
|       100       | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      32.62      |
|       200       | 	ResNet-32    |  CIFAR100 |   Softmax Loss	 |      27.84      |

- Loss Type: Focal Loss

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR10 |   Focal Loss   |      94.99      |
|       10        | 	ResNet-18    |  CIFAR10 |   Focal Loss	 |      90.53      |
|       20        | 	ResNet-18    |  CIFAR10 |   Focal Loss	 |      87.78      |
|       50        | 	ResNet-18    |  CIFAR10 |   Focal Loss	 |      83.07      |
|       100       | 	ResNet-18    |  CIFAR10 |   Focal Loss	 |      77.56      |
|       200       | 	ResNet-18    |  CIFAR10 |   Focal Loss	 |      71.90      |
|       1         | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      94.84      |
|       10        | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      88.80      |
|       20        | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      85.56      |
|       50        | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      78.50      |
|       100       | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      72.52      |
|       200       | 	ResNet-32    |  CIFAR10 |   Focal Loss	 |      65.35      |

| Imbalance Ratio |  Network    |     Dataset |  Loss Type   | Test Performance |
|:---------------:|:-----------:|:------------:|:---------------:|:---------------:|
|        1        | 	ResNet-18	 |  CIFAR100 |   Focal Loss  |     	77.29      |
|       10        | 	ResNet-18    |  CIFAR100 |   Focal Loss	 |      59.57      |
|       20        | 	ResNet-18    |  CIFAR100 |   Focal Loss	 |      49.58      |
|       50        | 	ResNet-18    |  CIFAR100 |   Focal Loss	 |      31.10      |
|       100       | 	ResNet-18    |  CIFAR100 |   Focal Loss	 |      26.90      |
|       200       | 	ResNet-18    |  CIFAR100 |   Focal Loss	 |      24.76      |
|       1         | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      77.60      |
|       10        | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      59.69      |
|       20        | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      49.16      |
|       50        | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      42.94      |
|       100       | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      36.78      |
|       200       | 	ResNet-32    |  CIFAR100 |   Focal Loss	 |      32.93      |

## Reference

[1] [https://github.com/vandit15/Class-balanced-loss-pytorch](https://github.com/vandit15/Class-balanced-loss-pytorch)

[2] [https://github.com/richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss) (Offical codes with Tensorflow)

