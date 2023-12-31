# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2023/11/24
# @author : ''
# @FileName: load_data.py
import pickle
import torch
from torchvision.datasets.cifar import CIFAR10, CIFAR100
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as f:
        dict = pickle.load(f, encoding='bytes')
    return dict


def get_data(file_path, train=False, num_classes=10, imbalance_ratio=1):
    data, labels, new_data = None, None, None
    new_labels = []
    if train:
        num_per_classes = np.array(
            [int(np.floor(5000 * ((1 / imbalance_ratio) ** (1 / 9)) ** (i))) for i in range(num_classes)])
        print(num_per_classes)
        for i in range(1, 6):
            batch = unpickle(file_path + "/cifar-10-batches-py/data_batch_" + str(i))
            if i == 1:
                data = batch[b'data']
                labels = batch[b'labels']
            else:
                data = np.concatenate([data, batch[b'data']])
                labels = np.concatenate([labels, batch[b'labels']])

        if imbalance_ratio != 1:
            count = np.zeros((num_classes), dtype=np.int32)
            for i in range(len(labels)):
                data[i] = data[i].reshape((1, -1))
                # set n = n_i * u^i, u = (1 / 100) ** (1 / 9), 100 presents imbalance factor
                # int(np.floor(5000 * ((1 / 100) ** (1 / 9)) ** (i)))
                if count[labels[i]] < num_per_classes[labels[i]]:
                    count[labels[i]] += 1
                    if i == 0:
                        new_data = data[i]
                    else:
                        new_data = np.concatenate([new_data, data[i]])
                    new_labels.append(labels[i])
                else:
                    continue
            new_labels = np.array(new_labels)
            new_data = new_data.reshape(-1, 3072)
        else:
            new_data = data 
            new_labels = labels
    else:
        batch = unpickle(file_path + "cifar-10-batches-py/test_batch")
        new_data = batch[b'data']
        new_labels = batch[b'labels']

    return new_data, new_labels


def get_data100(file_path, train=False, num_classes=100, imbalance_ratio=1):
    data, labels, new_data = None, None, None
    new_labels = []
    if train:
        num_per_classes = np.array(
            [int(np.floor(500 * ((1 / imbalance_ratio) ** (1 / (num_classes - 1))) ** (i))) for i in range(num_classes)])
        print(num_per_classes)
        batch = unpickle(file_path + "/cifar-100-python/train")
        data = batch[b'data']
        labels = batch[b'fine_labels']

        count = np.zeros((num_classes), dtype=np.int32)
        for i in range(len(labels)):
            data[i] = data[i].reshape((1, -1))
            # set n = n_i * u^i, u = (1 / 100) ** (1 / 99), 100 presents imbalance factor
            # int(np.floor(500 * ((1 / 100) ** (1 / 99)) ** (i)))
            if count[labels[i]] < num_per_classes[labels[i]]:
                count[labels[i]] += 1
                if i == 0:
                    new_data = data[i]
                else:
                    new_data = np.concatenate([new_data, data[i]])
                new_labels.append(labels[i])
            else:
                continue
        new_labels = np.array(new_labels)
        new_data = new_data.reshape(-1, 3072)
    else:
        batch = unpickle(file_path + "cifar-100-python/class_balanced_loss")
        new_data = batch[b'data']
        new_labels = batch[b'fine_labels']

    return new_data, new_labels


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


class CIFAR10_Dataset(CIFAR10):
    def __init__(self, dataset_path, imbalance_ratio=1, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR10_Dataset, self).__init__(root=dataset_path, download=download)
        self.imbalance_ratio = imbalance_ratio
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.dataset_path = dataset_path
        if train:
            self.data, self.labels = get_data(self.dataset_path, self.train, num_classes=10,
                                              imbalance_ratio=self.imbalance_ratio)
        else:
            self.data, self.labels = get_data(self.dataset_path, self.train, num_classes=10)

        num = self.data.shape[0]
        self.data = self.data.reshape((num, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, item):
        img, label = self.data[item], self.labels[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)


class CIFAR100_Dataset(CIFAR100):
    def __init__(self, dataset_path, imbalance_ratio=1, train=True, transform=None, target_transform=None,
                 download=False):
        super(CIFAR100_Dataset, self).__init__(root=dataset_path, download=download)
        self.imbalance_ratio = imbalance_ratio
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.dataset_path = dataset_path
        if train:
            self.data, self.labels = get_data100(self.dataset_path, self.train, num_classes=100,
                                                 imbalance_ratio=self.imbalance_ratio)
        else:
            self.data, self.labels = get_data100(self.dataset_path, self.train, num_classes=100)

        num = self.data.shape[0]
        self.data = self.data.reshape((num, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))

    def __getitem__(self, item):
        img, label = self.data[item], self.labels[item]
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.data)


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = CIFAR100_Dataset(dataset_path="../datasets/cifar100/",
                                  imbalance_ratio=10,
                                  train=True,
                                  transform=transform,
                                  target_transform=target_transform,
                                  download=True)
    print(train_data.__len__())
    test_data = CIFAR100_Dataset(dataset_path="../datasets/cifar100/",
                                 train=False,
                                 transform=transform,
                                 target_transform=target_transform,
                                 download=True)
    print(test_data.__len__())
    print(train_data.classes)
    # classes = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle',
    # 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', '
    # chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin',
    # 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp',
    # 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain',
    # 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck',
    # 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road',
    # 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    # 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger',
    # 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

