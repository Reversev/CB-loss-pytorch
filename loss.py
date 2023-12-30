# !/usr/bin/python
# -*- coding: utf-8 -*-
# @Time : 2023/11/25
# @author : ''
# @FileName: loss.py
import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
      labels: A float tensor of size [batch, num_classes].
      logits: A float tensor of size [batch, num_classes].
      alpha: A float tensor of size [batch_size]
        specifying per-example weight for balanced cross entropy.
      gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
      focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(input=logits, target=labels, reduction="mean")

    if gamma == 0.0:
        modulator = 1.0
    else:
        modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(1 +
                                                                           torch.exp(-1.0 * logits)))

    loss = modulator * BCLoss

    weighted_loss = alpha * loss
    focal_loss = torch.sum(weighted_loss)
    # Normalize by the total number of positive samples.
    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(labels, logits, samples_per_cls, num_of_classes, loss_type, beta, gamma):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.

    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      num_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.

    Returns:
      class_balanced_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * num_of_classes

    labels_one_hot = F.one_hot(labels, num_of_classes).float().cuda()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0).cuda()

    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, num_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma).cuda()
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(logits, labels_one_hot, weights).cuda()
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(pred, labels_one_hot, weights).cuda()
    else:
        raise ValueError("Please input correct loss_type!")
    return cb_loss


if __name__ == '__main__':
    outputs = torch.randn((2, 10))
    labels = torch.ones(1)
    num_per_classes = np.array(
        [int(np.floor(5000 * ((1 / 100) ** (1 / 9)) ** (i))) for i in range(10)])
    loss = CB_loss(labels=labels, logits=outputs,
                   samples_per_cls=num_per_classes, num_of_classes=10,
                   loss_type="focal", beta=0.9999, gamma=2)
