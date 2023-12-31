import torch
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
import argparse
from resnet import ResNet18, ResNet32
import os
from loss import CB_loss
import numpy as np
from load_data import CIFAR10_Dataset, CIFAR100_Dataset
from torch.optim import lr_scheduler
from lr_warm_up import GradualWarmupScheduler


def target_transform(label):
    label = np.array(label)
    target = torch.from_numpy(label).long()
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
    parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 输出结果保存路径
    parser.add_argument('--imbalance_ratio', type=int, default=10,
                        help='input batch size for training and eval(default: 1)')
    parser.add_argument('--loss_type', type=str, default='softmax',
                        help='softmax, sigmoid, focal')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='cifar10 and cifar100')
    args = parser.parse_args()

    if not os.path.exists(args.outf):
        os.makedirs(args.outf)

    # hyper-parameter setting
    EPOCH = 200
    pre_epoch = 0
    BATCH_SIZE = 128
    LR = 0.1             # 0.02  lr * train_batch_size / 128
    imbalance_ratio = args.imbalance_ratio
    warm_up_iter = 5
    dataset_name = args.dataset
    loss_type = args.loss_type   # softmax, sigmoid, focal
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pre-processing
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset, testset = None, None
    beta, gamma = 0.0, 0.0
    sample_per_class = 0
    if dataset_name == "cifar10":
        trainset = CIFAR10_Dataset(dataset_path="./datasets/cifar10/",
                                   imbalance_ratio=imbalance_ratio,
                                   train=True,
                                   transform=transform_train,
                                   target_transform=target_transform,
                                   download=True)
        testset = CIFAR10_Dataset(dataset_path="./datasets/cifar10/",
                                  imbalance_ratio=1,
                                  train=False,
                                  transform=transform_test,
                                  target_transform=target_transform,
                                  download=True)
        beta, gamma = 0.9999, 2.0
        sample_per_class = 5000
    elif dataset_name == "cifar100":
        trainset = CIFAR100_Dataset(dataset_path="./datasets/cifar100/",
                                    imbalance_ratio=imbalance_ratio,
                                    train=True,
                                    transform=transform_train,
                                    target_transform=target_transform,
                                    download=True)
        testset = CIFAR100_Dataset(dataset_path="./datasets/cifar100/",
                                   imbalance_ratio=1,
                                   train=False,
                                   transform=transform_test,
                                   target_transform=target_transform,
                                   download=True)
        beta, gamma = 0.99, 0.5
        sample_per_class = 500
    else:
        raise ValueError("Please input correct dataset! ")

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print('The size of data: {}(train) / {}(test)'.format(trainset.__len__(), testset.__len__()))

    num_classes = len(trainset.classes)
    num = []
    for i in range(num_classes):
        num.append(int(np.floor(5000 * ((1 / imbalance_ratio) ** (1 / (num_classes - 1))) ** (i))))
    num = np.array(num)

    # define ResNet model
    net = ResNet18(num_classes=num_classes, loss_type=loss_type).to(device)

    # define loss function and optimizer
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [160, 180], 0.1)
    scheduler_warmup = GradualWarmupScheduler(optimizer,
                                              multiplier=1,
                                              total_epoch=warm_up_iter,
                                              after_scheduler=scheduler)

    best_acc = 0
    with open("log_" + loss_type + '_' + str(imbalance_ratio) + "_" + dataset_name + ".txt", "w") as f2:
        for epoch in range(pre_epoch, EPOCH):
            net.train()
            sum_loss = 0.0
            correct = 0.0
            total = 0.0
            for i, (inputs, labels) in enumerate(trainloader):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = CB_loss(labels=labels, logits=outputs,
                               samples_per_cls=num, num_of_classes=num_classes,
                               loss_type=loss_type, beta=beta, gamma=gamma)
                # loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += predicted.eq(labels.data).cpu().sum()

            with torch.no_grad():
                test_correct = 0
                test_total = 0
                net.eval()
                for (images, labels) in testloader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum()

                acc = 100. * float(test_correct) / test_total
                # save model with best accuracy
                if acc > best_acc:
                    best_acc = acc
                    torch.save(net.state_dict(), '%s/net_best.pth' % (args.outf))

                print('Epoch:%d Loss: %.04f | Acc: %.3f%% | Lr: %.03f | Acc: %.3f%% | Best_acc: %.3f%%'
                      % (epoch + 1, sum_loss / len(trainloader),
                         100. * float(correct) / total, optimizer.state_dict()['param_groups'][0]['lr'],
                         acc, best_acc))
                f2.write('Epoch:%d Loss: %.03f | Acc: %.3f%% | Lr: %.03f | Acc: %.3f%% | Best_acc: %.3f%%'
                         % (epoch + 1, sum_loss / len(trainloader),
                            100. * float(correct) / total, optimizer.state_dict()['param_groups'][0]['lr'],
                            acc, best_acc))
                f2.write('\n')
                f2.flush()
            scheduler_warmup.step()
        print("Training Finished")
