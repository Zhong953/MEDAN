# code in this file is adpated from
# https://github.com/iCGY96/ARPL
# https://github.com/wjun0830/Difficulty-Aware-Simulator

import os
import torch
import numpy as np
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10, CIFAR100, SVHN

from .tools import *

DATA_PATH = './datasets'
# DATA_PATH = '_'
TINYIMAGENET_PATH = DATA_PATH + '/tiny_imagenet/'
val_annotations_path = 'path/to/tiny-imagenet-200/val/val_annotations.txt'


# 对数据进行过滤，只保留目标值在known列表中的数据
class CIFAR10_Filter(CIFAR10):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)

# 初始化CIFAR10_Filter类
class CIFAR10_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)
        
        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)        
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class CIFAR100_Filter(CIFAR100):
    def __Filter__(self, known):
        datas, targets = np.array(self.data), np.array(self.targets)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.targets = np.squeeze(
            np.take(datas, mask, axis=0)), np.array(new_targets)

class CIFAR100_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 100))) - set(known))
        print('Selected Labels: ', known)

        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        testset = CIFAR100_Filter(root=dataroot, train=False, download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        

class SVHN_Filter(SVHN):
    """SVHN Dataset.
    """
    def __Filter__(self, known):
        targets = np.array(self.labels)
        mask, new_targets = [], []
        for i in range(len(targets)):
            if targets[i] in known:
                mask.append(i)
                new_targets.append(known.index(targets[i]))
        self.data, self.labels = self.data[mask], np.array(new_targets)

class SVHN_OSR(object):
    def __init__(self, known, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 10))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = SVHN_Filter(root=dataroot, split='train',
                               download=True, transform=train_transform)
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))


class Tiny_ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        #给每一张图打标签
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets


class Tiny_ImageNet_OSR(object):
    def __init__(self, known, dataroot=TINYIMAGENET_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        self.num_known = len(known)
        self.known = known
        self.unknown = list(set(list(range(0, 200))) - set(known))

        print('Selected Labels: ', known)

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'train'), train_transform)        
        trainset.__Filter__(known=self.known)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        testset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val_menu'), transform)
        testset.__Filter__(known=self.known)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        outset = Tiny_ImageNet_Filter(os.path.join(dataroot, 'tiny-imagenet-200', 'val_menu'), transform)
        outset.__Filter__(known=self.unknown)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset), 'Outlier Num: ', len(outset))
        print('All Test: ', (len(testset) + len(outset)))

class CIFAR10_OOD(object):
    def __init__(self,  dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):

        train_transform = predata(img_size)
        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        testset = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'Test Num: ', len(testset))
        print('All Test: ', (len(testset)))

class CIFAR100_OOD(object):
    def __init__(self,  dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):

        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        cifar100_train = CIFAR100(root=DATA_PATH, train=True, download=True, transform=transform)
        cifar100_test = CIFAR100(root=DATA_PATH, train=False, download=True, transform=transform)

        # 合并训练集和测试集
        cifar100_all = torch.utils.data.ConcatDataset([cifar100_train, cifar100_test])

        # 随机抽取样本
        def sample_from_dataset(dataset, num_samples_per_class):
            class_indices = [[] for _ in range(100)]
            for idx, (_, target) in enumerate(dataset):
                class_indices[target].append(idx)

            sampled_indices = []
            for class_idx in range(100):
                sampled_indices.extend(np.random.choice(class_indices[class_idx], num_samples_per_class, replace=False))

            return sampled_indices

        num_samples_per_class = 100
        sampled_indices = sample_from_dataset(cifar100_all, num_samples_per_class)

        # 创建子数据集
        ood_dataset = torch.utils.data.Subset(cifar100_all, sampled_indices)

        # 验证样本数量
        print(f"Number of samples in OOD dataset: {len(ood_dataset)}")  # 应该输出10000

        # 创建数据加载器
        self.out_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory)

class SVHN_OOD(object):
    def __init__(self,  dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):

        transform = test_transform(img_size)

        pin_memory = True if use_gpu else False

        outset = SVHN_Filter(root=dataroot, split='test', download=True, transform=transform)
        self.out_loader = torch.utils.data.DataLoader(
            outset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        svhn_train = SVHN(root=DATA_PATH, split='train', download=True, transform=transform)
        svhn_test = SVHN(root=DATA_PATH, split='test', download=True, transform=transform)

        # 合并训练集和测试集
        svhn_all = torch.utils.data.ConcatDataset([svhn_train, svhn_test])

        # 随机抽取样本
        def sample_from_dataset(dataset, num_samples):
            indices = list(range(len(dataset)))
            sampled_indices = np.random.choice(indices, num_samples, replace=False)
            return sampled_indices

        num_samples = 10000
        sampled_indices = sample_from_dataset(svhn_all, num_samples)

        # 创建子数据集
        ood_dataset = torch.utils.data.Subset(svhn_all, sampled_indices)

        # 验证样本数量
        print(f"Number of samples in OOD dataset: {len(ood_dataset)}")  # 应该输出10000

        # 创建数据加载器
        self.out_loader = torch.utils.data.DataLoader(ood_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# F1检测
class CIFAR10_F1(object):
    def __init__(self,  dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=32, options=None):
        train_transform = predata(img_size)
        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        trainset = CIFAR10_Filter(root=dataroot, train=True, download=True, transform=train_transform)
        self.train_loader = torch.utils.data.DataLoader(
            trainset, batch_size=batch_size, shuffle=True, num_workers=options['num_workers'], pin_memory=pin_memory
        )

        val_set = CIFAR10_Filter(root=dataroot, train=False, download=True, transform=transform)
        self.val_loader = torch.utils.data.DataLoader(
            val_set, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )

        print('Train Num: ', len(trainset), 'VAl Num: ', len(val_set))
        print('All VAl: ', (len(val_set)))

class ImageNet_Filter(ImageFolder):
    """Tiny_ImageNet Dataset.
    """
    def __Filter__(self, known):
        #给每一张图打标签
        datas, targets = self.imgs, self.targets
        new_datas, new_targets = [], []
        for i in range(len(datas)):
            if datas[i][1] in known:
                new_item = (datas[i][0], known.index(datas[i][1]))
                new_datas.append(new_item)
                new_targets.append(known.index(targets[i]))
        datas, targets = new_datas, new_targets
        self.samples, self.imgs, self.targets = datas, datas, targets
class ImageNet_F1(object):
    def __init__(self, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        train_transform = predata(img_size)
        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        testset = ImageNet_Filter(os.path.join(dataroot, 'Imagenet', 'Imagenet'), transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('Test Num: ', len(testset))

class ImageNet_resize_F1(object):
    def __init__(self, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        train_transform = predata(img_size)
        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        testset = ImageNet_Filter(os.path.join(dataroot, 'Imagenet_resize', 'Imagenet_resize'), transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('Test Num: ', len(testset))

class LSUN_F1(object):
    def __init__(self, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        train_transform = predata(img_size)
        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        testset = ImageNet_Filter(os.path.join(dataroot, 'LSUN', 'LSUN'), transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('Test Num: ', len(testset))

class LSUN_resize_F1(object):
    def __init__(self, dataroot=DATA_PATH, use_gpu=True, batch_size=128, img_size=64, options=None):
        train_transform = predata(img_size)
        transform = test_transform(img_size)
        pin_memory = True if use_gpu else False

        testset = ImageNet_Filter(os.path.join(dataroot, 'LSUN_resize', 'LSUN_resize'), transform)
        self.test_loader = torch.utils.data.DataLoader(
            testset, batch_size=batch_size, shuffle=False, num_workers=options['num_workers'], pin_memory=pin_memory,
        )
        print('Test Num: ', len(testset))
