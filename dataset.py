import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

from copy import deepcopy

from config import cfg


def get_dataloaders():
    train_data_transforms = transforms.Compose([
            transforms.RandomResizedCrop(cfg.DATASET.SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    val_data_transforms = transforms.Compose([
            transforms.Resize(cfg.DATASET.SIZE),
            transforms.CenterCrop(cfg.DATASET.SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(cfg.DATASET.ROOT, transform=train_data_transforms)
    val_dataset = datasets.ImageFolder(cfg.DATASET.ROOT, transform=val_data_transforms)
    trainval_dataset = datasets.ImageFolder(cfg.DATASET.ROOT, transform=val_data_transforms)
    generator = torch.Generator().manual_seed(cfg.DATASET.SEED)
    indices = torch.randperm(len(train_dataset), generator=generator)
    train_size = int(cfg.DATASET.TRAIN_RATIO * len(train_dataset))
    train_dataset = Subset(train_dataset, indices[:train_size])
    trainval_dataset = Subset(trainval_dataset, indices[:train_size])
    val_dataset = Subset(val_dataset, indices[train_size:])

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, drop_last=True, num_workers=cfg.TRAIN.NUM_WORKERS)

    trainval_dataloader = DataLoader(trainval_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.EVAL.NUM_WORKERS)

    val_dataloader = DataLoader(val_dataset, batch_size=cfg.EVAL.BATCH_SIZE, shuffle=False, drop_last=False, num_workers=cfg.EVAL.NUM_WORKERS)

    return train_dataloader, val_dataloader, trainval_dataloader
