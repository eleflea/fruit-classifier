import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from calflops import calculate_flops
import torcheval.metrics.functional as MF

import argparse
import time
import os
from copy import deepcopy

from config import cfg
from models.models import MODELS
import dataset


def get_model(cfg, loader):
    net: nn.Module = MODELS[cfg.MODEL.NAME](cfg.DATASET.NUM_CLASSES)
    if cfg.MODEL.NAME == 'svm':
        loader_iterator = iter(deepcopy(loader))
        data = []
        count = 0
        while count <= cfg.MODEL.X_SIZE:
            data.append(next(loader_iterator)[0])
            count += cfg.TRAIN.BATCH_SIZE
        net.set_x(torch.concat(data, dim=0)[:cfg.MODEL.X_SIZE])

    return net


def eval(net, dataloader, cfg, loss_criterion=None):
    device = torch.device(cfg.SYS.DEVICE)

    net.eval()
    net.to(device)

    losses_list = []
    labels_list = []
    predicted_list = []

    with torch.no_grad():
        for data in tqdm(dataloader):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)

            if loss_criterion is not None:
                loss = loss_criterion(outputs, labels)
                losses_list.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)

            labels_list.append(labels)
            predicted_list.append(predicted)
    
    mean_loss = torch.as_tensor(losses_list).mean()
    targets = torch.concat(labels_list, dim=0)
    preds = torch.concat(predicted_list, dim=0)
    conf_matrix = MF.multiclass_confusion_matrix(preds, targets, num_classes=cfg.DATASET.NUM_CLASSES)
    mask = torch.eye(cfg.DATASET.NUM_CLASSES, device=conf_matrix.device, dtype=conf_matrix.dtype)
    acc = (mask * conf_matrix).sum() / conf_matrix.sum()

    summary = {'acc': acc.cpu(), 'loss': mean_loss.cpu(), 'conf_matrix': conf_matrix.cpu()}

    return summary


def save_config(cfg):
    filename = os.path.join('experiments', cfg.TRAIN.EXP_NAME, 'config.yaml')
    with open(filename, 'w') as fw:
        fw.write(cfg.dump())


def save_model_meta(net, cfg):
    input_shape = (1, 3, cfg.DATASET.SIZE, cfg.DATASET.SIZE)
    flops, macs, params = calculate_flops(model=net, 
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4)
    text = "FLOPs:{}   MACs:{}   Params:{} ".format(flops, macs, params)
    filename = os.path.join('experiments', cfg.TRAIN.EXP_NAME, 'model-meta.txt')
    with open(filename, 'w') as fw:
        fw.write(text)
    print(text)


def save_weights(net, cfg):
    filename = os.path.join('experiments', cfg.TRAIN.EXP_NAME, 'best.pt')
    torch.save(net.state_dict(), filename)


def save_history(history, cfg):
    filename = os.path.join('experiments', cfg.TRAIN.EXP_NAME, 'history.pt')
    torch.save(history, filename)


def train(cfg):
    torch.manual_seed(cfg.DATASET.SEED)
    device = torch.device(cfg.SYS.DEVICE)

    train_loader, val_loader, trainval_loader = dataset.get_dataloaders()

    net = get_model(cfg, train_loader)

    os.makedirs(os.path.join('experiments', cfg.TRAIN.EXP_NAME), exist_ok=True)
    save_config(cfg)
    save_model_meta(net, cfg)

    net.to(device)
    criterion = nn.CrossEntropyLoss(reduction='mean')
    optimizer = optim.SGD(net.parameters(), lr=cfg.TRAIN.LR, momentum=cfg.TRAIN.MOMENTUM, weight_decay=0)

    start_time = time.time()
    best_acc = 0
    history = []

    for epoch in range(cfg.TRAIN.EPOCHES):
        net.train()
        with tqdm(train_loader) as tl:
            for data in tl:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                tl.set_description(f'E {epoch + 1}/{cfg.TRAIN.EPOCHES}')
                tl.set_postfix(loss=loss.item())

        val_summary = eval(net, val_loader, cfg, loss_criterion=criterion)
        val_acc = val_summary['acc']
        val_loss = val_summary['loss']
        if val_loss > best_acc:
            best_acc = val_loss
            save_weights(net, cfg)

        train_summary = eval(net, trainval_loader, cfg, loss_criterion=criterion)
        history.append({
            'train': train_summary,
            'val': val_summary
        })
        save_history({'history': history}, cfg)

        train_acc = train_summary['acc']
        train_loss = train_summary['loss']
        print(f'validation: accuracy: {100 * val_acc:.2f} %, loss: {val_loss:.2f}')
        print(f'training: accuracy: {100 * train_acc:.2f} %, loss: {train_loss:.2f}')

    end_time = time.time()
    print(f'Finished training in {end_time - start_time:.1f} s.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fruit classifier training script')  
    parser.add_argument('--yaml', default='configs/cnn.yaml', type=str,  help='config file name')  
    args = parser.parse_args()
    cfg.merge_from_file(args.yaml)
    cfg['yaml'] = args.yaml
    cfg.freeze()

    train(cfg)
