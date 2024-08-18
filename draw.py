from matplotlib import pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
import torch
import os

from config import cfg


def get_accuracy(cm):
    mask = torch.eye(cfg.DATASET.NUM_CLASSES, device=cm.device, dtype=cm.dtype)
    acc = (mask * cm).sum() / cm.sum()
    return acc.item()

def get_class_accuracy(cm):
    return (torch.diag(cm) / cm.sum(0)).numpy()


model_name = 'VIT'
history_path = f'experiments/{model_name.lower()}/history.pt'
classes = ['Apple', 'Banana', 'Carambola', 'Guava', 'Kiwi', 'Mango', 'Orange', 'Peach', 'Pear', 'Persimmon', 'Pitaya', 'Plum', 'Pomegranate', 'Tomatoes', 'Muskmelon']

save_dir = os.path.join(os.path.dirname(history_path), 'images')
os.makedirs(save_dir, exist_ok=True)

history = torch.load(history_path, map_location='cpu')

summaries = history['history']


train_cms = [s['train']['conf_matrix'] for s in summaries]
val_cms = [s['val']['conf_matrix'] for s in summaries]

trian_accs = [get_accuracy(cm) for cm in train_cms]
val_accs = [get_accuracy(cm) for cm in val_cms]
best_index = np.argmax(val_accs)
print(f'Best acc = {val_accs[best_index]}')
epoches = list(range(1, len(summaries) + 1))

train_losses = [s['train']['loss'] for s in summaries]
val_losses = [s['val']['loss'] for s in summaries]

fig, ax = plt.subplots()
ax.plot(epoches, trian_accs, label = "training", marker='x')
ax.plot(epoches, val_accs, label = "val", marker='+')
ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
ax.set_title(f'Accuracy of {model_name} on training set and validation set with epoch')
fig.savefig(os.path.join(save_dir, 'accuracy.png'), bbox_inches='tight')

fig, ax = plt.subplots()
ax.plot(epoches, train_losses, label = "training", marker='x')
ax.plot(epoches, val_losses, label = "val", marker='+')
ax.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
ax.set_title(f'Loss of {model_name} on training set and validation set with epoch')
fig.savefig(os.path.join(save_dir, 'loss.png'), bbox_inches='tight')

fig, ax = plt.subplots()
cm = val_cms[best_index]
df_cm = pd.DataFrame(cm.numpy(), classes, classes)
ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 6}, fmt='')
ax.set_title(f'Confusion matrix of {model_name} on validation set')
fig.savefig(os.path.join(save_dir, 'confusion matrix.png'), bbox_inches='tight')

with open(os.path.join(save_dir, 'accs.csv'), 'w') as fw:
    sl = ['Class Name,Accuracy (%)']
    accs = get_class_accuracy(cm).tolist()
    sl.extend([f'{c},{a * 100: .2f}' for c, a in zip(classes, accs)])
    print(sl)
    fw.write('\n'.join(sl))
