from yacs.config import CfgNode as CN


_C = CN()

_C.SYS = CN()
# device name for training
_C.SYS.DEVICE = 'cuda:1'

_C.DATASET = CN()
# dataset root path
_C.DATASET.ROOT = '/home/eleflea/datasets/friuts'
# number of classes of dataset
_C.DATASET.NUM_CLASSES = 15
# image training size
_C.DATASET.SIZE = 192
# training set partition
_C.DATASET.TRAIN_RATIO = 0.7
# dataset partition random seed
_C.DATASET.SEED = 104829403

_C.MODEL = CN()
# model name
_C.MODEL.NAME = 'cnn'
# size of x data for SVM model
_C.MODEL.X_SIZE = 512
# patch size for ViT model
_C.MODEL.PATCH_SIZE = 16

_C.TRAIN = CN()
# training batch size
_C.TRAIN.BATCH_SIZE = 256
# number of training data loader workers
_C.TRAIN.NUM_WORKERS = 16
# training learning rate
_C.TRAIN.LR = 0.001
# training momentum
_C.TRAIN.MOMENTUM = 0.9
# number of epoches
_C.TRAIN.EPOCHES = 10
# experiment name
_C.TRAIN.EXP_NAME = 'cnn'

_C.EVAL = CN()
# evaluation batch size
_C.EVAL.BATCH_SIZE = 512
# number of evaluation data loader workers
_C.EVAL.NUM_WORKERS = 8

cfg = _C
