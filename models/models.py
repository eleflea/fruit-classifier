from models.cnn import CNN
from models.vit import ViT
from models.svm import SVM
from models.mlp import MLP

MODELS = {
    'cnn': CNN,
    'vit': ViT,
    'svm': SVM,
    'mlp': MLP,
}
