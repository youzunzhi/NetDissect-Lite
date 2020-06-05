import settings
import torch
import torchvision
import torch.nn as nn
from loader.mff.mff_model import Model as MFFModel


def loadmodel(hook_fn):
    model = MFFModel()
    for name in settings.FEATURE_NAMES:
        model._modules.get(name).register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
