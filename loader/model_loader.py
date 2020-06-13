import settings
import torch
import torchvision
import torch.nn as nn
from loader.mff.mff_model import Model as MFFModel


def loadmodel(hook_fn):
    model = MFFModel()
    for layer in settings.FEATURE_NAMES:
        module_names = layer.split('_')
        module = model
        for module_name in module_names:
            module = module._modules.get(module_name)
        module.register_forward_hook(hook_fn)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
