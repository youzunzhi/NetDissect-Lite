import settings
import torch
import torch.nn as nn
import loader.mff.modules as modules
import loader.mff.resnet as resnet, loader.mff.densenet as densenet, loader.mff.senet as senet
from loader.utils import load_weights


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        encoder = settings.MODEL_NAME.split('_')[-1]
        if encoder == 'resnet':
            original_model = resnet.resnet50(pretrained=False)
            self.E = modules.E_resnet(original_model)
            num_features = 2048
            block_channel = [256, 512, 1024, 2048]
        elif encoder == 'densenet':
            original_model = densenet.densenet161(pretrained=False)
            self.E = modules.E_densenet(original_model)
            num_features = 2208
            block_channel = [192, 384, 1056, 2208]
        elif encoder == 'senet':
            original_model = senet.senet154(pretrained=None)
            self.E = modules.E_senet(original_model)
            num_features = 2048
            block_channel = [256, 512, 1024, 2048]
        else:
            raise NotImplementedError

        self.D = modules.D(num_features)
        self.MFF = modules.MFF(block_channel)
        self.R = modules.R(block_channel)
        # self.use_cuda = cfg.USE_CUDA
        load_weights(self, settings.MODEL_WEIGHTS_FILE, True)

    def forward(self, x):
        x_block1, x_block2, x_block3, x_block4 = self.E(x)
        x_decoder = self.D(x_block1, x_block2, x_block3, x_block4)
        x_mff = self.MFF(x_block1, x_block2, x_block3, x_block4, [x_decoder.size(2), x_decoder.size(3)])
        out = self.R(torch.cat((x_decoder, x_mff), 1))

        return out