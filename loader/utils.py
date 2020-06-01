import torch
import torch.nn as nn


def load_weights(network, load_weights, parallel):
    # logging.getLogger('MDE').info(f'loading weights {load_weights}')
    print(f'loading weights {load_weights}')
    if parallel:
        network = nn.DataParallel(network)
    if torch.cuda.is_available():
        # network.load_state_dict(torch.load(load_weights, map_location='cuda:0'))
        network.load_state_dict(torch.load(load_weights))
    else:
        network.load_state_dict(torch.load(load_weights, map_location='cpu'))
