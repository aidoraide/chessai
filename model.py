import os
from collections import OrderedDict
import torch
import torch.nn as nn
from utils.neural_utils import ResBlock, Flatten, piece2plane
from utils.constants import device, MODEL_PATH


class ChessNet(nn.Module):
    def __init__(self, n_res_blocks=19, learning_rate=0.01, bias=False, gpu_id=0):
        super(ChessNet, self).__init__()
        res_blocks = [(f'res_block{i+1}', ResBlock())
                      for i in range(n_res_blocks)]
        self.res_tower = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(len(piece2plane), 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            *res_blocks,
        ]))
        self.policy_head = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(256, 256, 3, padding=1)),
            ('batchnorm1', nn.BatchNorm2d(256)),
            ('relu1', nn.ReLU(inplace=True)),
            ('conv2', nn.Conv2d(256, 73, 1)),
            ('flatten', Flatten()),
            ('softmax', nn.Softmax(dim=1)),
        ]))
        self.value_head = nn.Sequential(OrderedDict([
            ('conv256+x_1', nn.Conv2d(256, 1, 1)),
            ('batchnorm256_1', nn.BatchNorm2d(1)),
            ('relu1', nn.ReLU(inplace=True)),
            ('flatten', Flatten()),
            ('fc64_256', nn.Linear(64, 256)),
            ('relu3', nn.ReLU(inplace=True)),
            ('fc256_1', nn.Linear(256, 1)),
            ('tanh', nn.Tanh()),
        ]))

    def forward(self, x):
        tower_out = self.res_tower(x)
        policy_out = self.policy_head(tower_out)
        value_out = self.value_head(tower_out)

        return policy_out.view(-1, 73, 8, 8), value_out.view(-1, 1)


def init_weights(m):
    classname = m.__class__.__name__
    if 'Linear' in classname:
        bound = 1.0 / m.weight.data.shape[1]
        m.weight.data.uniform_(-bound, bound)
        nn.init.zeros_(m.bias)
    # if 'Conv2d' in classname:
    #     m.weight.data.normal_(0, 1)
    # if 'BatchNorm2d' in classname:
    #     m.weight.data.normal_(0, 1)


def load_model():
    nnet = ChessNet(24)
    nnet.eval()
    print('Looking for model in', MODEL_PATH)
    if os.path.exists(MODEL_PATH):
        print(f'Loading model from "{MODEL_PATH}"')
        nnet.load_state_dict(torch.load(MODEL_PATH))
    else:
        raise Exception('Could not find model in', MODEL_PATH)
    nnet.to(device)
    return nnet
