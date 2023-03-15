import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pyramid_pooling(vec, pooling_sizes, pooling_way):
    all_pooling_vec = []
    for v in vec:
        pooling_vec = []
        for s in pooling_sizes:
            pooling_vec = np.concatenate([pooling_vec, pooling(v, s, pooling_way)])
        all_pooling_vec = np.concatenate([all_pooling_vec, pooling_vec])
    return all_pooling_vec


def pooling(vec, size, pooling_way):
    length = len(vec)
    vec = torch.tensor(vec).view(1, length).float()
    kernel = int(math.ceil(length / size))
    pad1 = int(math.floor((kernel * size - length) / 2))
    pad2 = int(math.ceil((kernel * size - length) / 2))
    assert pad1 + pad2 == (kernel * size - length)
    padded_input = F.pad(input=vec, pad=[0, pad1 + pad2], mode='constant', value=0)
    if pooling_way == "max":
        pool = nn.MaxPool1d(kernel_size=kernel, stride=kernel, padding=0)
    else:
        pool = nn.AvgPool1d(kernel_size=kernel, stride=kernel, padding=0)
    x = pool(padded_input).flatten().numpy().tolist()
    return x
