
import copy
import time

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset



def lecun_normal_(tensor):
    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(tensor)
    std = np.sqrt(1.0 / fan_in)

    with torch.no_grad():
        return tensor.normal_(0.0, std)