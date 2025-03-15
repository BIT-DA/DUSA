# convert channel rgb2bgr or bgr2rgb
from typing import List, Dict
import torch

def convert_channel(data):
    # check if data is list
    if isinstance(data, List):
        for i in range(len(data)):
            data[i] = convert_channel(data[i])
        return data
    elif isinstance(data, torch.Tensor):
        data = data[[2,1,0],...]
        return data
    # check if data is dict
    elif isinstance(data, Dict):
        for key in data:
            data[key] = convert_channel(data[key])
        return data