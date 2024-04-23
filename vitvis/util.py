import torch
from torch import nn as nn
from torch.nn import functional as F
import torchvision
import random

class ClipWrapper(torch.nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.clip.encode_image(x)

def fix_random_seed(seed: int = 6247423):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
        
   
def new_init(size: int, batch_size: int = 1, last: torch.nn = None, padding: int = -1, zero: bool = False, use_fixed_random_seed: bool = False) -> torch.nn:
    if use_fixed_random_seed:
        fix_random_seed(seed=6247423)
    output = torch.rand(size=(batch_size, 3, size, size)) if not zero else torch.zeros(size=(batch_size, 3, size, size))
    output = output.cuda()
    if last is not None:
        big_size = size if padding == -1 else size - padding
        up = torch.nn.Upsample(size=(big_size, big_size), mode='bilinear', align_corners=False).cuda()
        scaled = up(last)
        cx = (output.size(-1) - big_size) // 2
        output[:, :, cx:cx + big_size, cx:cx + big_size] = scaled
    output = output.detach().clone()
    output.requires_grad_()
    return output


def save_intermediate_step(tensor: torch.Tensor, step: int, layer: int, feature: int, clipname: str, base_path='steps'):
    """
    Saves an intermediate step image during visualization.

    Parameters:
    - tensor: A torch.Tensor object. Expected shape [1, C, H, W].
    - step: An integer, the current optimization step.
    - layer: An integer, the current layer being visualized.
    - feature: An integer, the specific feature within the layer being targeted.
    - base_path: A string, the base directory to save the images.
    """
    import os
    import torchvision.utils

    # Ensure the base path exists
    os.makedirs(base_path, exist_ok=True)

    # Construct the filename
    base_path = f'steps/{clipname}_L{layer}-F{feature}/'
    #base_path = f'steps/L{layer}-F{feature}/'
    os.makedirs(base_path, exist_ok=True)
    filename = f'step{step}.png'
    filepath = os.path.join(base_path, filename)

    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)

    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())

    # Save the image
    torchvision.utils.save_image(tensor, filepath)

def save_image(tensor: torch.Tensor, path: str):
    """
    Saves a tensor as an image.

    Parameters:
    - tensor: A torch.Tensor object. Expected shape [C, H, W] or [1, C, H, W].
    - path: A string, the path where the image will be saved.
    """
    # If the tensor has a batch dimension, remove it
    if tensor.dim() == 4 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    
    # Normalize the tensor to [0, 1] if it's not already
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = (tensor - tensor.min()) / (tensor.max() - tensor.min())
    
    # Save the image
    torchvision.utils.save_image(tensor, path)
    
