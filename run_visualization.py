from model import longclip
import torch
from torch import nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torchvision
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torchvision.utils
import numpy as np
import random
import pdb
import collections
from datetime import datetime
import datetime
import time
from typing import Any
import argparse
from argparse import Namespace
import os
import sys
from model.model_longclip import QuickGELU
import warnings
warnings.filterwarnings("ignore")
# Custom imports
from vitvis.image_net import TotalVariation, CrossEntropyLoss, MatchBatchNorm, BaseFakeBN, LayerActivationNorm
from vitvis.image_net import ActivationNorm, NormalVariation, ColorVariation, fix_random_seed
from vitvis.image_net import NetworkPass
from vitvis.image_net import LossArray, TotalVariation
from vitvis.image_net import ViTFeatHook, ViTEnsFeatHook
from vitvis.regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from vitvis.regularizers import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from vitvis.regularizers import ColorVariation as BaseColorVariation
from vitvis.hooks import ViTAttHookHolder, ViTGeLUHook, ClipGeLUHook, SpecialSaliencyClipGeLUHook
from vitvis.prepost import Clip, Tile, Jitter, RepeatBatch, ColorJitter, fix_random_seed
from vitvis.prepost import GaussianNoise
from vitvis.util import ClipWrapper
from vitvis.util import new_init, save_intermediate_step, save_image, fix_random_seed

_nums = '0123456789'

steps_folder = 'steps'
os.makedirs(steps_folder, exist_ok=True)

steps_folder = 'finals'
os.makedirs(steps_folder, exist_ok=True)

class ImageNetVisualizer:
    def __init__(self, loss_array: LossArray, pre_aug: nn.Module = None,
                 post_aug: nn.Module = None, steps: int = 2000, lr: float = 0.1, save_every: int = 200, saver: bool = True,
                 print_every: int = 5, **_):
        self.loss = loss_array
        self.saver = saver #None #saver
        print(self.saver)

        self.pre_aug = pre_aug
        self.post_aug = post_aug

        self.save_every = save_every
        self.print_every = print_every
        self.steps = steps
        self.lr = lr

    def __call__(self, img: torch.tensor = None, optimizer: optim.Optimizer = None, layer: int = None, feature: int = None, clipname: str = None):
        if not img.is_cuda or img.device != torch.device('cuda:0'):
            img = img.to('cuda:0')
        if not img.requires_grad:
            img.requires_grad_()
        
        # Mess with the optimizer here:
        # ['ASGD', 'Adadelta', 'Adagrad', 'Adam', 'AdamW', 'Adamax', 'LBFGS', 'NAdam', 'RAdam', 'RMSprop', 'Rprop', 'SGD', 'SparseAdam']        
        # Default:
        # optimizer = optimizer if optimizer is not None else optim.Adam([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        optimizer = optimizer if optimizer is not None else optim.Adamax([img], lr=self.lr, betas=(0.5, 0.99), eps=1e-8)
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.steps, 0.)

        print(f'#i\t{self.loss.header()}', flush=True)

        for i in range(self.steps + 1):
            optimizer.zero_grad()
            augmented = self.pre_aug(img) if self.pre_aug is not None else img
            loss = self.loss(augmented)

            if i % self.print_every == 0:
                print(f'{i}\t{self.loss}', flush=True)
            if i % self.save_every == 0 and self.saver is True:
                save_intermediate_step(img, i, layer, feature, clipname)

            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            img.data = (self.post_aug(img) if self.post_aug is not None else img).data

            self.loss.reset()

        optimizer.state = collections.defaultdict(dict)
        return img

def get_clip_dimensions(clipmodel):
    model, preprocess = longclip.load(clipmodel)
    model = model.eval()
    for transform in preprocess.transforms:
        if isinstance(transform, Resize):
            input_dims = transform.size
            break
    num_layers = None
    num_features = None
    if hasattr(model, 'visual') and hasattr(model.visual, 'transformer'):
        num_layers = len(model.visual.transformer.resblocks)
        last_block = model.visual.transformer.resblocks[-1]
        if hasattr(last_block, 'mlp'):
            c_proj_layer = last_block.mlp.c_proj
            num_features = c_proj_layer.in_features
    return input_dims, num_layers, num_features

def load_clip_model(device: str = 'cuda') -> torch.nn.Module:
    model, _ = longclip.load(clipmodel, device=device)
    model = ClipWrapper(model).to(device)
    return model

def parse_range(range_str):
    if '-' in range_str:
        start, end = map(int, range_str.split('-'))
        return list(range(start, end + 1))
    else:
        return list(map(int, range_str.split(',')))

def generate_visualizations(model, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed):
    layer_range = parse_range(layer_range_str)
    feature_range = parse_range(feature_range_str)

    for layer in layer_range:
        for feature in feature_range:
            print(f"Generating visualization for Layer {layer}, Feature {feature}...")
            loss = LossArray()
            loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
            loss += TotalVariation(2, image_size, coefficient * tv)

            pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                            GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
            image = new_init(image_size, 1, use_fixed_random_seed=use_fixed_random_seed)

            visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient, use_fixed_random_seed=use_fixed_random_seed)
            image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

            save_image(image, f'finals/{clipname}_L{layer}_F{feature}.png')

def generate_single(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed):
    loss = LossArray()
    loss += ViTEnsFeatHook(ClipGeLUHook(model, sl=slice(layer, layer + 1)), key='high', feat=feature, coefficient=1)
    loss += TotalVariation(2, image_size, coefficient * tv)

    pre, post = torch.nn.Sequential(RepeatBatch(8), ColorJitter(8, shuffle_every=True),
                                    GaussianNoise(8, True, 0.5, 400), Tile(image_size // image_size), Jitter()), Clip()
    image = new_init(image_size, 1, use_fixed_random_seed=use_fixed_random_seed)

    visualizer = ImageNetVisualizer(loss_array=loss, pre_aug=pre, post_aug=post, print_every=print_every, lr=lr, steps=steps, save_every=save_every, saver=saver, coefficient=coefficient, use_fixed_random_seed=use_fixed_random_seed)
    image.data = visualizer(image, layer=layer, feature=feature, clipname=clipname)

    save_image(image, f'finals/{clipname}_L{layer}_F{feature}.png')
    
    
    
# Choose a Long-CLIP ViT model here: ['longclip-L.pt', 'longclip-B.pt', 'my-fine-tune.pt']
longclipmodel = 'longclip-L.pt'
clipmodel = f'checkpoints/{longclipmodel}'
clipname = longclipmodel.replace(".pt", "")

input_dims, num_layers, num_features = get_clip_dimensions(clipmodel)
print(f"Selected input dimension for {clipmodel}: {input_dims}")
# This is the valid range for the selected model, alas pay attention to this output when you run the code:
print(f"Number of Layers: {num_layers} with {num_features} Features / Layer\n")

   
def main():
    model = load_clip_model()
    image_size = input_dims

    # "True" for all in layer / feature RANGE set below. "False" to generate single layer, feature.
    generate_multi = False
    
    layer, feature = 18, 3501  # Single layer and feature
    print(f"Visualizing Layer {layer}, Feature {feature}")
    
    layer_range_str = "19-20"  # continuous range ("5-10") or discrete value ("5,6,8")
    feature_range_str = "3500-3600"  # continuous range ("50-90") or discrete value ("500,1000,1555")
    
    # coefficient=0.0005 -> sharp and noisy features; coefficient=0.005 -> balanced; coefficient=0.05 -> soft, blurry, muddy
    tv = 1.0
    coefficient=0.0005
    lr = 1.0
    steps = 400
    print_every = 10
    save_every = 10
    # Set to "False" to disable saving intermediate steps:
    saver = True
    
    # Set to "True" to make the results deterministic via torch.backends.cudnn.deterministic; False to get randomly different image of same feature.
    # CURRENTLY DOESNT WORK! LEAVE AT FALSE!
    use_fixed_random_seed = False

    if generate_multi:
        generate_visualizations(model, clipname, layer_range_str, feature_range_str, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed)
    else:
        generate_single(model, clipname, layer, feature, image_size, tv, lr, steps, print_every, save_every, saver, coefficient, use_fixed_random_seed)


if __name__ == '__main__':
    main()
    


