import pdb
import torch
from vitvis.hooks import ViTAttHookHolder, ViTAbsHookHolder
from vitvis.hooks import TimedHookHolder, LayerHook
from vitvis.regularizers import TotalVariation as BaseTotalVariation, FakeColorDistribution as AbstractColorDistribution
from vitvis.regularizers import FakeBatchNorm as BaseFakeBN, NormalVariation as BaseNormalVariation
from vitvis.regularizers import ColorVariation as BaseColorVariation
import numpy as np
import torch.nn as nn
import random

_nums = '0123456789'

def _abbreviation(name: str) -> str:
    if len(name) <= 3:
        return name
    abr = ''.join(x for x in name if x.isupper() or x in _nums)
    return abr[:3]

def _round(num: float) -> str:
    if num > 100:
        return str(int(round(num, 0)))
    if num > 10:
        return str(round(num, 1))
    return str(round(num, 2))
    
def fix_random_seed(seed: int = 6247423):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


class InvLoss:
    def __init__(self, coefficient: float = 1.0):
        self.c = coefficient
        self.name = _abbreviation(self.__class__.__name__)
        self.last_value = 0

    def __call__(self, x: torch.tensor) -> torch.tensor:
        tensor = self.loss(x)
        self.last_value = tensor.item()
        return self.c * tensor

    def loss(self, x: torch.tensor):
        raise NotImplementedError

    def __str__(self):
        return f'{_round(self.c * self.last_value)}({_round(self.last_value)})'

    def reset(self) -> torch.tensor:
        return 0


class LossArray:
    def __init__(self):
        self.losses = []
        self.last_value = 0

    def __add__(self, other: InvLoss):
        self.losses.append(other)
        return self

    def __call__(self, x: torch.tensor):
        tensor = sum(l(x) for l in self.losses)
        self.last_value = tensor.item()
        return tensor

    def header(self) -> str:
        rest = '\t'.join(l.name for l in self.losses)
        return f'Loss\t{rest}'

    def __str__(self):
        rest = '\t'.join(str(l) for l in self.losses)
        return f'{_round(self.last_value)}\t{rest}'

    def reset(self):
        return sum(l.reset() for l in self.losses)


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.register_buffer('mean', torch.Tensor(mean).reshape((1, -1, 1, 1)))
        self.register_buffer('std', torch.Tensor(std).reshape((1, -1, 1, 1)))

    def forward(self, t: torch.tensor) -> torch.tensor:
        return self.get_normal(t)

    def get_normal(self, t: torch.Tensor) -> torch.Tensor:
        return (t - self.mean) / self.std

    def get_unit(self, t: torch.Tensor) -> torch.Tensor:
        return (t * self.std) + self.mean


class MatchBatchNorm(InvLoss):
    def __init__(self, bn: BaseFakeBN, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.bn = bn

    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.bn(x)


class TotalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseTotalVariation(p)
        self.size = size * size


class NormalVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseNormalVariation(p)
        self.size = size * size


class ColorVariation(InvLoss):
    def loss(self, x: torch.tensor):
        return self.tv(x) * np.prod(x.shape[-2:]) / self.size

    def __init__(self, p: int = 2, size: int = 224, coefficient: float = 1.):
        super().__init__(coefficient)
        self.tv = BaseColorVariation(p)
        self.size = size * size
        
        
class ColorDistribution(InvLoss):
    def loss(self, x: torch.tensor):
        return self.color_loss(x)

    def __init__(self, normalizer: Normalizer, coefficient: float = 1.):
        super().__init__(coefficient)
        self.color_loss = AbstractColorDistribution(normalizer)


class BatchAugment(InvLoss):
    def loss(self, x: torch.tensor):
        if self.aug is not None:
            x = self.aug(x)
        return self.other(x)

    def __init__(self, other: InvLoss, aug: torch.tensor = None):
        super().__init__(coefficient=1.0)
        self.other = other
        self.aug = aug


class NetworkPass(InvLoss):
    def __init__(self, model: torch.nn.Module):
        super().__init__(coefficient=0.0)
        self.model = model

    def loss(self, x: torch.tensor):
        self.model(x)
        return torch.tensor(0)      
     

class CrossEntropyLoss(InvLoss):
    def loss(self, x: torch.tensor):
        return self.xent(self.model(x), self.label)

    def __init__(self, model: torch.nn.Module, label: torch.tensor, coefficient: float = 1.):
        super().__init__(coefficient)
        self.model = model
        self.label = label
        self.xent = torch.nn.CrossEntropyLoss()

class ViTFeatHook(InvLoss):
    def __init__(self, hook: ViTAbsHookHolder, key: str, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook = hook
        self.key = key

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, :mn].diag().mean()


class ReconstructionLoss(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, x: torch.tensor, key: str, feat: int = 0,
                 coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.ref = self.hook(x).clone().detach()
        self.f = feat

    def loss(self, x: torch.tensor):
        return (self.hook(x) - self.ref).norm()
        

class BatchNorm1stLayer(InvLoss):
    def loss(self, x: torch.tensor) -> torch.tensor:
        return self.hook.get_layer(self.layer)

    def reset(self) -> torch.tensor:
        return self.hook.reset()

    def __init__(self, bn_hook: TimedHookHolder, layer: int = 0, coefficient: float = 1.):
        super().__init__(coefficient=coefficient)
        self.hook = bn_hook
        self.layer = layer

class LayerActivationNorm(InvLoss):
    def __init__(self, hook: LayerHook, model: torch.nn.Module, coefficient: float = 1.0):
        super().__init__(coefficient)
        self.hook, self.model = hook, model

    def loss(self, x: torch.tensor) -> torch.tensor:
        self.model(x)
        return - self.hook()


class ActivationNorm(InvLoss):
    def loss(self, x: torch.tensor):
        return - self.hook.get_layer(self.layer)

    def __init__(self, activation_hook: TimedHookHolder, layer: int, coefficient: float = 1.):
        super().__init__(coefficient)
        self.hook = activation_hook
        self.layer = layer

    def reset(self) -> torch.tensor:
        return self.hook.reset()


class ViTEnsFeatHook(ViTFeatHook):
    def __init__(self, hook: ViTAbsHookHolder, key: str, feat: int = 0, coefficient: float = 1.0):
        super().__init__(hook, key, coefficient)
        self.f = feat

    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS
        mn = min(all_feats.shape)
        return - all_feats[:mn, self.f].diag().mean()


class ViTHeadHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        all_feats = d[self.key][0][:, 1:, :].mean(dim=1)  # Exclude CLS and average over words, Result is BSx768
        return -all_feats.view(all_feats.shape[0], 12, -1).mean(dim=-1)[:, self.f].mean()


class ViTScoreHook(ViTEnsFeatHook):
    def loss(self, x: torch.tensor):
        d, o = self.hook(x)
        score_head = d[self.key][0][:, self.f, 1:, 1:]
        pw = int(np.sqrt(score_head.shape[-1]))
        patched = score_head.view(-1, pw, pw, pw, pw)
        ret_val = -patched[:, :, :pw // 2, :, pw // 2:].mean()
        return ret_val * 10000