import torch
from torch import nn as nn
from torch.nn import functional as F
from model.model_longclip import QuickGELU
import random 

# GENERIC HOOKS

def fix_random_seed(seed: int = 6247423):
    import torch
    import numpy as np
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

class ItemIterator:
    @property
    def iterator_item(self):
        raise NotImplementedError

    def __iter__(self):
        return iter(self.iterator_item)

    def __getitem__(self, item):
        print(self.iterator_item)
        return self.iterator_item[item]

    def __len__(self):
        return len(self.iterator_item)

class BasicHook:
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.base_hook_fn)
        self.activations = None

    def close(self):
        self.hook.remove()

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t
        x = x[0][0] if isinstance(x[0], tuple) else x[0]
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError
        
        
class HookHolder(ItemIterator):
    def __init__(self, classifier: nn.Module, hook_class, layer_class):
        self.hooks = [hook_class(m) for m in classifier.modules() if isinstance(m, layer_class)]

    @property
    def iterator_item(self):
        return self.hooks

    def check_for_attr(self, attr: str, hook_class):
        for h in self:
            if not hasattr(h, attr):
                raise AttributeError('Class {} does not have attribute {}'.format(hook_class.__name__, attr))

    def _broadcast(self, func_name: str, *to_propagate):
        for i in self:
            func = getattr(i, func_name)
            func(*to_propagate)

    def _gather(self, attr: str) -> list:
        return [getattr(l, attr) for l in self]

    def close(self):
        self._broadcast('close')
        
class TimedHookHolder(HookHolder):
    def __init__(self, classifier: nn.Module, hook_class, layer_class, use_fixed_random_seed: bool = False):
        super().__init__(classifier, hook_class, layer_class)
        if use_fixed_random_seed:
            fix_random_seed()
    def get_activations(self):
        all_values = []
        for h in self.hooks:
            all_values += h.activations
        return all_values

    def get_layer(self, item):
        all_values = sorted(self.get_activations())
        return all_values[item][1]

    def set_seed(self, seed: int):
        self._broadcast('set_seed', seed)

    def set_target(self, target: list):
        self._broadcast('set_target', target)

    def reset(self):
        all_values = self.get_activations()
        all_values = sum([v.sum() for _, v in all_values])
        self._broadcast('reset')
        return all_values


class ViTHook(BasicHook):
    def __init__(self, module: nn.Module, return_output: bool, name: str):
        super().__init__(module)
        self.mode = return_output
        self.name = name

    def base_hook_fn(self, model: nn.Module, input_t: torch.tensor, output_t: torch.tensor):
        x = input_t if not self.mode else output_t
        x = x[0] if isinstance(x, tuple) else x
        return self.hook_fn(model, x)

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        self.activations = x

class LayerHook:
    def __init__(self, classifier: nn.Module, layer_class, layer_depth: int, hook_cls):
        self.layer = [m for m in classifier.modules() if isinstance(m, layer_class)][layer_depth]
        self.hook = hook_cls(self.layer)

    def __call__(self) -> torch.tensor:
        return self.hook()

class FakeHookWrapper:
    def __init__(self, value):
        self.activations = value

class ViTAbsHookHolder(nn.Module):
    pass


# GELU HOOKS for feature visualization

class ViTAttHookHolder(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, in_feat: bool = False, keys: bool = False, queries: bool = False,
                 values: bool = False, scores: bool = False, out_feat: bool = False, sl: slice = None):
        super().__init__()
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, MultiHeadedSelfAttention)]
        self.attentions = self.just_save[sl]
        self.in_features = [ViTHook(m, False, 'in') for m in self.attentions] if in_feat else None
        self.keys = [ViTHook(a.proj_k, True, 'k') for a in self.attentions] if keys else None
        self.queries = [ViTHook(a.proj_q, True, 'q') for a in self.attentions] if queries else None
        self.value = [ViTHook(a.proj_v, True, 'v') for a in self.attentions] if values else None
        self.score_behaviour = scores
        self.out_features = [ViTHook(m, True, 'out') for m in self.attentions] if out_feat else None
        # print(in_feat, keys, queries, values, out_feat)

        self.model = classifier

    @property
    def scores(self):
        # for a in self.attentions:
        #     a.scores = None
        # return None
        return [FakeHookWrapper(a.scores) for a in self.attentions] if self.score_behaviour else None

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        # for a in self.just_save:
        #     a.scores = None
        out = None
        if x is not None:
            out = self.model(x)
        options = [self.in_features, self.keys, self.queries, self.value, self.scores, self.out_features]
        options = [[o.activations for o in l] if l is not None else None for l in options]
        names = ['in_feat', 'keys', 'queries', 'values', 'scores', 'out_feat']
        return {n: o for n, o in zip(names, options) if o is not None}, out


class ClipGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        #for m in classifier.modules():
        #    print(type(m))
        self.just_save = [m for m in classifier.modules() if isinstance(m, QuickGELU)]
        #print("\nself.just_save")
        #print(self.just_save)
        #print("\nself.attentions")
        self.attentions = self.just_save[sl]
        #print(self.attentions)
        self.high = [ViTHook(m, True, 'high') for m in self.attentions]
        #print("\nself.high")
        #print(self.high)

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        #print("\nin def forward\nout")
        #print(out)
        options = [self.high]
        options = [[o.activations.transpose(0, 1) for o in l if o.activations is not None] if l is not None else None
                   for l in options]
        names = ['high']
        #print("\nOptions")
        #print(options)
        return {n: o for n, o in zip(names, options) if o is not None}, out

        
class ViTGeLUHook(ViTAbsHookHolder):
    def __init__(self, classifier: nn.Module, sl: slice = None):
        super().__init__()
        self.cl = classifier
        sl = slice(None, None) if sl is None else sl
        self.just_save = [m for m in classifier.modules() if isinstance(m, PositionWiseFeedForward)]
        self.attentions = self.just_save[sl]
        self.high = [ViTHook(m.fc1, True, 'high') for m in self.attentions]

    def forward(self, x: torch.tensor) -> ({}, torch.tensor):
        out = self.cl(x)
        options = [self.high]
        options = [[F.gelu(o.activations) for o in l] if l is not None else None for l in options]
        names = ['high']
        return {n: o for n, o in zip(names, options) if o is not None}, out


# OTHER CLIP HOOKS

class ReconstructionClipGeLUHook(ClipGeLUHook):
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[0].activations.transpose(0, 1)
        return acts
        
class SaliencyClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        acts = self.high[l].activations.transpose(0, 1)[:, 1:, f]
        return acts
        
class SpecialSaliencyClipGeLUHook(ClipGeLUHook):
    def __init__(self, classifier: nn.Module, sl: slice = None, layer=None, feature=None):
        super().__init__(classifier, sl)
        # Now, `layer` and `feature` are stored as attributes of the instance
        self.layer = layer
        self.feature = feature
    
    @torch.no_grad()
    def forward(self, x: torch.tensor, l: int, f: int) -> torch.tensor:
        _ = self.cl(x)
        # Use self.layer and self.feature if they are supposed to override l and f
        acts = self.high[l].activations.transpose(0, 1)[:, 1:, f]
        return acts
        
class SimpleClipGeLUHook(ClipGeLUHook):
    @torch.no_grad()
    def forward(self, x: torch.tensor) -> torch.tensor:
        _ = self.cl(x)
        # :-1 excludes CLS token
        acts = torch.cat([((l.activations.transpose(0, 1))[:, 1:, :]).mean(dim=1).float() for l in self.high
                          if l.activations is not None], dim=-1).clone().detach()
        return acts        

      
        
# ACTIVATION HOOKS

class AbsActivationHook(BasicHook):
    def __init__(self, module: nn.Module, feature: int = 0, targets: list = None):
        super().__init__(module)
        self.activations = []
        self.feature = feature
        self.targets = targets

    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    def reset(self):
        if self.activations is not None:
            for _, v in self.activations:
                del v
            del self.activations
        self.activations = []

    def set_feature(self, feature: int):
        self.feature = feature

    def set_target(self, target: list):
        self.targets = target

    def __call__(self) -> torch.tensor:
        if isinstance(self.activations, list):
            return torch.tensor(0)
        return self.activations


class ActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()


class ActivationReluHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        input_t = torch.relu(input_t)
        diagonal = torch.arange(min(input_t.size()[:2]))
        feats = input_t[diagonal, diagonal]
        self.activations = feats.norm(p=2, dim=(1, 2)).mean()


class TargetActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        diagonal = torch.arange(min(input_t.patch_size()[:2]))
        feats = input_t[diagonal, self.targets]
        self.activations.append((datetime.now(), feats.norm(p=2, dim=(1, 2)).mean()))


class ContrastiveActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        value = size * feats.norm(p=2, dim=(1, 2)).mean() - input_t[diagonal].norm(p=2, dim=(2, 3)).mean()
        self.activations.append((datetime.now(), value))


class ViTCLSActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats[:, 0].mean() * feats.patch_size(-1)
        self.activations.append((datetime.now(), feats))


class ViTMeanActivationHook(AbsActivationHook):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        input_t = input_t.transpose(1, 2)
        input_t = input_t[:, self.feature:]
        size = min(input_t.patch_size()[:2])
        diagonal = torch.arange(size)
        feats = input_t[diagonal, diagonal]
        feats = feats.norm(p=2, dim=-1).mean() * 10 * 10
        self.activations.append((datetime.now(), feats))


# BATCH NORM HOOKS 

class BatchNormHookHookAbs(AbsActivationHook):
    def hook_fn(self, model: nn.Module, x: torch.tensor):
        raise NotImplementedError

    @staticmethod
    def get_mean_var(x: torch.tensor) -> (torch.tensor, torch.tensor):
        view = x.transpose(1, 0).contiguous().view([x.patch_size(1), -1]).to('cuda:0')
        return view.mean(1), view.var(1, unbiased=False)

    @staticmethod
    def normalize_eval(model: nn.Module, x: torch.tensor) -> torch.tensor:
        extra_dim = [1] * (x.dim() - 2)
        mean = model.running_mean.data.view(1, -1, *extra_dim)
        var = model.running_var.data.view(1, -1, *extra_dim)
        return (x - mean) / var


class MatchModelBNStatsHook(BatchNormHookHookAbs):
    def hook_fn(self, model: nn.Module, input_t: torch.Tensor):
        mean, var = self.get_mean_var(input_t)
        cur_value = torch.norm(model.running_var.data - var, 2) + torch.norm(model.running_mean.data - mean, 2)
        self.activations.append((datetime.now(), cur_value))