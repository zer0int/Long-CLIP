import imageio
import torchvision
import PIL.Image
#checkin_step = training_iterations - 1
checkin_step = 10
checkin_loss_step = 50
import os
import sys
import kornia
import torch
import torch.nn.functional as F
import torch.nn.functional as F
from torch import nn
from model import longclip
import random
import numpy as np
import argparse
from torchvision.transforms import Resize
import warnings
from colorama import Fore, Style
warnings.filterwarnings('ignore')
import gzip
from functools import lru_cache
import regex as re


tokens_folder = 'TOK'
os.makedirs(tokens_folder, exist_ok=True)

training_iterations = 300   # <50 will yield awfully imprecise results, >600 doesn't improve reasonably. Recommended 300-400.
batchsize = 12              # Try 8 if you get CUDA OOM, but prefer "Sysmem Fallback" -> small batch_size = degraded quality of CLIP's "opinion".
many_tokens = 5             # How many tokens to sample. Don't use too many -> will become meaningless "random grab" of close-by tokens.
print(f"Running for {training_iterations} iterations")

device = 'cuda'
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

# longclip-B.pt - longclip-L.pt - or path/to/your/finetune_state_dict.pt 
# See my instructions for fine-tuning -> script to convert to state_dict
model, preprocess = longclip.load("checkpoints/longclip-L.pt", device=device)
model = model.eval().float()

input_dims = 224

parser = argparse.ArgumentParser(description="CLIP Gradient Ascent")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image")
args = parser.parse_args()

@lru_cache()
def default_bpe():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "model/bpe_simple_vocab_16e6.txt.gz")


@lru_cache()
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


class SimpleTokenizer(object):
    def __init__(self, bpe_path: str = default_bpe()):
        self.byte_encoder = bytes_to_unicode()
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        merges = gzip.open(bpe_path).read().decode("utf-8").split('\n')
        merges = merges[1:49152-256-2+1]
        merges = [tuple(merge.split()) for merge in merges]
        vocab = list(bytes_to_unicode().values())
        vocab = vocab + [v+'</w>' for v in vocab]
        for merge in merges:
            vocab.append(''.join(merge))
        vocab.extend(['<|startoftext|>', '<|endoftext|>'])
        self.encoder = dict(zip(vocab, range(len(vocab))))
        self.decoder = {v: k for k, v in self.encoder.items()}
        self.bpe_ranks = dict(zip(merges, range(len(merges))))
        self.cache = {'<|startoftext|>': '<|startoftext|>', '<|endoftext|>': '<|endoftext|>'}
        self.pat = re.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""", re.IGNORECASE)

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + ( token[-1] + '</w>',)
        pairs = get_pairs(word)

        if not pairs:
            return token+'</w>'

        while True:
            bigram = min(pairs, key = lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word)-1 and word[i+1] == second:
                    new_word.append(first+second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word

    def encode(self, text):
        bpe_tokens = []
        text = whitespace_clean(basic_clean(text)).lower()
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors="replace").replace('</w>', ' ')
        return text

"""# Def"""

def displ(img, pre_scaled=True):
  img = np.array(img)[:,:,:]
  img = np.transpose(img, (1, 2, 0))
  if not pre_scaled:
    img = scale(img, 48*4, 32*4)
  imageio.imwrite(str(3) + '.png', np.array(img))
  return display.Image(str(3)+'.png')


"""# Internal tweaks"""

def clip_encode_text(gobble, text):
  x = torch.matmul(text, gobble.token_embedding.weight)  # [batch_size, n_ctx, d_model]

  x = x + gobble.positional_embedding
  x = x.permute(1, 0, 2)  # NLD -> LND

  x = gobble.transformer(x)
  x = x.permute(1, 0, 2)  # LND -> NLD
  x = gobble.ln_final(x)

  x = x[torch.arange(x.shape[0]), many_tokens + len(prompt) + 2] @ gobble.text_projection

  return x

"""# Settings"""

import warnings
warnings.filterwarnings('ignore')
batch_size = batchsize

# a prompt to use before the learned tokens/words
prompt = longclip.tokenize('''''').numpy().tolist()[0]
print("Tokenized Prompt:", prompt)
prompt = [i for i in prompt if i != 0 and i != 49406 and i != 49407]

sideX = input_dims
sideY = input_dims

# set the image to use
img_path = args.image_path
img_name = os.path.splitext(os.path.basename(img_path))[0]

im = torch.tensor(imageio.imread(img_path).copy()).cuda().unsqueeze(0).permute(0, 3, 1, 2) / 255 # 0,3,1,2 . 255
im = F.interpolate(im, (sideX, sideY))
print("Image Shape After Preprocessing:", im.shape)

"""
# Setup parameters"""

torch.cuda.empty_cache()

class Pars(torch.nn.Module):
    def __init__(self):
        super(Pars, self).__init__()
        
        st = torch.zeros(batch_size, many_tokens, 49408).normal_()
        self.normu = torch.nn.Parameter(st.cuda())
        self.much_hard = 1000

        self.start = torch.zeros(batch_size, 1, 49408).cuda()
        self.start[:, :, 49406] = 1

        ptt = prompt

        self.prompt = torch.zeros(batch_size, len(ptt), 49408).cuda()
        for jk, pt in enumerate(ptt):
          self.prompt[:, jk, pt] = 1 
        
        self.pad = torch.zeros(batch_size, 248 - (many_tokens + len(prompt) + 1), 49408).cuda()
        self.pad[:, :, 49407] = 1

      
    def forward(self):
      self.soft = F.gumbel_softmax(self.normu, tau=self.much_hard, dim=-1, hard=True)
      fin = torch.cat([self.start, self.prompt, self.soft, self.pad], 1)
      #print("Output shape after forward pass:", fin.shape)
      return fin


lats = Pars().cuda()
mapper = [lats.normu]
optimizer = torch.optim.Adam([{'params': mapper, 'lr': 5}])
eps = 0

nom = torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

augs = torch.nn.Sequential(
    kornia.augmentation.RandomAffine(degrees=10, translate=.1, p=.8).cuda(),
).cuda()

tok = SimpleTokenizer()

bests = {1000:'None', 1001:'None', 1002:'None', 1003:'None', 1004:'None'}

torch.argmax(lats(), 2)[0].clone().detach().cpu().numpy()

"""# Train"""

import warnings
warnings.filterwarnings('ignore')

def augment(into):
  into = augs(into)
  return into


def ascend_txt():
  global im
  iii = nom(augment(im[:,:3,:,:].expand(64, -1, -1, -1)))                                 
  iii = model.encode_image(iii).detach()                   
  lll = lats()
  tx = clip_encode_text(model, lll)
  return -100*torch.cosine_similarity(tx.unsqueeze(0), iii.unsqueeze(1), -1).view(-1, batch_size).T.mean(1), lll

def train():
    with autocast():
        loss1, lll = ascend_txt()
    loss = loss1.mean()
    optimizer.zero_grad()
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return loss1, lll
    
def remove_non_printable(text):
    # Regex to match any character that is not a printable character, including control characters
    # and other categories like "Other", "Separator", "Surrogate", etc.
    clean_text = re.sub(r'[^\P{C}\p{Zs}]', ' ', text)
    return clean_text

def checkin(loss, lll):
    unique_tokens = set()

    these = [tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist()).replace('', '').replace('', '') for kj in range(lll.shape[0])]

    for kj in range(lll.shape[0]):
        if loss[kj] < sorted(list(bests.keys()))[-1]:
            # Remove non-printable characters and replace them with a space
            cleaned_text = ''.join([c if c.isprintable() else ' ' for c in these[kj]])
            bests[loss[kj]] = cleaned_text
            bests.pop(sorted(list(bests.keys()))[-1], None)
            try:
                decoded_tokens = tok.decode(torch.argmax(lll, 2)[kj].clone().detach().cpu().numpy().tolist())
                decoded_tokens = decoded_tokens.replace('<|startoftext|>', '').replace('<|endoftext|>', '')
                decoded_tokens = ''.join(c for c in decoded_tokens if c.isprintable())
                decoded_tokens = remove_non_printable(decoded_tokens)
                print(Fore.WHITE + f"Sample {kj} Tokens: ")
                print(Fore.BLUE + Style.BRIGHT + f"{decoded_tokens}")
                with open(f"TOK/tokens_{img_name}_all.txt", "a", encoding='utf-8') as f:
                    f.write("".join(decoded_tokens))
            except Exception as e:
                print(f"Error decoding tokens for sample {kj}: {e}")
                continue

    for j, k in zip(list(bests.values())[:5], list(bests.keys())[:5]):
        j = j.replace('<|startoftext|>', '')
        j = j.replace('<|endoftext|>', '')
        j = remove_non_printable(j)
        j = j.replace('\ufffd', '')
        j = j.replace('.', '')
        j = j.replace(';', '')
        j = j.replace('?', '')
        j = j.replace('_', '')
        j = j.replace('-', '')
        j = j.replace('\\', '')
        j = j.replace('\'', '')
        j = j.replace('"', '')
        j = j.replace('(', '')
        j = j.replace(')', '')
        j = j.replace('^', '')
        j = j.replace('&', '')
        j = j.replace('#', '')
        j = j.replace('*', '')
        j = j.replace(',', '')
        tokens = j.split()
        unique_tokens.update(tokens)

    with open(f"TOK/tokens_{img_name}_best.txt", "w", encoding='utf-8') as f:
        f.write(" ".join(unique_tokens))


def loop():
  for i in range(training_iterations):
    loss, lll = train()
    if i % checkin_step == 0:
      checkin(loss, lll)
    if i % checkin_loss_step == 0:
      print(Fore.YELLOW + f"Iteration {i}: Average Loss: {loss.mean().item()}")  # Print average loss

loop()