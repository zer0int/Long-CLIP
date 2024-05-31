import os
import sys
import glob
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC
import longattention.clip as longclip
import numpy as np
import cv2
import matplotlib.pyplot as plt
from captum.attr import visualization
import torch
from longattention.clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

input_dims = 224
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load base model
model, preprocess = longclip.load("checkpoints/longclip-L.pt", device=device)

# Load the fine-tuned state_dict
# If you fine-tuned Long-CLIP using GmP, first use: exp-ft-C-convert-Long-GmP-back-to-weight.py, then
# Use ft-C-convert-for-SDXL-comfyUI-longCLIP.py to get the state_dict and use it here:
finetunedclip = 'I:/LongCLIPmodelGMP/ft-checkpoints/state_dict_converted_longclip_ft_10.pt'
state_dict = torch.load(finetunedclip, map_location=device)

# Apply the fine-tuned state_dict to the base model
model.load_state_dict(state_dict)

model = model.float()

start_layer = -1  # Layer @ Visual Transformer
start_layer_text = -1  # Layer @ Text Transformer

preprocess = Compose([Resize((input_dims, input_dims), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                      Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])

image_folder = "IMG_IN"
token_folder = "TOK"
heatmap_folder = "VIS"
os.makedirs(heatmap_folder, exist_ok=True)

def interpret(image, texts, model, device, start_layer=start_layer, start_layer_text=start_layer_text):
    model = model.to(device)
    image = image.to(device)
    texts = texts.to(device)
    
    batch_size = texts.shape[0]
    images = image.repeat(batch_size, 1, 1, 1)
    logits_per_image, logits_per_text = model(images, texts)
    probs = logits_per_image.softmax(dim=-1).detach().cpu().numpy()
    index = [i for i in range(batch_size)]
    one_hot = np.zeros((logits_per_image.shape[0], logits_per_image.shape[1]), dtype=np.float32)
    one_hot[torch.arange(logits_per_image.shape[0]), index] = 1
    one_hot = torch.from_numpy(one_hot).requires_grad_(True)
    one_hot = torch.sum(one_hot.cuda() * logits_per_image)
    model.zero_grad()

    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())

    if start_layer == -1:
        start_layer = len(image_attn_blocks) - 1

    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(image_attn_blocks):
        if i < start_layer:
            continue
        
        attn_probs = blk.attn_probs.requires_grad_(True)
        grad = torch.autograd.grad(one_hot, [attn_probs], retain_graph=True, allow_unused=True)[0]
        
        if grad is not None:
            grad = grad.detach()
            cam = attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R = R + torch.bmm(cam, R)
        else:
            print(f"{color.RED}Gradient for block {i} is None{color.END}")
    
    image_relevance = R[:, 0, 1:]

    text_attn_blocks = list(dict(model.transformer.resblocks.named_children()).values())

    if start_layer_text == -1:
        start_layer_text = len(text_attn_blocks) - 1

    num_tokens = text_attn_blocks[0].attn_probs.shape[-1]
    R_text = torch.eye(num_tokens, num_tokens, dtype=text_attn_blocks[0].attn_probs.dtype).to(device)
    R_text = R_text.unsqueeze(0).expand(batch_size, num_tokens, num_tokens)
    
    for i, blk in enumerate(text_attn_blocks):
        if i < start_layer_text:
            continue
        
        attn_probs = blk.attn_probs.requires_grad_(True)
        grad = torch.autograd.grad(one_hot, [attn_probs], retain_graph=True, allow_unused=True)[0]
        
        if grad is not None:
            grad = grad.detach()
            cam = attn_probs.detach()
            cam = cam.reshape(-1, cam.shape[-1], cam.shape[-1])
            grad = grad.reshape(-1, grad.shape[-1], grad.shape[-1])
            cam = grad * cam
            cam = cam.reshape(batch_size, -1, cam.shape[-1], cam.shape[-1])
            cam = cam.clamp(min=0).mean(dim=1)
            R_text = R_text + torch.bmm(cam, R_text)
        else:
            print(f"{color.RED}Gradient for text block {i} is None{color.END}")
    
    text_relevance = R_text

    return text_relevance, image_relevance

def show_heatmap_on_text(text, text_encoding, R_text):
    CLS_idx = text_encoding.argmax(dim=-1)
    R_text = R_text[CLS_idx, 1:CLS_idx]
    text_scores = R_text / R_text.sum()
    text_scores = text_scores.flatten()
    text_tokens = _tokenizer.encode(text)
    text_tokens_decoded = [_tokenizer.decode([a]) for a in text_tokens]
    vis_data_records = [visualization.VisualizationDataRecord(text_scores, 0, 0, 0, 0, 0, text_tokens_decoded, 1)]
    visualization.visualize_text(vis_data_records)


def show_image_relevance(image_relevance, image, orig_image, img_path):
    def show_cam_on_image(img, mask):
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        cam = heatmap + np.float32(img)
        cam = cam / np.max(cam)
        return cam
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    image = image[0].permute(1, 2, 0).data.cpu().numpy()
    image = (image - image.min()) / (image.max() - image.min())
    vis = show_cam_on_image(image, image_relevance)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis


def get_image_relevance(image_relevance, image):
    dim = int(image_relevance.numel() ** 0.5)
    image_relevance = image_relevance.reshape(1, 1, dim, dim)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bilinear')
    image_relevance = image_relevance.reshape(224, 224).cuda().data.cpu().numpy()
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

all_files = os.listdir(image_folder)
for file in all_files:
    file_path = os.path.join(image_folder, file)
    try:
        img = Image.open(file_path)
        img = img.convert('RGB')
        new_file_path = os.path.join(image_folder, os.path.splitext(file)[0] + '.png')
        img.save(new_file_path, "PNG")
        if not file_path.endswith('.png'):
            os.remove(file_path)
    except IOError:
        print(f"{file} is not a valid image.")

image_files = glob.glob(f"{image_folder}/*.png")


for img_file in image_files:
    img_name = os.path.basename(os.path.splitext(img_file)[0])
    
    token_file = f"{token_folder}/tokens_{img_name}_best.txt"
    with open(token_file, 'r') as f:
        tokens = f.read().split()

    img = preprocess(Image.open(img_file)).unsqueeze(0).to(device)
    print(f"Processing {img_file} tokens...")

    for token in tokens:
        texts = [token]
        text = longclip.tokenize(texts).to(device)

        R_text, R_image = interpret(model=model, image=img, texts=text, device=device)
        #print(f"{color.BLUE}Final image relevance: {R_image}{color.END}")
        #print(f"{color.BLUE}Final text relevance: {R_text}{color.END}")
        batch_size = text.shape[0]
        for i in range(batch_size):
            show_heatmap_on_text(texts[i], text[i], R_text[i])
            show_image_relevance(R_image[i], img, orig_image=Image.open(img_file), img_path=img_file)

        heatmap_filename = f"{heatmap_folder}/{img_name}_{token}.png"
        vis = show_image_relevance(R_image[i], img, orig_image=Image.open(img_file), img_path=img_file)
        vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
        cv2.imwrite(heatmap_filename, vis)

