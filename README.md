## ⭐ Summary: 
This repo is for fine-tuning Long-CLIP in the command line. It does not add custom nodes to ComfyUI; however, you can easily use your fine-tune with ComfyUI:
- First, fine-tune with exp-ft-B-LongGmP-finetune-LongCLIP-L.py (recommended)
- Or with ft-B-train-LongCLIP-ViT-L-14.py (likely inferior) (deprecated)
- If you used "exp-ft-B-LongGmP", use this to convert the model: exp-ft-C-convert-Long-GmP-back-to-weight.py
- Then, for both fine-tune scripts, use ft-C-convert-for-SDXL-comfyUI-longCLIP.py
- Now you have a state_dict you can plug into ComfyUI for use with SD / SDXL!
### For ComfyUI, use [SeaArtLab/ComfyUI-Long-CLIP](https://github.com/SeaArtLab/ComfyUI-Long-CLIP) custom nodes!
⬇️ Download my best-performing (ImageNet/ObjectNet accuracy of 0.89) GmP fine-tune here:
- ⬇️ [https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14](https://huggingface.co/zer0int/LongCLIP-GmP-ViT-L-14)
- It's a state-dict; use with ComfyUI as-is, or load it as the state_dict of the original LongCLIP-L for inference, to fine-tune, etc.
----
### Changes 31/May/24:

### ⭐ Geometric Parameterization (GmP) and fine-tuning on CoCo-40k-SPRIGHT: 
👉 ...Eliminates typographic attack vulnerability in Long-CLIP-L ✅🤯

- Add COCO-SPRIGHT dataset labels + training code for reproduction of my results
- Exact code I used for fine-tune (10 Epochs, 3 hours or so, RTX4090); see code comments
- Attention visualization 
----
- Use exp-ft-B-reproduce-LongGmP-finetune-LongCLIP-L-on-long-labels.py to reproduce my results with GmP-CoCo-SPRIGHT-40k
- Use ft-D-Long-eval-single-image-multi-labels.py to classify single image with multi-labels, e.g. typographic attack probing
----
- Simply use runall-longclip-explain.py to batch processes all images in "IMG_IN"; will do:
- Getting a Long-CLIP opinion about them / gradient ascent, longclipga_AMP-finetune.py
- Using Long-CLIP opinion + image -> attention heatmaps / what CLIP was 'looking at': longclip-explain-attention-visualization.py - adaptation of [hila-chefer/Transformer-MM-Explainability](https://github.com/hila-chefer/Transformer-MM-Explainability) for Long-CLIP
- Note: CLIP often predicts emojis. This can lead to [unicode] errors on certain OS. If this happens, edit the files in the "TOK" folder and remove CLIP's predicted emojis and other special characters, then run "python longclip-explain-attention-visualization.py" + the below scripts one after another manually. Emojis are important predictions in CLIP for salient features (lol - but it's true!), so I didn't want to strip them by default.
- Write CLIP opinion word into image as text, make group mosaics / runmakenames.py + runmakemosaic.py
- --> As always, check the code for comments / details and usage info! <--
----

Results of the above fine-tuning and evaluation / visualization:

![apple-ipod-demo](https://github.com/zer0int/Long-CLIP/assets/132047210/950199f6-59f2-47fb-a37b-af427e2259a2)

![poodle-demo](https://github.com/zer0int/Long-CLIP/assets/132047210/da8fdef7-b0b7-4f35-b28f-d5a3b0c41eef)

Limitation: Non-English typographic attack vulnerability; here: German, "*attention eavesdropping*" on an Alexa:
Microphone, Satire: Remarkably correct. "lichmachpuck": Nonsensical token-smasher word "light-make-puck". "schtenberg" - made-up "last-name-sound-alike" because "just prepend 'sch-' and add '-berg' or '-ung' and it appears as if it was German", right?! Well; CLIP found that pattern during its pre-training, too. =)

![exception-typographic-non-english](https://github.com/zer0int/Long-CLIP/assets/132047210/b267c83f-9633-45d5-8f62-36a6c2f215c7)

Would likely need to train CLIP on German (and other languages) to stop that, as it seems to be a "biased rant due to minority representation in training data". Maybe I'll spam GPT-4o @ OpenAI API with 40k labels for translating. =)

-----
### Changes 29/May/24:
- Added exp-ft-**.py and eval-*.py scripts

### ⚠️ Extremely experimental Geometric Parameterization (GmP) inspired by [this paper](https://arxiv.org/abs/2305.15912v4).

- Introduces sophisticated per-parameter learning needed for GmP fine-tune.
- Introduces gradient accumulation. Check "finetune" code for details.
- Otherwise, this mainly changes the model architecture (!) via custom Long-CLIP model code.
---
- ⚠️ It is normal / inevitable to get large or even 'inf' gradients in Epoch 0. But 'inf' should NOT happen in later epochs!
- Optional: Use "exp-ft-X-visualize-theta-r-barplots.py" to visualize distribution of 'theta' and 'r' components (only works with GmP fine-tuned model).
- Use "exp-ft-C-convert-Long-GmP-back-to-weight.py" to convert fine-tune to normal model object. Otherwise, the model won't be compatible with any third party code at all!
- Once converted back to ".weight", you can use the full model object "as normal" and e.g. convert to state_dict with "ft-C-convert-for-SDXL-comfyUI-longCLIP.py".
- See the code comments in the eval-*.py files for info how to use them. Check model accuracy by evaluating against a dataset.

My GmP-CLIP, fine-tuned on [https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco](https://huggingface.co/datasets/SPRIGHT-T2I/spright_coco)
Initially with short (<77 tokens) labels so I can compare it to CLIP (not-long, original-CLIP).
You can find the short labels (if you need them for whatever reason) here: [CLIP-fine-tune/tree/CLIP-vision/COCO](https://github.com/zer0int/CLIP-fine-tune/tree/CLIP-vision/COCO)

### Results:

![clip-loooong-wins](https://github.com/zer0int/Long-CLIP/assets/132047210/2785bfe5-b0f3-4bac-acd0-3d2f4303df5f)

----

## What's Geometric Parameterization / GmP, theta, r? 🤔

- GmP replaces linear layer ".weight" with GeometricLinear() for c_fc and c_proj in the MLP (multi-layer perceptron):

```
"Normal" CLIP MLP (multi-layer perceptron):

(mlp): Sequential(
  |-(c_fc): Linear(in_features=1024, out_features=4096, bias=True)
  | (gelu): QuickGELU()
|-}-(c_proj): Linear(in_features=4096, out_features=1024, bias=True)
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.weight
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.weight
|---- visual.transformer.resblocks.0.mlp.c_proj.bias


GmP CLIP MLP:

Weight decomposition into:
- radial component 'r' as norm of pre-trained weights
- angular component 'theta' as normalized direction
-> preserves weight vectors' directionality and magnitude

(mlp): Sequential(
  |-(c_fc): GeometricLinear()
  | (gelu): QuickGELU()
|-}-(c_proj): GeometricLinear()
| | 
| |-- visual.transformer.resblocks.0.mlp.c_fc.r
| |-- visual.transformer.resblocks.0.mlp.c_fc.theta
| |-- visual.transformer.resblocks.0.mlp.c_fc.bias
|
|---- visual.transformer.resblocks.0.mlp.c_proj.r
|---- visual.transformer.resblocks.0.mlp.c_proj.theta
|---- visual.transformer.resblocks.0.mlp.c_proj.bias

(Same thing for [text] transformer.resblocks)
```

## Changes: Added new Long-CLIP GA AMP scripts:

+ Refactored GA = Gradient Ascent, gets a CLIP "opinion" (text) about an image
+ (optimizes for cosine similarity of text embeddings with image embeddings)
+ Long-CLIP ViT-L/14 (the model guiding stable diffusion) now fits in <24 GB memory!
+ Approx. 1.5 minutes / image (RTX 4090) / uses torch.cuda.amp / autocast + GradScaler

+ To use: python longclipga_AMP.py --image_path "IMG_IN/catpiz.png"
+ Likewise, longclipga_AMP_anti.py gets the cosine "DIS-similarity" ("opposite of") an image

+ There is no antonym to "cat" in real life - but in CLIP's embeddings, there is!
+ Use run_longclipga_AMP_opposites.py for both (batch) -> "What's most ALIKE to the image?" + "What's most UNLIKE the image?"
+ Saves output (all + best words) to "TOK" folder / txt files. -- Pro Tip: Use "best" to prompt SDXL. =)

+ ⚠️ Highly recommended: Use "Sysmem Fallback" (NVIDIA Control Panel). It *should* fit in <24 GB VRAM - BUT that depends on what else is running on your box. Plus, you wouldn't want a CUDA OOM crash just because you opened your browser to a video. You can also lower the batch_size in the code, but that degrades CLIP's "opinion" quality (but try e.g. "8" if you absolutely must).
+ ![use-nv-cp](https://github.com/zer0int/Long-CLIP/assets/132047210/1258f304-13b5-4b62-8136-7d2367389964)

### Example (Long-CLIP "looking at" a CLIP neuron):

![banner-ga-amp](https://github.com/zer0int/Long-CLIP/assets/132047210/41e969f9-f14b-4045-bef7-7284c2fa156a)

---------

# Changes: 🥳 Added fine-tuning code for Long-CLIP! 🤩

## Optimized for *I have 1 NVIDIA GPU with 24 GB VRAM available...* 😅

### You won't win benchmarks with throwing small batch_sizes at a big model such as ViT-L/14; but using a finetune as the text encoder for e.g. Stable Diffusion SDXL, this CLIP will win some hearts! 💙🤖

+ Uses AMP (automatic mixed precision) + AdaBelief optimizer (optional: fall back to AdamW) + OneCycleLR scheduler with warmup
+ Gradually unfreeze CLIP (optional) or train whole model (default) + set Learning Rate for individual parameters (optional)
+ Debug print when exploding or vanishing gradients occur + Many fancy logs and plots with live training updates

# How to use:

### 0. Install the dependencies from requirements-finetune.txt.

### 1. ft-A-clip-interrogator-csv-to-json-labels.py
- Converts a "desc.csv" from [CLIP Interrogator](https://github.com/pharmapsychotic/clip-interrogator) to dataset labels .json.
- Example: ft-X-example-my-dataset-labels.json is the expected format for my fine-tuning script; if you have a different format - e.g. single text files next to images - explain that to GPT-4, Claude 3, or any other AI assistant + "and I need to convert them to be labels in a single .json file that should look like so:" *copy-paste the content of ft-X-example-my-dataset-labels.json into prompt as a one-shot example*
- If you load your dataset: dataset1 = ImageTextDataset("path/to/image/folder", "path/to/my-text-labels.json", transform=preprocess), and inside the .json images are: "subpath/to/0001.jpg" -> then the script dataloader will look for the image in "path/to/image/folder/subpath/to/0001.jpg".

### 2. ft-A-augment-data-color-jitter.py
- Data augmentation: If your dataset is ~1000 images, consider augmenting the images by flipping them horizontally etc.
- The script example will create a copy of your images with color jitter, which prevents CLIP from overfitting on specific colors.
- Use augmented images with .json labels and randomly select from multiple labels for a given image. See code in (3) for details.

### 3. ft-B-train-LongCLIP-ViT-L-14.py
- Fine-tune CLIP. Insert dataset .json and path to images as per previous step. See code # comments for details.
- 10,000 text-image pairs can archive good fine-tuning results within 1-2 hours (RTX 4090).

### 4. ft-C-convert-for-SDXL-comfyUI-longCLIP.py
- Convert the torch.save model .pt into a state_dict you can then just plug into SDXL as the text encoder.
- Easy as Pi with ComfyUI, see [SeaArtLab/ComfyUI-Long-CLIP](https://github.com/SeaArtLab/ComfyUI-Long-CLIP) for details!

### 5. Examples: Crazy "DeepDream of CLIP's own Neurons" dataset. Don't ask. ;-)
- Same random seed etc., just swapping out the original longCLIP-L model for my fine-tune. CFG scale 14 = high CLIP influence / guidance.
- Please note: The U-Net of SDXL was also trained on the same dataset, with a frozen CLIP (independent of CLIP).
- For fine-tuning the SDXL U-Net Diffusion Model to complement CLIP, please refer to [kohya-ss/sd-scripts](https://github.com/kohya-ss/sd-scripts)

![github-finetune-longclip](https://github.com/zer0int/Long-CLIP/assets/132047210/ca65f34f-5b80-4692-855e-d898a0425825)


-------

## Changes:

![longclip-features](https://github.com/zer0int/Long-CLIP/assets/132047210/af1e786f-1904-4653-b0ad-b470339f7c6a)


Added run_visualization.py / 'vitvis' for LongCLIP feature activation max visualization
- Check run_visualization.py code # comments for instructions
- Based on [hamidkazemi22/vit-visualization](https://github.com/hamidkazemi22/vit-visualization)

## Changes:

- Added longclipga.py -> Get 'opinion' text from model about an image
- (Optimize cosine similarity of text embeddings for image embeddings)

## To use, type: python longclipga.py --image_path "IMG_IN/catpiz.png"
+ Check the code, I left comments.
+ Original CLIP Gradient Ascent Script: Used with permission by Twitter / X: [@advadnoun](https://twitter.com/advadnoun)

![catpizza](https://github.com/zer0int/Long-CLIP/assets/132047210/f357cda2-d460-40da-8c74-e039eb78aac5)

## Changes:

- Added longclip-token-to-ID.py -> Get token <-> ID mapping

![mathematzanzigithub](https://github.com/zer0int/Long-CLIP/assets/132047210/ab8aebd4-ecfd-4303-8cbd-d93d8d43d69b)

----
## Original README.md:

# Long-CLIP
This repository is the official implementation of Long-CLIP

**Long-CLIP: Unlocking the Long-Text Capability of CLIP**\
[Beichen Zhang](https://beichenzbc.github.io), [Pan Zhang](https://panzhang0212.github.io/), [Xiaoyi Dong](https://lightdxy.github.io/), [Yuhang Zang](https://yuhangzang.github.io/), [Jiaqi Wang](https://myownskyw7.github.io/)

## 💡 Highlights
- 🔥 **Long Input length** Increase the maximum input length of CLIP from **77** to **248**.
- 🔥 **Strong Performace** Improve the R@5 of long-caption text-image retrieval by **20%** and traditional text-image retrieval by **6%**.
- 🔥 **Plug-in and play** Can be directly applied in **any work** that requires long-text capability.


## 📜 News
🚀 [2024/4/1] The training code is released!

🚀 [2024/3/25] The Inference code and models ([LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L)) are released!

🚀 [2024/3/25] The [paper](https://arxiv.org/abs/) is released!

## 👨‍💻 Todo
- [x] Training code for Long-CLIP based on OpenAI-CLIP
- [x] Evaluation code for Long-CLIP
- [x] evaluation code for zero-shot classification and text-image retrieval tasks.
- [x] Usage example of Long-CLIP
- [x] Checkpoints of Long-CLIP


## 🛠️ Usage

### Installation

Our model is based on [CLIP](https://github.com/openai/CLIP), please prepare environment for CLIP.


### how to use

Please first clone our [repo](https://github.com/beichenzbc/Long-CLIP) from github by running the following command.

```shell
git clone https://github.com/beichenzbc/Long-CLIP.git
cd Long-CLIP
```

Then, download the checkpoints of our model [LongCLIP-B](https://huggingface.co/BeichenZhang/LongCLIP-B) and/or [LongCLIP-L](https://huggingface.co/BeichenZhang/LongCLIP-L) and place it under `./checkpoints`

```python
from model import longclip
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = longclip.load("./checkpoints/longclip-B.pt", device=device)

text = longclip.tokenize(["A man is crossing the street with a red car parked nearby.", "A man is driving a car in an urban scene."]).to(device)
image = preprocess(Image.open("./img/demo.png")).unsqueeze(0).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs) # prints: [[0.982  0.01799]]
```

### Evaluation
#### Zero-shot classification

To run zero-shot classification on imagenet dataset, run the following command after preparing the data
```shell
cd eval/classification/imagenet
python imagenet.py
```

Similarly, run the following command for cifar datset
```shell
cd eval/classification/cifar
python cifar10.py               #cifar10
python cifar100.py              #cifar100
```

#### Retrieval
To run text-image retrieval on COCO2017 or Flickr30k, run the following command after preparing the data
```shell
cd eval/retrieval
python coco.py                  #COCO2017
python flickr30k.py             #Flickr30k
```
### Traning
Please refer to `train/train.md` for training details.

## ⭐ Demos
### Long-caption text-image retrieval 
<p align="center"> <a>  
<img src="./img/retrieval.png"  width="900" />
</a> </p>

### Plug-and-Play text to image generation 
<p align="center"> <a>  
<img src="./img/generation.png"  width="900" />
</a> </p>
