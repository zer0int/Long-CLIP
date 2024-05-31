import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.transforms.functional import to_tensor, normalize
from longgmp import longclip
import numpy as np
import torchvision.transforms as transforms
import warnings
warnings.filterwarnings('ignore')
# Load models
clipmodel = 'C:/Users/zer0int/LongCLIP/checkpoints/longclip-L.pt'
finetunedgmpclip = 'I:/LongCLIPmodelGMP/ft-checkpoints/converted_longclip_ft_10.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
original_model, preprocess = longclip.load(clipmodel, device=device)
finetuned_model = torch.load(finetunedgmpclip)
finetuned_model = finetuned_model.cuda()

# Transform for CLIP
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


image_path = "IMG_IN/appleipod.png"

# use your own suggestions and, recommended, CLIP's own "opinion" words; use longclipga_AMP-finetune.py to obtain CLIP's "opinion":
labels = ["Granny Smith", "iPod", "library", "pizza", "toaster", "dough", "apple", "apples", "ioapple", "pod", "pods", "eaten", "nutrient", "ripe", "hoax", "absurd", "💩pods", "advertise", "endorsed", "productfakespeare"]

# For the "piggy bank $$ poodle" / "IMG_IN/poodleadv.png". Same as the "ipod apple", includes both OpenAI's examples as well as CLIP's own predictions:
#labels = ["$$", "$", "piggy bank", "aggression", "alpacstampede", "costs", "freelance", "mowing", "poodle",  "pony", "clipping", "censor", "alpaca", "money", "prices", "situation", "stakes", "phenom"]

# Function to evaluate a single image with all labels at once -> softmax

def evaluate_single_image_all_labels(image_path, labels, model):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    text_tokens = longclip.tokenize(labels).to(device)

    with torch.no_grad():
        image_embedding = model.encode_image(image)
        text_embeddings = model.encode_text(text_tokens)
        logits_per_image = (image_embedding @ text_embeddings.T).softmax(dim=-1)

    probs = logits_per_image.squeeze().cpu().numpy()
    sorted_indices = np.argsort(probs)[::-1]

    sorted_labels = [(labels[i], probs[i]) for i in sorted_indices]
    
    return sorted_labels

# Evaluate with original model (all labels at once)
original_results_all = evaluate_single_image_all_labels(image_path, labels, original_model)
print("\nOriginal Model classification score (All Labels at Once):")
for label, prob in original_results_all:
    print(f"{label}: {prob:.4f}")

# Evaluate with fine-tuned model (all labels at once)
finetuned_results_all = evaluate_single_image_all_labels(image_path, labels, finetuned_model)
print("\nFine-tuned Model classification score (All Labels at Once):")
for label, prob in finetuned_results_all:
    print(f"{label}: {prob:.4f}")
