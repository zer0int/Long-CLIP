import torch
from torch.utils.data import DataLoader
# Import original CLIP code with modification to bypass SHA256 checksum verification
# Don't use this to load arbitrary third-party models, google "pickle vulnerability" for details
# However, this allows you to use clip.load on your own (safe) fine-tuned model:
from longgmp import longclip
from model.model_longclip import CLIP
from PIL import Image
import json
from tqdm import tqdm
import os
import random
from torch.utils.data import Dataset

class ImageTextDataset(Dataset):
    def __init__(self, image_folder, annotations_file, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        with open(annotations_file, 'r') as f:
            self.annotations = json.load(f)
        self.image_paths = list(self.annotations.keys())

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_folder, self.image_paths[idx])
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        if self.transform:
            image = self.transform(image)

        labels = self.annotations[self.image_paths[idx]]
        
        # Just used normal dataloader from training, set this to...
        # ...more labels than there are + alas use what is defined in "elif":
        if len(labels) >= 20:
            label = random.choice([labels[0], labels[1]])
        elif labels:
            label = labels[1]  # Use second label = short original CoCo dataset label
        else:
            label = ''  # Fallback if no labels are available

        return image, label


clipmodel = 'checkpoints/longclip-L.pt'
finetunedclip = 'ft-checkpoints/converted_longclip_ft_10.pt'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models with clip.load()
original_model, preprocess = longclip.load(clipmodel, device=device)
finetuned_model = torch.load(finetunedclip)
finetuned_model = finetuned_model.cuda()

# Load the validation dataset the fine-tune did NOT learn on:
val_dataset = ImageTextDataset(
    "path/to/images",
    "path/to/val-labels.json",
    transform=preprocess
)

# Load the train dataset the fine-tune has learned (overfit-eval):
train_dataset = ImageTextDataset(
    "path/to/images",
    "path/to/train-labels.json",
    transform=preprocess
)

batch_size = 48
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


def evaluate_model(model, data_loader):
    correct = 0
    total = 0

    for batch_images, batch_labels in tqdm(data_loader):
        batch_images = batch_images.to(device)
        batch_texts = longclip.tokenize(batch_labels).to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(batch_images)
            text_embeddings = model.encode_text(batch_texts)
            logits_per_image = (image_embeddings @ text_embeddings.T).softmax(dim=-1)

            # Get the top predictions
            _, top_indices = logits_per_image.topk(1, dim=-1)

            for i, label in enumerate(batch_labels):
                if label == batch_labels[top_indices[i]]:
                    correct += 1
                total += 1

    accuracy = correct / total
    return accuracy

original_accuracy = evaluate_model(original_model, val_loader)
finetuned_accuracy = evaluate_model(finetuned_model, val_loader)

print(f"Original Model Accuracy on val: {original_accuracy}")
print(f"Fine-tuned Model Accuracy on val: {finetuned_accuracy}")

original_accuracy_train = evaluate_model(original_model, train_loader)
finetuned_accuracy_train = evaluate_model(finetuned_model, train_loader)

print(f"Original Model Accuracy on train: {original_accuracy_train}")
print(f"Fine-tuned Model Accuracy on train: {finetuned_accuracy_train}")

print("\nNote: Your fine-tune should be better than the original model. However, if the difference on 'train' far exceeds the difference on 'val', this suggests overfit (bad).")