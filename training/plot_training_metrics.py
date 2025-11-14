import torch
import matplotlib.pyplot as plt

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

ck_dir = 'model_checkpoints'
output_dir = 'results/training'
os.makedirs(output_dir, exist_ok=True)

n_epochs  = 3

train_losses_all = []
val_losses_ep    = []
perplexities_ep  = []

for ep in range(1, n_epochs + 1):
    ck_path = ck_dir + f"/checkpoint_epoch_{ep}.pth"
    ck = torch.load(ck_path, map_location="cpu")

    train_losses_all.append(ck["train_losses"])
    val_losses_ep.append(ck["val_losses"][-1])      
    perplexities_ep.append(ck["perplexities"][-1]) 

# list of avg training loss of every batch per epoch 
train_losses_ep = [sum(ep_losses) / len(ep_losses) for ep_losses in train_losses_all]
epochs = list(range(1, n_epochs + 1))

fig, ax = plt.subplots(1, 3, figsize=(15, 4))

# 1) Training loss
ax[0].plot(epochs, train_losses_ep, label="Train loss", color="tab:blue")
ax[0].set_xlabel("Batch")
ax[0].set_ylabel("Loss")
ax[0].set_title("Training loss")
ax[0].grid(alpha=0.3)

# 2) Validation loss
ax[1].plot(epochs, val_losses_ep, marker="o", label="Val loss", color="tab:orange")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Loss")
ax[1].set_title("Validation loss")
ax[1].grid(alpha=0.3)

# 3) Perplexity
ax[2].plot(epochs, perplexities_ep, marker="o", label="Perplexity", color="tab:green")
ax[2].set_xlabel("Epoch")
ax[2].set_ylabel("Perplexity")
ax[2].set_title("Perplexity")
ax[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir+"/training_curves.png", dpi=150)