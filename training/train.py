import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import pickle

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from utils import Vocabulary, TinyStoriesDataset, Collate, get_embedding_matrix, CrossEntropyLoss, train_epoch, evaluate
from models.model import DecoderOnlyTransformer

D_MODEL = 300      
N_LAYERS = 3      
N_HEADS = 8       
D_FF = 512     
D_ATTN = 304   # d_attn % n_heads == 0 
CONTEXT_LEN = 64
DROPOUT = 0.1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
NUM_EPOCHS = 4
VOCAB_FILE = "vocab.pkl"
EMBEDDING_FILE = "embedding_matrix.pt"
CHECKPOINT_DIR = 'model_checkpoints'

device = None

if torch.cuda.is_available():
    device = torch.device("cuda")
print(f"Using device: {device}")

def main():
    print("Loading TinyStories dataset...")
    dataset = load_dataset("roneneldan/TinyStories", split='train')

    train_val_data = dataset.train_test_split(test_size=0.1)
    train_data = train_val_data['train']
    val_data = train_val_data['test']

    print(f"Training data size: {len(train_data)}")
    print(f"Validation data size: {len(val_data)}")

    if os.path.exists(VOCAB_FILE):
        print(f"Loading vocabulary from {VOCAB_FILE}...")
        with open(VOCAB_FILE, 'rb') as f:
            vocab = pickle.load(f)
    else:
        print(f"Vocabulary file not found. Bulding vocabulary from scratch...")
        vocab = Vocabulary()
        vocab.build_vocabulary(train_data['text'])
        print(f"Saving vocabulary to {VOCAB_FILE}...")
        with open(VOCAB_FILE, 'wb') as f:
            pickle.dump(vocab, f)

    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")
    PAD_IDX = vocab.stoi['<pad>']

    train_dataset = TinyStoriesDataset(train_data, vocab)
    val_dataset = TinyStoriesDataset(val_data, vocab)

    collate_fn = Collate(pad_idx=PAD_IDX, context_len=CONTEXT_LEN)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        d_ff=D_FF,
        max_seq_len=CONTEXT_LEN,
        d_attn = D_ATTN,
        dropout=DROPOUT
    )

    model.to(device)

    if os.path.exists(EMBEDDING_FILE):
        print(f"Loading pre-computed embedding matrix from {EMBEDDING_FILE}...")
        embedding_weights = torch.load(EMBEDDING_FILE)
    else:
        print("Embedding matrix file not found. Calculating and saving...")
        embedding_weights = get_embedding_matrix(vocab, D_MODEL)
        print(f"Saving embedding matrix to {EMBEDDING_FILE}...")
        torch.save(embedding_weights, EMBEDDING_FILE)

    model.token_embedding.weight.data.copy_(embedding_weights)
    model.token_embedding.weight.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = CrossEntropyLoss(pad_idx=PAD_IDX)

    train_losses = []
    val_losses = []
    perplexities = []

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"Epoch no {epoch} of {NUM_EPOCHS}\n")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, perplexity = evaluate(model, val_loader, loss_fn, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        perplexities.append(perplexity)
        
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Perplexity: {perplexity:.4f}")

        CHECKPOINT_PATH = CHECKPOINT_DIR + f"/checkpoint_epoch_{epoch}.pth"

        print(f"\nSaving checkpoint {epoch} to {CHECKPOINT_PATH}...")
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': optimizer.state_dict(),
            'model_state_dict': model.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'perplexities': perplexities,
        }, CHECKPOINT_PATH)
        print("Checkpoint saved !")

    print(f"\nTraining finished !")