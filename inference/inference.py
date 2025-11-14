import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
import evaluate  
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import itertools

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets import load_dataset
from models.model import DecoderOnlyTransformer
from utils import Vocabulary, TinyStoriesDataset 

D_MODEL = 300
N_LAYERS = 3
N_HEADS = 8
D_FF = 512
D_ATTN = 304
CONTEXT_LEN = 64
DROPOUT = 0.1

CHECKPOINT_FILE = os.path.join(parent_dir, "model_checkpoints", "checkpoint_epoch_3.pth")
VOCAB_FILE = os.path.join(parent_dir, "vocab.pkl")
BATCH_SIZE = 1 
NUM_SAMPLES_FOR_EVAL = 50
PROMPT_LENGTH = 5

MAX_GENERATION_LENGTH = 150 
TEMPERATURE = 0.8 

METRICS_OUTPUT_DIR = 'results/inference/evaluation_results'
METRICS_OUTPUT_FILE_NAME = "evaluation_metrics.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate(prompt_tokens: list[int], model: DecoderOnlyTransformer, vocab: Vocabulary, max_len: int, temp: float) -> list[int]:
    model.eval()
    
    input_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    generated_tokens = []

    with torch.no_grad():
        for _ in range(max_len):
            input_for_model = input_tensor[:, -CONTEXT_LEN:]
            
            output = model(input_for_model)
            last_token_logits = output[0, -1, :]
            
            scaled_logits = last_token_logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            
            next_token_idx = torch.multinomial(probs, num_samples=1).item()
            
            if next_token_idx == vocab.stoi['<eos>']:
                break
            
            generated_tokens.append(next_token_idx)
            
            next_token_tensor = torch.tensor([[next_token_idx]], device=device)
            input_tensor = torch.cat([input_tensor, next_token_tensor], dim=1)

    return generated_tokens


def calculate_perplexity(model: DecoderOnlyTransformer, prompt_tokens: list[int], reference_tokens: list[int], vocab: Vocabulary) -> float:
    model.eval()
    
    # Combine prompt and reference to form the full target sequence
    full_sequence = prompt_tokens + reference_tokens
    input_tensor = torch.tensor(full_sequence, dtype=torch.long, device=device).unsqueeze(0)

    # Truncate the input sequence if it exceeds the model's context length
    input_tensor = input_tensor[:, :CONTEXT_LEN]
    
    # The target for the loss function is the input sequence shifted by one
    target_tensor = input_tensor[:, 1:]
    
    with torch.no_grad():
        # Get logits for the entire sequence
        logits = model(input_tensor[:, :-1])

    # We only care about the loss on the 'continuation' part
    # Logits for prompt_len-1 onwards predict tokens from prompt_len onwards (the reference)
    continuation_logits = logits[:, len(prompt_tokens)-1:, :]
    continuation_targets = target_tensor[:, len(prompt_tokens)-1:]

    # Reshape for CrossEntropyLoss
    continuation_logits = continuation_logits.reshape(-1, len(vocab))
    continuation_targets = continuation_targets.reshape(-1)

    # Calculate cross-entropy loss, ignoring padding if any
    loss = F.cross_entropy(continuation_logits, continuation_targets, ignore_index=vocab.stoi['<pad>'])
    
    perplexity = torch.exp(loss).item()
    return perplexity


def visualize_attention(sentence: str, model: DecoderOnlyTransformer, vocab: Vocabulary):
    """
    Visualizes the self-attention weights for each head in each layer.
    
    **IMPORTANT**: This function requires your model's forward pass to be modified
    to accept a `return_attention` flag and return the attention weights.
    
    Example modification in your DecoderOnlyTransformer's forward method:
    
    def forward(self, x, return_attention=False):
        # ... your existing forward pass ...
        all_layer_attentions = []
        for layer in self.layers:
            x, attention_weights = layer(x) # Assume layer now returns weights
            all_layer_attentions.append(attention_weights)
        # ...
        if return_attention:
            return self.fc_out(x), all_layer_attentions
        return self.fc_out(x)
        
    And your DecoderLayer's forward pass should also be updated to return the weights
    from its multi-head attention sub-layer.
    """

    output_dir = "results/inference/attention_visualizations"
    os.makedirs(output_dir, exist_ok=True)
    
    sanitized_sentence = "_".join(sentence.split()[:5]) # Use first 5 words for filename

    print(f"\n--- Visualizing attention for: '{sentence}' ---")
    model.eval()

    tokens = ['<sos>'] + vocab.tokenize(sentence) + ['<eos>']
    token_ids = [vocab.stoi[token] for token in tokens]
    input_tensor = torch.tensor(token_ids, dtype=torch.long, device=device).unsqueeze(0)

    try:
        _, attention_weights = model(input_tensor, return_attention=True)
    except TypeError:
        print("\nERROR: Could not get attention weights.")
        print("Please modify your model's forward pass to accept `return_attention=True`")
        print("and return the attention weights from each layer.")
        return

    for layer_idx, layer_attention in enumerate(attention_weights):
        # Squeeze out the batch dimension, layer_attention shape: (n_heads, seq_len, seq_len)
        layer_attention = layer_attention.squeeze(0).cpu().detach()
        
        num_heads = layer_attention.shape[0]
        fig, axes = plt.subplots(1, num_heads, figsize=(20, 4))
        if num_heads == 1: axes = [axes] # Make it iterable for a single head
        fig.suptitle(f'Layer {layer_idx + 1} Attention Heads', fontsize=16)

        for head_idx in range(num_heads):
            ax = axes[head_idx]
            sns.heatmap(layer_attention[head_idx], ax=ax, cmap='viridis', xticklabels=tokens, yticklabels=tokens)
            ax.set_title(f'Head {head_idx + 1}')
            ax.tick_params(axis='x', rotation=90)
            ax.tick_params(axis='y', rotation=0)

        plt.tight_layout(rect=[0, 0, 1, 0.96])

        filename = f"layer_{layer_idx + 1}_{sanitized_sentence}.png"
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight') 
        print(f"Saved attention map to {filepath}")



def main():
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"Error: Vocabulary file not found at {VOCAB_FILE}")
    print(f"Loading vocabulary from {VOCAB_FILE}...")
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, max_seq_len=CONTEXT_LEN,
        d_attn=D_ATTN, dropout=DROPOUT
    ).to(device)

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Error: Checkpoint file not found at {CHECKPOINT_FILE}")
    print(f"Loading checkpoint from {CHECKPOINT_FILE}...")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Model state loaded successfully.")

    # Calc Metrics (Perplexity and BLEU)
    print("\n" + "="*50)
    print("Running evaluation for Perplexity and BLEU Score...")
    print("="*50)

    valid_data = load_dataset("roneneldan/TinyStories", split='validation')
    valid_dataset = TinyStoriesDataset(valid_data, vocab)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    bleu = evaluate.load("bleu")
    total_perplexity = 0.0
    total_bleu = 0.0
    
    print(f"Taking {NUM_SAMPLES_FOR_EVAL} samples from the validation set...")
    

    for i, full_sequence_tensor in enumerate(tqdm(itertools.islice(valid_loader, NUM_SAMPLES_FOR_EVAL), total=NUM_SAMPLES_FOR_EVAL)):
        full_sequence_ids = full_sequence_tensor[0].tolist()
            
        prompt_tokens = full_sequence_ids[:PROMPT_LENGTH]
        reference_tokens = full_sequence_ids[PROMPT_LENGTH:]

        generated_tokens = generate(
            prompt_tokens=prompt_tokens,
            model=model,
            vocab=vocab,
            max_len=len(reference_tokens) + 10, 
            temp=TEMPERATURE
        )
        
        # Calc Perplexity
        perplexity = calculate_perplexity(model, prompt_tokens, reference_tokens, vocab)
        total_perplexity += perplexity
        
        # Calc BLEU
        generated_text = vocab.indices_to_text(generated_tokens)
        reference_text = vocab.indices_to_text(reference_tokens)
        
        bleu_score = bleu.compute(predictions=[generated_text], references=[[reference_text]])
        total_bleu += bleu_score['bleu']

        if i == 0 or i == 1 : print(reference_text, '\n')
    
    # Calc Final Scores
    avg_perplexity = total_perplexity / NUM_SAMPLES_FOR_EVAL
    avg_bleu = total_bleu / NUM_SAMPLES_FOR_EVAL
    
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print(f"Average Perplexity per Token: {avg_perplexity:.4f}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print("="*50)

    # Save perplexity and bleu score
    os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
    results_file = os.path.join(METRICS_OUTPUT_DIR, METRICS_OUTPUT_FILE_NAME)

    with open(results_file, 'w') as f:
        f.write("="*50 + "\n")
        f.write("EVALUATION METRICS\n")
        f.write("="*50 + "\n")
        f.write(f"Number of Samples: {NUM_SAMPLES_FOR_EVAL}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write("-" * 50 + "\n")
        f.write(f"Average Perplexity per Token: {avg_perplexity:.4f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.4f}\n")
    
    print(f"Evaluation metrics saved to {results_file}")

    # attention visualization
    example_sentences = [
        "there was a brave explorer he wanted to explore",
        "there was a little girl she was so small",
    ]
    
    visualize_attention(example_sentences[0], model, vocab)
    visualize_attention(example_sentences[1], model, vocab)
    
if __name__ == '__main__':
    main()