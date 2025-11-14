import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import pickle
import time
import sys
import torch.nn.utils.rnn as rnn_utils

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets import load_dataset
from utils import Vocabulary, TinyStoriesDataset

from models.model import DecoderOnlyTransformer as DecoderOnlyTransformerBaseline
from models.model_kv import DecoderOnlyTransformer as DecoderOnlyTransformerKV

D_MODEL = 300
N_LAYERS = 3
N_HEADS = 8
D_FF = 512
D_ATTN = 304
CONTEXT_LEN = 64
DROPOUT = 0.1

CHECKPOINT_FILE = os.path.join(parent_dir, "model_checkpoints", "checkpoint_epoch_3.pth")
VOCAB_FILE = os.path.join(parent_dir, "vocab.pkl")

results_dir = os.path.join(parent_dir, "results", "kv_caching")
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, "kv_caching_evaluation.txt")

BATCH_SIZE = 20          
PROMPT_LENGTH = 5
MAX_GENERATION_LENGTH = 100 
TEMPERATURE = 0.8 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def generate_baseline_batched(prompts_tensor: torch.Tensor, model: DecoderOnlyTransformerBaseline, max_len: int, temp: float) -> int:
    """
    Generates text autoregressively WITHOUT KV cache, for a batch.
    Returns total number of tokens generated.
    """
    model.eval()
    
    input_tensor = prompts_tensor.to(device)
    total_tokens_generated = 0

    with torch.no_grad():
        for _ in range(max_len - input_tensor.size(1)):
            input_for_model = input_tensor[:, -CONTEXT_LEN:]
            
            output = model(input_for_model) 
            
            last_token_logits = output[:, -1, :]
            
            scaled_logits = last_token_logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            
            next_token_indices = torch.multinomial(probs, num_samples=1)
            
            input_tensor = torch.cat([input_tensor, next_token_indices], dim=1)
            total_tokens_generated += input_tensor.size(0)

    return total_tokens_generated


def generate_with_kv_cache(prompts_tensor: torch.Tensor, model: DecoderOnlyTransformerKV, max_len: int, temp: float) -> int:
    model.eval()
    
    input_tensor = prompts_tensor.to(device)
    batch_size = input_tensor.size(0)
    total_tokens_generated = 0
    past_kv_caches = None

    with torch.no_grad():
        logits, new_kv_caches = model(input_tensor, past_kv_caches=None)
        
        past_kv_caches = new_kv_caches
        
        last_token_logits = logits[:, -1, :]
        
        scaled_logits = last_token_logits / temp
        probs = torch.softmax(scaled_logits, dim=-1)
        
        next_token_indices = torch.multinomial(probs, num_samples=1)
        
        input_tensor = next_token_indices
        total_tokens_generated += batch_size

        for _ in range(max_len - prompts_tensor.size(1) - 1):
            logits, new_kv_caches = model(input_tensor, past_kv_caches=past_kv_caches)
            
            past_kv_caches = new_kv_caches
            
            # Logits shape is (batch, 1, vocab_size)
            last_token_logits = logits[:, -1, :] 
            
            scaled_logits = last_token_logits / temp
            probs = torch.softmax(scaled_logits, dim=-1)
            
            next_token_indices = torch.multinomial(probs, num_samples=1)
            
            input_tensor = next_token_indices
            total_tokens_generated += batch_size

    return total_tokens_generated


def main():
    print("="*50)
    print("KV Cache Runtime Efficiency Benchmark")
    print("="*50)
    
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"Error: Vocabulary file not found at {VOCAB_FILE}")
    print(f"Loading vocabulary from {VOCAB_FILE}...")
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)
    print(f"Vocabulary size: {VOCAB_SIZE}")

    pad_idx = vocab.stoi['<pad>']

    def collate_fn(batch_of_tensors):
        """
        Takes a list of tensors of varying lengths and pads them 
        to the max length in the batch.
        """
        return rnn_utils.pad_sequence(
            batch_of_tensors, 
            batch_first=True, 
            padding_value=pad_idx
        )

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Error: Checkpoint file not found at {CHECKPOINT_FILE}")

    print("Loading Baseline Model (from model.py)...")
    model_baseline = DecoderOnlyTransformerBaseline(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, max_seq_len=CONTEXT_LEN,
        d_attn=D_ATTN, dropout=DROPOUT
    ).to(device)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model_baseline.load_state_dict(checkpoint['model_state_dict'])
    model_baseline.eval()

    print("Loading KV Cache Model (from model_kv.py)...")
    model_kv = DecoderOnlyTransformerKV(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, max_seq_len=CONTEXT_LEN,
        d_attn=D_ATTN, dropout=DROPOUT
    ).to(device)
    model_kv.load_state_dict(checkpoint['model_state_dict'])
    model_kv.eval()
    print("Models loaded successfully.")

    print(f"Loading {BATCH_SIZE} validation samples...")
    valid_data = load_dataset("roneneldan/TinyStories", split='validation')
    valid_dataset = TinyStoriesDataset(valid_data, vocab)

    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=collate_fn 
    )
    
    full_sequences_tensor = next(iter(valid_loader))
    prompts_tensor = full_sequences_tensor[:, :PROMPT_LENGTH].to(device)
    
    print(f"Batch shape: {prompts_tensor.shape}")
    print(f"Generating {MAX_GENERATION_LENGTH - PROMPT_LENGTH} new tokens for each sample.")

    print("\nRunning GPU warmup...")
    _ = generate_baseline_batched(prompts_tensor, model_baseline, MAX_GENERATION_LENGTH, TEMPERATURE)
    _ = generate_with_kv_cache(prompts_tensor, model_kv, MAX_GENERATION_LENGTH, TEMPERATURE)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("Warmup complete.")

    print("\nBenchmarking: Baseline (No KV Cache)...")
    start_time_baseline = time.time()
    
    total_tokens_baseline = generate_baseline_batched(
        prompts_tensor, model_baseline, MAX_GENERATION_LENGTH, TEMPERATURE
    )
    
    if device.type == 'cuda':
        torch.cuda.synchronize() 
    end_time_baseline = time.time()
    
    time_baseline = end_time_baseline - start_time_baseline
    tokens_per_sec_baseline = total_tokens_baseline / time_baseline
    
    print(f"Finished in {time_baseline:.4f} seconds")
    print(f"Generated {total_tokens_baseline} total tokens")

    print("\nBenchmarking: With KV Cache...")
    start_time_kv = time.time()
    
    total_tokens_kv = generate_with_kv_cache(
        prompts_tensor, model_kv, MAX_GENERATION_LENGTH, TEMPERATURE
    )
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    end_time_kv = time.time()
    
    time_kv = end_time_kv - start_time_kv
    tokens_per_sec_kv = total_tokens_kv / time_kv
    
    print(f"Finished in {time_kv:.4f} seconds")
    print(f"Generated {total_tokens_kv} total tokens")

    print("\n" + "="*50)
    print("Benchmark Results")
    print("="*50)
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Prompt Length: {PROMPT_LENGTH}")
    print(f"Total Generation Length: {MAX_GENERATION_LENGTH}")
    print(f"New Tokens Generated per Sample: {MAX_GENERATION_LENGTH - PROMPT_LENGTH}")
    print("-"*50)
    print(f"Baseline (No Cache): {tokens_per_sec_baseline:.2f} tokens/sec")
    print(f"KV Cache:            {tokens_per_sec_kv:.2f} tokens/sec")
    print("-"*50)
    print(f"Speedup: {tokens_per_sec_kv / tokens_per_sec_baseline:.2f}x")
    print("="*50)

    with open(results_file, 'w') as f:
        f.write("KV Cache Runtime Efficiency Benchmark Results\n")
        f.write("=" * 50 + "\n")
        f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Device: {device}\n")
        f.write(f"Batch Size: {BATCH_SIZE}\n")
        f.write(f"Prompt Length: {PROMPT_LENGTH}\n")
        f.write(f"Total Generation Length: {MAX_GENERATION_LENGTH}\n")
        f.write(f"New Tokens Generated per Sample: {MAX_GENERATION_LENGTH - PROMPT_LENGTH}\n")
        f.write(f"Temperature: {TEMPERATURE}\n")
        f.write("-" * 50 + "\n")
        f.write("MODEL CONFIGURATION:\n")
        f.write(f"  d_model: {D_MODEL}\n")
        f.write(f"  n_layers: {N_LAYERS}\n")
        f.write(f"  n_heads: {N_HEADS}\n")
        f.write(f"  d_ff: {D_FF}\n")
        f.write(f"  d_attn: {D_ATTN}\n")
        f.write(f"  context_len: {CONTEXT_LEN}\n")
        f.write(f"  dropout: {DROPOUT}\n")
        f.write("-" * 50 + "\n")
        f.write("PERFORMANCE RESULTS:\n")
        f.write(f"Baseline (No KV Cache):\n")
        f.write(f"  Time: {time_baseline:.4f} seconds\n")
        f.write(f"  Total tokens generated: {total_tokens_baseline}\n")
        f.write(f"  Throughput: {tokens_per_sec_baseline:.2f} tokens/sec\n")
        f.write(f"KV Cache:\n")
        f.write(f"  Time: {time_kv:.4f} seconds\n")
        f.write(f"  Total tokens generated: {total_tokens_kv}\n")
        f.write(f"  Throughput: {tokens_per_sec_kv:.2f} tokens/sec\n")
        f.write("-" * 50 + "\n")
        f.write(f"SPEEDUP: {tokens_per_sec_kv / tokens_per_sec_baseline:.2f}x\n")
        f.write("=" * 50 + "\n")
    
    print(f"\nResults saved to: {results_file}")

if __name__ == '__main__':
    main()