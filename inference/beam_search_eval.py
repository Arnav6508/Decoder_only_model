import torch
from torch.utils.data import DataLoader
import os
import pickle
import evaluate
import time
import itertools
from tqdm import tqdm

import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from datasets import load_dataset
from models.model import DecoderOnlyTransformer
from utils import Vocabulary, TinyStoriesDataset
from beam_search_generator import generate_beam_search

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
PROMPT_LENGTH = 10 
MAX_GENERATION_LENGTH = 50 
TEMPERATURE = 0.8 

output_dir = "results/beam_search"
output_file_name = "beam_search_evaluation.txt"

NUM_PROMPTS_FOR_EVAL = 5
BEAM_WIDTHS_TO_TEST = [5, 10]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_sampling(prompt_tokens: list[int], model: DecoderOnlyTransformer, vocab: Vocabulary, max_len: int, temp: float) -> list[int]:
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

def main():
    print(f"Using device: {device}")
    if not os.path.exists(VOCAB_FILE):
        raise FileNotFoundError(f"Error: Vocabulary file not found at {VOCAB_FILE}")
    with open(VOCAB_FILE, 'rb') as f:
        vocab = pickle.load(f)
    VOCAB_SIZE = len(vocab)

    model = DecoderOnlyTransformer(
        vocab_size=VOCAB_SIZE, d_model=D_MODEL, n_layers=N_LAYERS,
        n_heads=N_HEADS, d_ff=D_FF, max_seq_len=CONTEXT_LEN,
        d_attn=D_ATTN, dropout=DROPOUT
    ).to(device)

    if not os.path.exists(CHECKPOINT_FILE):
        raise FileNotFoundError(f"Error: Checkpoint file not found at {CHECKPOINT_FILE}")
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model state loaded successfully.")

    valid_data = load_dataset("roneneldan/TinyStories", split='validation')
    valid_dataset = TinyStoriesDataset(valid_data, vocab)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True)
    bleu = evaluate.load("bleu")

    results = []
    print("\n" + "="*80)
    print(f"Running evaluation on {NUM_PROMPTS_FOR_EVAL} prompts...")
    print("="*80)
    
    prompts_iterator = itertools.islice(valid_loader, NUM_PROMPTS_FOR_EVAL)
    
    for _, full_sequence_tensor in enumerate(tqdm(prompts_iterator, total=NUM_PROMPTS_FOR_EVAL, desc="Evaluating Prompts")):
        full_sequence_ids = full_sequence_tensor[0].tolist()
        
        prompt_tokens = full_sequence_ids[:PROMPT_LENGTH]
        reference_tokens = full_sequence_ids[PROMPT_LENGTH : PROMPT_LENGTH + MAX_GENERATION_LENGTH]
        
        prompt_text = vocab.indices_to_text(prompt_tokens)
        reference_text = vocab.indices_to_text(reference_tokens)

        current_prompt_results = {'prompt': prompt_text, 'reference': reference_text, 'methods': {}}

        # Method 1: Multinomial Sampling (Baseline) 
        start_time = time.time()
        sampling_tokens = generate_sampling(prompt_tokens, model, vocab, MAX_GENERATION_LENGTH, TEMPERATURE)
        duration = time.time() - start_time
        
        tokens_per_sec = len(sampling_tokens) / duration if duration > 0 else float('inf')
        sampling_text = vocab.indices_to_text(sampling_tokens)
        bleu_score = bleu.compute(predictions=[sampling_text], references=[[reference_text]])['bleu']
        
        current_prompt_results['methods']['Sampling (temp=0.8)'] = {'text': sampling_text, 'bleu': bleu_score, 'tokens_per_sec': tokens_per_sec}

        # Method 2 & 3: Beam Search
        for k in BEAM_WIDTHS_TO_TEST:
            start_time = time.time()
            beam_tokens = generate_beam_search(prompt_tokens, model, vocab, MAX_GENERATION_LENGTH, k, CONTEXT_LEN)
            duration = time.time() - start_time

            tokens_per_sec = len(beam_tokens) / duration if duration > 0 else float('inf')
            beam_text = vocab.indices_to_text(beam_tokens)
            bleu_score = bleu.compute(predictions=[beam_text], references=[[reference_text]])['bleu']
            
            method_name = f'Beam Search (k={k})'
            current_prompt_results['methods'][method_name] = {'text': beam_text, 'bleu': bleu_score, 'tokens_per_sec': tokens_per_sec}
        
        results.append(current_prompt_results)

    # Summary Results
    os.makedirs(output_dir, exist_ok=True) 
    results_file = os.path.join(output_dir, output_file_name)
        
    save_results(results, BEAM_WIDTHS_TO_TEST, results_file)
        
    print(f"\n Detailed evaluation results have been saved to {results_file}")
    

def save_results(results, beam_widths, output_filename: str):
    avg_scores = { 'Sampling (temp=0.8)': {'bleu': 0.0, 'tokens_per_sec': 0.0} }
    for k in beam_widths:
        avg_scores[f'Beam Search (k={k})'] = {'bleu': 0.0, 'tokens_per_sec': 0.0}

    for i, res in enumerate(results):
        for method_name, metrics in res['methods'].items():
            avg_scores[method_name]['bleu'] += metrics['bleu']
            avg_scores[method_name]['tokens_per_sec'] += metrics['tokens_per_sec']

    with open(output_filename, 'w', encoding='utf-8') as f:

        header = "\n" + "="*80 + "\nAVERAGE PERFORMANCE METRICS\n" + "="*80
        f.write(header + "\n")
        print(header)

        for method_name, scores in avg_scores.items():
            avg_bleu = scores['bleu'] / len(results)
            avg_tps = scores['tokens_per_sec'] / len(results)

            method_summary_header = f"Method: {method_name}"
            bleu_summary = f"  - Average BLEU Score:      {avg_bleu:.4f}"
            tps_summary = f"  - Average Tokens/Second:   {avg_tps:.2f}\n"

            print(method_summary_header)
            print(bleu_summary)
            print(tps_summary)

            f.write(f"{method_summary_header}\n")
            f.write(f"{bleu_summary}\n")
            f.write(f"{tps_summary}\n")


if __name__ == '__main__':
    main()