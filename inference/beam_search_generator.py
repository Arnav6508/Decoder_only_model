import torch
import torch.nn.functional as F
from typing import List

# A helper class to manage the state of each beam
class Beam:
    """Helper class to store beam search state."""
    def __init__(self, tokens: torch.Tensor, log_prob: float, device: torch.device):
        self.tokens = tokens      # Tensor of token IDs for the sequence
        self.log_prob = log_prob  # Cumulative log probability of the sequence
        self.device = device

    def extend(self, token_id: int, new_log_prob: float) -> 'Beam':
        """Creates a new beam by extending the current one with a new token."""
        new_tokens = torch.cat([self.tokens, torch.tensor([[token_id]], device=self.device)], dim=1)
        return Beam(new_tokens, self.log_prob + new_log_prob, self.device)

def generate_beam_search(
    prompt_tokens: List[int], 
    model, 
    vocab, 
    max_len: int, 
    beam_width: int, 
    context_len: int
) -> List[int]:
    """
    Generates a sequence of tokens using beam search decoding.

    Args:
        prompt_tokens: A list of integers representing the initial prompt.
        model: The trained transformer model.
        vocab: The vocabulary object.
        max_len: The maximum number of tokens to generate.
        beam_width: The number of beams (k) to keep at each step.
        context_len: The maximum context length the model can handle.

    Returns:
        A list of generated token integers (excluding the prompt).
    """
    model.eval()
    device = next(model.parameters()).device
    
    initial_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Start with one beam: the prompt itself with a log probability of 0
    beams = [Beam(initial_tensor, 0.0, device)]
    completed_beams = []

    with torch.no_grad():
        for _ in range(max_len):
            potential_next_beams = []
            
            # Go through each current beam and find potential next steps
            for beam in beams:
                # Get model output for the current beam's sequence
                input_for_model = beam.tokens[:, -context_len:]
                output = model(input_for_model)
                last_token_logits = output[0, -1, :]
                
                # Use log_softmax for numerical stability and easier calculations
                next_token_log_probs = F.log_softmax(last_token_logits, dim=-1)
                
                # Get the top `k` candidates for this specific beam
                top_log_probs, top_token_ids = torch.topk(next_token_log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_token_ids[i].item()
                    log_prob = top_log_probs[i].item()
                    
                    # Create a new potential beam by extending the current one
                    new_beam = beam.extend(token_id, log_prob)
                    potential_next_beams.append(new_beam)

            # Prune the candidates: sort all potential new beams and keep only the top `k`
            sorted_beams = sorted(potential_next_beams, key=lambda x: x.log_prob, reverse=True)
            beams = sorted_beams[:beam_width]
            
            # Identify and move completed beams (those ending in <eos>)
            remaining_beams = []
            for beam in beams:
                if beam.tokens[0, -1].item() == vocab.stoi['<eos>']:
                    completed_beams.append(beam)
                else:
                    remaining_beams.append(beam)
            
            beams = remaining_beams
            
            # If all active beams have finished, we can stop early
            if not beams:
                break
    
    # If no beam produced <eos>, use the current best ones
    if not completed_beams:
        completed_beams = beams
        
    # Choose the best sequence from all completed beams (highest log probability)
    best_beam = sorted(completed_beams, key=lambda x: x.log_prob, reverse=True)[0]
    
    # Return only the newly generated tokens
    generated_tokens = best_beam.tokens.squeeze(0).tolist()[len(prompt_tokens):]
    
    return generated_tokens