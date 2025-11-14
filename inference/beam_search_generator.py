import torch
import torch.nn.functional as F
from typing import List

class Beam:
    """Helper class to store beam search state."""
    def __init__(self, tokens: torch.Tensor, log_prob: float, device: torch.device):
        self.tokens = tokens      
        self.log_prob = log_prob  
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

    model.eval()
    device = next(model.parameters()).device
    
    initial_tensor = torch.tensor(prompt_tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    beams = [Beam(initial_tensor, 0.0, device)]
    completed_beams = []

    with torch.no_grad():
        for _ in range(max_len):
            potential_next_beams = []
            
            for beam in beams:
                input_for_model = beam.tokens[:, -context_len:]
                output = model(input_for_model)
                last_token_logits = output[0, -1, :]
                
                next_token_log_probs = F.log_softmax(last_token_logits, dim=-1)
                
                top_log_probs, top_token_ids = torch.topk(next_token_log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_token_ids[i].item()
                    log_prob = top_log_probs[i].item()
                    
                    new_beam = beam.extend(token_id, log_prob)
                    potential_next_beams.append(new_beam)

            sorted_beams = sorted(potential_next_beams, key=lambda x: x.log_prob, reverse=True)
            beams = sorted_beams[:beam_width]
            
            remaining_beams = []
            for beam in beams:
                if beam.tokens[0, -1].item() == vocab.stoi['<eos>']:
                    completed_beams.append(beam)
                else:
                    remaining_beams.append(beam)
            
            beams = remaining_beams
            
            if not beams:
                break
    
    if not completed_beams:
        completed_beams = beams
        
    best_beam = sorted(completed_beams, key=lambda x: x.log_prob, reverse=True)[0]
    
    generated_tokens = best_beam.tokens.squeeze(0).tolist()[len(prompt_tokens):]
    
    return generated_tokens