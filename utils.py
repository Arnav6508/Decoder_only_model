import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math
from tqdm import tqdm
import re
import zipfile 
import requests
import os


class Vocabulary:
    def __init__(self):
        self.itos = {0: "<sos>", 1: "<eos>", 2: "<pad>", 3: "<unk>"}
        self.stoi = {"<sos>": 0, "<eos>": 1, "<pad>": 2, "<unk>": 3}

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_l):
        token_set = set()
        idx = 4 
        
        for sentence in tqdm(sentence_l, desc="Creating vocab"):
            tokens_l = self.tokenize(sentence)
            for token in tokens_l:
                token_set.add(token)
        
        for token in token_set:
            self.stoi[token] = idx
            self.itos[idx] = token
            idx += 1
    
    def tokenize(self, text):
        return [word.lower() for word in re.findall(r'\w+', text)]

    def vectorize(self, text):
        token_l = self.tokenize(text)
        vector = []
        for token in token_l:
            idx = self.stoi.get(token, self.stoi["<unk>"])
            vector.append(idx)
        return vector
    
    def indices_to_text(self, indices):
        return " ".join([self.itos.get(idx, "<unk>") for idx in indices])
    
class TinyStoriesDataset(Dataset):
    def __init__(self, data, vocab):
        self.data = data
        self.vocab = vocab

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data[index]['text']
        vector = self.vocab.vectorize(text)
        
        sos_tensor = torch.tensor([self.vocab.stoi["<sos>"]], dtype=torch.long)
        eos_tensor = torch.tensor([self.vocab.stoi["<eos>"]], dtype=torch.long)
        main_tensor = torch.tensor(vector, dtype=torch.long)
        
        return torch.cat((sos_tensor, main_tensor, eos_tensor), dim=0)

class Collate:
    def __init__(self, pad_idx, context_len):
        self.pad_idx = pad_idx
        self.context_len = context_len

    def __call__(self, batch):
        truncated_batch = [seq[:self.context_len] for seq in batch]
        padded_batch = torch.full((len(truncated_batch), self.context_len), self.pad_idx, dtype=torch.long)

        for i, seq in enumerate(batch):
            seq_len = len(seq)
            padded_batch[i, :seq_len] = seq[:self.context_len]

        # Teacher forcing (last token acts as prediction tokens)
        inputs = padded_batch[:, :-1]
        targets = padded_batch[:, 1:]
        
        return inputs, targets
    
def load_fasttext_vectors(file_name, url, zip_file_name, cache_dir='.'):
    """
    Download + unzip + parse FastText .vec file.
    Returns a dictionary mapping tokens to their embedding vectors.
    """
    vec_path = os.path.join(cache_dir, file_name)
    zip_path = os.path.join(cache_dir, zip_file_name)

    if not os.path.exists(vec_path):
        print(f"{file_name} not found. Checking for zip file...")
        if not os.path.exists(zip_path):
            print(f"Downloading from {url}...")
            response = requests.get(url, stream=True)
            total_size_in_bytes = int(response.headers.get('content-length', 0))
            block_size = 1024 # in bytes
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(zip_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong during download")

        # Unzip
        print(f"Unzipping {zip_file_name}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(cache_dir)
        print("Unzip complete.")

    # Parse the .vec file
    print(f"Parsing {file_name}...")
    word_vectors = {}
    with open(vec_path, 'r', encoding='utf-8') as f:
        # The first line is header info, skip it
        next(f) 
        for line in tqdm(f, desc="Loading vectors"):
            parts = line.strip().split(' ')
            word = parts[0]
            vector_values = [float(val) for val in parts[1:]]
            word_vectors[word] = torch.tensor(vector_values)
    
    print("Vector parsing complete.")
    return word_vectors

def get_embedding_matrix(vocab, d_model):
    print("Loading FastText vectors...")

    fasttext_vectors = load_fasttext_vectors(
        file_name='wiki.simple.vec', 
        url='https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.zip',
        zip_file_name='wiki.simple.zip'
    )
  
    embedding_matrix = torch.randn(len(vocab), d_model)
    embedding_matrix[vocab.stoi["<pad>"]] = torch.zeros(d_model) #embedding of padding token = 0
    
    cnt = 0
    for token, idx in tqdm(vocab.stoi.items(), desc="Creating embedding matrix"):
        if token in fasttext_vectors:
            embedding_matrix[idx] = fasttext_vectors[token]
            cnt += 1
            
    print(f"Found {cnt} out of {len(vocab)} wodrs in FastText vocab.")
    return embedding_matrix

def train_epoch(model, dataloader, optimizer, loss_fn, device = None):
    model.train()
    tot_loss = 0
    num_batches = len(dataloader)
    
    for (inputs, targets) in tqdm(dataloader, desc="Traning"):   
        inputs = inputs.to(device) 
        targets = targets.to(device)
        
        logits = model(inputs) 
        
        loss = loss_fn(logits, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        tot_loss += loss.item()
        
    return tot_loss/num_batches

def evaluate(model, dataloader, loss_fn, device = None):
    model.eval()
    tot_loss = 0
    num_batches = len(dataloader)
    
    with torch.no_grad():
        for (inputs, targets) in tqdm(dataloader, desc="Evaluating"):
            inputs = inputs.to(device) 
            targets = targets.to(device)
        
            logits = model(inputs)
            
            loss = loss_fn(logits, targets)

            tot_loss += loss.item()
            
    avg_loss = tot_loss/num_batches
    perplexity = math.exp(avg_loss)
    return avg_loss, perplexity

class CrossEntropyLoss(nn.Module):
    def __init__(self, pad_idx):
        super().__init__()
        self.pad_idx = pad_idx

    def forward(self, logits, targets):
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)  
        targets = targets.reshape(-1)

        log_probs = nn.functional.log_softmax(logits, dim=1)

        nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1))
        nll_loss.squeeze_(1)

        non_pad_mask = (targets != self.pad_idx)
        nll_loss_masked = nll_loss.where(non_pad_mask, torch.tensor(0.0, device=logits.device))
        
        total_loss = nll_loss_masked.sum()
        non_pad_tokens = non_pad_mask.sum()
        
        if non_pad_tokens == 0: return total_loss  
        else: return total_loss/non_pad_tokens