
File structure:

```
.
├── inference/
│   ├── beam_search_eval.py      # Runs inference using beam search (k=5, k=10, etc.)
│   ├── beam_search_generator.py # Core logic for the beam search algorithm
│   ├── inference.py             # Runs standard inference on the base model
│   └── kv_cache_eval.py         # Runs inference and evaluation for the KV-cached model
│
├── model_checkpoints/
│   ├── checkpoint_epoch_1.pth   # Model weights after 1 epoch
│   ├── checkpoint_epoch_2.pth   # Model weights after 2 epochs
│   └── checkpoint_epoch_3.pth   # Model weights after 3 epochs (used for final evaluation)
│
├── models/
│   ├── model.py                 # Contains the base model architecture
│   └── model_kv.py              # Model architecture optimized with KV Caching
│
├── results/
│   ├── beam_search/             # Stores BLEU/tokens-per-sec comparison for sampling vs. beam search
│   ├── inference/               # Stores attention visualizations and evaluation metrics (Perplexity, BLEU)
│   ├── kv_caching/              # Stores throughput increase and time reduction metrics
│   └── training/                # Stores training/validation loss and perplexity plots
│
├── training/
│   ├── plot_training_metrics.py # Generates plots (loss, perplexity) from training logs
│   └── train.py                 # Main script to train the model
│
├── .vector_cache/               # (Likely a cache for embeddings or similar)
├── __pycache__/                 # Python cache files
├── embedding_matrix.pt          # Pre-computed embedding matrix
├── requirements.txt             # Project dependencies
├── utils.py                     # Utility functions (e.g., data loaders, tokenizers)
└── vocab.pkl                    # Pickled vocabulary file
```

<br>
<br>

⚙️ How to Run the Code

<br>

Download the model weights from this google drive link:
https://drive.google.com/file/d/1LyU08_QZC-IiHygID3s2pOyPNjRNKa7w/view?usp=sharing

<br>

[1] Training the Model
To train the model from scratch, run the train.py script:

```bash
python training/train.py
```
This script will train the model and save the checkpoints in the model_checkpoints/ directory for each epoch.

<br>
<br>

[2] Plotting Training Metrics
After training, you can visualize the training/validation loss and perplexity curves.

```bash
python training/plot_training_metrics.py
```

<br>
<br>

[3] Running Inference and Evaluation
For all evaluation scripts, we'll use the 3-epoch checkpoint as you specified. (Note: The exact command-line arguments might differ slightly based on your code, but this is the general idea).

A. Standard Inference
To run inference with the base model and generate attention visualizations and evaluation scores (Perplexity, BLEU):

```bash
python inference/inference.py 
```
**Output**: Results will be saved in results/inference/.
<br>

B. KV Cache Evaluation
To evaluate the inference speed (throughput, time reduction) gained from using KV Caching:

```bash
python inference/kv_cache_eval.py 
```
**Output**: Performance metrics will be saved in results/kv_caching/.
<br>

C. Beam Search Evaluation
To run inference using beam search and compare its BLEU score and speed against standard sampling:

```bash
python inference/beam_search_eval.py 
```
**Output**: Comparison results will be saved in results/beam_search/.
<br>


📊 Results
All generated outputs (plots, metrics, and visualizations) are saved in the results/ directory, sorted into subfolders corresponding to the experiment that produced them.