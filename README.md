# text-generation-rnn-lstm-transformer

Implementation of three autoregressive language models trained on a mystery novel corpus from Project Gutenberg, including Vanilla RNN, LSTM, and a full Transformer LLM.

## Models
- **Vanilla RNN** — simple recurrent network for sequential text modeling
- **LSTM** — gated architecture to capture longer-range dependencies  
- **Transformer** — full attention-based language model

## Project Structure
```
src/
├── data/              # Tokenization and dataset pipeline
├── models/            # Model implementations
│   ├── RNNs.py        # Vanilla RNN and LSTM
│   └── transformer.py # Transformer LLM
└── training/          # Training loop and text generation
    ├── train.py
    └── language_model.py
data/
└── process_data.py    # Downloads and preprocesses corpus
main.py                # Entry point for training and generation
```

## Requirements
```bash
pip install tensorflow numpy wandb
```

## Setup
Download and preprocess the mystery novel corpus:
```bash
cd data
python process_data.py
```

## Training
```bash
# Transformer
python main.py --model-type transformer --epochs 4 --vocab-size 10000 --d-model 512

# LSTM
python main.py --model-type lstm --epochs 3 --vocab-size 5000

# Vanilla RNN
python main.py --model-type vanilla_rnn --epochs 2 --vocab-size 3000

# Continue from checkpoint
python main.py --model-type transformer --continue-training

# Disable wandb logging
python main.py --model-type transformer --no-wandb
```

## Generate Text
```bash
# Generate samples
python main.py --generate-only --model-type transformer

# Interactive mode
python main.py --generate-only --model-type transformer --interactive
```

## Results
| Model | Vocab Size | Epochs | Perplexity |
|-------|-----------|--------|-----------|
| Vanilla RNN | 3,000 | 2 | ~50-200 |
| LSTM | 5,000 | 3 | ~50-200 |
| Transformer | 10,000 | 4 | ~10-115 |
