"""
data.py - Data loading for Transformer Language Model
Simple loader for preprocessed mystery corpus pickle file
"""

import tensorflow as tf
import numpy as np
import pickle
from typing import List, Tuple, Dict
import re

class TextTokenizer:
    """
    Tokenizer wrapper for preprocessed vocabulary.
    """

    def __init__(self, vocab: List[str], word_to_idx: Dict[str, int]):
        """
        Initialize tokenizer with preprocessed vocabulary.

        Args:
            vocab: List of vocabulary tokens (i.e., words)
            word_to_idx: Dictionary mapping tokens to indices
        """
        self.vocab = vocab
        self.token_to_idx = word_to_idx
        self.idx_to_token = {idx: token for token, idx in word_to_idx.items()}

        self.pad_token = '<PAD>'
        self.unk_token = '<UNK>'
        self.eos_token = '<EOS>'

    def decode(self, indices: List[int]) -> str:
        """
        Convert token indices back to text.

        Args:
            indices: List of token indices

        Returns:
            Decoded text string
        """
        tokens = []
        for idx in indices:
            token = self.idx_to_token.get(idx, self.unk_token)
            if token == self.eos_token:
                break
            if token == self.pad_token:
                continue
            tokens.append(token)
        
        # Join tokens and fix punctuation spacing
        text = ' '.join(tokens)
        text = text.replace(' ,', ',').replace(' .', '.').replace(' !', '!').replace(' ?', '?')
        text = text.replace(' :', ':').replace(' ;', ';').replace(' "', '"')

        return text

    def encode(self, text: str) -> List[int]:
        """
        Convert text to token indices (basic implementation).

        Args:
            text: Input text to encode

        Returns:
            List of token indices
        """
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text.lower())
        unk_id = self.get_unk_token_id()
        indices = []
        for tok in tokens:
            indices.append(self.token_to_idx.get(tok, unk_id))
    
        return indices

    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab)

    def get_pad_token_id(self) -> int:
        """Get padding token ID."""
        return self.token_to_idx.get(self.pad_token, 0)

    def get_unk_token_id(self) -> int:
        """Get unknown token ID."""
        return self.token_to_idx.get(self.unk_token, 1)

    def get_eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return self.token_to_idx.get(self.eos_token, 2)

    def __len__(self) -> int:
        return len(self.vocab)

def limit_vocabulary(train_tokens: List[int], test_tokens: List[int],
                    tokenizer: TextTokenizer, vocab_size: int) -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Limit vocabulary size by keeping only the most frequent tokens.
    Remap all tokens to the reduced vocabulary.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        tokenizer: Original tokenizer
        vocab_size: Target vocabulary size

    Returns:
        Tuple of (remapped_train_tokens, remapped_test_tokens, new_tokenizer)
    """
    # Keep the most frequent vocab_size tokens (they should already be sorted by frequency)
    # Ensure we keep special tokens
    special_tokens = ['<PAD>', '<UNK>', '<EOS>']

    # Keep top vocab_size tokens
    new_vocab = tokenizer.vocab[:vocab_size]

    # Make sure special tokens are included
    for special_token in special_tokens:
        if special_token not in new_vocab and special_token in tokenizer.vocab:
            # Replace least frequent non-special token
            for i in range(len(new_vocab)-1, -1, -1):
                if new_vocab[i] not in special_tokens:
                    new_vocab[i] = special_token
                    break

    # Create new word_to_idx mapping
    new_word_to_idx = {token: idx for idx, token in enumerate(new_vocab)}
    unk_idx = new_word_to_idx.get('<UNK>', 1)  # Default to index 1 if UNK not found

    # Remap tokens any token not in new vocab becomes UNK
    def remap_tokens(tokens):
        remapped = []
        for token_idx in tokens:
            original_token = tokenizer.idx_to_token.get(token_idx, '<UNK>')
            if original_token in new_word_to_idx:
                remapped.append(new_word_to_idx[original_token])
            else:
                remapped.append(unk_idx)
        return remapped

    new_train_tokens = remap_tokens(train_tokens)
    new_test_tokens = remap_tokens(test_tokens)

    # Create new tokenizer
    new_tokenizer = TextTokenizer(new_vocab, new_word_to_idx)

    print(f"  Vocabulary reduced from {len(tokenizer.vocab)} to {len(new_vocab)} tokens")
    print(f"  Special tokens: {[t for t in special_tokens if t in new_vocab]}")

    return new_train_tokens, new_test_tokens, new_tokenizer

def load_mystery_data(pickle_path: str = 'mystery_data.pkl') -> Tuple[List[int], List[int], TextTokenizer]:
    """
    Load preprocessed mystery corpus data from pickle file.

    Args:
        pickle_path: Path to the mystery_data.pkl file

    Returns:
        Tuple of (train_tokens, test_tokens, tokenizer)
    """
    print(f"Loading mystery data from {pickle_path}...")

    with open(pickle_path, 'rb') as f:
        data_dict = pickle.load(f)

    train_data = data_dict['train_data']
    test_data = data_dict['test_data']
    vocab = data_dict['vocab']
    word_to_idx = data_dict['word_to_idx']

    # Create tokenizer
    tokenizer = TextTokenizer(vocab, word_to_idx)

    return train_data, test_data, tokenizer

def create_sequences(tokens: List[int], seq_length: int) -> np.ndarray:
    """
    Create non-overlapping input sequences for language modeling.
    Targets will be computed during training by shifting the inputs.

    Args:
        tokens: List of token indices
        seq_length: Sequence length for training

    Returns:
        Input sequences array
    """
    sequences = []
    total_len = len(tokens)
    step = seq_length + 1
    for i in range(0, total_len - step + 1, step):
        seq = tokens[i : i + step]
        if len(seq) == step:
            sequences.append(seq)
    
    return np.array(sequences, dtype=np.int32)

def create_tf_datasets(train_tokens: List[int], test_tokens: List[int],
                      seq_length: int = 256, batch_size: int = 16) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets from token sequences.

    Args:
        train_tokens: Training token sequence
        test_tokens: Test token sequence
        seq_length: Sequence length for training
        batch_size: Batch size

    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    train_seqs = create_sequences(train_tokens, seq_length)
    test_seqs = create_sequences(test_tokens, seq_length)

    train_tensor = tf.constant(train_seqs, dtype=tf.int32)
    test_tensor = tf.constant(test_seqs, dtype=tf.int32)

    train_ds = (tf.data.Dataset.from_tensor_slices(train_tensor)
                .shuffle(train_tensor.shape[0])
                .batch(batch_size, drop_remainder=True)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (tf.data.Dataset.from_tensor_slices(test_tensor)
               .batch(batch_size, drop_remainder=True)
               .prefetch(tf.data.AUTOTUNE))

    return train_ds, test_ds

def prepare_data(pickle_path: str = 'mystery_data.pkl', seq_length: int = 256,
                batch_size: int = 16, vocab_size: int = None) -> Tuple[tf.data.Dataset, tf.data.Dataset, TextTokenizer]:
    """
    Complete data preparation pipeline.

    Args:
        pickle_path: Path to mystery_data.pkl file
        seq_length: Sequence length for training
        batch_size: Batch size for datasets
        vocab_size: Maximum vocabulary size (None = use full vocab, smaller values = faster training)

    Returns:
        Tuple of (train_dataset, test_dataset, tokenizer)
    """
    print("=" * 60)
    print("PREPARING CORPUS DATA")
    print("=" * 60)

    # Load preprocessed data
    train_tokens, test_tokens, tokenizer = load_mystery_data(pickle_path)

    # Optionally limit vocabulary size for faster training
    if vocab_size is not None and vocab_size < len(tokenizer.vocab):
        print(f"Limiting vocabulary from {len(tokenizer.vocab)} to {vocab_size} tokens...")
        train_tokens, test_tokens, tokenizer = limit_vocabulary(
            train_tokens, test_tokens, tokenizer, vocab_size
        )

    # Create TensorFlow datasets
    train_dataset, test_dataset = create_tf_datasets(
        train_tokens, test_tokens, seq_length, batch_size
    )

    print("Data preparation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

ation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

t, test_dataset, tokenizer

ation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

("=" * 60)

    return train_dataset, test_dataset, tokenizer

ataset, test_dataset, tokenizer

ation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

t, test_dataset, tokenizer

ation complete!")
    print("=" * 60)

    return train_dataset, test_dataset, tokenizer

("=" * 60)

    return train_dataset, test_dataset, tokenizer

ataset, test_dataset, tokenizer

("=" * 60)

    return train_dataset, test_dataset, tokenizer

=" * 60)

    return train_dataset, test_dataset, tokenizer

ataset, tokenizer

("=" * 60)

    return train_dataset, test_dataset, tokenizer

ataset, test_dataset, tokenizer

("=" * 60)

    return train_dataset, test_dataset, tokenizer

=" * 60)

    return train_dataset, test_dataset, tokenizer

r

("=" * 60)

    return train_dataset, test_dataset, tokenizer

=" * 60)

    return train_dataset, test_dataset, tokenizer

taset, tokenizer

