import tensorflow as tf
import math
from typing import Optional, Tuple, Dict, Any
from src.models.transformer import TransformerLanguageModel, create_language_model



class TextSampler:
    """
    Advanced sampling methods for text generation.
    """

    @staticmethod
    def sample_top_k(logits: tf.Tensor, k: int, temperature: float = 1.0) -> tf.Tensor:
        """
        Sample from top-k most likely tokens.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            k: Number of top tokens to consider
            temperature: Sampling temperature (higher = more random)

        Returns:
            Sampled token indices [batch_size, 1]
        """
        scaled = logits / temperature
        vocab_size = tf.shape(scaled)[-1]
        k = tf.minimum(k, vocab_size)
        values, _ = tf.math.top_k(scaled, k=k)
        kth = values[:, -1:]
        mask = scaled >= kth
        neg_inf = tf.cast(-1e9, scaled.dtype)
        masked = tf.where(mask, scaled, neg_inf)
        return tf.random.categorical(masked, 1)

    @staticmethod
    def sample_top_p(logits: tf.Tensor, p: float, temperature: float = 1.0) -> tf.Tensor:
        """
        Sample from nucleus (top-p) of probability distribution.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            p: Cumulative probability threshold (0.0 to 1.0)
            temperature: Sampling temperature

        Returns:
            Sampled token indices [batch_size, 1]
        """
        scaled = logits / temperature
        sorted_idx = tf.argsort(scaled, direction="DESCENDING", axis=-1)
        sorted_logits = tf.gather(scaled, sorted_idx, batch_dims=1)
        probs = tf.nn.softmax(sorted_logits, axis=-1)
        cumprobs = tf.cumsum(probs, axis=-1)
        mask = cumprobs <= p
        mask = tf.concat([tf.ones_like(mask[:, :1], dtype=tf.bool), mask[:, 1:]], axis=-1)
        neg_inf = tf.cast(-1e9, sorted_logits.dtype)
        masked_sorted = tf.where(mask, sorted_logits, neg_inf)
        sampled_in_sorted = tf.random.categorical(masked_sorted, 1)
        return tf.gather(sorted_idx, sampled_in_sorted, batch_dims=1)

    @staticmethod
    def sample_categorical(logits: tf.Tensor, temperature: float = 1.0) -> tf.Tensor:
        """
        Basic categorical sampling (baseline implementation).

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            temperature: Sampling temperature

        Returns:
            Sampled token indices [batch_size, 1]
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive.")
        scaled_logits = logits / temperature
        return tf.random.categorical(scaled_logits, num_samples=1)

    @classmethod
    def sample(cls, logits: tf.Tensor, method: str = "temperature", **kwargs) -> tf.Tensor:
        """
        Unified sampling interface.

        Args:
            logits: Raw model outputs [batch_size, vocab_size]
            method: Sampling method ("temperature", "top_k", "top_p")
            **kwargs: Method-specific parameters

        Returns:
            Sampled token indices [batch_size, 1]
        """
        if method == "temperature":
            return cls.sample_categorical(logits, temperature=kwargs.get("temperature", 1.0))
        elif method == "top_k":
            return cls.sample_top_k(
                logits,
                k=kwargs.get("k", 50),
                temperature=kwargs.get("temperature", 1.0)
            )
        elif method == "top_p":
            return cls.sample_top_p(
                logits,
                p=kwargs.get("p", 0.9),
                temperature=kwargs.get("temperature", 1.0)
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")


class TextGenerator:
    """
    High-level text generation interface using TransformerLanguageModel.
    """

    def __init__(self, model: TransformerLanguageModel, tokenizer=None):
        """
        Initialize text generator.

        Args:
            model: Trained TransformerLanguageModel
            tokenizer: Tokenizer for encoding/decoding text
        """
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = TextSampler()

    def generate(self,
                 prompt: str = "",
                 max_length: int = 100,
                 method: str = "top_k",
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.9,
                 stop_tokens: Optional[list] = None) -> str:
        """
        Generate text continuation from a prompt.

        Args:
            prompt: Starting text (empty string for unconditioned generation)
            max_length: Maximum tokens to generate
            method: Sampling method ("temperature", "top_k", "top_p")
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            stop_tokens: List of token IDs to stop generation

        Returns:
            Generated text string
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")
        
        # Encode prompt
        if prompt:
            prompt_tokens = self.tokenizer.encode(prompt)
        else:
            prompt_tokens = [1]  # Start token

        # We initalize the generated sequence with the prompt tokens and generate step by step
        generated = tf.constant([prompt_tokens], dtype=tf.int64)

        # Get max sequence length from model
        if hasattr(self.model, 'max_seq_length'):
            model_max_seq_length = self.model.max_seq_length
        else:
            # Otherwise, we can just infer from input shape
            model_max_seq_length = self.model.input_shape[1]

        # Generate tokens
        for _ in range(max_length):
            # Get current sequence
            current_seq = generated
            seq_len = tf.shape(current_seq)[1]
            
            if seq_len > model_max_seq_length:
                # Truncate to max length
                current_seq = current_seq[:, -model_max_seq_length:]
            elif seq_len < model_max_seq_length:
                # Pad to max length
                padding_length = model_max_seq_length - seq_len
                padding = tf.zeros([1, padding_length], dtype=tf.int64)
                # Pad on the left to preserve recent context
                current_seq = tf.concat([padding, current_seq], axis=-1)

            # Forward pass
            logits = self.model(current_seq, training=False)
            
            # Get logits for the last non padded position
            # For padded sequences, we need to get logits at position (original_seq_len - 1)
            if seq_len < model_max_seq_length:
                # Account for padding so we get logits at the last actual token position
                next_token_logits = logits[:, seq_len - 1, :]
            else:
                # No padding, use last position
                next_token_logits = logits[:, -1, :]

            # Sample next token using specified method
            sampling_kwargs = {
                "temperature": temperature,
                "k": top_k,
                "p": top_p
            }

            next_token = self.sampler.sample(next_token_logits, method=method, **sampling_kwargs)

            # Check for stop tokens
            if stop_tokens and int(next_token[0, 0]) in stop_tokens:
                break

            # Append to sequence
            generated = tf.concat([generated, next_token], axis=-1)

        # Decode to text
        generated_tokens = generated.numpy()[0].tolist()
        return self.tokenizer.decode(generated_tokens)