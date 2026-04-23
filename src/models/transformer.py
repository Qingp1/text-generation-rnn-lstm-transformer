import tensorflow as tf
import keras

@keras.saving.register_keras_serializable(package="transformer")
class AttentionMatrix(keras.layers.Layer):
    """Compute attention matrix"""

    def __init__(self, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.use_causal_mask = use_causal_mask

    def call(self, inputs):
        """
        Compute attention weights from K and Q matrices.

        Args:
            inputs: [K, Q] where K and Q are [batch_size, seq_length, embed_size]

        Returns:
            attention_weights: [batch_size, seq_length, seq_length]
        """
        K, Q = inputs

        # 1. Ensure consistent dtypes (cast to tf.float32)
        K = tf.cast(K, tf.float32)
        Q = tf.cast(Q, tf.float32)
        head_size = tf.cast(tf.shape(K)[-1], tf.float32)

        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(head_size)

        if self.use_causal_mask:
            seq_len_q = tf.shape(scores)[-2]
            seq_len_k = tf.shape(scores)[-1]
            mask = tf.linalg.band_part(tf.ones((seq_len_q, seq_len_k), dtype=tf.bool), -1, 0)
            mask = tf.reshape(mask, (1, seq_len_q, seq_len_k))
            scores = tf.where(mask, scores, tf.cast(-1e9, scores.dtype))

        attention_weights = tf.nn.softmax(scores, axis=-1)

        return attention_weights

    def get_config(self):
        config = super().get_config()
        return config

@keras.saving.register_keras_serializable(package="transformer")
class AttentionHead(keras.layers.Layer):
    """Single attention head"""

    def __init__(self, input_size, output_size, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.input_size = input_size
        self.output_size = output_size
        self.use_causal_mask = use_causal_mask

        self.key_proj = keras.layers.Dense(self.output_size, use_bias=False)
        self.query_proj = keras.layers.Dense(self.output_size, use_bias=False)
        self.value_proj = keras.layers.Dense(self.output_size, use_bias=False)
        self.attn = AttentionMatrix(use_causal_mask=self.use_causal_mask)


    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply single attention head.

        Args:
            inputs_for_keys: [batch_size, seq_length, input_size]
            inputs_for_values: [batch_size, seq_length, input_size]
            inputs_for_queries: [batch_size, seq_length, input_size]

        Returns:
            output: [batch_size, seq_length, output_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        K = self.key_proj(inputs_for_keys)
        Q = self.query_proj(inputs_for_queries)
        V = self.value_proj(inputs_for_values)

        weights = self.attn([K, Q])
        output = tf.matmul(weights, V)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "output_size": self.output_size,
            "use_causal_mask": self.use_causal_mask
        })
        return config

@keras.saving.register_keras_serializable(package="transformer")
class MultiHeadAttention(keras.layers.Layer):
    """Multi-head attention mechanism"""

    def __init__(self, embed_size, num_heads=8, use_causal_mask=False, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.use_causal_mask = use_causal_mask

        assert embed_size % num_heads == 0, "embed_size must be divisible by num_heads"

        self.heads = [
            AttentionHead(input_size=embed_size, output_size=self.head_size, use_causal_mask=use_causal_mask)
            for _ in range(num_heads)
        ]
        self.output_proj = keras.layers.Dense(embed_size, use_bias=False)

    def call(self, inputs_for_keys, inputs_for_values, inputs_for_queries):
        """
        Apply multi-head attention.

        Args:
            inputs_for_keys: [batch_size, seq_length, embed_size]
            inputs_for_values: [batch_size, seq_length, embed_size]
            inputs_for_queries: [batch_size, seq_length, embed_size]

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtypes
        inputs_for_keys = tf.cast(inputs_for_keys, tf.float32)
        inputs_for_values = tf.cast(inputs_for_values, tf.float32)
        inputs_for_queries = tf.cast(inputs_for_queries, tf.float32)

        head_outputs = [head(inputs_for_keys, inputs_for_values, inputs_for_queries) for head in self.heads]
        concatenated = tf.concat(head_outputs, axis=-1)

        return self.output_proj(concatenated)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "use_causal_mask": self.use_causal_mask
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class PositionalEncoding(tf.keras.layers.Layer):
    """
    Positional encoding for transformer inputs.
    Uses sinusoidal position encodings as described in "Attention Is All You Need".
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        # Create positional encoding matrix
        pe = self.get_positional_encoding(max_seq_length, d_model)
        self.positional_encoding = tf.Variable(
            initial_value=pe, trainable=False, name='positional_encoding'
        )

    def get_positional_encoding(self, seq_length: int, d_model: int) -> tf.Tensor:
        """Generate sinusoidal positional encodings as in 'Attention Is All You Need'."""
        
        # This is a position matrix where each row is position 0,1,2,...seq_length-1
        position = tf.range(seq_length, dtype=tf.float32)[:, tf.newaxis]  # [seq_length, 1]

        # This is the division term for the angles (computed from the Attention is All You Need paper)
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))  # [d_model // 2]
        
        # This calculates the sine and cosine terms for even and odd indices respectively
        pe_sin = tf.sin(position * div_term)  # [seq_length, d_model // 2]
        pe_cos = tf.cos(position * div_term)  # [seq_length, d_model // 2]

        # Now we stack and reshape to get the final positional encoding matrix
        pe = tf.stack([pe_sin, pe_cos], axis=2)  # [seq_length, d_model // 2, 2]
        pe = tf.reshape(pe, [seq_length, d_model])  # [seq_length, d_model]

        # Add batch dimension for broadcasting
        return tf.expand_dims(pe, axis=0)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """
        Add positional encoding to input embeddings.

        Args:
            inputs: Input tensor of shape [batch_size, seq_len, d_model]

        Returns:
            inputs with positional encodings added
        """
        seq_length = tf.shape(inputs)[1]

        pe_slice = self.positional_encoding[:, :seq_length, :]

        return inputs + pe_slice

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "max_seq_length": self.max_seq_length
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class LanguageTransformerBlock(keras.layers.Layer):
    """Single transformer block optimized for language modeling (no cross-attention)"""

    def __init__(self, embed_size, num_heads=8, ff_hidden_size=None, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.ff_hidden_size = ff_hidden_size or 4 * embed_size
        self.dropout_rate = dropout_rate
        self.use_causal_mask = True  # Always use causal mask for language modeling

        self.self_attn = MultiHeadAttention(embed_size, num_heads=num_heads, use_causal_mask=True)

        self.ffn_1 = keras.layers.Dense(self.ff_hidden_size, activation="relu")
        self.ffn_2 = keras.layers.Dense(self.embed_size)

        self.ln_1 = keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln_2 = keras.layers.LayerNormalization(epsilon=1e-5)

        self.dropout_attn = keras.layers.Dropout(self.dropout_rate)
        self.dropout_ffn = keras.layers.Dropout(self.dropout_rate)

    def call(self, inputs, training=None):
        """
        Apply transformer block with residual connections and layer normalization.

        Args:
            inputs: [batch_size, seq_length, embed_size]
            training: Whether in training mode

        Returns:
            output: [batch_size, seq_length, embed_size]
        """
        # 1. Ensure consistent dtype
        inputs = tf.cast(inputs, tf.float32)

        attn_out = self.self_attn(inputs, inputs, inputs)
        attn_out = self.dropout_attn(attn_out, training=training)
        x = self.ln_1(inputs + attn_out)

        ffn_out = self.ffn_2(self.ffn_1(x))
        ffn_out = self.dropout_ffn(ffn_out, training=training)
        out = self.ln_2(x + ffn_out)

        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_size": self.embed_size,
            "num_heads": self.num_heads,
            "ff_hidden_size": self.ff_hidden_size,
            "dropout_rate": self.dropout_rate
        })
        return config
    
@keras.saving.register_keras_serializable(package="transformer")
class TransformerLanguageModel(keras.Model):
    """
    Complete Transformer Language Model
    """

    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=None,
                 max_seq_length=512, dropout_rate=0.1, pad_token_id=0, **kwargs):
        super().__init__(**kwargs)

        # Store configuration
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff or 4 * d_model
        self.max_seq_length = max_seq_length
        self.dropout_rate = dropout_rate
        self.pad_token_id = pad_token_id

        self.token_embedding = keras.layers.Embedding(self.vocab_size, self.d_model)
        self.positional_encoding = PositionalEncoding(self.d_model, self.max_seq_length)
        self.embedding_dropout = keras.layers.Dropout(self.dropout_rate)

        self.blocks = [
            LanguageTransformerBlock(
                embed_size=self.d_model,
                num_heads=self.n_heads,
                ff_hidden_size=self.d_ff,
                dropout_rate=self.dropout_rate
            )
            for _ in range(self.n_layers)
        ]

        self.final_layer_norm = keras.layers.LayerNormalization(epsilon=1e-5)
        self.transformer_dropout = keras.layers.Dropout(self.dropout_rate)
        self.output_projection = keras.layers.Dense(self.vocab_size)


    def call(self, inputs, training=None):
        """
        Forward pass through the language model.

        Args:
            inputs: Token indices [batch_size, seq_length]
            training: Whether in training mode

        Returns:
            Logits over vocabulary [batch_size, seq_length, vocab_size]
        """
        # 1. Get token embeddings and scale by sqrt(d_model)
        embeddings = self.token_embedding(inputs)  # [batch_size, seq_length, d_model]
        embeddings = embeddings * tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        x = self.positional_encoding(embeddings)
        x = self.embedding_dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training=training)

        x = self.final_layer_norm(x)
        x = self.transformer_dropout(x, training=training)
        logits = self.output_projection(x)
        
        return logits

    def get_config(self):
        """Get model configuration for saving."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_seq_length': self.max_seq_length,
            'dropout_rate': self.dropout_rate,
            'pad_token_id': self.pad_token_id
        }

def create_language_model(vocab_size: int, **kwargs) -> TransformerLanguageModel:
    """
    Factory function to create a language model with sensible defaults.

    Args:
        vocab_size: Size of the vocabulary
        **kwargs: Additional model parameters

    Returns:
        Initialized TransformerLanguageModel
    """
    default_config = {
        'd_model': 512,
        'n_heads': 8,
        'n_layers': 6,
        'd_ff': 2048,
        'max_seq_length': 256,
        'dropout_rate': 0.1,
        'pad_token_id': 0
    }

    # Update with provided kwargs
    config = {**default_config, **kwargs}
    config['vocab_size'] = vocab_size

    return TransformerLanguageModel(**config): 256,
        'dropout_rate': 0.1,
        'pad_token_id': 0
    }

    # Update with provided kwargs
    config = {**default_config, **kwargs}
    config['vocab_size'] = vocab_size

    return TransformerLanguageModel(**config)