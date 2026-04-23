import tensorflow as tf

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class VanillaRNN(tf.keras.Model):
    """
    Simple vanilla RNN implementation from scratch for text language modeling.
    Compatible with mystery dataset text modeling codebase.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        self.Wx = self.add_weight(name="Wx", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.Wh = self.add_weight(name="Wh", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="b", shape=(hidden_size,), initializer="zeros", trainable=True)

        self.output_projection = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        """
        Forward pass for text language modeling.

        Args:
            inputs: Input token indices of shape [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits of shape [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        # 1. First, we pass the input tokens through the embedding layer
        embedded = self.embedding(inputs)  # [batch_size, seq_length, hidden_size]

        h = tf.zeros([batch_size, self.hidden_size])

        outputs = []
        for t in range(seq_length):
            x_t = embedded[:, t, :]  # [batch_size, hidden_size]
            h = tf.tanh(tf.matmul(x_t, self.Wx) + tf.matmul(h, self.Wh) + self.b)
            outputs.append(h)

        stacked_outputs = tf.stack(outputs, axis=1)  # [batch_size, seq_length, hidden_size]
        logits = self.output_projection(stacked_outputs)  # [batch_size, seq_length, vocab_size]

        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)

########################################################################################

@tf.keras.utils.register_keras_serializable(package="MyLayers")
class LSTM(tf.keras.Model):
    """
    LSTM implementation for comparison with vanilla RNN.
    """

    def __init__(self, vocab_size, hidden_size, seq_length, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.max_seq_length = seq_length
        self.seq_length = seq_length

        self.embedding = tf.keras.layers.Embedding(vocab_size, hidden_size)

        self.Wx_i = self.add_weight("Wx_i", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.Wh_i = self.add_weight("Wh_i", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_i  = self.add_weight("b_i",  shape=(hidden_size,), initializer="zeros", trainable=True)

        self.Wx_f = self.add_weight("Wx_f", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.Wh_f = self.add_weight("Wh_f", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_f  = self.add_weight("b_f",  shape=(hidden_size,), initializer=tf.keras.initializers.Ones(), trainable=True)

        self.Wx_o = self.add_weight("Wx_o", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.Wh_o = self.add_weight("Wh_o", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_o  = self.add_weight("b_o",  shape=(hidden_size,), initializer="zeros", trainable=True)

        self.Wx_g = self.add_weight("Wx_g", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.Wh_g = self.add_weight("Wh_g", shape=(hidden_size, hidden_size), initializer="glorot_uniform", trainable=True)
        self.b_g  = self.add_weight("b_g",  shape=(hidden_size,), initializer="zeros", trainable=True)

        self.output_projection = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, training=None):
        """
        LSTM forward pass.

        Args:
            inputs: Input token indices [batch_size, seq_length]
            training: Training mode flag

        Returns:
            Logits [batch_size, seq_length, vocab_size]
        """
        batch_size = tf.shape(inputs)[0]
        seq_length = tf.shape(inputs)[1]

        embedded = self.embedding(inputs)

        h = tf.zeros([batch_size, self.hidden_size])
        c = tf.zeros([batch_size, self.hidden_size])
        outputs = []

        for t in range(seq_length):
            x_t = embedded[:, t, :]
            i_t = tf.sigmoid(tf.matmul(x_t, self.Wx_i) + tf.matmul(h, self.Wh_i) + self.b_i)
            f_t = tf.sigmoid(tf.matmul(x_t, self.Wx_f) + tf.matmul(h, self.Wh_f) + self.b_f)
            o_t = tf.sigmoid(tf.matmul(x_t, self.Wx_o) + tf.matmul(h, self.Wh_o) + self.b_o)
            g_t = tf.tanh(tf.matmul(x_t, self.Wx_g) + tf.matmul(h, self.Wh_g) + self.b_g)
            c = f_t * c + i_t * g_t
            h = o_t * tf.tanh(c)
            outputs.append(h)

        stacked_outputs = tf.stack(outputs, axis=1)
        logits = self.output_projection(stacked_outputs)
        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {k:getattr(self, k) for k in ["vocab_size", "hidden_size", "seq_length"]}
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    

def create_rnn_language_model(vocab_size, hidden_size, seq_length, model_type="vanilla"):
    """
    Create an RNN-based language model.

    Args:
        vocab_size: Size of vocabulary
        hidden_size: Hidden state dimension
        seq_length: Maximum sequence length
        model_type: Type of RNN ("vanilla", "lstm")

    Returns:
        Configured RNN language model
    """
    # 1. Create appropriate RNN layer based on model_type
    if model_type == "vanilla":
        model = VanillaRNN(vocab_size, hidden_size, seq_length)
    elif model_type == "lstm":
        model = LSTM(vocab_size, hidden_size, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return modelcab_size, hidden_size, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return modelcab_size, hidden_size, seq_length)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return modele: {model_type}. Choose 'vanilla' or 'lstm'")

    return model"Unknown model type: {model_type}. Choose 'vanilla' or 'lstm'")

    return modeldel type: {model_type}. Choose 'vanilla' or 'lstm'")

    return model