"""
train.py: Training utilities and loops for Transformer Language Model
TensorFlow implementation for mystery corpus training

Author: Eric Ewing
"""

import tensorflow as tf
import math
import os
import json
from typing import Tuple, Dict
import tqdm

# Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

def calculate_perplexity(loss: float) -> float:
    """Calculate perplexity from cross-entropy loss."""
    return math.exp(loss)

def train(model, train_dataset, test_dataset, epochs=5, learning_rate=1e-3,
          wandb_run=None, checkpoint_dir="checkpoints", continue_training=False, submission_tracker=None) -> Tuple[tf.keras.Model, Dict[str, list]]:
    """
    Complete training function for language models.

    Args:
        model: Language model to train
        train_dataset: Training dataset
        test_dataset: Test dataset
        epochs: Number of epochs
        learning_rate: Learning rate
        wandb_run: Wandb run for logging
        tokenizer: Tokenizer for text generation
        checkpoint_dir: Directory to save checkpoints
        continue_training: Whether to continue training from latest checkpoint
        submission_tracker: Submission tracker for logging epoch results

    Returns:
        model: Trained model
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction="none")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=3)
    
    # Handle checkpoint restoration for continue training
    start_epoch = 1
    if continue_training:
        latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
        if latest_checkpoint:
            checkpoint.restore(latest_checkpoint)
            # Extract epoch number from checkpoint name
            try:
                start_epoch = int(latest_checkpoint.split('-')[-1])
                print(f"Resuming from epoch {start_epoch}")
            except:
                print("Could not determine start epoch, starting from 0")
        else:
            print("No checkpoint found, starting fresh")
    
    history = {'train_loss': [], 'val_loss': [], 'perplexity': []}

    for current_epoch in tqdm.tqdm(range(start_epoch, start_epoch + epochs), desc="Training Progress", position=0):
        total_epochs = start_epoch + epochs - 1
        train_losses = []
        for batch in train_dataset:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[:2]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch.get("x"))
                targets = batch.get("targets", batch.get("y"))
            else:
                inputs, targets = batch[:, :-1], batch[:, 1:]
            with tf.GradientTape() as tape:
                logits = model(inputs, training=True)
                batch_loss = loss_fn(targets, logits)
                batch_loss = tf.reduce_mean(batch_loss)
            grads = tape.gradient(batch_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            train_losses.append(batch_loss)

        avg_train_loss = tf.reduce_mean(tf.stack(train_losses))

        val_losses = []
        for batch in test_dataset:
            if isinstance(batch, (tuple, list)):
                inputs, targets = batch[:2]
            elif isinstance(batch, dict):
                inputs = batch.get("inputs", batch.get("x"))
                targets = batch.get("targets", batch.get("y"))
            else:
                inputs, targets = batch[:, :-1], batch[:, 1:]
            logits = model(inputs, training=False)
            batch_loss = loss_fn(targets, logits)
            batch_loss = tf.reduce_mean(batch_loss)
            val_losses.append(batch_loss)

        avg_val_loss = tf.reduce_mean(tf.stack(val_losses))
        ppl = calculate_perplexity(float(avg_val_loss))
        
        history['train_loss'].append(float(avg_train_loss))
        history['val_loss'].append(float(avg_val_loss))
        history['perplexity'].append(float(ppl))

        if submission_tracker is not None:
            submission_tracker.log_epoch(current_epoch, float(avg_train_loss), float(avg_val_loss), float(ppl))

        checkpoint_manager.save(checkpoint_number=current_epoch)

        if wandb_run:
            wandb_run.log({
                "epoch": int(current_epoch),
                "train_loss": float(avg_train_loss),
                "val_loss": float(avg_val_loss),
                "perplexity": float(ppl),
            })
        
    return model, history
