#!/usr/bin/env python3
import sys
import os

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import tensorflow as tf
import numpy as np
import json
import argparse
import hashlib
import time
from datetime import datetime
from typing import Dict

# Import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    print("Wandb not available: install with: pip install wandb")
    WANDB_AVAILABLE = False

# Import modules
from src.training.language_model import TextGenerator, TextSampler
from src.training.train import train
from src.data.data import prepare_data
from src.models.RNNs import create_rnn_language_model
from src.models.transformer import create_language_model

# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)
tf.keras.backend.clear_session()
tf.keras.backend.set_floatx('float32')
tf.keras.mixed_precision.set_global_policy('float32')

class Config:
    """Configuration for training parameters."""
    def __init__(self):
        # Data
        self.data_path = 'data/mystery_data.pkl'
        self.seq_length = 256
        self.batch_size = 64

        # Model
        self.model_type = "transformer"
        self.vocab_size = 10000
        self.d_model = 512
        self.n_heads = 8
        self.n_layers = 6
        self.d_ff = 2048
        self.dropout_rate = 0.1

        # Training
        self.epochs = 1
        self.learning_rate = 1e-3
        self.use_lr_schedule = False
        self.continue_training = False

        # Paths
        self.submission_base_dir = 'submission'
        self.checkpoint_dir = None  # Will be set dynamically: checkpoints/{model_type}/{timestamp}/
        self.logs_dir = 'logs'
        self.model_save_path = 'final_model'

        # Generation
        self.generation_length = 100
        self.generation_temperature = 0.8
        self.generation_top_k = 40
        self.generation_top_p = 0.9

        # Wandb
        self.use_wandb = True
        self.wandb_project = "mystery-transformer"
        self.wandb_entity = None
        self.wandb_run_name = "mystery_run"

def count_parameters(model):
    """Count trainable parameters."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])

def setup_directories(config: Config):
    """Create necessary directories."""
    if config.checkpoint_dir is not None:
        os.makedirs(config.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logs_dir, exist_ok=True)

class SubmissionTracker:
    def __init__(self, config, timestamp: str, submission_dir: str = None):
        self.config = config
        self.timestamp = timestamp
        self.training_log = []
        self.start_time = time.time()
        self.submission_dir = submission_dir or "."
        self.chain_seed = self._create_genesis_hash()
        self.previous_hash = self.chain_seed
    
    def _create_genesis_hash(self) -> str:
        genesis_data = {
            'model_type': self.config.model_type,
            'vocab_size': self.config.vocab_size,
            'd_model': self.config.d_model,
            'n_heads': self.config.n_heads,
            'n_layers': self.config.n_layers,
            'start_timestamp': self.timestamp,
            'start_time': self.start_time
        }
        canonical = json.dumps(genesis_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, perplexity: float):
        """Log epoch metrics with chained hash."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        epoch_data = {
            'epoch': int(epoch),
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'perplexity': float(perplexity),
            'logged_at': datetime.now().isoformat(),
            'elapsed_time': elapsed,
            'previous_hash': self.previous_hash  # Link to previous epoch
        }
        
        # Create hash for this epoch
        current_hash = self._hash_epoch(epoch_data)
        epoch_data['epoch_hash'] = current_hash
        
        self.training_log.append(epoch_data)
        self.previous_hash = current_hash  # Update for next epoch
        
        self._save_incremental()
    
    def _hash_epoch(self, epoch_data: dict) -> str:
        """Create deterministic hash of epoch data including previous hash."""
        # Include only the data that matters for integrity
        hash_input = {
            'epoch': epoch_data['epoch'],
            'train_loss': epoch_data['train_loss'],
            'val_loss': epoch_data['val_loss'],
            'perplexity': epoch_data['perplexity'],
            'logged_at': epoch_data['logged_at'],
            'elapsed_time': epoch_data['elapsed_time'],
            'previous_hash': epoch_data['previous_hash']
        }
        canonical = json.dumps(hash_input, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(canonical.encode()).hexdigest()
    
    def _save_incremental(self):
        """Save incremental backup."""
        backup_data = {
            'chain_seed': self.chain_seed,
            'training_log': self.training_log,
            'status': 'in_progress'
        }
        try:
            backup_path = os.path.join(self.submission_dir, '.submission_backup.json')
            with open(backup_path, 'w') as f:
                json.dump(backup_data, f, indent=2)
        except:
            pass
    
    def finalize(self, submission_path: str, weights_path: str = None):
        """Finalize submission with hash chain integrity."""
        if not self.training_log:
            print("Warning: No training data logged!")
            return None
        
        # Verify the hash chain before finalizing
        if not self._verify_chain():
            print("ERROR: Hash chain verification failed!")
            return None
        
        final_epoch = self.training_log[-1]
        total_time = time.time() - self.start_time
        
        submission_data = {
            'metadata': {
                'timestamp': self.timestamp,
                'model_type': self.config.model_type,
                'submission_version': '3.0',
                'total_training_time_seconds': total_time,
                'chain_seed': self.chain_seed,
                'final_hash': self.previous_hash
            },
            'model_params': {
                'vocab_size': self.config.vocab_size,
                'd_model': self.config.d_model,
                'n_heads': self.config.n_heads,
                'n_layers': self.config.n_layers,
                'batch_size': self.config.batch_size,
                'seq_length': self.config.seq_length,
                'learning_rate': self.config.learning_rate,
                'epochs': self.config.epochs
            },
            'final_metrics': {
                'train_loss': final_epoch['train_loss'],
                'val_loss': final_epoch['val_loss'],
                'perplexity': final_epoch['perplexity']
            },
            'training_history': {
                'epochs': [e['epoch'] for e in self.training_log],
                'train_loss': [e['train_loss'] for e in self.training_log],
                'val_loss': [e['val_loss'] for e in self.training_log],
                'perplexity': [e['perplexity'] for e in self.training_log],
                'logged_at': [e['logged_at'] for e in self.training_log],
                'elapsed_time': [e['elapsed_time'] for e in self.training_log],
                'epoch_hashes': [e['epoch_hash'] for e in self.training_log],
                'previous_hashes': [e['previous_hash'] for e in self.training_log]
            }
        }
        
        if weights_path and os.path.exists(weights_path):
            submission_data['metadata']['weights_hash'] = self._hash_file(weights_path)
        
        with open(submission_path, 'w') as f:
            json.dump(submission_data, f, indent=2)
        
        print(f"Submission file saved: {submission_path}")
        print(f"Hash chain verified ({len(self.training_log)} epochs)")
        print(f"Total training time: {total_time/60:.1f} minutes")

        backup_path = os.path.join(self.submission_dir, '.submission_backup.json')
        if os.path.exists(backup_path):
            os.remove(backup_path)

        return submission_path
    
    def _verify_chain(self) -> bool:
        """Verify the integrity of the hash chain."""
        expected_hash = self.chain_seed
        
        for epoch_data in self.training_log:
            # Check if previous hash matches
            if epoch_data['previous_hash'] != expected_hash:
                print(f"Chain break at epoch {epoch_data['epoch']}")
                return False
            
            # Recompute this epoch's hash
            computed_hash = self._hash_epoch(epoch_data)
            if computed_hash != epoch_data['epoch_hash']:
                print(f"Hash mismatch at epoch {epoch_data['epoch']}")
                return False
            
            expected_hash = computed_hash
        
        return True
    
    def _hash_file(self, filepath: str) -> str:
        """Create SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(filepath, 'rb') as f:
            while chunk := f.read(8192):
                sha256.update(chunk)
        return sha256.hexdigest()

def save_config(config: Config, filepath: str):
    """Save configuration to JSON."""
    config_dict = {k: v.item() if hasattr(v, 'item') else v 
                   for k, v in config.__dict__.items() if not k.startswith('_')}
    with open(filepath, 'w') as f:
        json.dump(config_dict, f, indent=2)

def load_config_from_file(filepath: str, config: Config):
    """Load configuration from JSON."""
    try:
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        for k, v in config_dict.items():
            if hasattr(config, k):
                setattr(config, k, v)
        return True
    except Exception as e:
        print(f"Could not load config from {filepath}: {e}")
        return False

def find_latest_config(model_type: str):
    """Find most recent config file."""
    logs_dir = 'logs'
    if not os.path.exists(logs_dir):
        return None
    
    config_files = [os.path.join(logs_dir, f) for f in os.listdir(logs_dir)
                    if f.startswith('config_') and f.endswith('.json')]
    
    if not config_files:
        return None
    
    return max(config_files, key=os.path.getmtime)

def load_mystery_data(config: Config):
    """Load mystery corpus data."""
    print("LOADING MYSTERY CORPUS DATA")
    train_dataset, test_dataset, tokenizer = prepare_data(
        pickle_path=config.data_path,
        seq_length=config.seq_length,
        batch_size=config.batch_size,
        vocab_size=config.vocab_size
    )
    config.vocab_size = len(tokenizer)
    print(f"Vocabulary size from pickle: {config.vocab_size:,}")
    return train_dataset, test_dataset, tokenizer

def create_model(config: Config):
    """Create and initialize model."""
    print(f"CREATING {config.model_type.upper()} MODEL")

    if config.model_type == "transformer":
        model = create_language_model(
            vocab_size=config.vocab_size,
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            d_ff=config.d_ff,
            max_seq_length=config.seq_length,
            dropout_rate=config.dropout_rate
        )
    elif config.model_type in ["vanilla_rnn", "lstm"]:
        rnn_type = "vanilla" if config.model_type == "vanilla_rnn" else "lstm"
        model = create_rnn_language_model(
            vocab_size=config.vocab_size,
            hidden_size=config.d_model,
            seq_length=config.seq_length,
            model_type=rnn_type
        )
    else:
        raise ValueError(f"Unknown model_type: {config.model_type}")

    print("Building model...")
    dummy_input = tf.zeros((1, config.seq_length), dtype=tf.int32)
    output = model(dummy_input)
    print(f"Model built with input: {dummy_input.dtype} and output: {output.dtype}")
    assert output.dtype == tf.float32, f"Expected float32 output, got {output.dtype}"

    total_params = count_parameters(model)
    print(f"Model created with {total_params:,} trainable parameters")
    return model

def train_model(model, train_dataset, test_dataset, config: Config, wandb_run, tokenizer, continue_training=False):
    """Train model."""
    print("STARTING TRAINING AT ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    model, history = train(
        model, train_dataset, test_dataset,
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        wandb_run=wandb_run,
        checkpoint_dir=config.checkpoint_dir,
        continue_training=continue_training
    )

    print("TRAINING COMPLETE AT ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return history, None

def generate_sample_text(model, tokenizer, config: Config):
    """Generate and display sample text."""
    generator = TextGenerator(model, tokenizer)
    
    sample_prompts = [
        "The detective examined",
        "In the dimly lit room",
        "The murder weapon was",
        "Holmes deduced that",
        ""
    ]
    
    print("\n" + "=" * 60)
    print("SAMPLE TEXT GENERATION")
    print("=" * 60)
    
    for i, prompt in enumerate(sample_prompts):
        print(f"\nPrompt {i+1}: '{prompt}'" if prompt else f"\nPrompt {i+1}: [Random generation]")
        print("-" * 60)
        try:
            generated_text = generator.generate(
                prompt=prompt,
                max_length=config.generation_length,
                method="top_k",
                temperature=config.generation_temperature,
                top_k=config.generation_top_k,
                top_p=config.generation_top_p
            )
            print(generated_text)
        except Exception as e:
            print(f"Error generating text: {e}")
        print()
    print("=" * 60)

def interactive_generation(model, tokenizer, config: Config):
    """Interactive text generation session."""
    print("=" * 60)
    print("INTERACTIVE MYSTERY TEXT GENERATION")
    print("=" * 60)
    print("Commands: text prompt | 'random' | 'samples' | 'settings' | 'quit'\n")

    generator = TextGenerator(model, tokenizer)
    
    mystery_prompts = [
        "The detective examined", "In the dimly lit room", "The murder weapon was",
        "Holmes deduced that", "The evidence suggested", "At the crime scene",
        "The suspect claimed", "The mysterious letter read"
    ]
    
    gen_settings = {
        'max_length': config.generation_length,
        'temperature': config.generation_temperature,
        'top_k': config.generation_top_k,
        'top_p': config.generation_top_p,
        'method': 'top_k'
    }
    
    print(f"Current settings: {gen_settings}\n")
    
    while True:
        try:
            user_input = input("Enter prompt (or command): ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                print("Exiting interactive generation...")
                break
            
            elif user_input.lower() == 'settings':
                print(f"\nCurrent settings: {gen_settings}\n")
                continue
            
            elif user_input.lower() == 'samples':
                print("\nPre-defined Mystery Prompts")
                for i, prompt in enumerate(mystery_prompts, 1):
                    print(f"{i}. \"{prompt}\"")
                print("\nGenerating samples...\n")
                
                for prompt in mystery_prompts[:3]:
                    print(f"Prompt: '{prompt}'")
                    print("-" * 60)
                    try:
                        generated_text = generator.generate(prompt=prompt, **gen_settings)
                        print(generated_text)
                    except Exception as e:
                        print(f"Error: {e}")
                    print()
                continue
            
            elif user_input.lower() == 'random':
                print("\nRandom Generation (no prompt)")
                print("-" * 60)
                try:
                    generated_text = generator.generate(prompt="", **gen_settings)
                    print(generated_text)
                except Exception as e:
                    print(f"Error: {e}")
                print()
                continue
            
            else:
                print(f"\nPrompt: '{user_input}'")
                print("-" * 60)
                try:
                    generated_text = generator.generate(prompt=user_input, **gen_settings)
                    print(generated_text)
                except Exception as e:
                    print(f"Error: {e}")
                print()
        
        except KeyboardInterrupt:
            print("\n\nExiting interactive generation...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

def main():
    """Main training function."""
    print("Starting...")
    
    # Configure GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
        except RuntimeError as e:
            print(f"GPU memory growth setup failed: {e}")

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train Transformer Language Model')
    parser.add_argument('--epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--seq-length', type=int, help='Sequence length')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--simple', action='store_true', help='Use simple training mode')
    parser.add_argument('--interactive', action='store_true', help='Run interactive generation')
    parser.add_argument('--generate-only', action='store_true', help='Skip training, only generate')
    parser.add_argument('--continue-training', action='store_true', help='Continue from checkpoint')
    parser.add_argument('--force-fresh', action='store_true', help='Force fresh training')
    parser.add_argument('--model-type', type=str, choices=['transformer', 'vanilla_rnn', 'lstm'],
                       help='Type of model to train')
    parser.add_argument('--vocab-size', type=int, help='Vocabulary size')
    parser.add_argument('--d-model', type=int, help='Model dimension/hidden size')
    parser.add_argument('--n-heads', type=int, help='Number of attention heads')
    parser.add_argument('--n-layers', type=int, help='Number of model layers')
    parser.add_argument('--d-ff', type=int, help='Feed-forward dimension')
    parser.add_argument('--dropout-rate', type=float, help='Dropout rate')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')

    args = parser.parse_args()

    # Load config and override with args
    config = Config()
    
    # Override config with command line arguments using loop
    arg_mapping = {
        'epochs': 'epochs',
        'batch_size': 'batch_size',
        'seq_length': 'seq_length',
        'learning_rate': 'learning_rate',
        'model_type': 'model_type',
        'vocab_size': 'vocab_size',
        'd_model': 'd_model',
        'n_heads': 'n_heads',
        'n_layers': 'n_layers',
        'd_ff': 'd_ff',
        'dropout_rate': 'dropout_rate',
        'continue_training': 'continue_training'
    }
    
    for arg_name, config_attr in arg_mapping.items():
        arg_value = getattr(args, arg_name, None)
        if arg_value is not None:
            setattr(config, config_attr, arg_value)
    
    if args.no_wandb:
        config.use_wandb = False

    # Configure TensorFlow computation mode
    if config.model_type in ["vanilla_rnn", "lstm"]:
        tf.config.run_functions_eagerly(True)
        print(f"Eager execution enabled for {config.model_type} model")
    else:
        tf.config.run_functions_eagerly(False)
        print(f"Graph mode enabled for {config.model_type} model (better GPU performance)")

    setup_directories(config)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("MYSTERY CORPUS TRAINING", flush=True)
    print(f"Timestamp: {timestamp}", flush=True)
    print(f"Data file: {config.data_path}", flush=True)

    if not os.path.exists(config.data_path):
        print(f"Error: {config.data_path} not found!")
        print("Please run the data preprocessing first:")
        print("  cd data")
        print("  python process_data.py")
        return

    start_time = datetime.now()
    print("Starting data loading...", flush=True)

    # Initialize wandb
    wandb_run = None
    if not args.generate_only and config.use_wandb and WANDB_AVAILABLE:
        try:
            print("Initializing wandb...")
            wandb_run = wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                name=f"{config.wandb_run_name}_{timestamp}",
                config={
                    'model_type': config.model_type,
                    'vocab_size': config.vocab_size,
                    'd_model': config.d_model,
                    'n_heads': config.n_heads,
                    'n_layers': config.n_layers,
                    'd_ff': config.d_ff,
                    'dropout_rate': config.dropout_rate,
                    'seq_length': config.seq_length,
                    'batch_size': config.batch_size,
                    'epochs': config.epochs,
                    'learning_rate': config.learning_rate,
                }
            )
            print(f"Wandb initialized run: {wandb_run.name}")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
            print("Continuing without wandb logging...")
            wandb_run = None
    elif not args.generate_only and config.use_wandb and not WANDB_AVAILABLE:
        print("Warning: wandb requested but not installed")
        print("Continuing without wandb logging...")

    if not args.generate_only:
        submission_base_dir = "submission"
        model_submission_dir = os.path.join(submission_base_dir, config.model_type)
        os.makedirs(model_submission_dir, exist_ok=True)

        checkpoint_base_dir = "checkpoints"

        if config.continue_training:
            model_checkpoint_base = os.path.join(checkpoint_base_dir, config.model_type)
            if os.path.exists(model_checkpoint_base):
                # Find the most recent run directory (by modification time)
                run_dirs = [os.path.join(model_checkpoint_base, d)
                           for d in os.listdir(model_checkpoint_base)
                           if os.path.isdir(os.path.join(model_checkpoint_base, d))]
                if run_dirs:
                    latest_run_dir = max(run_dirs, key=os.path.getmtime)
                    config.checkpoint_dir = latest_run_dir
                    print(f"Continuing training from: {config.checkpoint_dir}/")
                else:
                    print(f"No existing checkpoints found for {config.model_type}, starting fresh")
                    config.checkpoint_dir = os.path.join(checkpoint_base_dir, config.model_type, timestamp)
                    os.makedirs(config.checkpoint_dir, exist_ok=True)
            else:
                print(f"No checkpoint directory found for {config.model_type}, starting fresh")
                config.checkpoint_dir = os.path.join(checkpoint_base_dir, config.model_type, timestamp)
                os.makedirs(config.checkpoint_dir, exist_ok=True)
        else:
            config.checkpoint_dir = os.path.join(checkpoint_base_dir, config.model_type, timestamp)
            os.makedirs(config.checkpoint_dir, exist_ok=True)

        print(f"Checkpoint directory: {config.checkpoint_dir}/")

        # Load data and create model
        train_dataset, test_dataset, tokenizer = load_mystery_data(config)
        model = create_model(config)

        # Save config to submission folder
        config_path = os.path.join(model_submission_dir, f"{config.model_type}_config.json")
        save_config(config, config_path)
        print(f"Configuration saved to: {config_path}")

        # Initialize submission tracker and train
        submission_tracker = SubmissionTracker(config, timestamp, submission_dir=model_submission_dir)
        print("Submission tracker initialized")

        print("Using simple training mode...")
        model, history = train(
            model, train_dataset, test_dataset,
            epochs=config.epochs, learning_rate=config.learning_rate,
            wandb_run=wandb_run, checkpoint_dir=config.checkpoint_dir,
            continue_training=config.continue_training,
            submission_tracker=submission_tracker
        )

        # Finish wandb
        if wandb_run is not None:
            try:
                wandb.finish()
                print("Wandb run finished")
            except Exception as e:
                print(f"Warning: Error finishing wandb run: {e}")

        # Save model and submission files
        print("\n" + "=" * 60)
        print("SAVING MODEL AND SUBMISSION FILES")
        print("=" * 60)
        print(f"\nSubmission directory: {model_submission_dir}/")

        weights_filename = f"{config.model_type}_model.weights.h5"
        submission_filename = "submission.json"

        weights_path = os.path.join(model_submission_dir, weights_filename)
        submission_path = os.path.join(model_submission_dir, submission_filename)

        saved_files = []

        # Save weights
        try:
            print(f"\nSaving model weights...")
            model.save_weights(weights_path)
            if os.path.exists(weights_path):
                file_size = os.path.getsize(weights_path) / (1024 * 1024)
                print(f"  Saved: {weights_path} ({file_size:.2f} MB)")
                saved_files.append((weights_path, file_size))
        except Exception as e:
            print(f"  Error saving {weights_path}: {e}")

        # Config already saved earlier, just add to summary
        if os.path.exists(config_path):
            saved_files.append((config_path, None))

        # Finalize submission
        print(f"\nFinalizing submission file...")
        try:
            submission_file = submission_tracker.finalize(submission_path, weights_path=weights_path)
            if os.path.exists(submission_path):
                print(f"  Saved: {submission_path}")
                saved_files.append((submission_path, None))
        except Exception as e:
            print(f"  Error finalizing submission: {e}")

        # Print summary
        print("\n" + "-" * 60)
        print("SUBMISSION FILES SUMMARY")
        print("-" * 60)
        print(f"Model Type: {config.model_type}")
        print(f"Directory: {model_submission_dir}/")
        print(f"\nFiles saved ({len(saved_files)} total):")
        for filepath, size in saved_files:
            if size is not None:
                print(f"  • {os.path.basename(filepath)} ({size:.2f} MB)")
            else:
                print(f"  • {os.path.basename(filepath)}")
        print("\n" + "=" * 60)
        print(f"SUBMIT THE ENTIRE DIRECTORY: {model_submission_dir}/")
        print("=" * 60)

        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        print(f"Training completed in {total_time/60:.1f} minutes")

    else:
        # Generate-only mode
        print("Generate-only mode: Loading existing model...")

        # Look for config in submission folder
        submission_base_dir = "submission"
        model_submission_dir = os.path.join(submission_base_dir, config.model_type)

        model_config_path = os.path.join(model_submission_dir, f"{config.model_type}_config.json")
        if os.path.exists(model_config_path):
            print(f"Found saved config: {model_config_path}")

            params_specified = {
                'vocab_size': args.vocab_size is not None,
                'd_model': args.d_model is not None,
                'seq_length': args.seq_length is not None
            }

            if load_config_from_file(model_config_path, config):
                print(f"Loaded model configuration:")
                print(f"  vocab_size: {config.vocab_size}")
                print(f"  d_model: {config.d_model}")
                print(f"  seq_length: {config.seq_length}")
                print(f"  model_type: {config.model_type}")

                for param, attr in [('vocab_size', 'vocab_size'), ('d_model', 'd_model'),
                                   ('seq_length', 'seq_length')]:
                    if params_specified[param]:
                        setattr(config, attr, getattr(args, param))
                        print(f"  ({attr} overridden to {getattr(args, param)})")
        else:
            print(f"No saved config found at {model_config_path}")
            print("Using parameters from command line or defaults")

        _, _, tokenizer = load_mystery_data(config)
        model = create_model(config)

        # Load weights from submission folder
        model_weights_path = os.path.join(model_submission_dir, f"{config.model_type}_model.weights.h5")

        loaded = False
        load_errors = []

        if os.path.exists(model_weights_path):
            try:
                model.load_weights(model_weights_path)
                print(f"Model weights loaded from: {model_weights_path}")
                loaded = True
            except Exception as e:
                load_errors.append(f"{model_weights_path}: {str(e)}")

        if not loaded:
            print(f"Could not load {config.model_type} model weights!")
            print(f"Expected path: {model_weights_path}")
            if load_errors:
                print("\nErrors encountered:")
                for error in load_errors:
                    print(f"  {error}")
            print("\nMake sure you:")
            print("  1. Have trained a model first (weights should be in submission folder)")
            print(f"  2. Check that {model_weights_path} exists")
            print("  3. Use the same --vocab-size, --d-model, and --seq-length as during training")
            return

    # Generate sample text
    print("Generating sample text...")
    generate_sample_text(model, tokenizer, config)

    # Interactive generation
    if args.interactive:
        interactive_generation(model, tokenizer, config)

    print("Mystery corpus training complete!")

SubmissionTracker = SubmissionTracker

if __name__ == "__main__":
    main()