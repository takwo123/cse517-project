import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, RandomSampler
from datasets import load_dataset
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
import re
import unicodedata
from collections import Counter

DEVICE = (
    'cuda' if torch.cuda.is_available() else
    'mps' if torch.backends.mps.is_available() else
    'cpu'
)

print(f"✅ Using device: {DEVICE}")

# ADAPTIVE CONFIGURATION - No hardcoded values!
class AdaptiveConfig:
    def __init__(self, vocab_size, data_size, device):
        self.device = device
        self.vocab_size = vocab_size
        self.data_size = data_size
        
        # Auto-configure based on data characteristics
        self.emb_size = self._adaptive_emb_size()
        self.hidden_size = self._adaptive_hidden_size()
        self.num_layers = self._adaptive_num_layers()
        self.seq_len = self._adaptive_seq_len()
        self.batch_size = self._adaptive_batch_size()
        self.lr = self._adaptive_learning_rate()
        self.epochs = self._adaptive_epochs()
        self.dropout = 0.2 if vocab_size > 100 else 0.1
        
        print(f"🤖 Auto-configured for {vocab_size} characters, {data_size} data points:")
        print(f"   Embedding: {self.emb_size}, Hidden: {self.hidden_size}, Layers: {self.num_layers}")
        print(f"   Sequence: {self.seq_len}, Batch: {self.batch_size}, LR: {self.lr}, Epochs: {self.epochs}")
    
    def _adaptive_emb_size(self):
        # Scale embedding size based on vocabulary diversity
        base = min(64, max(32, self.vocab_size // 2))
        if self.vocab_size > 200:  # Multilingual
            return min(256, int(base * 2))
        elif self.vocab_size > 100:  # Extended charset
            return min(128, int(base * 1.5))
        else:  # Simple charset
            return int(base)
    
    def _adaptive_hidden_size(self):
        # Scale hidden size based on data complexity
        if self.data_size > 500000:  # Large dataset
            return min(512, int(self.emb_size * 3))
        elif self.data_size > 200000:  # Medium dataset
            return min(256, int(self.emb_size * 2))
        else:  # Small dataset
            return min(128, int(self.emb_size * 1.5))
    
    def _adaptive_num_layers(self):
        # More layers for complex data, fewer for simple
        if self.data_size > 500000 and self.vocab_size > 150:
            return 3
        elif self.data_size > 200000:
            return 2
        else:
            return 1
    
    def _adaptive_seq_len(self):
        # Longer sequences for rich languages, shorter for simple
        if self.vocab_size > 200:  # Multilingual/complex
            return min(64, max(32, self.data_size // 10000))
        else:  # Simple
            return min(48, max(24, self.data_size // 15000))
    
    def _adaptive_batch_size(self):
        # Larger batches for GPU efficiency, based on model size
        model_complexity = self.emb_size * self.hidden_size * self.num_layers
        if self.device == 'cuda':
            if model_complexity < 50000:
                return 128
            elif model_complexity < 100000:
                return 64
            else:
                return 32
        else:  # CPU/MPS
            return min(32, max(16, 100000 // model_complexity))
    
    def _adaptive_learning_rate(self):
        # Lower LR for complex models, higher for simple
        if self.num_layers >= 3 and self.hidden_size >= 256:
            return 5e-4
        elif self.num_layers >= 2:
            return 1e-3
        else:
            return 2e-3
    
    def _adaptive_epochs(self):
        # More epochs for larger, more complex datasets
        if self.data_size > 500000:
            return 15
        elif self.data_size > 200000:
            return 12
        else:
            return 8

class MultilingualEmbeddingLoader:
    """Multilingual character embedding loader with Unicode awareness"""
    
    @staticmethod
    def create_unicode_aware_embeddings(vocab, emb_size=64):
        """Create embeddings that understand Unicode character properties"""
        print(f"🌍 Creating Unicode-aware embeddings for {len(vocab)} characters...")
        
        char_embeddings = []
        for char in vocab:
            # Base random embedding
            embedding = np.random.normal(0, 0.1, emb_size)
            
            # Add Unicode-based features
            try:
                # Character category (Letter, Number, Punctuation, etc.)
                category = unicodedata.category(char)
                if category.startswith('L'):  # Letter
                    embedding[0] = 1.0
                elif category.startswith('N'):  # Number
                    embedding[1] = 1.0
                elif category.startswith('P'):  # Punctuation
                    embedding[2] = 1.0
                elif category.startswith('S'):  # Symbol
                    embedding[3] = 1.0
                elif category.startswith('Z'):  # Separator (space, etc.)
                    embedding[4] = 1.0
                
                # Writing direction
                bidi = unicodedata.bidirectional(char)
                if bidi in ['L', 'LRE', 'LRO']:  # Left-to-right
                    embedding[5] = 1.0
                elif bidi in ['R', 'AL', 'RLE', 'RLO']:  # Right-to-left
                    embedding[6] = 1.0
                
                # Case information
                if char.isupper():
                    embedding[7] = 1.0
                elif char.islower():
                    embedding[8] = 1.0
                
                # ASCII vs non-ASCII
                if ord(char) < 128:  # ASCII
                    embedding[9] = 1.0
                else:  # Non-ASCII (multilingual)
                    embedding[10] = 1.0
                
            except (ValueError, TypeError):
                # Handle any Unicode errors gracefully
                pass
            
            char_embeddings.append(embedding)
        
        return np.array(char_embeddings)

class AdaptiveCharDataset(Dataset):
    """Adaptive dataset that handles variable sequence lengths"""
    
    def __init__(self, text, config, char2idx):
        self.config = config
        self.char2idx = char2idx
        # Convert to indices once for efficiency
        self.data = torch.tensor([char2idx.get(c, 0) for c in text], dtype=torch.long)
        
        # Create variable-length sequences for better learning
        self.sequences = []
        self.targets = []
        
        for i in range(len(text) - config.seq_len):
            seq_len = random.randint(config.seq_len // 2, config.seq_len)
            if i + seq_len < len(text):
                self.sequences.append(self.data[i:i+seq_len])
                self.targets.append(self.data[i+seq_len])
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        target = self.targets[idx]
        
        # Pad sequence to fixed length
        if len(seq) < self.config.seq_len:
            padding = torch.zeros(self.config.seq_len - len(seq), dtype=torch.long)
            seq = torch.cat([padding, seq])
        
        return seq, target

class AdaptiveCharLSTM(nn.Module):
    """Adaptive LSTM architecture optimized for character prediction"""
    
    def __init__(self, vocab_size, config, pretrained_embeddings=None):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, config.emb_size)
        if pretrained_embeddings is not None:
            print("🔥 Initializing with multilingual character embeddings!")
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embeddings).float())
        
        # LSTM layers
        self.lstm = nn.LSTM(
            config.emb_size,
            config.hidden_size,
            config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=False  # Simpler for character prediction
        )
        
        # Output layers with residual connection for better gradient flow
        self.dropout = nn.Dropout(config.dropout)
        self.output_norm = nn.LayerNorm(config.hidden_size)
        self.output_linear = nn.Linear(config.hidden_size, vocab_size)
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better convergence"""
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.zeros_(self.output_linear.bias)
    
    def forward(self, x):
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, emb_size)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last output for prediction
        last_output = lstm_out[:, -1, :]  # (batch, hidden_size)
        
        # Apply normalization and dropout
        normalized = self.output_norm(last_output)
        dropped = self.dropout(normalized)
        
        # Final prediction
        logits = self.output_linear(dropped)  # (batch, vocab_size)
        
        return logits

class CharacterPatternTrainer:
    """Specialized trainer focused on character prediction patterns"""
    
    @staticmethod
    def create_pattern_focused_data(target_size=200000):
        """Create data specifically optimized for character prediction learning"""
        print("🎯 Creating pattern-focused training data...")
        
        texts = []
        current_size = 0
        
        # 1. Common English words and patterns (40%)
        common_words_text = CharacterPatternTrainer._generate_common_words_data(int(target_size * 0.4))
        texts.append(common_words_text)
        current_size += len(common_words_text)
        print(f"   ✅ Common words: {len(common_words_text)} chars")
        
        # 2. Character transition patterns (30%)
        transition_text = CharacterPatternTrainer._generate_transition_patterns(int(target_size * 0.3))
        texts.append(transition_text)
        current_size += len(transition_text)
        print(f"   ✅ Transition patterns: {len(transition_text)} chars")
        
        # 3. Real text with good character patterns (30%)
        real_text = CharacterPatternTrainer._get_pattern_rich_text(target_size - current_size)
        texts.append(real_text)
        current_size += len(real_text)
        print(f"   ✅ Pattern-rich text: {len(real_text)} chars")
        
        # Combine and create character-level training examples
        combined = ''.join(texts)
        
        # Add repetition of common character patterns
        pattern_enhanced = CharacterPatternTrainer._enhance_with_patterns(combined)
        
        print(f"🎯 Pattern-focused data created: {len(pattern_enhanced)} characters")
        return pattern_enhanced[:target_size]
    
    @staticmethod
    def _generate_common_words_data(size):
        """Generate data from most common English words"""
        common_words = [
            "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", 
            "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
            "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
            "did", "its", "let", "put", "say", "she", "too", "use", "that", "with",
            "have", "this", "will", "your", "from", "they", "know", "want", "been",
            "good", "much", "some", "time", "very", "when", "come", "here", "just",
            "like", "long", "make", "many", "over", "such", "take", "than", "them",
            "well", "were", "what", "would", "there", "could", "other", "after",
            "first", "never", "these", "think", "where", "being", "every", "great",
            "might", "shall", "still", "those", "under", "while", "before", "should",
            "through", "another", "between", "nothing", "something", "everything"
        ]
        
        text = ""
        for _ in range(size):
            if len(text) >= size:
                break
            word = random.choice(common_words)
            text += word + " "
        
        return text[:size]
    
    @staticmethod
    def _generate_transition_patterns(size):
        """Generate character transition patterns"""
        patterns = [
            # Common prefixes and suffixes
            "ing", "tion", "ness", "ment", "able", "less", "ful", "pre", "un", "re",
            # Common letter combinations
            "th", "he", "in", "er", "an", "ed", "nd", "to", "en", "ti", "es", "or",
            "te", "of", "be", "ha", "as", "is", "wa", "et", "it", "on", "me", "at",
            # Vowel patterns
            "ea", "ai", "ie", "ou", "oo", "ee", "oa", "au", "oi", "ue", "ui", "eo",
            # Consonant clusters
            "st", "nd", "nt", "rt", "ch", "sh", "th", "wh", "ph", "gh", "ck", "ng"
        ]
        
        text = ""
        for _ in range(size // 4):  # Each pattern contributes ~4 chars
            if len(text) >= size:
                break
            pattern = random.choice(patterns)
            # Add some context around patterns
            prefix = random.choice(['', 'a', 'e', 'i', 'o', 'u'])
            suffix = random.choice(['', 'a', 'e', 'i', 'o', 'u', 's', 'd', 'r', 'n'])
            text += prefix + pattern + suffix + " "
        
        return text[:size]
    
    @staticmethod
    def _get_pattern_rich_text(size):
        """Get text rich in character patterns"""
        try:
            # Use children's books or simple texts with good patterns
            dataset = load_dataset("roneneldan/TinyStories", trust_remote_code=True)
            text = ""
            for story in dataset["train"]:
                if len(text) >= size:
                    break
                text += story["text"] + " "
            return text[:size]
        except:
            # Fallback to simple pattern text
            return "hello world test data pattern example simple text for training character prediction models " * (size // 100)
    
    @staticmethod
    def _enhance_with_patterns(text):
        """Enhance text by repeating common character patterns"""
        # Find common 2-3 character patterns
        pattern_counts = Counter()
        for i in range(len(text) - 2):
            pattern_counts[text[i:i+3]] += 1
        
        # Get most common patterns
        common_patterns = [p for p, c in pattern_counts.most_common(50) if c > 10]
        
        # Add repetitions of common patterns
        enhanced = text
        for pattern in common_patterns[:20]:  # Top 20 patterns
            enhanced += f" {pattern} " * min(10, pattern_counts[pattern] // 10)
        
        return enhanced

def create_smart_vocab(text):
    """Create vocabulary intelligently based on character frequency and type"""
    print("🧠 Creating intelligent vocabulary...")
    
    # Count character frequencies
    char_counts = Counter(text)
    
    # Categorize characters
    ascii_chars = []
    unicode_chars = []
    
    for char, count in char_counts.items():
        if ord(char) < 128:  # ASCII
            ascii_chars.append((char, count))
        else:  # Unicode (multilingual)
            unicode_chars.append((char, count))
    
    # Sort by frequency
    ascii_chars.sort(key=lambda x: x[1], reverse=True)
    unicode_chars.sort(key=lambda x: x[1], reverse=True)
    
    # Select most important characters
    vocab = set()
    
    # Always include basic ASCII
    for char, count in ascii_chars:
        if len(vocab) < 150:  # Reserve space for Unicode
            vocab.add(char)
        elif count > 10:  # Only frequent ASCII beyond limit
            vocab.add(char)
    
    # Add most frequent Unicode characters
    for char, count in unicode_chars[:100]:  # Top 100 Unicode chars
        if count > 5:  # Only reasonably frequent
            vocab.add(char)
    
    # Convert to sorted list
    vocab_list = sorted(list(vocab))
    
    print(f"🧠 Smart vocabulary created: {len(vocab_list)} characters")
    print(f"   ASCII: {sum(1 for c in vocab_list if ord(c) < 128)}")
    print(f"   Unicode: {sum(1 for c in vocab_list if ord(c) >= 128)}")
    
    return vocab_list

class MyModel:
    """Compatible with original structure but with adaptive improvements"""
    model = None
    char2idx = None
    idx2char = None
    vocab = None

    @classmethod
    def load_training_data(cls):
        print("🚀 Loading character-level language model data...")
        
        # Use pattern-focused data for better character prediction
        text_data = CharacterPatternTrainer.create_pattern_focused_data(target_size=300000)
        
        # Create smart vocabulary
        cls.vocab = create_smart_vocab(text_data)
        cls.char2idx = {char: idx for idx, char in enumerate(cls.vocab)}
        cls.idx2char = {idx: char for idx, char in enumerate(cls.vocab)}
        
        print(f"✅ Dataset loaded: {len(text_data)} characters, {len(cls.vocab)} unique characters")
        return text_data

    @classmethod
    def load_test_data(cls, fname):
        """Load test data from file"""
        data = []
        with open(fname) as f:
            for line in f:
                inp = line.rstrip('\n')
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        """Write predictions to file"""
        with open(fname, 'wt') as f:
            for p in preds:
                f.write(f"{p}\n")

    def run_train(self, data, work_dir):
        # Use adaptive configuration for character prediction
        config = AdaptiveConfig(len(self.char2idx), len(data), DEVICE)
        dataset = AdaptiveCharDataset(data, config, self.char2idx)
        
        # Create data loader optimized for character learning
        data_loader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2 if DEVICE == 'cuda' else 0,
            pin_memory=True if DEVICE == 'cuda' else False,
            drop_last=True
        )

        # Create multilingual embeddings with proper size
        emb_size = config.emb_size
        embeddings = MultilingualEmbeddingLoader.create_unicode_aware_embeddings(self.vocab, emb_size)

        # Initialize model
        self.model = AdaptiveCharLSTM(
            len(self.char2idx), 
            config,
            pretrained_embeddings=embeddings
        ).to(DEVICE)
        
        print(f"🎯 Character-focused model initialized:")
        print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Focus: Character pattern learning")
        
        # Loss function and optimizer optimized for character prediction
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=config.lr, weight_decay=0.005)
        
        # Focused learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        
        # Training variables
        best_loss = float('inf')
        patience_counter = 0
        
        self.model.train()
        
        for epoch in range(config.epochs):
            total_loss = 0
            batch_count = 0
            epoch_correct = 0
            epoch_total = 0
            
            print(f"\n🎯 Epoch {epoch+1}/{config.epochs} - Character Pattern Learning...")
            
            for batch_idx, (input_seq, target) in enumerate(data_loader):
                input_seq, target = input_seq.to(DEVICE, non_blocking=True), target.to(DEVICE, non_blocking=True)
                
                optimizer.zero_grad()
                output = self.model(input_seq)
                loss = criterion(output, target)
                loss.backward()
                
                # Focused gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
                # Calculate accuracy
                pred = output.argmax(dim=1)
                epoch_correct += (pred == target).sum().item()
                epoch_total += target.size(0)
                
                if batch_count % 50 == 0:
                    avg_loss = total_loss / batch_count
                    acc = epoch_correct / epoch_total * 100
                    print(f"   Batch {batch_count}: Loss={avg_loss:.4f}, Acc={acc:.1f}%")
            
            # Epoch statistics
            epoch_loss = total_loss / batch_count
            epoch_acc = epoch_correct / epoch_total * 100
            scheduler.step()
            
            print(f"\n📊 Epoch {epoch+1}/{config.epochs}: Loss={epoch_loss:.4f}, Acc={epoch_acc:.2f}%")
            
            # Save best model with more patience for pattern learning
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
                
                checkpoint = {
                    'model_state_dict': self.model.state_dict(),
                    'char2idx': self.char2idx,
                    'idx2char': self.idx2char,
                    'vocab': self.vocab,
                    'embeddings': embeddings,
                    'config': {
                        'emb_size': config.emb_size,
                        'hidden_size': config.hidden_size,
                        'num_layers': config.num_layers,
                        'seq_len': config.seq_len,
                        'dropout': config.dropout
                    }
                }
                torch.save(checkpoint, os.path.join(work_dir, 'model.checkpoint'))
                print(f"   💾 Best character model saved (loss: {best_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= 6:  # More patience for character learning
                    print(f"   ⏰ Early stopping triggered after {epoch + 1} epochs")
                    break
        
        print(f"\n🎯 Character-focused training completed! Best loss: {best_loss:.4f}")

    def run_pred(self, data):
        """Generate predictions for the given data"""
        if self.model is None:
            print("❌ Error: No trained model available for prediction")
            return []
        
        preds = []
        self.model.eval()
        
        # Get sequence length from model config or use reasonable default
        seq_len = getattr(self.model, 'config', None)
        if seq_len and hasattr(seq_len, 'seq_len'):
            seq_len = seq_len.seq_len
        else:
            seq_len = 32  # Default fallback
        
        with torch.no_grad():
            for context in data:
                # Use adaptive context length
                context = context[-seq_len:] if len(context) > seq_len else context
                
                # Pad context if too short
                if len(context) < seq_len:
                    context = ' ' * (seq_len - len(context)) + context
                
                context_encoded = torch.tensor(
                    [[self.char2idx.get(c, 0) for c in context]], 
                    dtype=torch.long
                ).to(DEVICE)
                
                output = self.model(context_encoded)
                
                # Get top 3 predictions with smart filtering
                probs = torch.softmax(output, dim=1)
                top_probs, top_indices = torch.topk(probs, k=min(10, len(self.vocab)), dim=1)
                
                # Filter and select top 3
                top_chars = []
                for i in range(top_indices.size(1)):
                    idx = top_indices[0, i].item()
                    prob = top_probs[0, i].item()
                    char = self.idx2char[idx]
                    
                    # Smart filtering: prefer printable, meaningful characters
                    if (len(top_chars) < 3 and 
                        char not in top_chars and 
                        prob > 0.005 and
                        (char.isprintable() or char in [' ', '\t', '\n'])):
                        top_chars.append(char)
                
                # Ensure we have 3 predictions
                while len(top_chars) < 3:
                    remaining_chars = [c for c in self.vocab 
                                     if c not in top_chars and c.isprintable()]
                    if remaining_chars:
                        top_chars.append(random.choice(remaining_chars))
                    else:
                        top_chars.append(' ')
                
                pred_string = "".join(top_chars[:3])
                preds.append(pred_string)
        
        return preds

    def save(self, work_dir):
        """Save is handled automatically during training"""
        pass  # Model is saved during training as model.checkpoint

    @classmethod
    def load(cls, work_dir):
        """Load a trained model from the work directory"""
        checkpoint_path = os.path.join(work_dir, 'model.checkpoint')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No model checkpoint found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        model = cls()
        
        # Load vocabulary and mappings
        model.char2idx = checkpoint['char2idx']
        model.idx2char = checkpoint['idx2char']
        model.vocab = checkpoint.get('vocab', sorted(model.char2idx.keys()))
        
        # Load embeddings
        embeddings = checkpoint.get('embeddings', None)
        if embeddings is None:
            print("⚠️ No embeddings in checkpoint, recreating...")
            emb_size = checkpoint.get('config', {}).get('emb_size', 64)
            print(f"🔧 Recreating embeddings with size: {emb_size}")
            embeddings = MultilingualEmbeddingLoader.create_unicode_aware_embeddings(model.vocab, emb_size)
        
        # Create model with saved configuration
        if 'config' in checkpoint:
            saved_config = checkpoint['config']
            print(f"📊 Loading model with saved configuration:")
            for key, value in saved_config.items():
                print(f"   {key}: {value}")
            
            # Create a temporary config with saved parameters
            temp_config = AdaptiveConfig(len(model.char2idx), len(model.vocab), DEVICE)
            temp_config.emb_size = saved_config.get('emb_size', temp_config.emb_size)
            temp_config.hidden_size = saved_config.get('hidden_size', temp_config.hidden_size)
            temp_config.num_layers = saved_config.get('num_layers', temp_config.num_layers)
            temp_config.seq_len = saved_config.get('seq_len', temp_config.seq_len)
            temp_config.dropout = saved_config.get('dropout', temp_config.dropout)
            
            model.model = AdaptiveCharLSTM(
                len(model.char2idx), 
                temp_config,
                pretrained_embeddings=embeddings
            ).to(DEVICE)
        else:
            print("⚠️ Using adaptive configuration (may cause compatibility issues)")
            config = AdaptiveConfig(len(model.char2idx), len(model.vocab), DEVICE)
            
            # If embeddings were recreated, make sure they match the adaptive config size
            if embeddings is not None and embeddings.shape[1] != config.emb_size:
                print(f"🔧 Recreating embeddings to match adaptive config size: {config.emb_size}")
                embeddings = MultilingualEmbeddingLoader.create_unicode_aware_embeddings(model.vocab, config.emb_size)
            
            model.model = AdaptiveCharLSTM(
                len(model.char2idx), 
                config,
                pretrained_embeddings=embeddings
            ).to(DEVICE)
        
        # Load model state
        try:
            model.model.load_state_dict(checkpoint['model_state_dict'])
            print("✅ Model state loaded successfully")
        except Exception as e:
            print(f"⚠️ Model state loading failed: {e}")
            print("   This may happen when hyperparameters change between training and loading")
            raise e
            
        model.model.eval()
        print(f"✅ Model loaded with {sum(p.numel() for p in model.model.parameters()):,} parameters")
        
        return model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print(f'Making working directory {args.work_dir}')
            os.makedirs(args.work_dir)
        print('Instantiating model')
        model = MyModel()
        print('Loading training data')
        train_data = MyModel.load_training_data()
        print('Training')
        model.run_train(train_data, args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print(f'Loading test data from {args.test_data}')
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print(f'Writing predictions to {args.test_output}')
        assert len(pred) == len(test_data), f'Expected {len(test_data)} predictions but got {len(pred)}'
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError(f'Unknown mode {args.mode}')

