"""
Deep Learning Models for Predictive Quantum Stability
Implements LSTM, Transformer, and TCN architectures for decoherence prediction
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

# Check if CUDA is available
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class ModelConfig:
    """Configuration for deep learning models"""
    # Common parameters
    input_size: int = 30
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.3
    bidirectional: bool = True
    
    # LSTM specific
    lstm_type: str = 'lstm'  # 'lstm', 'gru'
    
    # Transformer specific
    num_heads: int = 4
    dim_feedforward: int = 256
    max_seq_length: int = 100
    
    # TCN specific
    kernel_size: int = 3
    dilation_base: int = 2
    
    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    num_epochs: int = 50
    weight_decay: float = 1e-5


class LSTMPredictor(nn.Module):
    """
    LSTM-based model for temporal pattern recognition in qubit telemetry
    """
    
    def __init__(self, config: ModelConfig):
        super(LSTMPredictor, self).__init__()
        
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(config.hidden_size * (2 if config.bidirectional else 1), 
                     config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 1)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * (2 if config.bidirectional else 1), 
                     config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, 1) - probability of decoherence
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden*2)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Weighted sum
        context = torch.sum(lstm_out * attention_weights, dim=1)  # (batch, hidden*2)
        
        # Classification
        output = self.classifier(context)
        
        return output


class TransformerPredictor(nn.Module):
    """
    Transformer-based model for long-range dependency modeling
    """
    
    def __init__(self, config: ModelConfig):
        super(TransformerPredictor, self).__init__()
        
        self.config = config
        
        # Input embedding
        self.input_embedding = nn.Linear(config.input_size, config.hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(
            d_model=config.hidden_size,
            max_len=config.max_seq_length
        )
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_layers
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        # Embedding
        x = self.input_embedding(x)  # (batch, seq_len, hidden)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer encoder
        x = self.transformer_encoder(x)  # (batch, seq_len, hidden)
        
        # Global average pooling
        x = x.transpose(1, 2)  # (batch, hidden, seq_len)
        x = self.global_pool(x).squeeze(-1)  # (batch, hidden)
        
        # Classification
        output = self.classifier(x)
        
        return output


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TemporalConvNet(nn.Module):
    """
    Temporal Convolutional Network (TCN) for efficient sequence modeling
    """
    
    def __init__(self, config: ModelConfig):
        super(TemporalConvNet, self).__init__()
        
        self.config = config
        
        # Build TCN layers with dilated convolutions
        layers = []
        num_levels = config.num_layers
        
        in_channels = config.input_size
        out_channels = config.hidden_size
        
        for i in range(num_levels):
            dilation = config.dilation_base ** i
            padding = (config.kernel_size - 1) * dilation // 2
            
            # Dilated causal convolution
            conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=config.kernel_size,
                padding=padding,
                dilation=dilation,
                bias=False
            )
            
            # Weight norm
            conv = nn.utils.weight_norm(conv)
            
            layers.append(conv)
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(config.dropout))
            
            in_channels = out_channels
            
        self.tcn = nn.Sequential(*layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(out_channels, out_channels // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(out_channels // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor of shape (batch, seq_len, input_size)
            
        Returns:
            Output tensor of shape (batch, 1)
        """
        # Transpose for Conv1d: (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        
        # TCN forward
        x = self.tcn(x)  # (batch, hidden, seq_len)
        
        # Take the last time step
        x = x[:, :, -1]  # (batch, hidden)
        
        # Classification
        output = self.classifier(x)
        
        return output


class EnsemblePredictor:
    """
    Ensemble of multiple models for robust prediction
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        """
        Initialize ensemble
        
        Args:
            models: List of trained models
            weights: Optional weights for each model (default: equal weights)
        """
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Make ensemble prediction
        
        Args:
            x: Input tensor
            
        Returns:
            Ensemble prediction probabilities
        """
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            model.eval()
            with torch.no_grad():
                pred = model(x).cpu().numpy()
                predictions.append(pred * weight)
                
        ensemble_pred = np.sum(predictions, axis=0)
        
        return ensemble_pred


class DeepLearningTrainer:
    """
    Trainer for deep learning models
    """
    
    def __init__(self, 
                 model: nn.Module,
                 config: ModelConfig,
                 device: torch.device = DEVICE):
        """
        Initialize trainer
        
        Args:
            model: PyTorch model to train
            config: Model configuration
            device: Device to use for training
        """
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Loss function with class weighting
        self.criterion = nn.BCELoss()
        
        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        
    def prepare_data(self, 
                    X: np.ndarray, 
                    y: np.ndarray,
                    val_split: float = 0.2,
                    shuffle: bool = True) -> Tuple[DataLoader, DataLoader]:
        """
        Prepare data loaders
        
        Args:
            X: Feature array (samples, seq_len, features) or (samples, features)
            y: Labels array
            val_split: Validation split ratio
            shuffle: Whether to shuffle data
            
        Returns:
            Train and validation data loaders
        """
        # Convert to tensors
        if len(X.shape) == 2:
            # Reshape for sequence models: add sequence dimension
            X = X.reshape(X.shape[0], 1, X.shape[1])
            
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        
        # Split data
        num_samples = X.shape[0]
        num_val = int(num_samples * val_split)
        indices = np.arange(num_samples)
        
        if shuffle:
            np.random.shuffle(indices)
            
        train_indices = indices[num_val:]
        val_indices = indices[:num_val]
        
        X_train = X_tensor[train_indices]
        y_train = y_tensor[train_indices]
        X_val = X_tensor[val_indices]
        y_val = y_tensor[val_indices]
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item() * batch_X.size(0)
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_X.size(0)
            
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                
                total_loss += loss.item() * batch_X.size(0)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_X.size(0)
                
        avg_loss = total_loss / total
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader,
              val_loader: DataLoader,
              num_epochs: Optional[int] = None) -> Dict:
        """
        Full training loop
        
        Returns:
            Training history
        """
        num_epochs = num_epochs or self.config.num_epochs
        
        print(f"Training on {self.device}")
        print(f"Epochs: {num_epochs}, Batch size: {self.config.batch_size}")
        print("-" * 60)
        
        best_val_loss = float('inf')
        patience_counter = 0
        patience = 10
        
        for epoch in range(num_epochs):
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print progress
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs}: "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                    
        print("-" * 60)
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Final validation accuracy: {self.history['val_acc'][-1]:.4f}")
        
        return self.history
    
    def load_best_model(self, path: str = 'best_model.pth'):
        """Load the best model from checkpoint"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print(f"Loaded best model from {path}")


def create_sample_sequence_data(num_samples: int = 1000,
                               seq_length: int = 50,
                               num_features: int = 18,
                               noise_level: float = 0.1,
                               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic sequential data for testing deep learning models
    
    Args:
        num_samples: Number of samples
        seq_length: Length of each sequence
        num_features: Number of features per time step
        noise_level: Amount of noise to add
        seed: Random seed
        
    Returns:
        X: Sequence data (num_samples, seq_length, num_features)
        y: Binary labels
    """
    if seed is not None:
        np.random.seed(seed)
        
    X = np.zeros((num_samples, seq_length, num_features))
    y = np.zeros(num_samples)
    
    for i in range(num_samples):
        # Generate base signal with different patterns for positive/negative samples
        label = np.random.randint(0, 2)
        y[i] = label
        
        if label == 1:
            # Positive sample: increasing trend with fluctuations (precursor pattern)
            base_trend = np.linspace(0, 1, seq_length)
            pattern = base_trend + 0.3 * np.sin(np.linspace(0, 4*np.pi, seq_length))
        else:
            # Negative sample: stable signal
            base_trend = np.ones(seq_length) * 0.5
            pattern = base_trend + 0.1 * np.sin(np.linspace(0, 2*np.pi, seq_length))
            
        # Add feature variations
        for f in range(num_features):
            feature_pattern = pattern * (1 + 0.1 * f)
            noise = np.random.normal(0, noise_level, seq_length)
            X[i, :, f] = feature_pattern + noise
            
        # Add some temporal correlations
        if seq_length > 1:
            X[i, 1:, :] += 0.3 * X[i, :-1, :]
            
    return X, y


if __name__ == "__main__":
    print("=" * 60)
    print("Deep Learning Models for PQSS")
    print("=" * 60)
    
    # Configuration
    config = ModelConfig(
        input_size=18,
        hidden_size=64,
        num_layers=2,
        dropout=0.3,
        batch_size=32,
        num_epochs=30
    )
    
    print(f"\nModel Configuration:")
    print(f"  Input size: {config.input_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Num layers: {config.num_layers}")
    print(f"  Device: {DEVICE}")
    
    # Generate synthetic sequential data
    print("\nGenerating synthetic sequential data...")
    X, y = create_sample_sequence_data(
        num_samples=500,
        seq_length=50,
        num_features=18,
        seed=42
    )
    
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"Class distribution: {np.sum(y)} positive, {len(y) - np.sum(y)} negative")
    
    # Test LSTM model
    print("\n" + "=" * 60)
    print("Testing LSTM Predictor")
    print("=" * 60)
    
    lstm_model = LSTMPredictor(config)
    lstm_trainer = DeepLearningTrainer(lstm_model, config)
    
    train_loader, val_loader = lstm_trainer.prepare_data(X, y, val_split=0.2)
    
    print("\nTraining LSTM model...")
    lstm_history = lstm_trainer.train(train_loader, val_loader, num_epochs=20)
    
    # Test Transformer model
    print("\n" + "=" * 60)
    print("Testing Transformer Predictor")
    print("=" * 60)
    
    transformer_config = ModelConfig(
        input_size=18,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        dim_feedforward=128,
        dropout=0.3,
        batch_size=32,
        num_epochs=20
    )
    
    transformer_model = TransformerPredictor(transformer_config)
    transformer_trainer = DeepLearningTrainer(transformer_model, transformer_config)
    
    train_loader, val_loader = transformer_trainer.prepare_data(X, y, val_split=0.2)
    
    print("\nTraining Transformer model...")
    transformer_history = transformer_trainer.train(train_loader, val_loader, num_epochs=20)
    
    # Test TCN model
    print("\n" + "=" * 60)
    print("Testing Temporal Convolutional Network")
    print("=" * 60)
    
    tcn_config = ModelConfig(
        input_size=18,
        hidden_size=64,
        num_layers=3,
        kernel_size=3,
        dropout=0.3,
        batch_size=32,
        num_epochs=20
    )
    
    tcn_model = TemporalConvNet(tcn_config)
    tcn_trainer = DeepLearningTrainer(tcn_model, tcn_config)
    
    train_loader, val_loader = tcn_trainer.prepare_data(X, y, val_split=0.2)
    
    print("\nTraining TCN model...")
    tcn_history = tcn_trainer.train(train_loader, val_loader, num_epochs=20)
    
    # Create ensemble
    print("\n" + "=" * 60)
    print("Creating Ensemble Predictor")
    print("=" * 60)
    
    # Load best models
    try:
        lstm_trainer.load_best_model()
        transformer_trainer.load_best_model()
        tcn_trainer.load_best_model()
        
        ensemble = EnsemblePredictor([
            lstm_model,
            transformer_model,
            tcn_model
        ])
        
        # Test ensemble on validation set
        X_val = torch.FloatTensor(X[-100:].reshape(-1, 50, 18))
        y_val = y[-100:]
        
        ensemble_pred = ensemble.predict(X_val)
        ensemble_acc = np.mean((ensemble_pred > 0.5).flatten() == y_val)
        
        print(f"\nEnsemble accuracy on test set: {ensemble_acc:.4f}")
        
    except Exception as e:
        print(f"Could not create ensemble: {e}")
    
    print("\n" + "=" * 60)
    print("Deep Learning Models Training Complete")
    print("=" * 60)
