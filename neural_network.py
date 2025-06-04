import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Optional

class DiabetesNN(nn.Module):
    """Neural network for diabetes prediction"""
    
    def __init__(self, input_size: int, hidden_sizes: List[int] = [64, 32], 
                 output_size: int = 1, dropout_rate: float = 0.2):
        """
        Initialize the neural network
        
        Args:
            input_size: Number of input features
            hidden_sizes: List of hidden layer sizes
            output_size: Number of output units
            dropout_rate: Dropout rate for regularization
        """
        super(DiabetesNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.dropout_rate = dropout_rate
        
        # Build network layers
        layers = []
        current_size = input_size
        
        # Hidden layers
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(current_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                nn.Dropout(dropout_rate)
            ])
            current_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(current_size, output_size))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        
        # Store layer references for easy access
        self.fc1 = layers[0]  # First linear layer
        
        # Initialize weights
        self._initialize_weights()
        
        print(f"Initialized DiabetesNN: {input_size} -> {hidden_sizes} -> {output_size}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        return self.network(x)
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def get_layer_weights(self) -> dict:
        """
        Get weights from all layers
        
        Returns:
            Dictionary of layer weights
        """
        weights = {}
        for name, param in self.named_parameters():
            weights[name] = param.data.clone()
        return weights
    
    def set_layer_weights(self, weights: dict):
        """
        Set weights for all layers
        
        Args:
            weights: Dictionary of layer weights
        """
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in weights:
                    param.copy_(weights[name])
    
    def get_gradients(self) -> dict:
        """
        Get gradients from all parameters
        
        Returns:
            Dictionary of parameter gradients
        """
        gradients = {}
        for name, param in self.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
            else:
                gradients[name] = torch.zeros_like(param.data)
        return gradients
    
    def compute_loss(self, outputs: torch.Tensor, targets: torch.Tensor, 
                    loss_type: str = 'bce') -> torch.Tensor:
        """
        Compute loss for given outputs and targets
        
        Args:
            outputs: Model outputs
            targets: Target values
            loss_type: Type of loss function ('bce', 'mse')
            
        Returns:
            Computed loss
        """
        if loss_type.lower() == 'bce':
            criterion = nn.BCELoss()
        elif loss_type.lower() == 'mse':
            criterion = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")
        
        return criterion(outputs, targets)
    
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """
        Make binary predictions
        
        Args:
            x: Input tensor
            threshold: Decision threshold
            
        Returns:
            Binary predictions
        """
        with torch.no_grad():
            outputs = self.forward(x)
            predictions = (outputs > threshold).float()
        return predictions
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities
        
        Args:
            x: Input tensor
            
        Returns:
            Prediction probabilities
        """
        with torch.no_grad():
            probabilities = self.forward(x)
        return probabilities
    
    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> dict:
        """
        Evaluate model performance
        
        Args:
            x: Input features
            y: Target labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        with torch.no_grad():
            # Get predictions
            outputs = self.forward(x)
            predictions = (outputs > 0.5).float()
            
            # Calculate metrics
            accuracy = (predictions == y.unsqueeze(1) if y.dim() == 1 else y).float().mean().item()
            loss = self.compute_loss(outputs, y.unsqueeze(1) if y.dim() == 1 else y).item()
            
            # Calculate additional metrics
            tp = ((predictions == 1) & (y.unsqueeze(1) == 1)).sum().item()
            tn = ((predictions == 0) & (y.unsqueeze(1) == 0)).sum().item()
            fp = ((predictions == 1) & (y.unsqueeze(1) == 0)).sum().item()
            fn = ((predictions == 0) & (y.unsqueeze(1) == 1)).sum().item()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            return {
                'accuracy': accuracy,
                'loss': loss,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'true_positives': tp,
                'true_negatives': tn,
                'false_positives': fp,
                'false_negatives': fn
            }
    
    def get_model_size(self) -> int:
        """
        Get total number of parameters in the model
        
        Returns:
            Number of parameters
        """
        return sum(p.numel() for p in self.parameters())
    
    def get_model_info(self) -> dict:
        """
        Get detailed model information
        
        Returns:
            Dictionary of model information
        """
        info = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'dropout_rate': self.dropout_rate,
            'total_parameters': self.get_model_size(),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }
        
        # Layer-wise parameter count
        layer_params = {}
        for name, param in self.named_parameters():
            layer_params[name] = param.numel()
        info['layer_parameters'] = layer_params
        
        return info
    
    def freeze_layers(self, layer_names: List[str]):
        """
        Freeze specified layers
        
        Args:
            layer_names: List of layer names to freeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = False
                print(f"Frozen layer: {name}")
    
    def unfreeze_layers(self, layer_names: List[str]):
        """
        Unfreeze specified layers
        
        Args:
            layer_names: List of layer names to unfreeze
        """
        for name, param in self.named_parameters():
            if any(layer_name in name for layer_name in layer_names):
                param.requires_grad = True
                print(f"Unfrozen layer: {name}")
    
    def get_activation_maps(self, x: torch.Tensor, layer_names: Optional[List[str]] = None) -> dict:
        """
        Get activation maps from specified layers
        
        Args:
            x: Input tensor
            layer_names: List of layer names to extract activations from
            
        Returns:
            Dictionary of activation maps
        """
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in self.named_modules():
            if layer_names is None or name in layer_names:
                if isinstance(module, (nn.Linear, nn.ReLU, nn.BatchNorm1d)):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
        
        # Forward pass
        with torch.no_grad():
            _ = self.forward(x)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return activations
    
    def save_model(self, filepath: str):
        """
        Save model state
        
        Args:
            filepath: Path to save the model
        """
        torch.save({
            'state_dict': self.state_dict(),
            'model_config': {
                'input_size': self.input_size,
                'hidden_sizes': self.hidden_sizes,
                'output_size': self.output_size,
                'dropout_rate': self.dropout_rate
            }
        }, filepath)
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load_model(cls, filepath: str) -> 'DiabetesNN':
        """
        Load model from file
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded DiabetesNN instance
        """
        checkpoint = torch.load(filepath)
        config = checkpoint['model_config']
        
        model = cls(
            input_size=config['input_size'],
            hidden_sizes=config['hidden_sizes'],
            output_size=config['output_size'],
            dropout_rate=config['dropout_rate']
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        print(f"Model loaded from {filepath}")
        
        return model
