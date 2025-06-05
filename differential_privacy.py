import numpy as np
from typing import List, Dict, Any
import math

class DifferentialPrivacyManager:
    """Manages differential privacy for federated learning"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.noise_scale = self._calculate_noise_scale()
    
    def _calculate_noise_scale(self):
        """Calculate noise scale for Gaussian mechanism"""
        # For (ε, δ)-differential privacy using Gaussian mechanism
        if self.delta == 0:
            # Pure ε-differential privacy (Laplace mechanism)
            return self.sensitivity / self.epsilon
        else:
            # (ε, δ)-differential privacy (Gaussian mechanism)
            c = math.sqrt(2 * math.log(1.25 / self.delta))
            return c * self.sensitivity / self.epsilon
    
    def add_noise(self, client_updates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add differential privacy noise to client updates"""
        if not client_updates:
            return client_updates
        
        noisy_updates = []
        
        for update in client_updates:
            try:
                noisy_update = self._add_noise_to_update(update)
                noisy_updates.append(noisy_update)
            except Exception as e:
                print(f"DP noise addition failed for client {update.get('client_id', 'unknown')}: {e}")
                # Include original update if noise addition fails
                noisy_updates.append(update)
        
        return noisy_updates
    
    def _add_noise_to_update(self, update: Dict[str, Any]) -> Dict[str, Any]:
        """Add noise to a single client update"""
        noisy_update = update.copy()
        
        parameters = update['parameters']
        
        if self.delta == 0:
            # Laplace noise for pure ε-differential privacy
            noise = np.random.laplace(0, self.noise_scale, size=parameters.shape)
        else:
            # Gaussian noise for (ε, δ)-differential privacy
            noise = np.random.normal(0, self.noise_scale, size=parameters.shape)
        
        # Add noise to parameters
        noisy_parameters = parameters + noise
        noisy_update['parameters'] = noisy_parameters
        
        # Add metadata about privacy
        noisy_update['dp_applied'] = True
        noisy_update['epsilon'] = self.epsilon
        noisy_update['delta'] = self.delta
        noisy_update['noise_scale'] = self.noise_scale
        
        return noisy_update
    
    def clip_gradients(self, parameters: np.ndarray, clip_norm: float = 1.0) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        param_norm = np.linalg.norm(parameters)
        
        if param_norm > clip_norm:
            return parameters * (clip_norm / param_norm)
        else:
            return parameters
    
    def add_noise_to_aggregated_model(self, parameters: np.ndarray) -> np.ndarray:
        """Add noise directly to aggregated model parameters"""
        if self.delta == 0:
            noise = np.random.laplace(0, self.noise_scale, size=parameters.shape)
        else:
            noise = np.random.normal(0, self.noise_scale, size=parameters.shape)
        
        return parameters + noise
    
    def calculate_privacy_budget(self, num_rounds: int, composition_method: str = 'advanced') -> Dict[str, float]:
        """Calculate privacy budget consumption over multiple rounds"""
        if composition_method == 'basic':
            # Basic composition
            total_epsilon = num_rounds * self.epsilon
            total_delta = num_rounds * self.delta
        elif composition_method == 'advanced':
            # Advanced composition (tighter bounds)
            if self.delta > 0:
                # Using moments accountant approximation
                total_epsilon = self.epsilon * math.sqrt(num_rounds * math.log(1/self.delta))
                total_delta = num_rounds * self.delta
            else:
                total_epsilon = num_rounds * self.epsilon
                total_delta = 0
        else:
            # Conservative estimate
            total_epsilon = num_rounds * self.epsilon
            total_delta = num_rounds * self.delta
        
        return {
            'total_epsilon': total_epsilon,
            'total_delta': total_delta,
            'rounds': num_rounds,
            'per_round_epsilon': self.epsilon,
            'per_round_delta': self.delta
        }
    
    def get_privacy_parameters(self) -> Dict[str, float]:
        """Get current privacy parameters"""
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'sensitivity': self.sensitivity,
            'noise_scale': self.noise_scale
        }
    
    def update_privacy_parameters(self, epsilon: float = None, delta: float = None, 
                                 sensitivity: float = None):
        """Update privacy parameters"""
        if epsilon is not None:
            self.epsilon = epsilon
        if delta is not None:
            self.delta = delta
        if sensitivity is not None:
            self.sensitivity = sensitivity
        
        # Recalculate noise scale
        self.noise_scale = self._calculate_noise_scale()

class LocalDifferentialPrivacy:
    """Local differential privacy for individual client data"""
    
    def __init__(self, epsilon=1.0):
        self.epsilon = epsilon
    
    def randomized_response(self, data: np.ndarray, probability: float = None) -> np.ndarray:
        """Apply randomized response mechanism"""
        if probability is None:
            # Calculate optimal probability for binary data
            probability = math.exp(self.epsilon) / (1 + math.exp(self.epsilon))
        
        # Apply randomized response
        random_mask = np.random.random(data.shape) < probability
        noisy_data = np.where(random_mask, data, 1 - data)
        
        return noisy_data
    
    def add_laplace_noise(self, data: np.ndarray, sensitivity: float = 1.0) -> np.ndarray:
        """Add Laplace noise for numerical data"""
        noise_scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, noise_scale, size=data.shape)
        
        return data + noise
    
    def privatize_counts(self, counts: Dict[Any, int], domain_size: int) -> Dict[Any, float]:
        """Privatize count queries"""
        noisy_counts = {}
        noise_scale = 1.0 / self.epsilon
        
        for key, count in counts.items():
            noise = np.random.laplace(0, noise_scale)
            noisy_counts[key] = max(0, count + noise)  # Ensure non-negative
        
        return noisy_counts
