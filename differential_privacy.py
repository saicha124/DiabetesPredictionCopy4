import numpy as np
from typing import List, Dict, Any, Tuple
import math
import scipy.stats as stats
from abc import ABC, abstractmethod

class PrivacyMechanism(ABC):
    """Abstract base class for privacy mechanisms"""
    
    @abstractmethod
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 0) -> np.ndarray:
        pass

class GaussianMechanism(PrivacyMechanism):
    """Gaussian mechanism for differential privacy"""
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 1e-5) -> np.ndarray:
        if delta == 0:
            raise ValueError("Gaussian mechanism requires delta > 0")
        
        sigma = math.sqrt(2 * math.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

class LaplaceMechanism(PrivacyMechanism):
    """Laplace mechanism for differential privacy"""
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 0) -> np.ndarray:
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

class ExponentialMechanism(PrivacyMechanism):
    """Exponential mechanism for differential privacy"""
    
    def add_noise(self, data: np.ndarray, sensitivity: float, epsilon: float, delta: float = 0) -> np.ndarray:
        # Simplified exponential mechanism implementation
        probabilities = np.exp(epsilon * data / (2 * sensitivity))
        probabilities = probabilities / np.sum(probabilities)
        
        # Sample based on probabilities
        indices = np.random.choice(len(data), size=len(data), p=probabilities)
        return data[indices]

class PrivacyAccountant:
    """Advanced privacy accounting for composition"""
    
    def __init__(self, total_epsilon: float, total_delta: float):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.query_history = []
    
    def consume_privacy_budget(self, epsilon: float, delta: float = 0) -> bool:
        """Check if privacy budget allows this query"""
        if self.used_epsilon + epsilon > self.total_epsilon:
            return False
        if self.used_delta + delta > self.total_delta:
            return False
        
        self.used_epsilon += epsilon
        self.used_delta += delta
        self.query_history.append({'epsilon': epsilon, 'delta': delta})
        return True
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return self.total_epsilon - self.used_epsilon, self.total_delta - self.used_delta

class DifferentialPrivacyManager:
    """Advanced differential privacy manager for federated learning"""
    
    def __init__(self, epsilon=1.0, delta=1e-5, sensitivity=1.0, mechanism='gaussian'):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism_type = mechanism
        self.gradient_clip_norm = 1.0
        self.privacy_accountant = PrivacyAccountant(epsilon, delta)
        
        # Initialize privacy mechanism
        if mechanism == 'gaussian':
            self.mechanism = GaussianMechanism()
        elif mechanism == 'laplace':
            self.mechanism = LaplaceMechanism()
        elif mechanism == 'exponential':
            self.mechanism = ExponentialMechanism()
        else:
            self.mechanism = GaussianMechanism()
        
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
        """Add differential privacy noise to client updates with advanced mechanisms"""
        if not client_updates:
            return client_updates
        
        noisy_updates = []
        
        # Calculate per-round privacy budget
        epsilon_per_round = self.epsilon / len(client_updates) if len(client_updates) > 0 else self.epsilon
        delta_per_round = self.delta / len(client_updates) if len(client_updates) > 0 else self.delta
        
        for update in client_updates:
            try:
                # Check privacy budget
                if not self.privacy_accountant.consume_privacy_budget(epsilon_per_round, delta_per_round):
                    print(f"Privacy budget exhausted for client {update.get('client_id', 'unknown')}")
                    noisy_updates.append(update)  # Return original if budget exhausted
                    continue
                
                noisy_update = self._add_noise_to_update_advanced(update, epsilon_per_round, delta_per_round)
                noisy_updates.append(noisy_update)
            except Exception as e:
                print(f"DP noise addition failed for client {update.get('client_id', 'unknown')}: {e}")
                noisy_updates.append(update)
        
        return noisy_updates
    
    def clip_gradients(self, parameters: np.ndarray, clip_norm: float = None) -> np.ndarray:
        """Clip gradients to bound sensitivity"""
        if clip_norm is None:
            clip_norm = self.gradient_clip_norm
        
        param_norm = np.linalg.norm(parameters)
        if param_norm > clip_norm:
            return parameters * (clip_norm / param_norm)
        return parameters
    
    def _add_noise_to_update_advanced(self, update: Dict[str, Any], epsilon: float, delta: float) -> Dict[str, Any]:
        """Add noise to update using advanced mechanisms"""
        noisy_update = update.copy()
        parameters = update['parameters']
        
        # Gradient clipping
        if isinstance(parameters, np.ndarray):
            clipped_params = self.clip_gradients(parameters)
            
            # Add noise using selected mechanism
            noisy_params = self.mechanism.add_noise(
                clipped_params, 
                self.sensitivity, 
                epsilon, 
                delta
            )
            
            noisy_update['parameters'] = noisy_params
            noisy_update['clipped'] = True
            noisy_update['privacy_mechanism'] = self.mechanism_type
        
        return noisy_update
    
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
