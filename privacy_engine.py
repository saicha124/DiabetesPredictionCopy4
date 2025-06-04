import numpy as np
import random
from typing import Union, List, Dict
from diffprivlib.mechanisms import Laplace, Gaussian
from diffprivlib.utils import copy_docstring

class PrivacyEngine:
    """Differential privacy engine for federated learning"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        """
        Initialize privacy engine
        
        Args:
            epsilon: Privacy budget parameter
            delta: Probability of privacy breach
            sensitivity: Sensitivity of the query/function
        """
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        
        # Initialize mechanisms
        self.laplace_mechanism = Laplace(epsilon=epsilon, sensitivity=sensitivity)
        self.gaussian_mechanism = Gaussian(epsilon=epsilon, delta=delta, sensitivity=sensitivity)
        
        print(f"Privacy engine initialized with ε={epsilon}, δ={delta}")
    
    def add_noise(self, value: Union[float, np.ndarray], mechanism: str = "laplace") -> Union[float, np.ndarray]:
        """
        Add differential privacy noise to a value or array
        
        Args:
            value: Value or array to add noise to
            mechanism: Type of noise mechanism ("laplace" or "gaussian")
            
        Returns:
            Noisy value or array
        """
        try:
            if isinstance(value, (int, float)):
                if mechanism.lower() == "laplace":
                    return self.laplace_mechanism.randomise(float(value))
                else:
                    return self.gaussian_mechanism.randomise(float(value))
            
            elif isinstance(value, np.ndarray):
                noisy_array = np.zeros_like(value, dtype=float)
                flat_value = value.flatten()
                
                for i, v in enumerate(flat_value):
                    if mechanism.lower() == "laplace":
                        noisy_array.flat[i] = self.laplace_mechanism.randomise(float(v))
                    else:
                        noisy_array.flat[i] = self.gaussian_mechanism.randomise(float(v))
                
                return noisy_array.reshape(value.shape)
            
            else:
                raise ValueError(f"Unsupported value type: {type(value)}")
                
        except Exception as e:
            print(f"Error adding noise: {e}")
            return value
    
    def add_noise_to_gradients(self, gradients: Dict[str, np.ndarray], 
                             mechanism: str = "gaussian") -> Dict[str, np.ndarray]:
        """
        Add differential privacy noise to model gradients
        
        Args:
            gradients: Dictionary of parameter names to gradient arrays
            mechanism: Noise mechanism to use
            
        Returns:
            Dictionary of noisy gradients
        """
        try:
            noisy_gradients = {}
            
            for param_name, grad_array in gradients.items():
                if isinstance(grad_array, np.ndarray):
                    noisy_gradients[param_name] = self.add_noise(grad_array, mechanism)
                else:
                    # Handle other types (like tensors)
                    grad_np = np.array(grad_array)
                    noisy_grad_np = self.add_noise(grad_np, mechanism)
                    noisy_gradients[param_name] = type(grad_array)(noisy_grad_np)
            
            return noisy_gradients
            
        except Exception as e:
            print(f"Error adding noise to gradients: {e}")
            return gradients
    
    def add_noise_to_weights(self, weights: Dict[str, np.ndarray], 
                           mechanism: str = "gaussian") -> Dict[str, np.ndarray]:
        """
        Add differential privacy noise to model weights
        
        Args:
            weights: Dictionary of parameter names to weight arrays
            mechanism: Noise mechanism to use
            
        Returns:
            Dictionary of noisy weights
        """
        try:
            noisy_weights = {}
            
            for param_name, weight_array in weights.items():
                if hasattr(weight_array, 'numpy'):
                    # Handle PyTorch tensors
                    weight_np = weight_array.detach().numpy()
                    noisy_weight_np = self.add_noise(weight_np, mechanism)
                    noisy_weights[param_name] = type(weight_array)(noisy_weight_np)
                elif isinstance(weight_array, np.ndarray):
                    noisy_weights[param_name] = self.add_noise(weight_array, mechanism)
                else:
                    # Try to convert to numpy
                    weight_np = np.array(weight_array)
                    noisy_weight_np = self.add_noise(weight_np, mechanism)
                    noisy_weights[param_name] = noisy_weight_np
            
            return noisy_weights
            
        except Exception as e:
            print(f"Error adding noise to weights: {e}")
            return weights
    
    def mask_reputation_scores(self, scores: List[float]) -> List[float]:
        """
        Apply differential privacy to reputation scores
        
        Args:
            scores: List of reputation scores
            
        Returns:
            List of noisy reputation scores
        """
        try:
            noisy_scores = []
            for score in scores:
                # Ensure score is in valid range [0, 1]
                clamped_score = max(0.0, min(1.0, float(score)))
                noisy_score = self.add_noise(clamped_score, mechanism="laplace")
                # Re-clamp after noise addition
                noisy_score = max(0.0, min(1.0, noisy_score))
                noisy_scores.append(noisy_score)
            
            return noisy_scores
            
        except Exception as e:
            print(f"Error masking reputation scores: {e}")
            return scores
    
    def compute_privacy_loss(self, num_queries: int) -> float:
        """
        Compute total privacy loss for a given number of queries
        
        Args:
            num_queries: Number of queries made
            
        Returns:
            Total privacy loss (epsilon)
        """
        # Simple composition - in practice, use advanced composition
        return self.epsilon * num_queries
    
    def update_privacy_budget(self, consumed_epsilon: float):
        """
        Update remaining privacy budget
        
        Args:
            consumed_epsilon: Amount of privacy budget consumed
        """
        self.epsilon = max(0.0, self.epsilon - consumed_epsilon)
        
        # Update mechanisms with new epsilon
        if self.epsilon > 0:
            self.laplace_mechanism = Laplace(epsilon=self.epsilon, sensitivity=self.sensitivity)
            self.gaussian_mechanism = Gaussian(
                epsilon=self.epsilon, 
                delta=self.delta, 
                sensitivity=self.sensitivity
            )
    
    def get_privacy_stats(self) -> Dict[str, float]:
        """
        Get current privacy statistics
        
        Returns:
            Dictionary of privacy statistics
        """
        return {
            'epsilon': self.epsilon,
            'delta': self.delta,
            'sensitivity': self.sensitivity,
            'mechanism_type': 'Laplace + Gaussian'
        }
    
    def add_discrete_noise(self, value: int, mechanism: str = "laplace") -> int:
        """
        Add noise to discrete values (e.g., counts)
        
        Args:
            value: Integer value to add noise to
            mechanism: Noise mechanism
            
        Returns:
            Noisy integer value
        """
        try:
            # Add continuous noise and round
            noisy_value = self.add_noise(float(value), mechanism)
            return max(0, int(round(noisy_value)))
            
        except Exception as e:
            print(f"Error adding discrete noise: {e}")
            return value
    
    def secure_aggregation_noise(self, aggregated_value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Add noise specifically for secure aggregation
        
        Args:
            aggregated_value: Value from aggregation
            
        Returns:
            Noisy aggregated value
        """
        # Use Gaussian mechanism for secure aggregation
        return self.add_noise(aggregated_value, mechanism="gaussian")
    
    def generate_random_mask(self, shape: tuple, seed: int = None) -> np.ndarray:
        """
        Generate random mask for secret sharing
        
        Args:
            shape: Shape of the mask
            seed: Random seed for reproducibility
            
        Returns:
            Random mask array
        """
        if seed is not None:
            np.random.seed(seed)
        
        return np.random.normal(0, 1, shape)
    
    def is_privacy_budget_exhausted(self) -> bool:
        """
        Check if privacy budget is exhausted
        
        Returns:
            True if budget is exhausted
        """
        return self.epsilon <= 0.0
    
    def reset_privacy_budget(self, new_epsilon: float = None):
        """
        Reset privacy budget to initial value or new value
        
        Args:
            new_epsilon: New epsilon value, if None uses original
        """
        if new_epsilon is not None:
            self.epsilon = new_epsilon
        
        # Reinitialize mechanisms
        self.laplace_mechanism = Laplace(epsilon=self.epsilon, sensitivity=self.sensitivity)
        self.gaussian_mechanism = Gaussian(
            epsilon=self.epsilon, 
            delta=self.delta, 
            sensitivity=self.sensitivity
        )
        
        print(f"Privacy budget reset to ε={self.epsilon}")
