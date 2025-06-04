import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import copy
from collections import OrderedDict

class FogAggregator:
    """Fog-level aggregation for federated learning"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        """
        Initialize fog aggregator
        
        Args:
            aggregation_method: Aggregation method to use ("fedavg", "weighted_avg")
        """
        self.aggregation_method = aggregation_method.lower()
        self.supported_methods = ["fedavg", "weighted_avg", "median", "trimmed_mean"]
        
        if self.aggregation_method not in self.supported_methods:
            raise ValueError(f"Unsupported aggregation method: {aggregation_method}")
        
        print(f"Fog aggregator initialized with method: {aggregation_method}")
    
    def aggregate(self, client_updates: List[Dict]) -> Dict:
        """
        Aggregate client updates
        
        Args:
            client_updates: List of client update dictionaries
            
        Returns:
            Aggregated weights dictionary
        """
        try:
            if not client_updates:
                raise ValueError("No client updates provided")
            
            # Extract weights and metadata
            weights_list = []
            sample_counts = []
            
            for update in client_updates:
                if 'weights' not in update:
                    continue
                
                weights_list.append(update['weights'])
                sample_counts.append(update.get('num_samples', 1))
            
            if not weights_list:
                raise ValueError("No valid weights found in client updates")
            
            # Perform aggregation based on method
            if self.aggregation_method == "fedavg":
                aggregated_weights = self._federated_averaging(weights_list, sample_counts)
            elif self.aggregation_method == "weighted_avg":
                aggregated_weights = self._weighted_averaging(weights_list, sample_counts)
            elif self.aggregation_method == "median":
                aggregated_weights = self._median_aggregation(weights_list)
            elif self.aggregation_method == "trimmed_mean":
                aggregated_weights = self._trimmed_mean_aggregation(weights_list)
            else:
                aggregated_weights = self._federated_averaging(weights_list, sample_counts)
            
            return aggregated_weights
            
        except Exception as e:
            print(f"Fog aggregation failed: {e}")
            return {}
    
    def _federated_averaging(self, weights_list: List[Dict], sample_counts: List[int]) -> Dict:
        """
        Perform FedAvg aggregation
        
        Args:
            weights_list: List of weight dictionaries
            sample_counts: List of sample counts for each client
            
        Returns:
            Aggregated weights
        """
        if not weights_list:
            return {}
        
        # Get total samples
        total_samples = sum(sample_counts)
        
        # Initialize aggregated weights
        aggregated_weights = OrderedDict()
        
        # Get parameter names from first client
        param_names = list(weights_list[0].keys())
        
        for param_name in param_names:
            # Initialize parameter sum
            param_sum = None
            
            for i, weights in enumerate(weights_list):
                if param_name in weights:
                    param_tensor = weights[param_name]
                    weight = sample_counts[i] / total_samples
                    
                    if param_sum is None:
                        param_sum = weight * param_tensor.clone()
                    else:
                        param_sum += weight * param_tensor
            
            if param_sum is not None:
                aggregated_weights[param_name] = param_sum
        
        return aggregated_weights
    
    def _weighted_averaging(self, weights_list: List[Dict], sample_counts: List[int]) -> Dict:
        """
        Perform weighted averaging (same as FedAvg)
        """
        return self._federated_averaging(weights_list, sample_counts)
    
    def _median_aggregation(self, weights_list: List[Dict]) -> Dict:
        """
        Perform median aggregation (robust to outliers)
        
        Args:
            weights_list: List of weight dictionaries
            
        Returns:
            Aggregated weights using median
        """
        if not weights_list:
            return {}
        
        aggregated_weights = OrderedDict()
        param_names = list(weights_list[0].keys())
        
        for param_name in param_names:
            # Collect all parameter values
            param_values = []
            
            for weights in weights_list:
                if param_name in weights:
                    param_tensor = weights[param_name]
                    param_values.append(param_tensor.detach().numpy())
            
            if param_values:
                # Convert to numpy and compute median
                param_array = np.stack(param_values, axis=0)
                median_array = np.median(param_array, axis=0)
                
                # Convert back to tensor
                aggregated_weights[param_name] = torch.from_numpy(median_array).float()
        
        return aggregated_weights
    
    def _trimmed_mean_aggregation(self, weights_list: List[Dict], trim_ratio: float = 0.2) -> Dict:
        """
        Perform trimmed mean aggregation
        
        Args:
            weights_list: List of weight dictionaries
            trim_ratio: Ratio of extreme values to trim
            
        Returns:
            Aggregated weights using trimmed mean
        """
        if not weights_list:
            return {}
        
        aggregated_weights = OrderedDict()
        param_names = list(weights_list[0].keys())
        
        for param_name in param_names:
            # Collect all parameter values
            param_values = []
            
            for weights in weights_list:
                if param_name in weights:
                    param_tensor = weights[param_name]
                    param_values.append(param_tensor.detach().numpy())
            
            if param_values:
                # Convert to numpy
                param_array = np.stack(param_values, axis=0)
                
                # Calculate trim count
                num_clients = len(param_values)
                trim_count = int(num_clients * trim_ratio / 2)
                
                if trim_count > 0:
                    # Sort along client dimension and trim
                    sorted_array = np.sort(param_array, axis=0)
                    trimmed_array = sorted_array[trim_count:-trim_count]
                    mean_array = np.mean(trimmed_array, axis=0)
                else:
                    mean_array = np.mean(param_array, axis=0)
                
                # Convert back to tensor
                aggregated_weights[param_name] = torch.from_numpy(mean_array).float()
        
        return aggregated_weights
    
    def validate_weights(self, weights: Dict) -> bool:
        """
        Validate aggregated weights
        
        Args:
            weights: Weight dictionary to validate
            
        Returns:
            True if weights are valid
        """
        try:
            if not weights:
                return False
            
            for param_name, param_tensor in weights.items():
                # Check for NaN or inf values
                if torch.isnan(param_tensor).any() or torch.isinf(param_tensor).any():
                    print(f"Invalid values found in parameter {param_name}")
                    return False
                
                # Check for extremely large values
                if torch.abs(param_tensor).max() > 1000:
                    print(f"Extremely large values in parameter {param_name}")
                    return False
            
            return True
            
        except Exception as e:
            print(f"Weight validation failed: {e}")
            return False

class GlobalAggregator:
    """Global aggregation at leader fog level"""
    
    def __init__(self, aggregation_method: str = "fedavg"):
        """
        Initialize global aggregator
        
        Args:
            aggregation_method: Aggregation method to use
        """
        self.aggregation_method = aggregation_method.lower()
        self.fog_aggregator = FogAggregator(aggregation_method)
        
        print(f"Global aggregator initialized with method: {aggregation_method}")
    
    def aggregate(self, fog_updates: List[Dict]) -> Dict:
        """
        Aggregate fog-level updates
        
        Args:
            fog_updates: List of fog update dictionaries
            
        Returns:
            Global aggregated weights
        """
        try:
            if not fog_updates:
                raise ValueError("No fog updates provided")
            
            # Convert fog updates to client-like format for reuse
            processed_updates = []
            
            for fog_update in fog_updates:
                if 'aggregated_weights' in fog_update:
                    processed_update = {
                        'weights': fog_update['aggregated_weights'],
                        'num_samples': fog_update.get('num_clients', 1)
                    }
                    processed_updates.append(processed_update)
            
            if not processed_updates:
                raise ValueError("No valid fog updates found")
            
            # Use fog aggregator for global aggregation
            global_weights = self.fog_aggregator.aggregate(processed_updates)
            
            # Validate global weights
            if not self.fog_aggregator.validate_weights(global_weights):
                raise ValueError("Invalid global weights produced")
            
            return global_weights
            
        except Exception as e:
            print(f"Global aggregation failed: {e}")
            return {}
    
    def compute_aggregation_stats(self, fog_updates: List[Dict]) -> Dict:
        """
        Compute statistics about the aggregation process
        
        Args:
            fog_updates: List of fog update dictionaries
            
        Returns:
            Aggregation statistics
        """
        try:
            stats = {
                'num_fogs': len(fog_updates),
                'total_clients': 0,
                'avg_clients_per_fog': 0,
                'weight_diversity': 0.0
            }
            
            if not fog_updates:
                return stats
            
            # Count total clients
            client_counts = [update.get('num_clients', 0) for update in fog_updates]
            stats['total_clients'] = sum(client_counts)
            stats['avg_clients_per_fog'] = np.mean(client_counts) if client_counts else 0
            
            # Calculate weight diversity (variance across fogs)
            if len(fog_updates) >= 2:
                weights_list = []
                for update in fog_updates:
                    if 'aggregated_weights' in update:
                        weights_list.append(update['aggregated_weights'])
                
                if len(weights_list) >= 2:
                    # Calculate diversity for first parameter as proxy
                    first_param_key = list(weights_list[0].keys())[0]
                    param_values = []
                    
                    for weights in weights_list:
                        if first_param_key in weights:
                            param_tensor = weights[first_param_key]
                            param_values.append(param_tensor.detach().numpy().flatten())
                    
                    if len(param_values) >= 2:
                        param_matrix = np.stack(param_values)
                        stats['weight_diversity'] = float(np.var(param_matrix, axis=0).mean())
            
            return stats
            
        except Exception as e:
            print(f"Stats computation failed: {e}")
            return {'num_fogs': 0, 'total_clients': 0, 'avg_clients_per_fog': 0, 'weight_diversity': 0.0}

class SecureAggregator:
    """Secure aggregation with additional security measures"""
    
    def __init__(self, base_aggregator: Union[FogAggregator, GlobalAggregator], 
                 privacy_engine=None):
        """
        Initialize secure aggregator
        
        Args:
            base_aggregator: Base aggregator to wrap
            privacy_engine: Privacy engine for differential privacy
        """
        self.base_aggregator = base_aggregator
        self.privacy_engine = privacy_engine
        
        print("Secure aggregator initialized")
    
    def secure_aggregate(self, updates: List[Dict], security_committee: List[int] = None) -> Dict:
        """
        Perform secure aggregation with committee validation
        
        Args:
            updates: List of updates to aggregate
            security_committee: List of trusted committee member IDs
            
        Returns:
            Securely aggregated weights
        """
        try:
            # Filter updates by committee if specified
            if security_committee is not None:
                validated_updates = []
                for update in updates:
                    client_id = update.get('client_id', -1)
                    if client_id in security_committee:
                        validated_updates.append(update)
                updates = validated_updates
            
            if not updates:
                raise ValueError("No validated updates after committee filtering")
            
            # Perform base aggregation
            aggregated_weights = self.base_aggregator.aggregate(updates)
            
            # Apply differential privacy if available
            if self.privacy_engine and aggregated_weights:
                noisy_weights = self.privacy_engine.add_noise_to_weights(
                    aggregated_weights, mechanism="gaussian"
                )
                return noisy_weights
            
            return aggregated_weights
            
        except Exception as e:
            print(f"Secure aggregation failed: {e}")
            return {}
    
    def detect_poisoning_attacks(self, updates: List[Dict]) -> List[int]:
        """
        Detect potential poisoning attacks in updates
        
        Args:
            updates: List of client updates
            
        Returns:
            List of suspicious client IDs
        """
        try:
            suspicious_clients = []
            
            if len(updates) < 2:
                return suspicious_clients
            
            # Calculate statistics for each parameter
            param_stats = {}
            param_names = list(updates[0].get('weights', {}).keys())
            
            for param_name in param_names:
                param_values = []
                client_ids = []
                
                for update in updates:
                    if 'weights' in update and param_name in update['weights']:
                        param_tensor = update['weights'][param_name]
                        param_norm = torch.norm(param_tensor).item()
                        param_values.append(param_norm)
                        client_ids.append(update.get('client_id', -1))
                
                if len(param_values) >= 2:
                    param_mean = np.mean(param_values)
                    param_std = np.std(param_values)
                    
                    # Detect outliers (values > 3 standard deviations from mean)
                    for i, value in enumerate(param_values):
                        if abs(value - param_mean) > 3 * param_std:
                            client_id = client_ids[i]
                            if client_id not in suspicious_clients:
                                suspicious_clients.append(client_id)
            
            return suspicious_clients
            
        except Exception as e:
            print(f"Poisoning detection failed: {e}")
            return []
    
    def byzantine_robust_aggregation(self, updates: List[Dict], 
                                   byzantine_ratio: float = 0.33) -> Dict:
        """
        Perform Byzantine-robust aggregation
        
        Args:
            updates: List of client updates
            byzantine_ratio: Maximum ratio of Byzantine clients
            
        Returns:
            Byzantine-robust aggregated weights
        """
        try:
            if not updates:
                return {}
            
            # Detect suspicious updates
            suspicious_clients = self.detect_poisoning_attacks(updates)
            
            # Filter out suspicious updates
            clean_updates = []
            for update in updates:
                client_id = update.get('client_id', -1)
                if client_id not in suspicious_clients:
                    clean_updates.append(update)
            
            # Ensure we have enough clean updates
            max_byzantine = int(len(updates) * byzantine_ratio)
            if len(suspicious_clients) > max_byzantine:
                print(f"Warning: Detected {len(suspicious_clients)} suspicious clients, "
                      f"exceeds expected maximum of {max_byzantine}")
            
            # Use remaining clean updates for aggregation
            if clean_updates:
                return self.base_aggregator.aggregate(clean_updates)
            else:
                print("Warning: No clean updates available, using original updates")
                return self.base_aggregator.aggregate(updates)
                
        except Exception as e:
            print(f"Byzantine-robust aggregation failed: {e}")
            return self.base_aggregator.aggregate(updates)
