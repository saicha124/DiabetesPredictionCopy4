import numpy as np
from typing import List, Dict, Any, Tuple
import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, f1_score

class FogNode:
    """Represents a fog computing node in hierarchical federated learning"""
    
    def __init__(self, fog_id: int, client_ids: List[int], aggregation_method: str = "FedAvg", mu: float = 0.01):
        self.fog_id = fog_id
        self.client_ids = client_ids
        self.aggregation_method = aggregation_method
        self.mu = mu  # FedProx proximal parameter
        self.local_model = None
        self.aggregation_history = []
        self.performance_metrics = {
            'accuracy': [],
            'loss': [],
            'f1_score': [],
            'aggregation_time': [],
            'communication_overhead': []
        }
        self.client_round_metrics = {}  # Track per-client per-round metrics
    
    def aggregate_client_updates(self, client_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """Aggregate updates from clients assigned to this fog node"""
        start_time = time.time()
        
        # Filter updates for clients assigned to this fog node
        fog_client_updates = [update for update in client_updates if update['client_id'] in self.client_ids]
        
        if not fog_client_updates:
            return None
        
        # Perform aggregation based on method
        if self.aggregation_method == "FedAvg":
            aggregated_update = self._fedavg_aggregation(fog_client_updates, global_model)
        elif self.aggregation_method == "FedProx":
            aggregated_update = self._fedprox_aggregation(fog_client_updates, global_model)
        elif self.aggregation_method == "WeightedAvg":
            aggregated_update = self._weighted_aggregation(fog_client_updates, global_model)
        elif self.aggregation_method == "Median":
            aggregated_update = self._median_aggregation(fog_client_updates, global_model)
        else:
            aggregated_update = self._fedavg_aggregation(fog_client_updates, global_model)
        
        # Calculate metrics
        aggregation_time = time.time() - start_time
        self.performance_metrics['aggregation_time'].append(aggregation_time)
        
        # Store aggregation info
        aggregation_info = {
            'fog_id': self.fog_id,
            'num_clients': len(fog_client_updates),
            'aggregation_method': self.aggregation_method,
            'aggregation_time': aggregation_time,
            'client_ids': self.client_ids,
            'update': aggregated_update
        }
        
        self.aggregation_history.append(aggregation_info)
        
        return aggregation_info
    
    def _fedavg_aggregation(self, client_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """Federated Averaging at fog level"""
        if not client_updates:
            return None
        
        # Calculate weighted average based on number of samples
        total_samples = sum(update['num_samples'] for update in client_updates)
        
        if total_samples == 0:
            return None
        
        # Initialize aggregated parameters
        aggregated_params = {}
        
        # Handle different parameter formats
        first_update = client_updates[0]
        
        if isinstance(first_update['parameters'], dict):
            param_keys = first_update['parameters'].keys()
            
            for key in param_keys:
                weighted_sum = np.zeros_like(first_update['parameters'][key])
                
                for update in client_updates:
                    weight = update['num_samples'] / total_samples
                    weighted_sum += weight * update['parameters'][key]
                
                aggregated_params[key] = weighted_sum
        else:
            # Handle array-based parameters
            param_array = first_update['parameters']
            weighted_sum = np.zeros_like(param_array)
            
            for update in client_updates:
                weight = update['num_samples'] / total_samples
                weighted_sum += weight * update['parameters']
            
            aggregated_params = weighted_sum
        
        return {
            'parameters': aggregated_params,
            'num_samples': total_samples,
            'client_count': len(client_updates)
        }
    
    def _fedprox_aggregation(self, client_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """FedProx aggregation with proximal regularization"""
        if not client_updates:
            return None
        
        # Get global model parameters for proximal term
        try:
            if hasattr(global_model, 'coef_') and hasattr(global_model, 'intercept_'):
                global_params = np.concatenate([
                    global_model.coef_.flatten(),
                    global_model.intercept_.flatten()
                ])
            else:
                global_params = None
        except:
            global_params = None
        
        total_samples = sum(update['num_samples'] for update in client_updates)
        if total_samples == 0:
            return None
        
        # Aggregate with proximal regularization
        aggregated_params = None
        
        for update in client_updates:
            weight = update['num_samples'] / total_samples
            params = update['parameters']
            
            # Apply proximal regularization if global params available
            if global_params is not None and isinstance(params, np.ndarray) and len(params) == len(global_params):
                regularized_params = params - self.mu * (params - global_params)
            else:
                regularized_params = params
            
            if aggregated_params is None:
                aggregated_params = weight * regularized_params
            else:
                aggregated_params += weight * regularized_params
        
        return {
            'parameters': aggregated_params,
            'num_samples': total_samples,
            'client_count': len(client_updates),
            'proximal_mu': self.mu
        }
    
    def _weighted_aggregation(self, client_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """Weighted aggregation based on client performance"""
        if not client_updates:
            return None
        
        # Calculate weights based on accuracy (higher accuracy = higher weight)
        weights = []
        for update in client_updates:
            accuracy = update.get('accuracy', 0.5)
            # Transform accuracy to weight (avoid zero weights)
            weight = max(accuracy, 0.1)
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Aggregate parameters
        aggregated_params = {}
        first_update = client_updates[0]
        
        if isinstance(first_update['parameters'], dict):
            param_keys = first_update['parameters'].keys()
            
            for key in param_keys:
                weighted_sum = np.zeros_like(first_update['parameters'][key])
                
                for i, update in enumerate(client_updates):
                    weighted_sum += normalized_weights[i] * update['parameters'][key]
                
                aggregated_params[key] = weighted_sum
        else:
            # Handle array-based parameters
            param_array = first_update['parameters']
            weighted_sum = np.zeros_like(param_array)
            
            for i, update in enumerate(client_updates):
                weighted_sum += normalized_weights[i] * update['parameters']
            
            aggregated_params = weighted_sum
        
        return {
            'parameters': aggregated_params,
            'num_samples': sum(update['num_samples'] for update in client_updates),
            'client_count': len(client_updates),
            'weights_used': normalized_weights
        }
    
    def _median_aggregation(self, client_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """Median aggregation for robustness against outliers"""
        if not client_updates:
            return None
        
        aggregated_params = {}
        first_update = client_updates[0]
        
        if isinstance(first_update['parameters'], dict):
            param_keys = first_update['parameters'].keys()
            
            for key in param_keys:
                # Collect all parameter values for this key
                param_values = [update['parameters'][key] for update in client_updates]
                param_stack = np.stack(param_values, axis=0)
                
                # Calculate median along the client axis
                median_params = np.median(param_stack, axis=0)
                aggregated_params[key] = median_params
        else:
            # Handle array-based parameters
            param_values = [update['parameters'] for update in client_updates]
            param_stack = np.stack(param_values, axis=0)
            aggregated_params = np.median(param_stack, axis=0)
        
        return {
            'parameters': aggregated_params,
            'num_samples': sum(update['num_samples'] for update in client_updates),
            'client_count': len(client_updates)
        }

class HierarchicalFederatedLearning:
    """Hierarchical federated learning with fog nodes"""
    
    def __init__(self, num_clients: int, num_fog_nodes: int = 3, fog_aggregation_method: str = "FedAvg"):
        self.num_clients = num_clients
        self.num_fog_nodes = num_fog_nodes
        self.fog_aggregation_method = fog_aggregation_method
        self.fog_nodes = []
        self.global_aggregation_history = []
        self.loss_tracking = {
            'global_loss': [],
            'fog_losses': {},
            'client_losses': {},
            'round_losses': []
        }
        
        # Create fog nodes and assign clients
        self._create_fog_topology()
    
    def _create_fog_topology(self):
        """Create fog nodes and assign clients to them"""
        clients_per_fog = self.num_clients // self.num_fog_nodes
        remaining_clients = self.num_clients % self.num_fog_nodes
        
        current_client = 0
        for fog_id in range(self.num_fog_nodes):
            # Assign clients to this fog node
            num_clients_for_fog = clients_per_fog + (1 if fog_id < remaining_clients else 0)
            client_ids = list(range(current_client, current_client + num_clients_for_fog))
            
            # Create fog node with configurable aggregation method
            fog_node = FogNode(fog_id, client_ids, self.fog_aggregation_method, mu=0.01)
            self.fog_nodes.append(fog_node)
            
            current_client += num_clients_for_fog
    
    def fog_level_aggregation(self, client_updates: List[Dict[str, Any]], global_model) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Perform aggregation at fog level"""
        fog_updates = []
        fog_metrics = {
            'fog_performances': [],
            'aggregation_times': [],
            'client_distribution': {},
            'methods_used': []
        }
        
        for fog_node in self.fog_nodes:
            fog_update = fog_node.aggregate_client_updates(client_updates, global_model)
            
            if fog_update:
                fog_updates.append(fog_update)
                
                # Track fog performance
                fog_metrics['fog_performances'].append({
                    'fog_id': fog_node.fog_id,
                    'clients_served': len(fog_node.client_ids),
                    'aggregation_method': fog_node.aggregation_method,
                    'aggregation_time': fog_update['aggregation_time']
                })
                
                fog_metrics['aggregation_times'].append(fog_update['aggregation_time'])
                fog_metrics['client_distribution'][fog_node.fog_id] = fog_node.client_ids
                fog_metrics['methods_used'].append(fog_node.aggregation_method)
        
        return fog_updates, fog_metrics
    
    def leader_fog_aggregation(self, fog_updates: List[Dict[str, Any]], global_model) -> Dict[str, Any]:
        """Final aggregation at leader fog level"""
        if not fog_updates:
            return None
        
        start_time = time.time()
        
        # Use FedAvg for leader aggregation
        total_samples = sum(update['update']['num_samples'] for update in fog_updates)
        
        if total_samples == 0:
            return None
        
        # Initialize aggregated parameters
        aggregated_params = {}
        first_update = fog_updates[0]['update']
        
        if isinstance(first_update['parameters'], dict):
            param_keys = first_update['parameters'].keys()
            
            for key in param_keys:
                weighted_sum = np.zeros_like(first_update['parameters'][key])
                
                for fog_update in fog_updates:
                    weight = fog_update['update']['num_samples'] / total_samples
                    weighted_sum += weight * fog_update['update']['parameters'][key]
                
                aggregated_params[key] = weighted_sum
        else:
            # Handle array-based parameters
            param_array = first_update['parameters']
            weighted_sum = np.zeros_like(param_array)
            
            for fog_update in fog_updates:
                weight = fog_update['update']['num_samples'] / total_samples
                weighted_sum += weight * fog_update['update']['parameters']
            
            aggregated_params = weighted_sum
        
        leader_aggregation_time = time.time() - start_time
        
        # Store global aggregation info
        aggregation_info = {
            'total_fog_nodes': len(fog_updates),
            'total_samples': total_samples,
            'leader_aggregation_time': leader_aggregation_time,
            'fog_methods_used': [fu['aggregation_method'] for fu in fog_updates]
        }
        
        self.global_aggregation_history.append(aggregation_info)
        
        return {
            'parameters': aggregated_params,
            'aggregation_info': aggregation_info
        }
    
    def calculate_hierarchical_loss(self, global_model, client_data: List[Dict], round_num: int):
        """Calculate loss at different levels of the hierarchy"""
        round_losses = {
            'round': round_num,
            'global_loss': 0.0,
            'fog_losses': {},
            'client_losses': {},
            'timestamp': time.time()
        }
        
        total_samples = 0
        total_loss = 0.0
        
        # Calculate loss for each fog node
        for fog_node in self.fog_nodes:
            fog_loss = 0.0
            fog_samples = 0
            fog_client_losses = {}
            
            for client_id in fog_node.client_ids:
                if client_id < len(client_data):
                    client_X = client_data[client_id]['X_test']
                    client_y = client_data[client_id]['y_test']
                    
                    if len(client_X) > 0 and len(client_y) > 0:
                        try:
                            # Calculate predictions and loss
                            predictions = global_model.predict_proba(client_X)
                            if predictions.shape[1] > 1:
                                client_loss = log_loss(client_y, predictions)
                            else:
                                # Binary case fallback
                                pred_proba = global_model.predict_proba(client_X)[:, 1]
                                client_loss = log_loss(client_y, pred_proba)
                            
                            fog_client_losses[client_id] = client_loss
                            fog_loss += client_loss * len(client_X)
                            fog_samples += len(client_X)
                            
                            total_loss += client_loss * len(client_X)
                            total_samples += len(client_X)
                            
                        except Exception as e:
                            print(f"Error calculating loss for client {client_id}: {e}")
                            fog_client_losses[client_id] = 0.0
            
            # Average fog loss
            if fog_samples > 0:
                avg_fog_loss = fog_loss / fog_samples
            else:
                avg_fog_loss = 0.0
            
            round_losses['fog_losses'][fog_node.fog_id] = {
                'loss': avg_fog_loss,
                'samples': fog_samples,
                'client_losses': fog_client_losses
            }
            
            # Update fog node metrics
            fog_node.performance_metrics['loss'].append(avg_fog_loss)
        
        # Calculate global loss
        if total_samples > 0:
            global_loss = total_loss / total_samples
        else:
            global_loss = 0.0
        
        round_losses['global_loss'] = global_loss
        
        # Store in tracking
        self.loss_tracking['global_loss'].append(global_loss)
        self.loss_tracking['round_losses'].append(round_losses)
        
        return round_losses
    
    def get_topology_info(self) -> Dict[str, Any]:
        """Get information about the fog topology"""
        topology = {
            'num_fog_nodes': len(self.fog_nodes),
            'total_clients': self.num_clients,
            'fog_assignments': {},
            'aggregation_methods': {}
        }
        
        for fog_node in self.fog_nodes:
            topology['fog_assignments'][fog_node.fog_id] = fog_node.client_ids
            topology['aggregation_methods'][fog_node.fog_id] = fog_node.aggregation_method
        
        return topology
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        summary = {
            'fog_performance': {},
            'global_metrics': {
                'total_rounds': len(self.loss_tracking['global_loss']),
                'final_global_loss': self.loss_tracking['global_loss'][-1] if self.loss_tracking['global_loss'] else 0.0,
                'avg_global_loss': np.mean(self.loss_tracking['global_loss']) if self.loss_tracking['global_loss'] else 0.0
            },
            'aggregation_efficiency': {}
        }
        
        # Fog node performance
        for fog_node in self.fog_nodes:
            summary['fog_performance'][fog_node.fog_id] = {
                'method': fog_node.aggregation_method,
                'clients_served': len(fog_node.client_ids),
                'avg_aggregation_time': np.mean(fog_node.performance_metrics['aggregation_time']) if fog_node.performance_metrics['aggregation_time'] else 0.0,
                'avg_loss': np.mean(fog_node.performance_metrics['loss']) if fog_node.performance_metrics['loss'] else 0.0,
                'total_aggregations': len(fog_node.aggregation_history)
            }
        
        # Global aggregation efficiency
        if self.global_aggregation_history:
            summary['aggregation_efficiency'] = {
                'avg_leader_aggregation_time': np.mean([h['leader_aggregation_time'] for h in self.global_aggregation_history]),
                'total_fog_participations': sum([h['total_fog_nodes'] for h in self.global_aggregation_history])
            }
        
        return summary