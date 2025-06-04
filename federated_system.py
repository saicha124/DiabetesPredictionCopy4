import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import random
import time
from typing import Dict, List, Tuple, Optional, Callable
from datetime import datetime

from neural_network import DiabetesNN
from aggregation import FogAggregator, GlobalAggregator
from privacy_engine import PrivacyEngine
from security_committee import SecurityCommittee
from data_loader import partition_data

class FederatedClient:
    """Individual federated learning client"""
    
    def __init__(self, client_id: int, data: Tuple[np.ndarray, np.ndarray], model_params: Dict):
        self.client_id = client_id
        self.X_train, self.y_train = data
        self.model = DiabetesNN(**model_params)
        self.reputation_score = 1.0  # Initial reputation
        self.training_history = []
        
    def local_training(self, global_weights: Optional[Dict] = None, epochs: int = 5) -> Dict:
        """Perform local training"""
        try:
            # Update model with global weights if provided
            if global_weights:
                self.model.load_state_dict(global_weights)
            
            # Training
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            X_tensor = torch.FloatTensor(self.X_train)
            y_tensor = torch.FloatTensor(self.y_train).unsqueeze(1)
            
            epoch_losses = []
            
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            
            # Calculate accuracy
            with torch.no_grad():
                predictions = (self.model(X_tensor) > 0.5).float()
                accuracy = (predictions == y_tensor).float().mean().item()
            
            training_result = {
                'client_id': self.client_id,
                'weights': self.model.state_dict(),
                'loss': np.mean(epoch_losses),
                'accuracy': accuracy,
                'num_samples': len(self.X_train)
            }
            
            self.training_history.append(training_result)
            return training_result
            
        except Exception as e:
            print(f"Client {self.client_id} training failed: {e}")
            return None
    
    def update_reputation(self, delta: float):
        """Update client reputation score"""
        self.reputation_score = max(0.0, min(1.0, self.reputation_score + delta))

class FogNode:
    """Fog node for intermediate aggregation"""
    
    def __init__(self, fog_id: int, privacy_engine: PrivacyEngine):
        self.fog_id = fog_id
        self.privacy_engine = privacy_engine
        self.assigned_clients = []
        self.aggregator = FogAggregator()
        
    def assign_clients(self, client_ids: List[int]):
        """Assign clients to this fog node"""
        self.assigned_clients = client_ids
    
    def aggregate_client_updates(self, client_updates: List[Dict]) -> Dict:
        """Aggregate updates from assigned clients"""
        try:
            # Filter updates from assigned clients
            relevant_updates = [
                update for update in client_updates 
                if update['client_id'] in self.assigned_clients
            ]
            
            if not relevant_updates:
                return None
            
            # Perform aggregation
            aggregated_weights = self.aggregator.aggregate(relevant_updates)
            
            return {
                'fog_id': self.fog_id,
                'aggregated_weights': aggregated_weights,
                'num_clients': len(relevant_updates),
                'client_ids': [update['client_id'] for update in relevant_updates]
            }
            
        except Exception as e:
            print(f"Fog {self.fog_id} aggregation failed: {e}")
            return None

class LeaderFog:
    """Leader fog node for global aggregation"""
    
    def __init__(self, privacy_engine: PrivacyEngine):
        self.privacy_engine = privacy_engine
        self.global_aggregator = GlobalAggregator()
        self.global_model_weights = None
        
    def global_aggregation(self, fog_updates: List[Dict]) -> Dict:
        """Perform global aggregation of fog updates"""
        try:
            valid_updates = [update for update in fog_updates if update is not None]
            
            if not valid_updates:
                return None
            
            # Global aggregation
            global_weights = self.global_aggregator.aggregate(valid_updates)
            self.global_model_weights = global_weights
            
            return {
                'global_weights': global_weights,
                'num_fogs': len(valid_updates),
                'total_clients': sum(update['num_clients'] for update in valid_updates)
            }
            
        except Exception as e:
            print(f"Leader fog global aggregation failed: {e}")
            return None

class FederatedLearningSystem:
    """Main federated learning system orchestrator"""
    
    def __init__(self, num_clients: int, num_fogs: int, privacy_epsilon: float, data: pd.DataFrame):
        self.num_clients = num_clients
        self.num_fogs = num_fogs
        self.privacy_epsilon = privacy_epsilon
        
        # Initialize components
        self.privacy_engine = PrivacyEngine(epsilon=privacy_epsilon)
        self.security_committee = SecurityCommittee(privacy_engine=self.privacy_engine)
        
        # Initialize clients
        self.clients = self._initialize_clients(data)
        
        # Initialize fog nodes
        self.fog_nodes = self._initialize_fog_nodes()
        
        # Initialize leader fog
        self.leader_fog = LeaderFog(self.privacy_engine)
        
        # Assign clients to fog nodes
        self._assign_clients_to_fogs()
        
        # Initialize committees
        self.current_committee = []
        self._form_committee()
        
        print(f"Initialized federated system with {num_clients} clients, {num_fogs} fogs")
    
    def _initialize_clients(self, data: pd.DataFrame) -> List[FederatedClient]:
        """Initialize federated clients with partitioned data"""
        # Partition data among clients
        client_data = partition_data(data, self.num_clients)
        
        # Model parameters
        model_params = {
            'input_size': len(data.columns) - 1,  # Exclude target column
            'hidden_sizes': [64, 32],
            'output_size': 1
        }
        
        clients = []
        for i, (X, y) in enumerate(client_data):
            client = FederatedClient(i, (X, y), model_params)
            clients.append(client)
        
        return clients
    
    def _initialize_fog_nodes(self) -> List[FogNode]:
        """Initialize fog nodes"""
        fog_nodes = []
        for i in range(self.num_fogs):
            fog = FogNode(i, self.privacy_engine)
            fog_nodes.append(fog)
        
        return fog_nodes
    
    def _assign_clients_to_fogs(self):
        """Assign clients to fog nodes"""
        clients_per_fog = self.num_clients // self.num_fogs
        extra_clients = self.num_clients % self.num_fogs
        
        client_idx = 0
        for i, fog in enumerate(self.fog_nodes):
            # Determine number of clients for this fog
            num_clients_for_fog = clients_per_fog + (1 if i < extra_clients else 0)
            
            # Assign clients
            assigned_client_ids = list(range(client_idx, client_idx + num_clients_for_fog))
            fog.assign_clients(assigned_client_ids)
            
            client_idx += num_clients_for_fog
    
    def _form_committee(self):
        """Form security committee based on reputation scores"""
        try:
            # Get reputation scores with differential privacy
            reputation_scores = []
            for client in self.clients:
                noisy_reputation = self.privacy_engine.add_noise(client.reputation_score)
                reputation_scores.append((client.client_id, noisy_reputation))
            
            # Select committee members
            self.current_committee = self.security_committee.select_committee(
                reputation_scores, committee_size=min(5, len(self.clients) // 2)
            )
            
        except Exception as e:
            print(f"Committee formation failed: {e}")
            self.current_committee = []
    
    def execute_round(self, progress_callbacks: Optional[Dict[str, Callable]] = None) -> Dict:
        """Execute a complete federated learning round"""
        round_start_time = time.time()
        
        try:
            # Phase 1: Client Training
            if progress_callbacks and 'client' in progress_callbacks:
                progress_callbacks['client'](0.1)
            
            client_updates = []
            for i, client in enumerate(self.clients):
                update = client.local_training(
                    global_weights=self.leader_fog.global_model_weights,
                    epochs=3
                )
                if update:
                    client_updates.append(update)
                
                # Update progress
                if progress_callbacks and 'client' in progress_callbacks:
                    progress = 0.1 + (0.8 * (i + 1) / len(self.clients))
                    progress_callbacks['client'](progress)
            
            if progress_callbacks and 'client' in progress_callbacks:
                progress_callbacks['client'](1.0)
            
            # Phase 2: Fog Aggregation
            if progress_callbacks and 'fog' in progress_callbacks:
                progress_callbacks['fog'](0.1)
            
            fog_aggregation_start = time.time()
            fog_updates = []
            
            for i, fog in enumerate(self.fog_nodes):
                fog_update = fog.aggregate_client_updates(client_updates)
                if fog_update:
                    fog_updates.append(fog_update)
                
                # Update progress
                if progress_callbacks and 'fog' in progress_callbacks:
                    progress = 0.1 + (0.8 * (i + 1) / len(self.fog_nodes))
                    progress_callbacks['fog'](progress)
            
            fog_aggregation_time = time.time() - fog_aggregation_start
            
            if progress_callbacks and 'fog' in progress_callbacks:
                progress_callbacks['fog'](1.0)
            
            # Phase 3: Global Aggregation
            if progress_callbacks and 'global' in progress_callbacks:
                progress_callbacks['global'](0.1)
            
            global_aggregation_start = time.time()
            global_result = self.leader_fog.global_aggregation(fog_updates)
            global_aggregation_time = time.time() - global_aggregation_start
            
            if progress_callbacks and 'global' in progress_callbacks:
                progress_callbacks['global'](1.0)
            
            # Update committee and reputations
            self._form_committee()
            self._update_reputations(client_updates)
            
            # Calculate round statistics
            total_time = time.time() - round_start_time
            avg_client_loss = np.mean([update['loss'] for update in client_updates if update])
            global_accuracy = self._calculate_global_accuracy()
            
            round_results = {
                'client_updates': len(client_updates),
                'fog_updates': len(fog_updates),
                'global_accuracy': global_accuracy,
                'avg_client_loss': avg_client_loss,
                'training_time': total_time,
                'fog_aggregation_time': fog_aggregation_time,
                'global_aggregation_time': global_aggregation_time,
                'num_committees': len(self.current_committee),
                'timestamp': datetime.now().isoformat()
            }
            
            return round_results
            
        except Exception as e:
            print(f"Round execution failed: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _update_reputations(self, client_updates: List[Dict]):
        """Update client reputation scores based on performance"""
        try:
            if not client_updates:
                return
            
            # Calculate performance metrics
            avg_loss = np.mean([update['loss'] for update in client_updates])
            
            for update in client_updates:
                client = self.clients[update['client_id']]
                
                # Reputation delta based on relative performance
                performance_ratio = avg_loss / (update['loss'] + 1e-8)
                reputation_delta = (performance_ratio - 1.0) * 0.1
                
                # Apply reputation update
                client.update_reputation(reputation_delta)
                
        except Exception as e:
            print(f"Reputation update failed: {e}")
    
    def _calculate_global_accuracy(self) -> float:
        """Calculate global model accuracy on test data"""
        try:
            if self.leader_fog.global_model_weights is None:
                return 0.0
            
            # Use first client's data for global evaluation
            # In practice, this would be a separate test dataset
            test_client = self.clients[0]
            
            # Create model with global weights
            model = DiabetesNN(
                input_size=test_client.model.fc1.in_features,
                hidden_sizes=[64, 32],
                output_size=1
            )
            model.load_state_dict(self.leader_fog.global_model_weights)
            
            # Calculate accuracy
            with torch.no_grad():
                X_tensor = torch.FloatTensor(test_client.X_train)
                y_tensor = torch.FloatTensor(test_client.y_train).unsqueeze(1)
                
                predictions = (model(X_tensor) > 0.5).float()
                accuracy = (predictions == y_tensor).float().mean().item()
            
            return accuracy
            
        except Exception as e:
            print(f"Global accuracy calculation failed: {e}")
            return 0.0
    
    def get_client_reputation(self, client_id: int) -> float:
        """Get client reputation with differential privacy"""
        try:
            if 0 <= client_id < len(self.clients):
                reputation = self.clients[client_id].reputation_score
                return self.privacy_engine.add_noise(reputation)
            return 0.0
        except:
            return 0.0
    
    def get_system_stats(self) -> Dict:
        """Get current system statistics"""
        return {
            'num_clients': len(self.clients),
            'num_fogs': len(self.fog_nodes),
            'committee_size': len(self.current_committee),
            'privacy_epsilon': self.privacy_epsilon,
            'global_model_ready': self.leader_fog.global_model_weights is not None
        }
