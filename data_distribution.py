import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns

class DataDistributionStrategy:
    """Base class for data distribution strategies"""
    
    def __init__(self, num_clients: int, random_state: int = 42):
        self.num_clients = num_clients
        self.random_state = random_state
        np.random.seed(random_state)
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Distribute data among clients"""
        raise NotImplementedError
    
    def get_distribution_stats(self, client_data: List[Dict[str, np.ndarray]]) -> Dict[str, Any]:
        """Get statistics about the data distribution"""
        stats = {
            'strategy': self.__class__.__name__,
            'num_clients': len(client_data),
            'total_samples': sum(len(data['X_train']) + len(data['X_test']) for data in client_data),
            'client_sizes': [len(data['X_train']) + len(data['X_test']) for data in client_data],
            'class_distributions': []
        }
        
        for i, data in enumerate(client_data):
            all_y = np.concatenate([data['y_train'], data['y_test']])
            unique, counts = np.unique(all_y, return_counts=True)
            class_dist = dict(zip(unique.astype(int), counts))
            stats['class_distributions'].append(class_dist)
        
        return stats

class IIDDistribution(DataDistributionStrategy):
    """Independent and Identically Distributed (IID) data distribution"""
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Distribute data randomly among clients (IID)"""
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        # Split data evenly among clients
        client_data = []
        samples_per_client = len(X) // self.num_clients
        
        for i in range(self.num_clients):
            start_idx = i * samples_per_client
            if i == self.num_clients - 1:
                # Last client gets remaining samples
                end_idx = len(X)
            else:
                end_idx = (i + 1) * samples_per_client
            
            client_X = X_shuffled[start_idx:end_idx]
            client_y = y_shuffled[start_idx:end_idx]
            
            # Split into train/test
            if len(client_X) > 1 and len(np.unique(client_y)) > 1:
                try:
                    X_train, X_test, y_train, y_test = train_test_split(
                        client_X, client_y, test_size=0.2, random_state=self.random_state + i, 
                        stratify=client_y
                    )
                except ValueError:
                    # Fallback if stratification fails
                    split_idx = int(0.8 * len(client_X))
                    X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                    y_train, y_test = client_y[:split_idx], client_y[split_idx:]
            else:
                split_idx = max(1, int(0.8 * len(client_X)))
                X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                y_train, y_test = client_y[:split_idx], client_y[split_idx:]
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
        return client_data

class NonIIDDistribution(DataDistributionStrategy):
    """Non-IID distribution with class imbalance"""
    
    def __init__(self, num_clients: int, alpha: float = 0.5, random_state: int = 42):
        super().__init__(num_clients, random_state)
        self.alpha = alpha  # Controls the degree of non-IID-ness
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Distribute data using Dirichlet distribution for non-IID"""
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        # Initialize client data storage
        client_X_parts = [[] for _ in range(self.num_clients)]
        client_y_parts = [[] for _ in range(self.num_clients)]
        
        for class_label in unique_classes:
            # Get indices for this class
            class_indices = np.where(y == class_label)[0]
            class_X = X[class_indices]
            class_y = y[class_indices]
            
            # Sample proportions from Dirichlet distribution
            proportions = np.random.dirichlet([self.alpha] * self.num_clients)
            
            # Distribute class samples according to proportions
            start_idx = 0
            for i, prop in enumerate(proportions):
                num_samples = int(prop * len(class_X))
                if i == self.num_clients - 1:
                    # Last client gets remaining samples
                    end_idx = len(class_X)
                else:
                    end_idx = start_idx + num_samples
                
                if start_idx < end_idx:
                    client_class_X = class_X[start_idx:end_idx]
                    client_class_y = class_y[start_idx:end_idx]
                    
                    # Add to client lists
                    client_X_parts[i].append(client_class_X)
                    client_y_parts[i].append(client_class_y)
                
                start_idx = end_idx
        
        # Combine data parts for each client and create train/test splits
        final_client_data = []
        for i in range(self.num_clients):
            if len(client_X_parts[i]) > 0:
                # Concatenate all parts for this client
                X_client = np.vstack(client_X_parts[i])
                y_client = np.concatenate(client_y_parts[i])
                
                if len(X_client) > 1 and len(np.unique(y_client)) > 1:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_client, y_client, test_size=0.2, random_state=self.random_state + i,
                            stratify=y_client
                        )
                    except ValueError:
                        split_idx = max(1, int(0.8 * len(X_client)))
                        X_train, X_test = X_client[:split_idx], X_client[split_idx:]
                        y_train, y_test = y_client[:split_idx], y_client[split_idx:]
                else:
                    split_idx = max(1, int(0.8 * len(X_client)))
                    X_train, X_test = X_client[:split_idx], X_client[split_idx:]
                    y_train, y_test = y_client[:split_idx], y_client[split_idx:]
            else:
                # Empty client - assign minimal data from overall dataset
                min_samples = max(2, len(X) // (self.num_clients * 5))
                if len(X) >= min_samples:
                    indices = np.random.choice(len(X), min_samples, replace=False)
                    X_client = X[indices]
                    y_client = y[indices]
                    
                    split_idx = max(1, int(0.8 * len(X_client)))
                    X_train, X_test = X_client[:split_idx], X_client[split_idx:]
                    y_train, y_test = y_client[:split_idx], y_client[split_idx:]
                else:
                    # Fallback with proper data structure
                    X_train = X[:1].copy() if len(X) > 0 else np.zeros((1, X.shape[1] if len(X.shape) > 1 else 8))
                    y_train = y[:1].copy() if len(y) > 0 else np.array([0])
                    X_test = X_train.copy()
                    y_test = y_train.copy()
            
            final_client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
        return final_client_data

class PathologicalNonIIDDistribution(DataDistributionStrategy):
    """Pathological non-IID where each client has only a few classes"""
    
    def __init__(self, num_clients: int, classes_per_client: int = 1, random_state: int = 42):
        super().__init__(num_clients, random_state)
        self.classes_per_client = classes_per_client
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Each client gets only specific classes"""
        unique_classes = np.unique(y)
        num_classes = len(unique_classes)
        
        # Assign classes to clients
        client_classes = {}
        for i in range(self.num_clients):
            # Randomly select classes for this client
            start_class = (i * self.classes_per_client) % num_classes
            client_class_indices = []
            for j in range(self.classes_per_client):
                class_idx = (start_class + j) % num_classes
                client_class_indices.append(unique_classes[class_idx])
            client_classes[i] = client_class_indices
        
        # Distribute data
        client_data = []
        
        for i in range(self.num_clients):
            client_X_list = []
            client_y_list = []
            
            for class_label in client_classes[i]:
                class_indices = np.where(y == class_label)[0]
                
                # Divide class samples among clients that have this class
                clients_with_class = [j for j in range(self.num_clients) if class_label in client_classes[j]]
                samples_per_client = len(class_indices) // len(clients_with_class)
                
                client_idx_in_class = clients_with_class.index(i)
                start_idx = client_idx_in_class * samples_per_client
                if client_idx_in_class == len(clients_with_class) - 1:
                    end_idx = len(class_indices)
                else:
                    end_idx = start_idx + samples_per_client
                
                if start_idx < end_idx:
                    selected_indices = class_indices[start_idx:end_idx]
                    client_X_list.append(X[selected_indices])
                    client_y_list.append(y[selected_indices])
            
            if client_X_list:
                client_X = np.vstack(client_X_list)
                client_y = np.concatenate(client_y_list)
                
                # Create train/test split
                if len(client_X) > 1:
                    if len(np.unique(client_y)) > 1:
                        try:
                            X_train, X_test, y_train, y_test = train_test_split(
                                client_X, client_y, test_size=0.2, random_state=self.random_state + i,
                                stratify=client_y
                            )
                        except ValueError:
                            split_idx = max(1, int(0.8 * len(client_X)))
                            X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                            y_train, y_test = client_y[:split_idx], client_y[split_idx:]
                    else:
                        split_idx = max(1, int(0.8 * len(client_X)))
                        X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                        y_train, y_test = client_y[:split_idx], client_y[split_idx:]
                else:
                    X_train = X_test = client_X
                    y_train = y_test = client_y
            else:
                # Empty client
                X_train = X_test = np.array([]).reshape(0, X.shape[1])
                y_train = y_test = np.array([])
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
        return client_data

class QuantitySkewDistribution(DataDistributionStrategy):
    """Distribution with varying data quantities per client"""
    
    def __init__(self, num_clients: int, skew_factor: float = 2.0, random_state: int = 42):
        super().__init__(num_clients, random_state)
        self.skew_factor = skew_factor
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Distribute data with quantity skew"""
        # Generate client sizes using power law distribution
        sizes = np.random.power(self.skew_factor, self.num_clients)
        sizes = sizes / np.sum(sizes)  # Normalize
        sizes = (sizes * len(X)).astype(int)
        sizes[-1] = len(X) - np.sum(sizes[:-1])  # Ensure all data is used
        
        # Shuffle data
        indices = np.random.permutation(len(X))
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        client_data = []
        start_idx = 0
        
        for i, size in enumerate(sizes):
            if size > 0:
                end_idx = start_idx + size
                client_X = X_shuffled[start_idx:end_idx]
                client_y = y_shuffled[start_idx:end_idx]
                
                # Create train/test split
                if len(client_X) > 1 and len(np.unique(client_y)) > 1:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            client_X, client_y, test_size=0.2, random_state=self.random_state + i,
                            stratify=client_y
                        )
                    except ValueError:
                        split_idx = max(1, int(0.8 * len(client_X)))
                        X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                        y_train, y_test = client_y[:split_idx], client_y[split_idx:]
                else:
                    split_idx = max(1, int(0.8 * len(client_X)))
                    X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                    y_train, y_test = client_y[:split_idx], client_y[split_idx:]
                
                start_idx = end_idx
            else:
                # Empty client
                X_train = X_test = np.array([]).reshape(0, X.shape[1])
                y_train = y_test = np.array([])
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
        return client_data

class GeographicDistribution(DataDistributionStrategy):
    """Simulate geographic distribution patterns"""
    
    def __init__(self, num_clients: int, correlation_strength: float = 0.8, random_state: int = 42):
        super().__init__(num_clients, random_state)
        self.correlation_strength = correlation_strength
    
    def distribute_data(self, X: np.ndarray, y: np.ndarray) -> List[Dict[str, np.ndarray]]:
        """Distribute data based on simulated geographic correlation"""
        # Create geographic clusters based on feature similarity
        from sklearn.cluster import KMeans
        
        # Use KMeans to create geographic clusters
        kmeans = KMeans(n_clusters=self.num_clients, random_state=self.random_state)
        cluster_labels = kmeans.fit_predict(X)
        
        client_data = []
        
        for i in range(self.num_clients):
            # Get data for this geographic region
            cluster_indices = np.where(cluster_labels == i)[0]
            
            if len(cluster_indices) > 0:
                client_X = X[cluster_indices]
                client_y = y[cluster_indices]
                
                # Create train/test split
                if len(client_X) > 1 and len(np.unique(client_y)) > 1:
                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            client_X, client_y, test_size=0.2, random_state=self.random_state + i,
                            stratify=client_y
                        )
                    except ValueError:
                        split_idx = max(1, int(0.8 * len(client_X)))
                        X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                        y_train, y_test = client_y[:split_idx], client_y[split_idx:]
                else:
                    split_idx = max(1, int(0.8 * len(client_X)))
                    X_train, X_test = client_X[:split_idx], client_X[split_idx:]
                    y_train, y_test = client_y[:split_idx], client_y[split_idx:]
            else:
                # Empty cluster
                X_train = X_test = np.array([]).reshape(0, X.shape[1])
                y_train = y_test = np.array([])
            
            client_data.append({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            })
        
        return client_data

def get_distribution_strategy(strategy_name: str, num_clients: int, **kwargs) -> DataDistributionStrategy:
    """Factory function to create distribution strategies"""
    strategies = {
        'IID': IIDDistribution,
        'Non-IID (Dirichlet)': NonIIDDistribution,
        'Pathological Non-IID': PathologicalNonIIDDistribution,
        'Quantity Skew': QuantitySkewDistribution,
        'Geographic': GeographicDistribution
    }
    
    if strategy_name not in strategies:
        raise ValueError(f"Unknown strategy: {strategy_name}. Available: {list(strategies.keys())}")
    
    return strategies[strategy_name](num_clients=num_clients, **kwargs)

def visualize_data_distribution(client_data: List[Dict[str, np.ndarray]], strategy_stats: Dict[str, Any]):
    """Create visualizations for data distribution"""
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    
    # Client size distribution
    client_sizes = strategy_stats['client_sizes']
    
    fig_sizes = go.Figure()
    # Create meaningful medical facility names
    facility_names = [
        "City General Hospital",
        "Regional Medical Center", 
        "Community Health Clinic",
        "University Hospital",
        "Rural Health Center",
        "Specialty Care Institute",
        "Emergency Care Facility",
        "Primary Care Network",
        "Diagnostic Center",
        "Wellness Clinic"
    ]
    
    # Use facility names up to the number of clients, then fall back to numbered format
    station_labels = []
    for i in range(len(client_sizes)):
        if i < len(facility_names):
            station_labels.append(facility_names[i])
        else:
            station_labels.append(f"Medical Facility {i+1}")
    
    fig_sizes.add_trace(go.Bar(
        x=station_labels,
        y=client_sizes,
        marker_color='lightblue',
        text=client_sizes,
        textposition='auto'
    ))
    fig_sizes.update_layout(
        title="Data Distribution Across Medical Stations",
        xaxis_title="Medical Station",
        yaxis_title="Number of Samples",
        template="plotly_white"
    )
    
    # Class distribution heatmap
    class_distributions = strategy_stats['class_distributions']
    all_classes = set()
    for dist in class_distributions:
        all_classes.update(dist.keys())
    all_classes = sorted(list(all_classes))
    
    # Create matrix for heatmap
    heatmap_data = []
    for i, dist in enumerate(class_distributions):
        row = [dist.get(cls, 0) for cls in all_classes]
        heatmap_data.append(row)
    
    fig_heatmap = go.Figure(data=go.Heatmap(
        z=heatmap_data,
        x=[f"Class {cls}" for cls in all_classes],
        y=station_labels,
        colorscale='Viridis',
        text=heatmap_data,
        texttemplate="%{text}",
        textfont={"size": 12},
    ))
    fig_heatmap.update_layout(
        title="Class Distribution Heatmap",
        xaxis_title="Classes",
        yaxis_title="Medical Stations",
        template="plotly_white"
    )
    
    return fig_sizes, fig_heatmap