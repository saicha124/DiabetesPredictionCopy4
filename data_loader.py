import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
import os

def load_diabetes_data(file_path: str = "attached_assets/diabetes.csv") -> pd.DataFrame:
    """
    Load diabetes dataset
    
    Args:
        file_path: Path to the diabetes CSV file
        
    Returns:
        Loaded DataFrame
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Diabetes dataset not found at {file_path}")
        
        # Load data
        data = pd.read_csv(file_path)
        
        print(f"Loaded diabetes dataset: {data.shape[0]} samples, {data.shape[1]} features")
        print(f"Target distribution: {data['Outcome'].value_counts().to_dict()}")
        
        return data
        
    except Exception as e:
        print(f"Error loading diabetes data: {e}")
        # Return empty DataFrame if loading fails
        return pd.DataFrame()

def preprocess_data(data: pd.DataFrame, 
                   target_column: str = 'Outcome',
                   test_size: float = 0.2,
                   random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess diabetes data for federated learning
    
    Args:
        data: Raw DataFrame
        target_column: Name of target column
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    try:
        if data.empty:
            raise ValueError("Empty dataset provided")
        
        # Separate features and target
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"Preprocessed data - Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train.values, y_test.values
        
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        # Return empty arrays if preprocessing fails
        return np.array([]), np.array([]), np.array([]), np.array([])

def partition_data(data: pd.DataFrame, 
                  num_clients: int,
                  target_column: str = 'Outcome',
                  partition_method: str = 'iid',
                  alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data among federated clients
    
    Args:
        data: Dataset to partition
        num_clients: Number of clients
        target_column: Name of target column
        partition_method: Partitioning method ('iid', 'non_iid', 'pathological')
        alpha: Dirichlet distribution parameter for non-IID partitioning
        
    Returns:
        List of (X, y) tuples for each client
    """
    try:
        if data.empty:
            raise ValueError("Empty dataset provided")
        
        # Preprocess data
        X_train, _, y_train, _ = preprocess_data(data, target_column)
        
        if len(X_train) == 0:
            raise ValueError("No training data after preprocessing")
        
        # Shuffle data
        X_train, y_train = shuffle(X_train, y_train, random_state=42)
        
        if partition_method.lower() == 'iid':
            client_data = _partition_iid(X_train, y_train, num_clients)
        elif partition_method.lower() == 'non_iid':
            client_data = _partition_non_iid(X_train, y_train, num_clients, alpha)
        elif partition_method.lower() == 'pathological':
            client_data = _partition_pathological(X_train, y_train, num_clients)
        else:
            print(f"Unknown partition method {partition_method}, using IID")
            client_data = _partition_iid(X_train, y_train, num_clients)
        
        print(f"Partitioned data among {num_clients} clients using {partition_method} method")
        for i, (X_client, y_client) in enumerate(client_data):
            print(f"  Client {i}: {len(X_client)} samples, "
                  f"class distribution: {np.bincount(y_client.astype(int)).tolist()}")
        
        return client_data
        
    except Exception as e:
        print(f"Error partitioning data: {e}")
        # Return empty partitions if partitioning fails
        return [(np.array([]), np.array([])) for _ in range(num_clients)]

def _partition_iid(X: np.ndarray, y: np.ndarray, num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data in an IID manner
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        
    Returns:
        List of client data partitions
    """
    n_samples = len(X)
    samples_per_client = n_samples // num_clients
    
    client_data = []
    start_idx = 0
    
    for i in range(num_clients):
        if i == num_clients - 1:
            # Last client gets remaining samples
            end_idx = n_samples
        else:
            end_idx = start_idx + samples_per_client
        
        X_client = X[start_idx:end_idx]
        y_client = y[start_idx:end_idx]
        
        client_data.append((X_client, y_client))
        start_idx = end_idx
    
    return client_data

def _partition_non_iid(X: np.ndarray, y: np.ndarray, num_clients: int, 
                      alpha: float = 0.5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data in a non-IID manner using Dirichlet distribution
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        
    Returns:
        List of client data partitions
    """
    n_classes = len(np.unique(y))
    n_samples = len(X)
    
    # Create class-wise sample indices
    class_indices = {}
    for class_id in range(n_classes):
        class_indices[class_id] = np.where(y == class_id)[0]
    
    client_data = []
    
    for client_id in range(num_clients):
        client_indices = []
        
        # For each class, determine how many samples this client gets
        for class_id in range(n_classes):
            class_samples = class_indices[class_id]
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet([alpha] * num_clients)
            client_proportion = proportions[client_id]
            
            # Determine number of samples for this client from this class
            n_client_samples = int(client_proportion * len(class_samples))
            
            # Randomly select samples
            if n_client_samples > 0:
                selected_indices = np.random.choice(
                    class_samples, size=min(n_client_samples, len(class_samples)), 
                    replace=False
                )
                client_indices.extend(selected_indices)
        
        # Ensure each client has at least some samples
        if len(client_indices) == 0:
            # Give random samples
            remaining_indices = list(range(n_samples))
            client_indices = np.random.choice(
                remaining_indices, size=max(1, n_samples // (num_clients * 2)), 
                replace=False
            )
        
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        client_data.append((X_client, y_client))
    
    return client_data

def _partition_pathological(X: np.ndarray, y: np.ndarray, 
                          num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Partition data pathologically (each client has samples from limited classes)
    
    Args:
        X: Feature array
        y: Target array
        num_clients: Number of clients
        
    Returns:
        List of client data partitions
    """
    n_classes = len(np.unique(y))
    classes_per_client = max(1, n_classes // 2)  # Each client gets samples from limited classes
    
    # Create class-wise sample indices
    class_indices = {}
    for class_id in range(n_classes):
        class_indices[class_id] = np.where(y == class_id)[0]
    
    client_data = []
    
    for client_id in range(num_clients):
        # Assign classes to this client
        start_class = (client_id * classes_per_client) % n_classes
        client_classes = []
        
        for i in range(classes_per_client):
            class_id = (start_class + i) % n_classes
            client_classes.append(class_id)
        
        # Collect samples from assigned classes
        client_indices = []
        for class_id in client_classes:
            if class_id in class_indices:
                class_samples = class_indices[class_id]
                # Split class samples among clients that have this class
                samples_per_client = len(class_samples) // (num_clients // classes_per_client + 1)
                start_idx = (client_id // classes_per_client) * samples_per_client
                end_idx = start_idx + samples_per_client
                
                if start_idx < len(class_samples):
                    selected_samples = class_samples[start_idx:min(end_idx, len(class_samples))]
                    client_indices.extend(selected_samples)
        
        # Ensure each client has some samples
        if len(client_indices) == 0:
            # Give random samples as fallback
            all_indices = list(range(len(X)))
            client_indices = np.random.choice(
                all_indices, size=len(X) // num_clients, replace=False
            )
        
        X_client = X[client_indices]
        y_client = y[client_indices]
        
        client_data.append((X_client, y_client))
    
    return client_data

def get_data_statistics(data: pd.DataFrame, target_column: str = 'Outcome') -> dict:
    """
    Get comprehensive statistics about the dataset
    
    Args:
        data: Dataset DataFrame
        target_column: Target column name
        
    Returns:
        Dictionary of dataset statistics
    """
    try:
        if data.empty:
            return {}
        
        stats = {
            'total_samples': len(data),
            'num_features': len(data.columns) - 1,
            'target_distribution': data[target_column].value_counts().to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'feature_stats': {}
        }
        
        # Feature statistics
        for column in data.columns:
            if column != target_column:
                col_stats = {
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'median': float(data[column].median())
                }
                stats['feature_stats'][column] = col_stats
        
        return stats
        
    except Exception as e:
        print(f"Error computing data statistics: {e}")
        return {}

def validate_partition_quality(client_data: List[Tuple[np.ndarray, np.ndarray]]) -> dict:
    """
    Validate the quality of data partitioning
    
    Args:
        client_data: List of client data partitions
        
    Returns:
        Dictionary of partition quality metrics
    """
    try:
        if not client_data:
            return {}
        
        total_samples = sum(len(X) for X, y in client_data)
        sample_counts = [len(X) for X, y in client_data]
        
        # Calculate distribution metrics
        min_samples = min(sample_counts)
        max_samples = max(sample_counts)
        mean_samples = np.mean(sample_counts)
        std_samples = np.std(sample_counts)
        
        # Calculate class distribution variance across clients
        class_distributions = []
        for X, y in client_data:
            if len(y) > 0:
                class_dist = np.bincount(y.astype(int), minlength=2)
                class_dist = class_dist / len(y)  # Normalize
                class_distributions.append(class_dist)
        
        if class_distributions:
            class_dist_variance = np.var(class_distributions, axis=0).mean()
        else:
            class_dist_variance = 0.0
        
        quality_metrics = {
            'total_samples': total_samples,
            'num_clients': len(client_data),
            'min_samples_per_client': min_samples,
            'max_samples_per_client': max_samples,
            'mean_samples_per_client': mean_samples,
            'std_samples_per_client': std_samples,
            'sample_distribution_cv': std_samples / mean_samples if mean_samples > 0 else 0,
            'class_distribution_variance': float(class_dist_variance),
            'balance_score': 1.0 - (std_samples / mean_samples) if mean_samples > 0 else 0
        }
        
        return quality_metrics
        
    except Exception as e:
        print(f"Error validating partition quality: {e}")
        return {}
