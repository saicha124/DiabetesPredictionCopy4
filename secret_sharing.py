"""
Hierarchical Secret Sharing Implementation for Federated Learning
================================================================

This module implements Shamir's Secret Sharing scheme for distributing
client model weights across fog nodes in a hierarchical federated learning
architecture.
"""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any
import random
from dataclasses import dataclass
from scipy.interpolate import lagrange


@dataclass
class SecretSharingConfig:
    """Configuration for secret sharing parameters"""
    num_fog_nodes: int = 5
    threshold: int = 3
    prime_modulus: int = 2**31 - 1  # Large prime for finite field arithmetic
    polynomial_degree: int = 2  # Will be set to threshold - 1
    
    def __post_init__(self):
        if self.polynomial_degree is None:
            self.polynomial_degree = max(1, self.threshold - 1)
        
        # Validate configuration
        if self.threshold > self.num_fog_nodes:
            raise ValueError("Threshold cannot exceed number of fog nodes")
        if self.threshold < 2:
            raise ValueError("Threshold must be at least 2")


class ShamirSecretSharing:
    """
    Shamir's Secret Sharing implementation for federated learning weights
    
    This class divides client model weights into shares that are distributed
    across fog nodes. The original weights can only be reconstructed when
    a threshold number of fog nodes cooperate.
    """
    
    def __init__(self, config: SecretSharingConfig):
        self.config = config
        self.prime = config.prime_modulus
    
    def _generate_polynomial_coefficients(self, secret: float) -> List[float]:
        """Generate random polynomial coefficients with secret as constant term"""
        coefficients = [secret]  # Secret is the constant term
        
        # Generate random coefficients for higher degree terms
        for _ in range(self.config.polynomial_degree):
            coeff = random.randint(1, self.prime - 1)
            coefficients.append(coeff)
        
        return coefficients
    
    def _evaluate_polynomial(self, coefficients: List[float], x: int) -> float:
        """Evaluate polynomial at point x using Horner's method"""
        result = coefficients[-1]
        for i in range(len(coefficients) - 2, -1, -1):
            result = (result * x + coefficients[i]) % self.prime
        return result
    
    def create_shares(self, secret: float) -> List[Tuple[int, float]]:
        """
        Create secret shares for a single weight value
        
        Args:
            secret: The secret weight value to be shared
            
        Returns:
            List of (fog_node_id, share_value) tuples
        """
        # Scale secret to work with integer arithmetic
        scaled_secret = int(secret * 1000000) % self.prime
        
        # Generate polynomial coefficients
        coefficients = self._generate_polynomial_coefficients(scaled_secret)
        
        # Create shares by evaluating polynomial at different points
        shares = []
        for fog_node_id in range(1, self.config.num_fog_nodes + 1):
            share_value = self._evaluate_polynomial(coefficients, fog_node_id)
            shares.append((fog_node_id, share_value))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, float]]) -> float:
        """
        Reconstruct secret from threshold number of shares using Lagrange interpolation
        
        Args:
            shares: List of (fog_node_id, share_value) tuples
            
        Returns:
            Reconstructed secret value
        """
        if len(shares) < self.config.threshold:
            raise ValueError(f"Need at least {self.config.threshold} shares to reconstruct")
        
        # Use only threshold number of shares
        shares = shares[:self.config.threshold]
        
        # Lagrange interpolation to find polynomial value at x=0 (the secret)
        secret = 0
        
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial
            basis = 1
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    basis *= (-xj) * pow(xi - xj, -1, self.prime)
                    basis %= self.prime
            
            secret += yi * basis
            secret %= self.prime
        
        # Scale back to original range
        return (secret / 1000000) % self.prime
    
    def share_model_weights(self, weights: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Share all model weights across fog nodes
        
        Args:
            weights: Model weights as numpy array
            
        Returns:
            Dictionary mapping fog_node_id to its share of all weights
        """
        fog_shares = {i: [] for i in range(1, self.config.num_fog_nodes + 1)}
        
        # Flatten weights for processing
        flat_weights = weights.flatten()
        
        # Create shares for each weight parameter
        for weight in flat_weights:
            shares = self.create_shares(weight)
            for fog_node_id, share_value in shares:
                fog_shares[fog_node_id].append(share_value)
        
        # Convert lists back to numpy arrays
        result = {}
        for fog_node_id in fog_shares:
            result[fog_node_id] = np.array(fog_shares[fog_node_id])
        
        return result
    
    def reconstruct_model_weights(self, fog_shares: Dict[int, np.ndarray], 
                                original_shape: Tuple) -> np.ndarray:
        """
        Reconstruct model weights from fog node shares
        
        Args:
            fog_shares: Dictionary mapping fog_node_id to weight shares
            original_shape: Original shape of weight matrix
            
        Returns:
            Reconstructed weights as numpy array
        """
        # Get threshold number of fog nodes
        available_nodes = list(fog_shares.keys())[:self.config.threshold]
        
        if len(available_nodes) < self.config.threshold:
            raise ValueError(f"Need shares from {self.config.threshold} fog nodes")
        
        # Get number of parameters
        num_params = len(fog_shares[available_nodes[0]])
        reconstructed_weights = []
        
        # Reconstruct each weight parameter
        for param_idx in range(num_params):
            shares = [(node_id, fog_shares[node_id][param_idx]) 
                     for node_id in available_nodes]
            reconstructed_weight = self.reconstruct_secret(shares)
            reconstructed_weights.append(reconstructed_weight)
        
        # Reshape back to original form
        return np.array(reconstructed_weights).reshape(original_shape)


class HierarchicalSecretSharing:
    """
    Hierarchical Secret Sharing for Federated Learning
    
    Manages secret sharing across the entire federated learning hierarchy:
    - Clients share weights with fog nodes
    - Fog nodes aggregate shares and forward to global server
    - Global server reconstructs final model
    """
    
    def __init__(self, config: SecretSharingConfig):
        self.config = config
        self.secret_sharing = ShamirSecretSharing(config)
        self.client_shares = {}  # client_id -> fog_shares
        self.fog_aggregated_shares = {}  # fog_id -> aggregated_shares
        
    def client_share_weights(self, client_id: int, weights: np.ndarray) -> Dict[int, np.ndarray]:
        """Client creates and distributes weight shares to fog nodes"""
        fog_shares = self.secret_sharing.share_model_weights(weights)
        self.client_shares[client_id] = {
            'original_shape': weights.shape,
            'fog_shares': fog_shares
        }
        return fog_shares
    
    def fog_aggregate_shares(self, fog_node_id: int, client_shares: Dict[int, np.ndarray]) -> np.ndarray:
        """Fog node aggregates shares from multiple clients"""
        if not client_shares:
            return np.array([])
        
        # Simple averaging of shares (can be more sophisticated)
        aggregated = np.zeros_like(list(client_shares.values())[0])
        
        for client_id, shares in client_shares.items():
            aggregated += shares
        
        aggregated /= len(client_shares)
        self.fog_aggregated_shares[fog_node_id] = aggregated
        
        return aggregated
    
    def global_reconstruct_model(self, fog_aggregated_shares: Dict[int, np.ndarray], 
                               original_shape: Tuple) -> np.ndarray:
        """Global server reconstructs final model from fog aggregated shares"""
        return self.secret_sharing.reconstruct_model_weights(
            fog_aggregated_shares, original_shape
        )
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Calculate security and performance metrics"""
        return {
            'information_theoretic_security': True,
            'threshold_security': f"{self.config.threshold}/{self.config.num_fog_nodes}",
            'collusion_resistance': f"Secure against {self.config.threshold-1} colluding nodes",
            'fault_tolerance': f"Tolerates {self.config.num_fog_nodes - self.config.threshold} node failures",
            'computational_overhead': f"O({self.config.threshold}²) per reconstruction",
            'communication_overhead': f"{self.config.num_fog_nodes}x parameter transmission"
        }


def create_secret_sharing_demo():
    """Create interactive demo of secret sharing functionality"""
    
    st.subheader("🔧 Secret Sharing Configuration")
    
    # Configuration panel
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_fog_nodes = st.slider("Number of Fog Nodes", min_value=3, max_value=10, value=5, key="ss_num_fog_nodes")
    
    with col2:
        threshold = st.slider("Reconstruction Threshold", min_value=2, max_value=num_fog_nodes, 
                            value=min(3, num_fog_nodes), key="ss_threshold")
    
    with col3:
        demo_weights_size = st.slider("Demo Weight Matrix Size", min_value=5, max_value=50, value=20, key="ss_demo_size")
    
    # Create configuration
    try:
        config = SecretSharingConfig(
            num_fog_nodes=num_fog_nodes,
            threshold=threshold
        )
        
        # Initialize hierarchical secret sharing
        hierarchical_ss = HierarchicalSecretSharing(config)
        
        st.success(f"✅ Configuration valid: {threshold}/{num_fog_nodes} threshold scheme")
        
    except ValueError as e:
        st.error(f"❌ Configuration error: {e}")
        return
    
    # Demo section
    st.subheader("🎮 Interactive Secret Sharing Demo")
    
    if st.button("🚀 Run Secret Sharing Demo", type="primary"):
        
        # Generate demo model weights
        np.random.seed(42)  # For reproducible demo
        original_weights = np.random.randn(demo_weights_size, demo_weights_size) * 0.1
        
        # Display original weights
        with st.expander("📊 Original Model Weights", expanded=False):
            fig = px.imshow(original_weights, title="Original Model Weights Matrix",
                          color_continuous_scale="RdBu", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Weight Matrix Shape", f"{original_weights.shape}")
                st.metric("Total Parameters", f"{original_weights.size:,}")
            with col2:
                st.metric("Mean Weight", f"{np.mean(original_weights):.6f}")
                st.metric("Weight Std Dev", f"{np.std(original_weights):.6f}")
        
        # Client shares weights
        st.write("### 1️⃣ Client Weight Sharing")
        client_id = 1
        fog_shares = hierarchical_ss.client_share_weights(client_id, original_weights)
        
        # Display shares distribution
        shares_data = []
        for fog_id, shares in fog_shares.items():
            shares_data.append({
                'Fog Node': f"Fog {fog_id}",
                'Share Size': len(shares),
                'Share Mean': np.mean(shares),
                'Share Std': np.std(shares)
            })
        
        import pandas as pd
        shares_df = pd.DataFrame(shares_data)
        st.dataframe(shares_df, use_container_width=True)
        
        # Visualize share distribution
        fig = go.Figure()
        for fog_id, shares in fog_shares.items():
            fig.add_trace(go.Histogram(
                x=shares[:100],  # Show first 100 shares for visibility
                name=f'Fog Node {fog_id}',
                opacity=0.7,
                nbinsx=20
            ))
        
        fig.update_layout(
            title="Distribution of Weight Shares Across Fog Nodes",
            xaxis_title="Share Values",
            yaxis_title="Frequency",
            barmode='overlay'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Fog aggregation simulation
        st.write("### 2️⃣ Fog Node Aggregation")
        
        # Simulate multiple clients
        num_clients = 3
        all_client_fog_shares = {}
        
        for client_id in range(1, num_clients + 1):
            client_weights = np.random.randn(demo_weights_size, demo_weights_size) * 0.1
            client_fog_shares = hierarchical_ss.client_share_weights(client_id, client_weights)
            all_client_fog_shares[client_id] = client_fog_shares
        
        # Each fog node aggregates its shares
        fog_aggregated = {}
        for fog_id in range(1, num_fog_nodes + 1):
            client_shares_for_fog = {
                client_id: client_fog_shares[fog_id] 
                for client_id, client_fog_shares in all_client_fog_shares.items()
            }
            aggregated_shares = hierarchical_ss.fog_aggregate_shares(fog_id, client_shares_for_fog)
            fog_aggregated[fog_id] = aggregated_shares
        
        st.success(f"✅ {num_fog_nodes} fog nodes aggregated shares from {num_clients} clients")
        
        # Global reconstruction
        st.write("### 3️⃣ Global Model Reconstruction")
        
        try:
            reconstructed_weights = hierarchical_ss.global_reconstruct_model(
                fog_aggregated, original_weights.shape
            )
            
            # Calculate reconstruction accuracy
            reconstruction_error = np.mean(np.abs(original_weights - reconstructed_weights))
            relative_error = reconstruction_error / np.mean(np.abs(original_weights))
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Reconstruction Error", f"{reconstruction_error:.8f}")
                st.metric("Relative Error", f"{relative_error:.6%}")
            with col2:
                if relative_error < 0.01:
                    st.success("🎯 Excellent reconstruction quality")
                elif relative_error < 0.05:
                    st.warning("⚠️ Good reconstruction quality")
                else:
                    st.error("❌ Poor reconstruction quality")
            
            # Visualize reconstruction comparison
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            im1 = axes[0].imshow(original_weights, cmap='RdBu', vmin=-0.3, vmax=0.3)
            axes[0].set_title('Original Weights')
            axes[0].axis('off')
            
            im2 = axes[1].imshow(reconstructed_weights, cmap='RdBu', vmin=-0.3, vmax=0.3)
            axes[1].set_title('Reconstructed Weights')
            axes[1].axis('off')
            
            difference = original_weights - reconstructed_weights
            im3 = axes[2].imshow(difference, cmap='RdBu', vmin=-0.1, vmax=0.1)
            axes[2].set_title('Reconstruction Error')
            axes[2].axis('off')
            
            plt.tight_layout()
            st.pyplot(plt.gcf())
            
        except Exception as e:
            st.error(f"Reconstruction failed: {e}")
        
        # Security metrics
        st.write("### 🔒 Security Analysis")
        security_metrics = hierarchical_ss.get_security_metrics()
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"🛡️ **Information-Theoretic Security:** {security_metrics['information_theoretic_security']}")
            st.info(f"🎯 **Threshold Security:** {security_metrics['threshold_security']}")
            st.info(f"🤝 **Collusion Resistance:** {security_metrics['collusion_resistance']}")
        
        with col2:
            st.info(f"⚡ **Fault Tolerance:** {security_metrics['fault_tolerance']}")
            st.info(f"💻 **Computational Overhead:** {security_metrics['computational_overhead']}")
            st.info(f"📡 **Communication Overhead:** {security_metrics['communication_overhead']}")


def integrate_secret_sharing_with_federated_learning(fl_manager, config: SecretSharingConfig):
    """
    Integrate secret sharing with existing federated learning manager
    
    Args:
        fl_manager: Existing FederatedLearningManager instance
        config: Secret sharing configuration
    """
    hierarchical_ss = HierarchicalSecretSharing(config)
    
    # Store original aggregation method
    original_aggregate = fl_manager.aggregator.aggregate
    
    def secret_sharing_aggregate(global_model, client_updates):
        """Modified aggregation with secret sharing"""
        
        # Extract weights from client updates
        client_weights = []
        for update in client_updates:
            if 'parameters' in update:
                weights = np.array(update['parameters'])
                client_weights.append(weights)
        
        if not client_weights:
            return original_aggregate(global_model, client_updates)
        
        # Apply secret sharing
        all_fog_shares = {}
        original_shape = client_weights[0].shape
        
        # Each client shares weights
        for client_id, weights in enumerate(client_weights):
            fog_shares = hierarchical_ss.client_share_weights(client_id, weights)
            all_fog_shares[client_id] = fog_shares
        
        # Fog nodes aggregate shares
        fog_aggregated = {}
        for fog_id in range(1, config.num_fog_nodes + 1):
            client_shares_for_fog = {
                client_id: client_fog_shares[fog_id] 
                for client_id, client_fog_shares in all_fog_shares.items()
            }
            if client_shares_for_fog:
                aggregated_shares = hierarchical_ss.fog_aggregate_shares(fog_id, client_shares_for_fog)
                fog_aggregated[fog_id] = aggregated_shares
        
        # Global reconstruction
        if len(fog_aggregated) >= config.threshold:
            reconstructed_weights = hierarchical_ss.global_reconstruct_model(
                fog_aggregated, original_shape
            )
            
            # Update global model with reconstructed weights
            # This is a simplified version - actual implementation would depend on model type
            return {
                'parameters': reconstructed_weights.tolist() if isinstance(reconstructed_weights, np.ndarray) else reconstructed_weights,
                'num_samples': sum(update.get('num_samples', 1) for update in client_updates),
                'secret_sharing_applied': True
            }
        else:
            # Fallback to original aggregation if insufficient fog nodes
            return original_aggregate(global_model, client_updates)
    
    # Replace aggregation method
    fl_manager.aggregator.aggregate = secret_sharing_aggregate
    
    return hierarchical_ss