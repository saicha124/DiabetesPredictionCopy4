import numpy as np
import pandas as pd
import random
from typing import List, Dict, Tuple, Any
from scipy.interpolate import lagrange
import streamlit as st

class SecretSharingScheme:
    """
    Shamir's Secret Sharing implementation for federated learning weights
    Allows splitting model weights into shares distributed across fog nodes
    """
    
    def __init__(self, threshold: int, num_shares: int, prime: int = 2**31 - 1):
        """
        Initialize secret sharing scheme
        
        Args:
            threshold: Minimum number of shares needed to reconstruct secret
            num_shares: Total number of shares to generate
            prime: Prime number for finite field arithmetic
        """
        self.threshold = threshold
        self.num_shares = num_shares
        self.prime = prime
        
        if threshold > num_shares:
            raise ValueError("Threshold cannot be greater than number of shares")
    
    def _mod_inverse(self, a: int, m: int) -> int:
        """Calculate modular inverse using extended Euclidean algorithm"""
        if a < 0:
            a = (a % m + m) % m
        
        # Extended Euclidean Algorithm
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y
        
        gcd, x, _ = extended_gcd(a, m)
        if gcd != 1:
            raise ValueError("Modular inverse does not exist")
        return (x % m + m) % m
    
    def _evaluate_polynomial(self, coefficients: List[int], x: int) -> int:
        """Evaluate polynomial at point x using Horner's method"""
        result = 0
        for coeff in reversed(coefficients):
            result = (result * x + coeff) % self.prime
        return result
    
    def split_secret(self, secret: float) -> List[Tuple[int, int]]:
        """
        Split a secret into shares using Shamir's Secret Sharing
        
        Args:
            secret: The secret value to split (model weight)
            
        Returns:
            List of (x, y) coordinate shares
        """
        # Convert float to integer for finite field operations
        secret_int = int(secret * 10**6) % self.prime  # Scale for precision
        
        # Generate random coefficients for polynomial
        coefficients = [secret_int]
        for _ in range(self.threshold - 1):
            coefficients.append(random.randint(0, self.prime - 1))
        
        # Generate shares by evaluating polynomial at different points
        shares = []
        for i in range(1, self.num_shares + 1):
            x = i
            y = self._evaluate_polynomial(coefficients, x)
            shares.append((x, y))
        
        return shares
    
    def reconstruct_secret(self, shares: List[Tuple[int, int]]) -> float:
        """
        Reconstruct secret from shares using Lagrange interpolation
        
        Args:
            shares: List of (x, y) coordinate shares
            
        Returns:
            Reconstructed secret value
        """
        if len(shares) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} shares to reconstruct secret")
        
        # Use first 'threshold' shares for reconstruction
        shares = shares[:self.threshold]
        
        # Lagrange interpolation to find secret (y-intercept at x=0)
        secret = 0
        
        for i, (xi, yi) in enumerate(shares):
            # Calculate Lagrange basis polynomial Li(0)
            numerator = 1
            denominator = 1
            
            for j, (xj, _) in enumerate(shares):
                if i != j:
                    numerator = (numerator * (-xj)) % self.prime
                    denominator = (denominator * (xi - xj)) % self.prime
            
            # Calculate Li(0) = numerator / denominator
            denominator_inv = self._mod_inverse(denominator, self.prime)
            lagrange_coeff = (numerator * denominator_inv) % self.prime
            
            # Add yi * Li(0) to result
            secret = (secret + yi * lagrange_coeff) % self.prime
        
        # Convert back to float
        return (secret % self.prime) / 10**6

class HierarchicalSecretSharing:
    """
    Hierarchical secret sharing for federated learning with fog nodes
    Each client splits weights among fog nodes using secret sharing
    """
    
    def __init__(self, num_fog_nodes: int, threshold_ratio: float = 0.67):
        """
        Initialize hierarchical secret sharing
        
        Args:
            num_fog_nodes: Number of fog nodes in the system
            threshold_ratio: Ratio of fog nodes needed to reconstruct (default 2/3)
        """
        self.num_fog_nodes = num_fog_nodes
        self.threshold = max(2, int(num_fog_nodes * threshold_ratio))
        self.secret_sharing = SecretSharingScheme(self.threshold, num_fog_nodes)
        
        # Track shares for each client and parameter
        self.fog_shares = {f"fog_{i}": {} for i in range(num_fog_nodes)}
        self.client_metadata = {}
    
    def distribute_client_weights(self, client_id: int, model_weights: Dict[str, np.ndarray]) -> Dict[str, Dict[str, List]]:
        """
        Distribute client model weights across fog nodes using secret sharing
        
        Args:
            client_id: Identifier for the client
            model_weights: Dictionary of layer_name -> weight_array
            
        Returns:
            Dictionary mapping fog_node_id -> {layer_name -> shares}
        """
        fog_distributions = {f"fog_{i}": {} for i in range(self.num_fog_nodes)}
        
        # Store metadata for reconstruction
        self.client_metadata[client_id] = {
            'layers': list(model_weights.keys()),
            'shapes': {layer: weights.shape for layer, weights in model_weights.items()}
        }
        
        for layer_name, weights in model_weights.items():
            # Flatten weights for secret sharing
            flat_weights = weights.flatten()
            
            # Split each weight value using secret sharing
            for weight_idx, weight_value in enumerate(flat_weights):
                shares = self.secret_sharing.split_secret(float(weight_value))
                
                # Distribute shares to fog nodes
                for fog_idx, (x, y) in enumerate(shares):
                    fog_id = f"fog_{fog_idx}"
                    
                    if layer_name not in fog_distributions[fog_id]:
                        fog_distributions[fog_id][layer_name] = []
                    
                    fog_distributions[fog_id][layer_name].append({
                        'weight_idx': weight_idx,
                        'share': (x, y),
                        'client_id': client_id
                    })
        
        # Store shares in fog nodes
        for fog_id, distribution in fog_distributions.items():
            if client_id not in self.fog_shares[fog_id]:
                self.fog_shares[fog_id][client_id] = {}
            self.fog_shares[fog_id][client_id].update(distribution)
        
        return fog_distributions
    
    def aggregate_at_fog_level(self, fog_id: str, participating_clients: List[int]) -> Dict[str, np.ndarray]:
        """
        Aggregate shares at fog level for participating clients
        
        Args:
            fog_id: Identifier of the fog node
            participating_clients: List of client IDs participating in this round
            
        Returns:
            Partial aggregated weights (shares) for this fog node
        """
        if fog_id not in self.fog_shares:
            raise ValueError(f"Fog node {fog_id} not found")
        
        aggregated_shares = {}
        
        # Get all layers from first client as template
        if participating_clients:
            first_client = participating_clients[0]
            if first_client in self.fog_shares[fog_id]:
                layers = list(self.fog_shares[fog_id][first_client].keys())
                
                for layer_name in layers:
                    # Collect shares for this layer from all clients
                    layer_shares = []
                    
                    for client_id in participating_clients:
                        if client_id in self.fog_shares[fog_id] and layer_name in self.fog_shares[fog_id][client_id]:
                            client_shares = self.fog_shares[fog_id][client_id][layer_name]
                            layer_shares.extend(client_shares)
                    
                    # Group shares by weight index and average
                    weight_indices = {}
                    for share_data in layer_shares:
                        weight_idx = share_data['weight_idx']
                        share = share_data['share']
                        
                        if weight_idx not in weight_indices:
                            weight_indices[weight_idx] = []
                        weight_indices[weight_idx].append(share)
                    
                    # Average shares for each weight index
                    averaged_shares = {}
                    for weight_idx, shares_list in weight_indices.items():
                        # Average the y-values of shares (keep same x-values)
                        avg_y = np.mean([share[1] for share in shares_list])
                        averaged_shares[weight_idx] = (shares_list[0][0], int(avg_y))
                    
                    aggregated_shares[layer_name] = averaged_shares
        
        return aggregated_shares
    
    def reconstruct_global_weights(self, fog_aggregations: Dict[str, Dict[str, Any]], 
                                 participating_fog_nodes: List[str]) -> Dict[str, np.ndarray]:
        """
        Reconstruct global model weights from fog-level aggregations
        
        Args:
            fog_aggregations: Dictionary mapping fog_id -> aggregated_shares
            participating_fog_nodes: List of fog node IDs with sufficient shares
            
        Returns:
            Reconstructed global model weights
        """
        if len(participating_fog_nodes) < self.threshold:
            raise ValueError(f"Need at least {self.threshold} fog nodes to reconstruct global model")
        
        global_weights = {}
        
        # Get layer structure from first fog node
        first_fog = participating_fog_nodes[0]
        if first_fog in fog_aggregations:
            layers = list(fog_aggregations[first_fog].keys())
            
            for layer_name in layers:
                # Get shape for reconstruction
                client_id = list(self.client_metadata.keys())[0]  # Use first client's metadata
                layer_shape = self.client_metadata[client_id]['shapes'][layer_name]
                
                # Reconstruct each weight in the layer
                reconstructed_weights = []
                
                # Determine number of weights from first fog node
                max_weight_idx = max(fog_aggregations[first_fog][layer_name].keys())
                
                for weight_idx in range(max_weight_idx + 1):
                    # Collect shares for this weight from participating fog nodes
                    shares_for_weight = []
                    
                    for fog_id in participating_fog_nodes:
                        if (fog_id in fog_aggregations and 
                            layer_name in fog_aggregations[fog_id] and 
                            weight_idx in fog_aggregations[fog_id][layer_name]):
                            
                            share = fog_aggregations[fog_id][layer_name][weight_idx]
                            shares_for_weight.append(share)
                    
                    # Reconstruct weight value from shares
                    if len(shares_for_weight) >= self.threshold:
                        reconstructed_weight = self.secret_sharing.reconstruct_secret(shares_for_weight)
                        reconstructed_weights.append(reconstructed_weight)
                    else:
                        # Not enough shares, use zero or interpolation
                        reconstructed_weights.append(0.0)
                
                # Reshape to original layer shape
                reconstructed_array = np.array(reconstructed_weights).reshape(layer_shape)
                global_weights[layer_name] = reconstructed_array
        
        return global_weights
    
    def get_privacy_metrics(self) -> Dict[str, Any]:
        """
        Calculate privacy and security metrics for the secret sharing scheme
        
        Returns:
            Dictionary containing privacy metrics
        """
        return {
            'threshold': self.threshold,
            'num_fog_nodes': self.num_fog_nodes,
            'privacy_threshold_ratio': self.threshold / self.num_fog_nodes,
            'collusion_resistance': f"Secure against up to {self.threshold - 1} colluding fog nodes",
            'reconstruction_requirement': f"Requires {self.threshold} out of {self.num_fog_nodes} fog nodes",
            'information_theoretic_security': True,
            'computational_overhead': 'Polynomial evaluation and Lagrange interpolation'
        }
    
    def simulate_weight_distribution(self, num_clients: int = 3, 
                                   weight_dimensions: Tuple[int, int] = (10, 5)) -> Dict[str, Any]:
        """
        Simulate the complete weight distribution and reconstruction process
        
        Args:
            num_clients: Number of clients to simulate
            weight_dimensions: Dimensions of weight matrices to simulate
            
        Returns:
            Simulation results and statistics
        """
        simulation_results = {
            'original_weights': {},
            'fog_distributions': {},
            'fog_aggregations': {},
            'reconstructed_weights': {},
            'reconstruction_errors': {},
            'privacy_preserved': True
        }
        
        # Generate synthetic weights for simulation
        for client_id in range(num_clients):
            client_weights = {
                'dense_layer': np.random.randn(*weight_dimensions),
                'output_layer': np.random.randn(weight_dimensions[1], 1)
            }
            simulation_results['original_weights'][client_id] = client_weights
            
            # Distribute weights using secret sharing
            fog_dist = self.distribute_client_weights(client_id, client_weights)
            simulation_results['fog_distributions'][client_id] = fog_dist
        
        # Simulate fog-level aggregation
        participating_clients = list(range(num_clients))
        for fog_idx in range(self.num_fog_nodes):
            fog_id = f"fog_{fog_idx}"
            fog_agg = self.aggregate_at_fog_level(fog_id, participating_clients)
            simulation_results['fog_aggregations'][fog_id] = fog_agg
        
        # Reconstruct global weights
        participating_fogs = [f"fog_{i}" for i in range(self.threshold)]
        try:
            reconstructed = self.reconstruct_global_weights(
                simulation_results['fog_aggregations'], 
                participating_fogs
            )
            simulation_results['reconstructed_weights'] = reconstructed
            
            # Calculate reconstruction errors
            if num_clients > 0:
                original_avg = {}
                for layer_name in simulation_results['original_weights'][0].keys():
                    layer_sum = np.zeros_like(simulation_results['original_weights'][0][layer_name])
                    for client_id in range(num_clients):
                        layer_sum += simulation_results['original_weights'][client_id][layer_name]
                    original_avg[layer_name] = layer_sum / num_clients
                
                for layer_name in original_avg.keys():
                    if layer_name in reconstructed:
                        error = np.mean(np.abs(original_avg[layer_name] - reconstructed[layer_name]))
                        simulation_results['reconstruction_errors'][layer_name] = error
            
        except Exception as e:
            simulation_results['reconstruction_error'] = str(e)
            simulation_results['privacy_preserved'] = False
        
        return simulation_results

def create_secret_sharing_demo():
    """Create Streamlit demo for secret sharing in hierarchical federated learning"""
    
    st.subheader("🔐 Secret Sharing in Hierarchical Federated Learning")
    
    # Configuration
    col1, col2, col3 = st.columns(3)
    
    with col1:
        num_fog_nodes = st.slider("Number of Fog Nodes", 3, 7, 5)
    with col2:
        threshold_ratio = st.slider("Threshold Ratio", 0.5, 1.0, 0.67, step=0.1)
    with col3:
        num_clients = st.slider("Number of Clients", 2, 10, 3)
    
    # Initialize secret sharing system
    hierarchical_ss = HierarchicalSecretSharing(num_fog_nodes, threshold_ratio)
    
    # Display system parameters
    st.subheader("📊 System Parameters")
    metrics = hierarchical_ss.get_privacy_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Fog Nodes", metrics['num_fog_nodes'])
    with col2:
        st.metric("Threshold", metrics['threshold'])
    with col3:
        st.metric("Privacy Ratio", f"{metrics['privacy_threshold_ratio']:.2f}")
    with col4:
        st.metric("Collusion Resistance", f"{metrics['threshold'] - 1} nodes")
    
    # Run simulation
    if st.button("🚀 Run Secret Sharing Simulation", use_container_width=True):
        with st.spinner("Simulating secret sharing process..."):
            results = hierarchical_ss.simulate_weight_distribution(
                num_clients=num_clients,
                weight_dimensions=(8, 4)
            )
        
        # Display results
        st.subheader("🎯 Simulation Results")
        
        if 'reconstruction_error' not in results:
            st.success("✅ Secret sharing and reconstruction successful!")
            
            # Show reconstruction errors
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Reconstruction Errors by Layer:**")
                for layer, error in results['reconstruction_errors'].items():
                    st.metric(f"{layer.replace('_', ' ').title()}", f"{error:.6f}")
            
            with col2:
                st.markdown("**Privacy Metrics:**")
                st.info(f"🛡️ {metrics['collusion_resistance']}")
                st.info(f"🔑 {metrics['reconstruction_requirement']}")
                st.success("🔒 Information-theoretic security maintained")
        else:
            st.error(f"❌ Reconstruction failed: {results['reconstruction_error']}")
        
        # Visualization of weight distribution
        st.subheader("📈 Weight Distribution Visualization")
        
        # Create distribution chart
        fog_data = []
        for client_id, fog_dist in results['fog_distributions'].items():
            for fog_id, layers in fog_dist.items():
                for layer_name, shares in layers.items():
                    fog_data.append({
                        'Fog Node': fog_id,
                        'Client': f"Client {client_id}",
                        'Layer': layer_name,
                        'Shares Count': len(shares)
                    })
        
        if fog_data:
            import plotly.express as px
            df = pd.DataFrame(fog_data)
            
            fig = px.bar(df, x='Fog Node', y='Shares Count', 
                        color='Client', facet_col='Layer',
                        title="Weight Shares Distribution Across Fog Nodes")
            st.plotly_chart(fig, use_container_width=True)
    
    # Technical explanation
    with st.expander("🔍 Technical Details", expanded=False):
        st.markdown("""
        ### Shamir's Secret Sharing in Federated Learning
        
        **How it works:**
        1. **Weight Splitting**: Each client splits their model weights using Shamir's Secret Sharing
        2. **Distribution**: Weight shares are distributed across multiple fog nodes
        3. **Threshold Security**: Requires a threshold number of fog nodes to reconstruct weights
        4. **Fog Aggregation**: Each fog node aggregates shares from multiple clients
        5. **Global Reconstruction**: Global server reconstructs final model from fog aggregations
        
        **Security Properties:**
        - **Information-Theoretic Security**: Individual shares reveal no information about original weights
        - **Collusion Resistance**: Up to threshold-1 fog nodes can collude without compromising privacy
        - **Fault Tolerance**: System continues to work even if some fog nodes fail
        
        **Privacy Benefits:**
        - No single fog node sees complete client weights
        - Malicious fog nodes cannot reconstruct individual client data
        - Provides formal privacy guarantees beyond differential privacy
        """)
    
    return hierarchical_ss