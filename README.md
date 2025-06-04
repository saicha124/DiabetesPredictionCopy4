# Hierarchical Federated Deep Learning System

A comprehensive federated learning system with fog-level aggregation, differential privacy, and committee-based security for diabetes prediction using neural networks.

## 🏗️ System Architecture

### Core Components

#### 1. **Hierarchical Structure**
- **Clients**: Individual federated learning participants that perform local training on partitioned diabetes data
- **Fog Nodes**: Intermediate aggregation nodes that collect and aggregate client updates (no training performed)
- **Leader Fog**: Central coordinator that performs global aggregation of fog-level updates
- **Security Committee**: Dynamic committee system for secure validation and reputation management

#### 2. **Neural Network**
- Deep neural network specifically designed for diabetes prediction
- Configurable architecture with multiple hidden layers
- Dropout and batch normalization for regularization
- Binary classification output with sigmoid activation

#### 3. **Privacy & Security**
- **Differential Privacy**: Uses Laplace and Gaussian mechanisms for reputation masking
- **Secret Sharing**: Secure communication between nodes
- **Committee-based Validation**: Rotating committees for secure update validation
- **Attack Detection**: Protection against Sybil and Byzantine attacks

## 🔄 Training Workflow

### Round Execution Process

1. **Client Training Phase**
   - Each client performs local training on their data partition
   - Clients use global model weights from previous round (if available)
   - Local training produces updated model weights, loss, and accuracy metrics

2. **Fog Aggregation Phase**
   - Fog nodes collect updates from their assigned clients
   - Perform federated averaging (FedAvg) on client updates
   - No training occurs at fog level - only aggregation

3. **Global Aggregation Phase**
   - Leader fog collects aggregated updates from all fog nodes
   - Performs global aggregation using weighted averaging
   - Distributes new global model weights to all clients

4. **Committee & Reputation Update**
   - Form new security committee based on DP-masked reputation scores
   - Update client reputations based on training performance
   - Validate updates for potential attacks

### Communication Patterns

