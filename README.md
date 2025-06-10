# Hierarchical Federated Deep Learning for Diabetes Prediction

A cutting-edge Streamlit-powered hierarchical federated learning platform for advanced diabetes prediction, combining medical and agricultural insights with innovative machine learning techniques.

## 🏥 Overview

This system implements a comprehensive 3-tier federated learning architecture (Patient → Fog → Global) designed specifically for healthcare applications. The platform enables collaborative machine learning across multiple medical facilities while preserving patient privacy through advanced differential privacy mechanisms.

## 🔬 Key Features

### Core Federated Learning Architecture
- **Hierarchical 3-Tier Federation**: Patient-level → Fog nodes → Global aggregation
- **Mathematical Protocol Implementation**: Complete implementation of polynomial parameter division and gradient-based updates
- **Committee-Based Security**: Multi-validator consensus mechanism for robust security
- **Advanced Differential Privacy**: Gaussian, Laplace, and Exponential mechanisms with privacy accounting

### Patient Risk Prediction System
- **Real-Time Risk Assessment**: Instant diabetes risk prediction with clinical decision support
- **Feature Importance Analysis**: SHAP-like explainable AI for medical interpretation
- **Population Comparison**: Comparative analysis against demographic cohorts
- **Clinical Decision Support**: Actionable insights for healthcare professionals

### Data Distribution Strategies
- **IID Distribution**: Independent and identically distributed data across clients
- **Non-IID Variants**: Dirichlet-based, pathological, quantity skew, and geographic distributions
- **Adaptive Client Management**: Automatic adjustment for varying dataset sizes
- **Robust Validation**: Multi-layer data integrity verification

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Streamlit
- NumPy, Pandas, Scikit-learn
- Plotly for visualizations

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit numpy pandas scikit-learn plotly matplotlib seaborn
   ```

### Running the Application
```bash
streamlit run app.py --server.port 5000
```

Access the application at `http://localhost:5000`

## 📊 System Architecture

### Hierarchical Protocol Flow
1. **Client Data Selection**: Each client selects portion d'i from local dataset Di
2. **Local Training**: Trains M_global_init with local data to get M_local_i
3. **Polynomial Division**: Divides model parameters using polynomial representation
4. **Fog Aggregation**: Partial FederatedAveraging at fog node level
5. **Gradient Updates**: M_local_i = M_local_i - η ∇ Fk M_global
6. **Global Convergence**: Iterative refinement until target accuracy achieved

### Privacy Protection
- **Differential Privacy**: Multiple mechanisms (Gaussian, Laplace, Exponential)
- **Privacy Accounting**: Advanced composition tracking for budget management
- **Gradient Clipping**: Sensitivity bounding for robust privacy guarantees
- **Committee Validation**: Consensus-based anomaly detection

## 🏥 Medical Application Features

### Training Configuration
- **Model Types**: Logistic Regression, Neural Networks, CNN, SVM, Random Forest
- **Client Simulation**: 3-10 medical facilities with realistic data distributions
- **Fog Topology**: Configurable hierarchical aggregation (2-5 fog nodes)
- **Privacy Settings**: Adjustable epsilon/delta parameters for differential privacy

### Performance Monitoring
- **Real-Time Metrics**: Accuracy, loss, convergence tracking across all levels
- **Hierarchical Visualization**: Performance metrics at client, fog, and global levels
- **Medical Insights**: Feature importance and clinical correlations
- **Training Progress**: Detailed round-by-round performance analysis

### Patient Risk Prediction
- **Risk Assessment**: Comprehensive diabetes risk evaluation
- **Feature Analysis**: Individual feature contribution analysis
- **Population Comparison**: Demographic and risk factor comparisons
- **Clinical Recommendations**: Actionable medical insights

## 📁 File Structure

### Core Components
- `app.py` - Main Streamlit application with complete UI and training orchestration
- `federated_learning.py` - Core federated learning manager with committee-based security
- `hierarchical_fl_protocol.py` - Mathematical protocol implementation
- `fog_aggregation.py` - Hierarchical fog node management and aggregation

### Data and Privacy
- `data_distribution.py` - Multiple data distribution strategies for realistic federation
- `data_preprocessing.py` - Medical data preprocessing pipeline
- `differential_privacy.py` - Advanced privacy mechanisms and accounting
- `client_simulator.py` - Realistic client behavior simulation

### Algorithms
- `aggregation_algorithms.py` - FedAvg, FedProx, and secure aggregation implementations
- `utils.py` - Utility functions for data handling and visualization

## 🔧 Configuration Options

### Training Parameters
- **Number of Clients**: 3-10 (representing medical facilities)
- **Training Rounds**: 10-50 rounds with early stopping
- **Model Selection**: Multiple ML algorithms optimized for medical data
- **Fog Nodes**: 2-5 hierarchical aggregation points

### Privacy Settings
- **Differential Privacy**: Epsilon (0.1-10.0), Delta (1e-8 to 1e-3)
- **Gradient Clipping**: Configurable L2 norm bounds
- **Privacy Mechanisms**: Gaussian, Laplace, Exponential noise addition
- **Committee Size**: 3-7 validators for security consensus

### Data Distribution
- **IID**: Balanced data distribution across clients
- **Non-IID**: Realistic medical facility data heterogeneity
- **Pathological**: Extreme non-IID for robustness testing
- **Geographic**: Location-based correlation patterns

## 📈 Performance Features

### Training Visualization
- **Real-Time Progress**: Live training metrics and convergence tracking
- **Hierarchical Metrics**: Performance at client, fog, and global levels
- **Privacy Budget**: Real-time privacy consumption monitoring
- **Feature Importance**: Medical feature correlation analysis

### Medical Insights
- **Risk Prediction**: Individual patient diabetes risk assessment
- **Population Analytics**: Demographic trend analysis
- **Clinical Decision Support**: Evidence-based medical recommendations
- **Comparative Analysis**: Cross-population risk factor evaluation

## 🔒 Security and Privacy

### Differential Privacy Implementation
- **Multiple Mechanisms**: Gaussian, Laplace, and Exponential noise addition
- **Advanced Composition**: Sophisticated privacy budget accounting
- **Adaptive Clipping**: Dynamic gradient norm adjustment
- **Local Privacy**: Individual client-level privacy protection

### Committee-Based Security
- **Consensus Validation**: Multi-validator agreement for update acceptance
- **Anomaly Detection**: Statistical outlier identification and filtering
- **Byzantine Robustness**: Protection against malicious client behavior
- **Secure Aggregation**: Cryptographically secure parameter combination

## 💡 Medical Use Cases

### Healthcare Applications
- **Multi-Hospital Collaboration**: Joint model training across medical networks
- **Privacy-Preserving Research**: Clinical research without data sharing
- **Population Health**: Large-scale diabetes risk assessment
- **Clinical Decision Support**: Evidence-based treatment recommendations

### Research Applications
- **Federated Medical Research**: Collaborative studies across institutions
- **Privacy-Preserving Analytics**: Statistical analysis without data exposure
- **Algorithm Validation**: Cross-institutional model validation
- **Demographic Studies**: Population-level health trend analysis

## 🎯 Future Enhancements

### Planned Features
- **Advanced Model Architectures**: Deep learning and transformer models
- **Real-Time Clinical Integration**: Direct EHR system connectivity
- **Expanded Privacy Mechanisms**: Homomorphic encryption and secure multi-party computation
- **Advanced Visualization**: Interactive medical data exploration tools

### Research Directions
- **Personalized Medicine**: Individual patient model customization
- **Predictive Analytics**: Advanced disease progression modeling
- **Multi-Modal Learning**: Integration of imaging, lab, and clinical data
- **Real-Time Monitoring**: Continuous patient health assessment

## 📚 Technical References

### Federated Learning
- Federated Averaging (FedAvg) algorithm implementation
- Hierarchical federated learning with fog computing
- Non-IID data distribution strategies for realistic medical scenarios
- Advanced differential privacy mechanisms for healthcare applications

### Medical AI
- Diabetes risk prediction using machine learning
- Explainable AI for clinical decision support
- Population health analytics and comparative analysis
- Privacy-preserving medical research methodologies

## 🤝 Contributing

This project demonstrates advanced federated learning techniques for healthcare applications. Contributions are welcome for:
- Additional privacy mechanisms
- Enhanced medical model architectures
- Improved clinical decision support features
- Extended visualization and analytics capabilities

## 📄 License

This project is designed for educational and research purposes, demonstrating state-of-the-art federated learning techniques in healthcare applications.

---

**Note**: This system demonstrates advanced federated learning concepts and should be adapted for production medical applications with appropriate regulatory compliance and additional security measures.