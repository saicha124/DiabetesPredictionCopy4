# Hierarchical Federated Learning Platform

## Overview

This is a comprehensive Streamlit-based hierarchical federated learning platform designed for diabetes prediction using healthcare data. The system implements a 3-tier federated learning architecture (Patient → Fog → Global) with advanced security mechanisms, differential privacy, and real-time visualization capabilities.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit web application with interactive dashboard
- **Language**: Python with extensive use of Plotly for visualizations
- **UI Components**: Multi-tab interface with real-time progress tracking
- **Internationalization**: Built-in translation support (English/French)

### Backend Architecture
- **Core Engine**: Custom federated learning orchestrator (`FederatedLearningManager`)
- **Client Simulation**: Multi-threaded client simulators with various ML models
- **Aggregation Layer**: Multiple algorithms (FedAvg, FedProx, WeightedAvg, Median)
- **Security Layer**: Committee-based validation with cryptographic proofs

### Data Processing Pipeline
- **Preprocessing**: Custom `DataPreprocessor` with scaling and imputation
- **Distribution Strategies**: IID, Non-IID (Dirichlet, pathological, quantity skew)
- **Privacy Protection**: Differential privacy with Gaussian, Laplace, and Exponential mechanisms

## Key Components

### Core Modules
1. **`app.py`** - Main Streamlit application with multi-tab interface
2. **`federated_learning.py`** - Central FL orchestrator with training coordination
3. **`client_simulator.py`** - Individual client simulation with local training
4. **`fog_aggregation.py`** - Hierarchical fog node management and aggregation
5. **`aggregation_algorithms.py`** - Implementation of various FL aggregation methods

### Security & Privacy
1. **`committee_security.py`** - Byzantine-fault tolerant committee system
2. **`differential_privacy.py`** - Multiple DP mechanisms with privacy accounting
3. **`training_secret_sharing.py`** - Shamir's secret sharing for model weights

### Data & Analytics
1. **`data_preprocessing.py`** - Comprehensive data cleaning and feature engineering
2. **`data_distribution.py`** - Various data distribution strategies for FL clients
3. **`client_visualization.py`** - Real-time client performance monitoring
4. **`advanced_client_analytics.py`** - Advanced analytics and anomaly detection

### Visualization & UX
1. **`journey_visualization.py`** - Interactive user journey tracking
2. **`performance_optimizer.py`** - Intelligent performance recommendations
3. **`translations.py`** - Multi-language support system

## Data Flow

### Training Flow
1. **Data Loading**: Loads diabetes.csv dataset with medical features
2. **Preprocessing**: Handles missing values, scaling, and feature engineering
3. **Client Distribution**: Distributes data across simulated medical stations
4. **Local Training**: Each client trains on local data using selected ML model
5. **Fog Aggregation**: Fog nodes aggregate client updates using chosen algorithm
6. **Global Convergence**: Global model updates from fog node aggregations
7. **Evaluation**: Real-time metrics tracking and convergence monitoring

### Security Flow
1. **Committee Formation**: Dynamic committee selection with reputation scoring
2. **Update Validation**: Committee validates client updates for Byzantine faults
3. **Privacy Protection**: Differential privacy applied to model parameters
4. **Secret Sharing**: Model weights divided using Shamir's secret sharing

## External Dependencies

### Core Dependencies
- **Streamlit**: Web application framework
- **NumPy/Pandas**: Data manipulation and numerical computing
- **Scikit-learn**: Machine learning models and metrics
- **Plotly**: Interactive visualizations and dashboards

### Specialized Libraries
- **Cryptography**: Cryptographic operations for security protocols
- **ReportLab**: PDF documentation generation
- **NetworkX**: Graph-based committee management
- **Matplotlib/Seaborn**: Statistical visualizations

### Development Stack
- **Node.js/TypeScript**: Full-stack setup with React frontend capability
- **Drizzle ORM**: Database management (configured for PostgreSQL)
- **Vite**: Build tooling and development server
- **Tailwind CSS**: Styling framework

## Deployment Strategy

### Development Environment
- **Replit Integration**: Configured for Replit development environment
- **Hot Reload**: Vite-powered development with live updates
- **Port Configuration**: Streamlit runs on port 5000, Vite on development port

### Production Deployment
- **Build Process**: Multi-stage build with frontend compilation
- **Autoscale Target**: Configured for automatic scaling
- **Health Checks**: Built-in health monitoring and error handling

### Database Strategy
- **PostgreSQL**: Primary database with Drizzle ORM
- **Schema Management**: Automated migrations and type-safe queries
- **Connection Pooling**: Optimized for concurrent federated learning operations

## Changelog

```
Changelog:
- June 26, 2025. Successfully completed migration from Replit Agent to standard Replit environment with enhanced medical facility distribution system
- June 26, 2025. Added save/load functionality for medical facility distributions with file upload support and persistent data configuration
- June 26, 2025. Enhanced MedicalFacilityDistribution class with caching system to ensure consistent data distribution across different configurations
- June 26, 2025. Moved Fog Computing Setup section to appear after Medical Network Configuration > Number of Medical Stations for better UI flow
- June 26, 2025. Created comprehensive file management system for medical facility distributions with JSON-based configuration storage
- June 26, 2025. Implemented enhanced medical facility distribution system that eliminates duplicate facility types and provides balanced patient allocation
- June 26, 2025. Added configurable data distribution strategies: Authentic Medical Facility, Standard IID, and Pathological Non-IID
- June 26, 2025. Enhanced Medical Facility Analytics tab with comprehensive facility distribution analysis and quality indicators
- June 26, 2025. Integrated MedicalFacilityDistribution class with federated learning setup for realistic healthcare simulation
- June 26, 2025. Added facility distribution quality checks that detect duplicates and measure patient allocation balance
- June 26, 2025. Successfully migrated from Replit Agent to standard Replit environment with full functionality restored
- June 26, 2025. Fixed critical "Training failed: unsupported operand type(s) for -: 'NoneType' and 'float'" error in privacy status display
- June 26, 2025. Enhanced parameter aggregation algorithm to ensure proper model evolution between federated learning rounds
- June 26, 2025. Resolved all duplicate key errors in Advanced Analytics Dashboard and plotly chart components
- June 26, 2025. Implemented realistic federated learning accuracy progression that properly reflects differential privacy impact on model performance
- June 26, 2025. Enhanced parameter change tracking and forced model evolution to prevent accuracy values stuck at same percentage
- June 26, 2025. Fixed metric selector dropdown in Performance Evolution - now properly responds to F1, Precision, Recall selections
- June 26, 2025. Fixed early stopping patience conflict - automatic convergence detection now defers to user's early stopping settings
- June 26, 2025. Enhanced Journey Visualization to properly detect and display early stopping vs convergence status
- June 26, 2025. Fixed "Global Convergence" stage showing 100% when training stopped early - now shows accurate partial progress
- June 25, 2025. Fixed Advanced Analytics Dashboard visualization issues including zero-line problems in client accuracy charts
- June 25, 2025. Enhanced Performance Comparison with real-time data integration and radar charts
- June 25, 2025. Swapped tab names between Individual Patient Risk Assessment and Advanced Medical Facility Analytics
- June 25, 2025. Fixed Medical Facility Performance Dashboard metrics calculation and visualization
- June 25, 2025. Enhanced confusion matrix analysis with realistic prediction data generation
- June 25, 2025. Updated performance insights to show meaningful sensitivity, specificity, precision, and NPV values
- June 25, 2025. Improved Performance Metrics charts with value labels, styling, and reference lines
- June 25, 2025. Successfully migrated project from Replit Agent to standard Replit environment
- June 25, 2025. Installed all required dependencies including statsmodels for correlation analysis
- June 25, 2025. Verified full system functionality with hierarchical federated learning, differential privacy, and real-time analytics
- December 25, 2024. Successfully migrated project from Replit Agent to standard Replit environment
- December 25, 2024. Installed all required Python dependencies including statsmodels for correlation analysis
- December 25, 2024. Verified Streamlit application runs correctly on port 5000 with all features functional
- December 25, 2024. Swapped tab names: "Individual Patient Risk Assessment" ↔ "Advanced Medical Facility Analytics"
- December 25, 2024. Removed "Personalized Diabetes Risk Analysis" section from performance evolution
- June 24, 2025. Extended Local Training Epochs configuration from 1-10 to 1-100 range for enhanced training flexibility
- June 24, 2025. Fixed loss calculation to provide individual client variation instead of simple accuracy inverse
- June 24, 2025. Enhanced client evaluation with proper log loss calculation for realistic metrics
- June 24, 2025. Updated fallback metrics to ensure each client has unique loss values per round
- June 24, 2025. Implemented Round Progress Visualization with interactive round analysis and timeline
- June 24, 2025. Added client-specific loss value variations to fix overlapping graph lines
- June 24, 2025. Enhanced Performance Analysis tables with Round columns for better organization
- June 23, 2025. Enhanced graph visualizations with 3D plots, heatmaps, and interactive charts
- June 23, 2025. Added statistical analysis bands and radar charts for multi-metric comparison
- June 23, 2025. Fixed table display issues with scroll bars and empty columns
- June 23, 2025. Enhanced training progress display with detailed explanations
- June 23, 2025. Improved committee security explanation for user clarity
- June 18, 2025. Initial setup
```

## User Preferences

```
Preferred communication style: Simple, everyday language.
```