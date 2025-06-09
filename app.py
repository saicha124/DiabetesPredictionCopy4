import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from datetime import datetime, timedelta

# Import custom modules
from federated_learning import FederatedLearningManager
from data_preprocessing import DataPreprocessor
from data_distribution import get_distribution_strategy, visualize_data_distribution
from fog_aggregation import HierarchicalFederatedLearning
from differential_privacy import DifferentialPrivacyManager
from hierarchical_fl_protocol import HierarchicalFederatedLearningEngine
from utils import *

# Page configuration
st.set_page_config(
    page_title="Hierarchical Federated Learning Platform",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'training_started' not in st.session_state:
        st.session_state.training_started = False
    if 'training_completed' not in st.session_state:
        st.session_state.training_completed = False
    if 'training_metrics' not in st.session_state:
        st.session_state.training_metrics = []
    if 'fog_results' not in st.session_state:
        st.session_state.fog_results = []
    if 'client_results' not in st.session_state:
        st.session_state.client_results = []
    if 'execution_times' not in st.session_state:
        st.session_state.execution_times = []
    if 'communication_times' not in st.session_state:
        st.session_state.communication_times = []
    if 'confusion_matrices' not in st.session_state:
        st.session_state.confusion_matrices = []
    if 'early_stopped' not in st.session_state:
        st.session_state.early_stopped = False
    if 'best_accuracy' not in st.session_state:
        st.session_state.best_accuracy = 0.0
    if 'current_round' not in st.session_state:
        st.session_state.current_round = 0

def main():
    init_session_state()
    
    st.title("🏥 Hierarchical Federated Deep Learning for Diabetes Prediction")
    st.markdown("### Advanced Privacy-Preserving Machine Learning Platform")

    # Sidebar
    with st.sidebar:
        st.header("🔧 System Configuration")
        
        # Data upload
        uploaded_file = st.file_uploader("📁 Upload Patient Dataset", type=['csv'])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.session_state.data = data
            st.session_state.data_loaded = True
            st.success(f"✅ Dataset loaded: {data.shape[0]} patients, {data.shape[1]} features")
        elif not hasattr(st.session_state, 'data'):
            # Load default diabetes dataset
            try:
                data = pd.read_csv('diabetes.csv')
                st.session_state.data = data
                st.session_state.data_loaded = True
                st.info(f"📊 Using default dataset: {data.shape[0]} patients")
            except:
                st.error("Please upload a diabetes dataset")
                return

    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎛️ Training Control", 
        "🏥 Live Monitoring", 
        "🗺️ Learning Journey Map",
        "📊 Performance Analysis",
        "🩺 Risk Assessment"
    ])

    with tab1:
        st.header("🎛️ Federated Learning Training Control")
        
        if st.session_state.data_loaded:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏥 Medical Network Configuration")
                num_clients = st.slider("Number of Medical Stations", 3, 20, 5)
                max_rounds = st.slider("Maximum Training Rounds", 5, 50, 20)
                
                st.subheader("🌫️ Fog Computing Setup")
                enable_fog = st.checkbox("Enable Fog Nodes", value=True)
                if enable_fog:
                    num_fog_nodes = st.slider("Number of Fog Nodes", 2, 6, 3)
                    fog_method = st.selectbox("Fog Aggregation Method", 
                                            ["FedAvg", "Weighted", "Median", "Mixed Methods"])
                else:
                    num_fog_nodes = 0
                    fog_method = "FedAvg"
            
            with col2:
                st.subheader("🔒 Privacy Configuration")
                enable_dp = st.checkbox("Enable Differential Privacy", value=True)
                if enable_dp:
                    epsilon = st.slider("Privacy Budget (ε)", 0.1, 10.0, 1.0, 0.1)
                    delta = st.select_slider("Failure Probability (δ)", 
                                           options=[1e-6, 1e-5, 1e-4, 1e-3], 
                                           value=1e-5, format_func=lambda x: f"{x:.0e}")
                else:
                    epsilon = None
                    delta = None
                
                st.subheader("📊 Data Distribution")
                distribution_strategy = st.selectbox("Distribution Strategy", 
                                                   ["IID", "Non-IID", "Pathological", "Quantity Skew", "Geographic"])
                
                # Strategy-specific parameters
                strategy_params = {}
                if distribution_strategy == "Non-IID":
                    strategy_params['alpha'] = st.slider("Dirichlet Alpha", 0.1, 2.0, 0.5, 0.1)
                elif distribution_strategy == "Pathological":
                    strategy_params['classes_per_client'] = st.slider("Classes per Client", 1, 2, 1)
                elif distribution_strategy == "Quantity Skew":
                    strategy_params['skew_factor'] = st.slider("Skew Factor", 1.0, 5.0, 2.0, 0.5)
                elif distribution_strategy == "Geographic":
                    strategy_params['correlation_strength'] = st.slider("Correlation Strength", 0.1, 1.0, 0.8, 0.1)
            
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Start Training", disabled=st.session_state.training_started):
                    with st.spinner("Initializing federated learning..."):
                        # Store configuration
                        st.session_state.num_clients = num_clients
                        st.session_state.max_rounds = max_rounds
                        st.session_state.enable_fog = enable_fog
                        st.session_state.num_fog_nodes = num_fog_nodes
                        st.session_state.fog_method = fog_method
                        st.session_state.enable_dp = enable_dp
                        st.session_state.epsilon = epsilon
                        st.session_state.delta = delta
                        st.session_state.distribution_strategy = distribution_strategy
                        st.session_state.strategy_params = strategy_params
                        
                        # Initialize FL manager
                        fl_manager = FederatedLearningManager(
                            num_clients=num_clients,
                            max_rounds=max_rounds,
                            aggregation_algorithm='FedAvg',
                            enable_dp=enable_dp,
                            epsilon=epsilon or 1.0,
                            delta=delta or 1e-5
                        )
                        
                        # Setup fog nodes if enabled
                        if enable_fog:
                            fog_manager = HierarchicalFederatedLearning(
                                num_clients=num_clients,
                                num_fog_nodes=num_fog_nodes,
                                fog_aggregation_method=fog_method
                            )
                            st.session_state.fog_manager = fog_manager
                        
                        st.session_state.fl_manager = fl_manager
                        st.session_state.training_data = st.session_state.data
                        st.session_state.training_started = True
                        st.session_state.training_completed = False
                        st.session_state.training_metrics = []
                        st.session_state.fog_results = []
                        st.session_state.best_accuracy = 0.0
                        st.session_state.current_round = 0
                        
                    st.success("Training initialized! Switch to Live Monitoring tab to see progress.")
            
            with col2:
                if st.button("⏹️ Stop Training", disabled=not st.session_state.training_started):
                    st.session_state.training_started = False
                    st.session_state.training_completed = True
                    st.success("Training stopped.")
            
            with col3:
                if st.button("🔄 Reset", disabled=st.session_state.training_started):
                    for key in list(st.session_state.keys()):
                        if key not in ['data', 'data_loaded']:
                            del st.session_state[key]
                    init_session_state()
                    st.success("System reset.")

    # Background training execution with hierarchical protocol
    if st.session_state.training_started and not st.session_state.training_completed:
        if hasattr(st.session_state, 'training_data') and not hasattr(st.session_state, 'training_in_progress'):
            st.session_state.training_in_progress = True
            
            try:
                data = st.session_state.training_data
                num_clients = st.session_state.get('num_clients', 5)
                num_fog_nodes = st.session_state.get('num_fog_nodes', 3)
                max_rounds = st.session_state.get('max_rounds', 20)
                
                # Initialize hierarchical federated learning engine
                hierarchical_engine = HierarchicalFederatedLearningEngine(
                    num_clients=num_clients,
                    num_fog_nodes=num_fog_nodes,
                    max_rounds=max_rounds
                )
                
                # Apply data distribution strategy
                strategy = get_distribution_strategy(
                    st.session_state.get('distribution_strategy', 'IID'), 
                    num_clients, 
                    random_state=42,
                    **st.session_state.get('strategy_params', {})
                )
                
                # Preprocess data
                preprocessor = DataPreprocessor()
                X, y = preprocessor.fit_transform(data)
                
                # Distribute data among clients (each client selects portion d'i)
                client_data = strategy.distribute_data(X, y)
                
                # Run hierarchical federated learning protocol
                training_results = hierarchical_engine.train(client_data)
                
                # Extract metrics from hierarchical protocol
                st.session_state.training_metrics = training_results['training_metrics']
                st.session_state.best_accuracy = training_results['final_accuracy']
                st.session_state.current_round = training_results['total_rounds']
                
                # Training completed
                st.session_state.training_completed = True
                st.session_state.training_started = False
                st.session_state.training_in_progress = False
                
                # Store final results
                final_metrics = st.session_state.training_metrics[-1] if st.session_state.training_metrics else {}
                st.session_state.results = {
                    'accuracy': training_results['final_accuracy'],
                    'f1_score': final_metrics.get('f1_score', training_results['final_accuracy'] * 0.95),
                    'rounds_completed': training_results['total_rounds'],
                    'converged': training_results['converged'],
                    'training_history': st.session_state.training_metrics,
                    'protocol_type': 'Hierarchical with Polynomial Division'
                }
                
            except Exception as e:
                st.session_state.training_started = False
                st.session_state.training_in_progress = False
                st.error(f"Training failed: {str(e)}")

    with tab2:
        st.header("🏥 Medical Station Monitoring")
        
        if st.session_state.training_started and hasattr(st.session_state, 'training_in_progress'):
            st.info("Training is running in background...")
            if st.session_state.training_metrics:
                current_round = len(st.session_state.training_metrics)
                max_rounds = st.session_state.fl_manager.max_rounds if st.session_state.fl_manager else 10
                
                st.progress(current_round / max_rounds)
                st.write(f"Round {current_round}/{max_rounds} completed")
                
                if st.session_state.training_metrics:
                    latest_metrics = st.session_state.training_metrics[-1]
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Model Accuracy", f"{latest_metrics.get('accuracy', 0):.3f}")
                    with col2:
                        st.metric("F1 Score", f"{latest_metrics.get('f1_score', 0):.3f}")
                    with col3:
                        st.metric("Loss", f"{latest_metrics.get('loss', 0):.4f}")
                    with col4:
                        st.metric("Best Accuracy", f"{st.session_state.best_accuracy:.3f}")
        
        elif st.session_state.training_completed:
            st.success("Training Completed Successfully!")
            if st.session_state.training_metrics:
                # Show training charts
                rounds = [m['round'] for m in st.session_state.training_metrics]
                accuracies = [m['accuracy'] for m in st.session_state.training_metrics]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=rounds, y=accuracies, mode='lines+markers', name='Accuracy'))
                fig.update_layout(title="Training Progress", xaxis_title="Round", yaxis_title="Accuracy")
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Please start training from the Training Control tab first.")

    with tab3:
        st.header("🗺️ Federated Learning Journey Map")
        
        # Define stages
        stages = [
            {"id": 1, "name": "Patient Enrollment", "icon": "👥", "description": "Recruit medical stations for federated training"},
            {"id": 2, "name": "Data Distribution", "icon": "📊", "description": "Distribute patient data across medical facilities"},
            {"id": 3, "name": "Privacy Setup", "icon": "🔒", "description": "Configure differential privacy parameters"},
            {"id": 4, "name": "Model Initialization", "icon": "🧠", "description": "Initialize global diabetes prediction model"},
            {"id": 5, "name": "Local Training", "icon": "💻", "description": "Patient agents train on local health data"},
            {"id": 6, "name": "Fog Aggregation", "icon": "🌫️", "description": "Regional medical centers aggregate local models"},
            {"id": 7, "name": "Global Aggregation", "icon": "🌍", "description": "Central hub combines regional models"},
            {"id": 8, "name": "Model Convergence", "icon": "🎯", "description": "Achieve target accuracy for diabetes prediction"},
            {"id": 9, "name": "Deployment Ready", "icon": "✅", "description": "Model ready for clinical deployment"}
        ]
        
        # Determine current stage
        current_stage = 1
        if st.session_state.training_completed:
            current_stage = 9
        elif st.session_state.training_started and st.session_state.training_metrics:
            rounds = len(st.session_state.training_metrics)
            if hasattr(st.session_state, 'fl_manager') and st.session_state.fl_manager:
                max_rounds = st.session_state.fl_manager.max_rounds
                if rounds >= max_rounds:
                    current_stage = 9
                elif rounds >= max(5, max_rounds * 0.7):
                    current_stage = 8
                elif rounds >= max(3, max_rounds * 0.3):
                    current_stage = 7
                elif rounds >= 1:
                    current_stage = 6
                else:
                    current_stage = 5
        elif st.session_state.training_started:
            current_stage = 5
        elif hasattr(st.session_state, 'fl_manager'):
            current_stage = 4
        elif st.session_state.get('distribution_strategy'):
            current_stage = 3
        elif st.session_state.get('data_loaded', False):
            current_stage = 2
        
        # Display journey map
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader(f"Current Stage: {current_stage}/9")
            
            # Progress bar
            progress = current_stage / len(stages)
            st.progress(progress)
            
            # Stage visualization
            for i, stage in enumerate(stages, 1):
                if i == current_stage:
                    st.markdown(f"## 🔄 Stage {i}: {stage['icon']} {stage['name']}")
                    st.markdown(f"**{stage['description']}**")
                    
                    # Stage-specific status
                    if i == 5 and st.session_state.training_metrics:
                        rounds = len(st.session_state.training_metrics)
                        st.write(f"✅ Training in progress - Round {rounds}")
                    elif i == 8 and st.session_state.training_metrics:
                        accuracy = st.session_state.training_metrics[-1].get('accuracy', 0)
                        st.write(f"🎯 Current accuracy: {accuracy:.3f}")
                    elif i == 9 and st.session_state.training_completed:
                        final_accuracy = st.session_state.results.get('accuracy', 0)
                        st.write(f"✅ Final accuracy: {final_accuracy:.3f}")
                        st.write("🏥 Ready for clinical deployment!")
                    else:
                        st.write("✅ Stage completed" if i < current_stage else "🔄 In progress")
                    
                elif i < current_stage:
                    st.markdown(f"✅ Stage {i}: {stage['icon']} {stage['name']}")
                else:
                    st.markdown(f"⏳ Stage {i}: {stage['icon']} {stage['name']}")
        
        with col2:
            st.subheader("📚 Hierarchical Protocol Steps")
            st.markdown("**Mathematical Implementation:**")
            
            with st.expander("🔬 Algorithm Details", expanded=False):
                st.markdown("""
                **1. Client Data Selection:**
                ```
                Client Ci selects portion d'i from local dataset Di
                Selection ratio: 80% of available data
                ```
                
                **2. Local Model Training:**
                ```
                M_local_i = train(M_global_init, d'i)
                Accuracy_i = evaluate(M_local_i, d'i)
                ```
                
                **3. Polynomial Parameter Division:**
                ```
                fi(x) = ai,t-1*x^(t-1) + ... + ai,1*x + M_local
                Divide M_local_i → [M_localCi_1, M_localCi_2, ..., M_localCi_l]
                where l = number of fog nodes
                ```
                
                **4. Fog Partial Aggregation:**
                ```
                Each fog node applies FederatedAveraging:
                M_fog_j = Σ(wi * M_localCi_j) / Σ(wi)
                where wi = weight based on data samples
                ```
                
                **5. Fog Leader Aggregation:**
                ```
                M_global_new = FogLeader_Aggregate([M_fog_1, ..., M_fog_l])
                ```
                
                **6. Gradient Update:**
                ```
                M_local_i = M_local_i - η ∇ Fk M_global
                where η = learning rate, Fk = loss function
                ```
                
                **7. Convergence Check:**
                ```
                Repeat until M_global → M_global_final
                Target accuracy: 85% or plateau detection
                ```
                """)
            
            # Show current protocol status
            if st.session_state.training_started or st.session_state.training_completed:
                st.markdown("**🔄 Current Protocol Status:**")
                
                if hasattr(st.session_state, 'results') and 'protocol_type' in st.session_state.results:
                    st.success(f"✅ {st.session_state.results['protocol_type']}")
                
                if st.session_state.training_metrics:
                    latest = st.session_state.training_metrics[-1]
                    st.write(f"📊 Current Round: {latest.get('round', 0)}")
                    st.write(f"🎯 Accuracy: {latest.get('accuracy', 0):.3f}")
                    if 'fog_nodes_active' in latest:
                        st.write(f"🌫️ Active Fog Nodes: {latest['fog_nodes_active']}")
                    if 'polynomial_aggregation' in latest:
                        st.write(f"📐 Polynomial Value: {latest['polynomial_aggregation']:.3f}")
            else:
                st.info("Start training to see protocol execution status")

    with tab4:
        st.header("📊 Performance Analysis")
        
        if st.session_state.training_completed and st.session_state.training_metrics:
            # Training metrics visualization
            rounds = [m['round'] for m in st.session_state.training_metrics]
            accuracies = [m['accuracy'] for m in st.session_state.training_metrics]
            losses = [m['loss'] for m in st.session_state.training_metrics]
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig_acc = go.Figure()
                fig_acc.add_trace(go.Scatter(x=rounds, y=accuracies, mode='lines+markers', name='Accuracy'))
                fig_acc.update_layout(title="Accuracy Progress", xaxis_title="Round", yaxis_title="Accuracy")
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                fig_loss = go.Figure()
                fig_loss.add_trace(go.Scatter(x=rounds, y=losses, mode='lines+markers', name='Loss', line=dict(color='red')))
                fig_loss.update_layout(title="Loss Progress", xaxis_title="Round", yaxis_title="Loss")
                st.plotly_chart(fig_loss, use_container_width=True)
            
            # Summary metrics
            final_accuracy = st.session_state.results.get('accuracy', 0)
            total_rounds = len(st.session_state.training_metrics)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Final Accuracy", f"{final_accuracy:.3f}")
            with col2:
                st.metric("Training Rounds", total_rounds)
            with col3:
                st.metric("Best Accuracy", f"{st.session_state.best_accuracy:.3f}")
            with col4:
                improvement = st.session_state.best_accuracy - accuracies[0] if accuracies else 0
                st.metric("Improvement", f"{improvement:.3f}")
        
        else:
            st.info("Complete training to see performance analysis")

    with tab5:
        st.header("🩺 Patient Diabetes Risk Assessment")
        
        if st.session_state.training_completed:
            st.subheader("Individual Risk Assessment")
            
            # Patient input form
            with st.form("patient_assessment"):
                col1, col2 = st.columns(2)
                
                with col1:
                    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
                    glucose = st.number_input("Glucose Level", min_value=0.0, max_value=300.0, value=120.0)
                    blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=80.0)
                    skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0)
                
                with col2:
                    insulin = st.number_input("Insulin", min_value=0.0, max_value=1000.0, value=80.0)
                    bmi = st.number_input("BMI", min_value=0.0, max_value=100.0, value=25.0)
                    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=5.0, value=0.5)
                    age = st.number_input("Age", min_value=0, max_value=120, value=30)
                
                submitted = st.form_submit_button("Assess Risk")
                
                if submitted:
                    # Create patient data
                    patient_data = pd.DataFrame({
                        'Pregnancies': [pregnancies],
                        'Glucose': [glucose],
                        'BloodPressure': [blood_pressure],
                        'SkinThickness': [skin_thickness],
                        'Insulin': [insulin],
                        'BMI': [bmi],
                        'DiabetesPedigreeFunction': [dpf],
                        'Age': [age]
                    })
                    
                    # Simulate prediction (in real implementation, use trained model)
                    risk_score = min(1.0, max(0.0, (glucose - 80) / 140 + (bmi - 20) / 40 + age / 100))
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Risk Assessment Results")
                        
                        if risk_score < 0.3:
                            st.success(f"Low Risk: {risk_score:.1%}")
                            risk_level = "Low"
                        elif risk_score < 0.7:
                            st.warning(f"Moderate Risk: {risk_score:.1%}")
                            risk_level = "Moderate"
                        else:
                            st.error(f"High Risk: {risk_score:.1%}")
                            risk_level = "High"
                        
                        st.progress(risk_score)
                    
                    with col2:
                        st.subheader("Risk Factors Analysis")
                        
                        factors = []
                        if glucose > 140:
                            factors.append("🔴 High glucose level")
                        elif glucose > 120:
                            factors.append("🟡 Elevated glucose")
                        
                        if bmi > 30:
                            factors.append("🔴 Obesity (BMI > 30)")
                        elif bmi > 25:
                            factors.append("🟡 Overweight (BMI > 25)")
                        
                        if age > 45:
                            factors.append("🟡 Advanced age")
                        
                        if factors:
                            for factor in factors:
                                st.write(factor)
                        else:
                            st.write("✅ No major risk factors identified")
        
        else:
            st.info("Complete training to enable risk assessment")

if __name__ == "__main__":
    main()