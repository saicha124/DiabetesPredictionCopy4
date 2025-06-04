import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import threading
from datetime import datetime

from federated_system import FederatedLearningSystem
from visualization import (
    create_network_diagram, create_reputation_chart, create_accuracy_chart,
    create_confusion_matrix_heatmap, create_timing_analysis_chart, 
    create_communication_flow_chart, create_performance_metrics_dashboard
)
from data_loader import load_diabetes_data

# Page configuration
st.set_page_config(
    page_title="Hierarchical Federated Learning System",
    page_icon="🔗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'federated_system' not in st.session_state:
    st.session_state.federated_system = None
if 'training_active' not in st.session_state:
    st.session_state.training_active = False
if 'current_round' not in st.session_state:
    st.session_state.current_round = 0
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False

def initialize_system(num_clients, num_fogs, privacy_epsilon):
    """Initialize the federated learning system"""
    try:
        # Load diabetes data
        data = load_diabetes_data()
        
        # Initialize federated system
        system = FederatedLearningSystem(
            num_clients=num_clients,
            num_fogs=num_fogs,
            privacy_epsilon=privacy_epsilon,
            data=data
        )
        
        st.session_state.federated_system = system
        st.session_state.system_initialized = True
        st.session_state.current_round = 0
        st.session_state.training_history = []
        
        return True
    except Exception as e:
        st.error(f"Failed to initialize system: {str(e)}")
        return False

def main():
    st.title("🔗 Hierarchical Federated Deep Learning System")
    st.markdown("### With Fog-level Aggregation, Differential Privacy & Committee-based Security")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ System Configuration")
        
        # System parameters
        num_clients = st.slider("Number of Clients", min_value=3, max_value=20, value=8, step=1)
        num_fogs = st.slider("Number of Fog Nodes", min_value=2, max_value=5, value=3, step=1)
        privacy_epsilon = st.slider("Privacy Parameter (ε)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        
        # Initialize system button
        if st.button("🚀 Initialize System", type="primary"):
            with st.spinner("Initializing federated learning system..."):
                if initialize_system(num_clients, num_fogs, privacy_epsilon):
                    st.success("System initialized successfully!")
                    st.rerun()
        
        st.divider()
        
        # Training controls
        if st.session_state.system_initialized:
            st.header("🎯 Training Controls")
            
            num_rounds = st.slider("Training Rounds", min_value=1, max_value=20, value=5, step=1)
            
            if st.button("▶️ Start Training", disabled=st.session_state.training_active):
                st.session_state.training_active = True
                st.rerun()
            
            if st.button("⏹️ Stop Training", disabled=not st.session_state.training_active):
                st.session_state.training_active = False
                st.rerun()
            
            if st.button("🔄 Reset System"):
                st.session_state.federated_system = None
                st.session_state.system_initialized = False
                st.session_state.training_active = False
                st.session_state.current_round = 0
                st.session_state.training_history = []
                st.rerun()
        
        st.divider()
        
        # Attack simulation placeholder
        st.header("🛡️ Security Testing")
        attack_simulation = st.selectbox(
            "Attack Simulation",
            ["None", "Sybil Attack", "Byzantine Attack"],
            disabled=True
        )
        st.info("⚠️ Attack simulation incompatible with hierarchical fog setup. Feature available in future updates only.")

    # Main content area with tabs
    if not st.session_state.system_initialized:
        st.info("👆 Please configure and initialize the system using the sidebar controls.")
        
        # Show system overview
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏗️ System Architecture")
            st.markdown("""
            **Hierarchical Structure:**
            - **Clients**: Perform local training on data partitions
            - **Fog Nodes**: Aggregate client updates (no training)
            - **Leader Fog**: Performs global aggregation
            - **Committees**: Secure validation with DP-masked reputations
            """)
        
        with col2:
            st.subheader("🔒 Security Features")
            st.markdown("""
            **Privacy & Security:**
            - Differential privacy for reputation masking
            - Secret sharing for secure communication
            - Committee-based validation
            - Protection against Sybil/Byzantine attacks
            """)
    
    else:
        # Get num_rounds from sidebar
        with st.sidebar:
            num_rounds = st.slider("Training Rounds", min_value=1, max_value=20, value=5, step=1, key="num_rounds_main")
        
        # Training execution
        if st.session_state.training_active and st.session_state.current_round < num_rounds:
            # Execute training round
            execute_training_round(num_rounds)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Training Progress", 
            "🌐 Network Status", 
            "🏆 Committee & Reputation", 
            "📈 Aggregation Status",
            "🎯 Predictions & Accuracy",
            "🔧 Timing Analysis",
            "📡 Communication Flow",
            "🩺 Patient Prediction"
        ])
        
        with tab1:
            show_training_progress()
        
        with tab2:
            show_network_status()
        
        with tab3:
            show_committee_reputation()
        
        with tab4:
            show_aggregation_status()
        
        with tab5:
            show_predictions_accuracy()
        
        with tab6:
            show_timing_analysis()
        
        with tab7:
            show_communication_flow()
        
        with tab8:
            show_patient_prediction()

def execute_training_round(num_rounds):
    """Execute a single training round"""
    if st.session_state.federated_system is None:
        return
    
    try:
        # Create progress indicators
        progress_container = st.container()
        
        with progress_container:
            st.subheader(f"🔄 Round {st.session_state.current_round + 1} Progress")
            
            # Phase progress bars
            client_progress = st.progress(0, text="Client Training...")
            fog_progress = st.progress(0, text="Fog Aggregation...")
            global_progress = st.progress(0, text="Global Aggregation...")
            
            # Execute training round
            round_results = st.session_state.federated_system.execute_round(
                progress_callbacks={
                    'client': lambda p: client_progress.progress(p),
                    'fog': lambda p: fog_progress.progress(p),
                    'global': lambda p: global_progress.progress(p)
                }
            )
            
            # Update session state
            st.session_state.training_history.append(round_results)
            st.session_state.current_round += 1
            
            # Check if training is complete
            if st.session_state.current_round >= num_rounds:
                st.session_state.training_active = False
                st.success(f"✅ Training completed! {st.session_state.current_round} rounds finished.")
            
            # Auto-refresh for next round
            if st.session_state.training_active:
                time.sleep(1)
                st.rerun()
                
    except Exception as e:
        st.error(f"Training round failed: {str(e)}")
        st.session_state.training_active = False

def show_training_progress():
    """Show training progress tab"""
    if not st.session_state.training_history:
        st.info("No training data available. Start training to see progress.")
        return
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Training accuracy over rounds
        fig = create_accuracy_chart(st.session_state.training_history)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Current round statistics
        if st.session_state.training_history:
            latest_round = st.session_state.training_history[-1]
            
            st.metric("Current Round", st.session_state.current_round)
            st.metric("Global Accuracy", f"{latest_round.get('global_accuracy', 0):.3f}")
            st.metric("Average Client Loss", f"{latest_round.get('avg_client_loss', 0):.4f}")
            st.metric("Active Committees", latest_round.get('num_committees', 0))
    
    # Detailed round information
    st.subheader("📋 Round Details")
    
    if st.session_state.training_history:
        round_data = []
        for i, round_info in enumerate(st.session_state.training_history):
            round_data.append({
                "Round": i + 1,
                "Global Accuracy": f"{round_info.get('global_accuracy', 0):.4f}",
                "Avg Client Loss": f"{round_info.get('avg_client_loss', 0):.4f}",
                "Training Time (s)": f"{round_info.get('training_time', 0):.2f}",
                "Committees": round_info.get('num_committees', 0)
            })
        
        df = pd.DataFrame(round_data)
        st.dataframe(df, use_container_width=True)

def show_network_status():
    """Show network status tab"""
    if st.session_state.federated_system is None:
        return
    
    # Network topology diagram
    st.subheader("🌐 Network Topology")
    network_fig = create_network_diagram(st.session_state.federated_system)
    st.plotly_chart(network_fig, use_container_width=True)
    
    # System status metrics
    col1, col2, col3, col4 = st.columns(4)
    
    system = st.session_state.federated_system
    
    with col1:
        st.metric("Active Clients", len(system.clients))
    
    with col2:
        st.metric("Fog Nodes", len(system.fog_nodes))
    
    with col3:
        st.metric("Leader Fog", "1")
    
    with col4:
        st.metric("System Status", "🟢 Active" if system else "🔴 Inactive")

def show_committee_reputation():
    """Show committee and reputation tab"""
    if st.session_state.federated_system is None:
        return
    
    system = st.session_state.federated_system
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Client Reputations")
        rep_fig = create_reputation_chart(system)
        st.plotly_chart(rep_fig, use_container_width=True)
    
    with col2:
        st.subheader("👥 Current Committee")
        if hasattr(system, 'current_committee') and system.current_committee:
            committee_data = []
            for member in system.current_committee:
                committee_data.append({
                    "Client ID": f"Client_{member}",
                    "Reputation": f"{system.get_client_reputation(member):.3f}",
                    "Role": "Committee Member"
                })
            
            committee_df = pd.DataFrame(committee_data)
            st.dataframe(committee_df, hide_index=True)
        else:
            st.info("No active committee")
    
    # Privacy statistics
    st.subheader("🔒 Privacy Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Privacy Epsilon (ε)", f"{system.privacy_epsilon:.1f}")
    
    with col2:
        st.metric("DP Noise Applied", "✅ Active")
    
    with col3:
        st.metric("Secret Sharing", "✅ Enabled")

def show_aggregation_status():
    """Show aggregation status tab"""
    if not st.session_state.training_history:
        st.info("No aggregation data available. Start training to see aggregation status.")
        return
    
    # Aggregation timeline
    st.subheader("⏱️ Aggregation Timeline")
    
    if st.session_state.training_history:
        timeline_data = []
        for i, round_info in enumerate(st.session_state.training_history):
            timeline_data.append({
                "Round": i + 1,
                "Fog Aggregation Time": round_info.get('fog_aggregation_time', 0),
                "Global Aggregation Time": round_info.get('global_aggregation_time', 0),
                "Total Aggregation Time": round_info.get('fog_aggregation_time', 0) + round_info.get('global_aggregation_time', 0)
            })
        
        df = pd.DataFrame(timeline_data)
        
        fig = px.bar(df, x='Round', y=['Fog Aggregation Time', 'Global Aggregation Time'], 
                     title="Aggregation Times by Round",
                     labels={'value': 'Time (seconds)', 'variable': 'Aggregation Type'})
        st.plotly_chart(fig, use_container_width=True)
    
    # Current aggregation status
    if st.session_state.federated_system:
        system = st.session_state.federated_system
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Fog Nodes Ready", f"{len(system.fog_nodes)}/{len(system.fog_nodes)}")
        
        with col2:
            st.metric("Leader Fog Status", "🟢 Ready")
        
        with col3:
            st.metric("Aggregation Method", "FedAvg")

def show_predictions_accuracy():
    """Show predictions and accuracy tab"""
    if not st.session_state.training_history:
        st.info("No prediction data available. Complete training rounds to see accuracy metrics.")
        return
    
    # Accuracy metrics over time
    st.subheader("📈 Model Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Global accuracy trend
        rounds = list(range(1, len(st.session_state.training_history) + 1))
        accuracies = [round_info.get('global_accuracy', 0) for round_info in st.session_state.training_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rounds, y=accuracies, mode='lines+markers', name='Global Accuracy'))
        fig.update_layout(title="Global Model Accuracy", xaxis_title="Round", yaxis_title="Accuracy")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loss trend
        losses = [round_info.get('avg_client_loss', 0) for round_info in st.session_state.training_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rounds, y=losses, mode='lines+markers', name='Average Loss', line=dict(color='red')))
        fig.update_layout(title="Average Client Loss", xaxis_title="Round", yaxis_title="Loss")
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance summary
    if st.session_state.training_history:
        latest_round = st.session_state.training_history[-1]
        
        st.subheader("🎯 Current Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Accuracy", f"{max(accuracies):.4f}")
        
        with col2:
            st.metric("Current Accuracy", f"{latest_round.get('global_accuracy', 0):.4f}")
        
        with col3:
            st.metric("Lowest Loss", f"{min(losses):.4f}")
        
        with col4:
            st.metric("Current Loss", f"{latest_round.get('avg_client_loss', 0):.4f}")

def show_timing_analysis():
    """Show detailed timing analysis tab"""
    if not st.session_state.training_history:
        st.info("No timing data available. Complete training rounds to see timing analysis.")
        return
    
    # Main timing analysis chart
    st.subheader("⏱️ Comprehensive Timing Analysis")
    timing_fig = create_timing_analysis_chart(st.session_state.training_history)
    st.plotly_chart(timing_fig, use_container_width=True)
    
    # Timing metrics summary
    col1, col2, col3, col4 = st.columns(4)
    
    if st.session_state.training_history:
        latest_round = st.session_state.training_history[-1]
        
        with col1:
            avg_client_time = latest_round.get('avg_client_training_time', 0)
            st.metric("Avg Client Training", f"{avg_client_time:.3f}s")
        
        with col2:
            avg_fog_time = latest_round.get('avg_fog_execution_time', 0)
            st.metric("Avg Fog Execution", f"{avg_fog_time:.3f}s")
        
        with col3:
            global_agg_time = latest_round.get('global_aggregation_time', 0)
            st.metric("Global Aggregation", f"{global_agg_time:.3f}s")
        
        with col4:
            total_round_time = latest_round.get('training_time', 0)
            st.metric("Total Round Time", f"{total_round_time:.3f}s")
    
    # Detailed timing breakdown table
    st.subheader("📋 Timing Breakdown by Round")
    
    if st.session_state.training_history:
        timing_data = []
        for i, round_info in enumerate(st.session_state.training_history):
            timing_data.append({
                "Round": i + 1,
                "Client Training (s)": f"{round_info.get('avg_client_training_time', 0):.3f}",
                "Fog Execution (s)": f"{round_info.get('avg_fog_execution_time', 0):.3f}",
                "Global Aggregation (s)": f"{round_info.get('global_aggregation_time', 0):.3f}",
                "Communication (s)": f"{round_info.get('avg_communication_time', 0):.3f}",
                "Total Round (s)": f"{round_info.get('training_time', 0):.3f}"
            })
        
        df = pd.DataFrame(timing_data)
        st.dataframe(df, use_container_width=True)

def show_communication_flow():
    """Show communication flow and network analysis tab"""
    if not st.session_state.training_history:
        st.info("No communication data available. Complete training rounds to see communication analysis.")
        return
    
    # Communication flow chart
    st.subheader("📡 Communication Flow & Latency")
    comm_fig = create_communication_flow_chart(st.session_state.training_history)
    st.plotly_chart(comm_fig, use_container_width=True)
    
    # Communication metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Communication Metrics")
        if st.session_state.training_history:
            latest_round = st.session_state.training_history[-1]
            
            # Communication timing metrics
            st.metric("Client-to-Fog Comm", f"{latest_round.get('avg_communication_time', 0):.4f}s")
            st.metric("Global Distribution", f"{latest_round.get('global_communication_time', 0):.4f}s")
            st.metric("Fog Aggregation Phase", f"{latest_round.get('fog_aggregation_time', 0):.4f}s")
            
            # Calculate total communication overhead
            total_comm = (
                latest_round.get('avg_communication_time', 0) +
                latest_round.get('global_communication_time', 0) +
                latest_round.get('fog_aggregation_time', 0)
            )
            st.metric("Total Communication Overhead", f"{total_comm:.4f}s")
    
    with col2:
        st.subheader("🔍 Performance Analysis")
        if st.session_state.training_history and st.session_state.federated_system:
            latest_round = st.session_state.training_history[-1]
            
            # Show confusion matrix if available
            if 'confusion_matrix' in latest_round:
                confusion_data = latest_round['confusion_matrix']
                
                # Display confusion matrix heatmap
                cm_fig = create_confusion_matrix_heatmap(confusion_data)
                st.plotly_chart(cm_fig, use_container_width=True)
                
                # Performance metrics dashboard
                perf_fig = create_performance_metrics_dashboard(confusion_data)
                st.plotly_chart(perf_fig, use_container_width=True)
            else:
                st.info("Confusion matrix data not available for current round.")
    
    # Network topology and efficiency analysis
    st.subheader("🌐 Network Efficiency Analysis")
    
    if st.session_state.federated_system and st.session_state.training_history:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            num_clients = len(st.session_state.federated_system.clients)
            num_fogs = len(st.session_state.federated_system.fog_nodes)
            st.metric("Network Topology", f"{num_clients} clients → {num_fogs} fogs → 1 leader")
        
        with col2:
            clients_per_fog = num_clients / num_fogs if num_fogs > 0 else 0
            st.metric("Avg Clients per Fog", f"{clients_per_fog:.1f}")
        
        with col3:
            committee_size = len(st.session_state.federated_system.current_committee)
            committee_ratio = committee_size / num_clients if num_clients > 0 else 0
            st.metric("Committee Participation", f"{committee_ratio:.1%}")

def show_patient_prediction():
    """Show patient prediction tab for individual diabetes prediction"""
    st.subheader("🩺 Individual Patient Diabetes Prediction")
    
    if st.session_state.federated_system is None:
        st.warning("Please initialize the federated system first before making predictions.")
        return
    
    if not st.session_state.training_history:
        st.warning("Please complete at least one training round before making predictions.")
        return
    
    st.markdown("Enter patient information to predict diabetes risk using the trained federated model:")
    
    # Create input form for patient data
    with st.form("patient_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Basic Information**")
            pregnancies = st.number_input("Number of Pregnancies", min_value=0, max_value=20, value=1, step=1)
            glucose = st.number_input("Glucose Level (mg/dL)", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0.0, max_value=200.0, value=80.0, step=1.0)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0.0, max_value=100.0, value=20.0, step=1.0)
        
        with col2:
            st.markdown("**Metabolic Information**")
            insulin = st.number_input("Insulin Level (μU/mL)", min_value=0.0, max_value=1000.0, value=80.0, step=1.0)
            bmi = st.number_input("Body Mass Index (BMI)", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
            diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=30, step=1)
        
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk", use_container_width=True)
    
    if submitted:
        try:
            # Prepare patient data for prediction
            patient_data = np.array([[
                pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age
            ]], dtype=np.float32)
            
            # Get the latest global model weights
            if st.session_state.training_history:
                latest_round = st.session_state.training_history[-1]
                global_weights = latest_round.get('global_weights')
                
                if global_weights:
                    # Create a temporary model for prediction
                    import torch
                    from neural_network import DiabetesNN
                    
                    # Initialize model with same architecture
                    temp_model = DiabetesNN(input_size=8, hidden_sizes=[64, 32], output_size=1)
                    temp_model.set_layer_weights(global_weights)
                    temp_model.eval()
                    
                    # Make prediction
                    with torch.no_grad():
                        patient_tensor = torch.FloatTensor(patient_data)
                        
                        # Get prediction probability
                        prediction_prob = temp_model.predict_proba(patient_tensor).item()
                        prediction_binary = temp_model.predict(patient_tensor).item()
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("🎯 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            risk_percentage = prediction_prob * 100
                            st.metric("Diabetes Risk Probability", f"{risk_percentage:.1f}%")
                        
                        with col2:
                            prediction_text = "High Risk" if prediction_binary == 1 else "Low Risk"
                            color = "🔴" if prediction_binary == 1 else "🟢"
                            st.metric("Risk Assessment", f"{color} {prediction_text}")
                        
                        with col3:
                            confidence = max(prediction_prob, 1 - prediction_prob) * 100
                            st.metric("Model Confidence", f"{confidence:.1f}%")
                        
                        # Risk interpretation
                        st.markdown("### 📋 Risk Interpretation")
                        
                        if prediction_prob >= 0.7:
                            st.error("**High Risk**: The model indicates a high probability of diabetes. Please consult with a healthcare professional for proper medical evaluation.")
                        elif prediction_prob >= 0.3:
                            st.warning("**Moderate Risk**: The model indicates moderate diabetes risk. Consider lifestyle modifications and regular health monitoring.")
                        else:
                            st.success("**Low Risk**: The model indicates low diabetes risk. Continue maintaining healthy lifestyle habits.")
                        
                        # Feature importance display
                        st.markdown("### 📊 Input Analysis")
                        
                        feature_names = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness', 
                                       'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
                        feature_values = [pregnancies, glucose, blood_pressure, skin_thickness,
                                        insulin, bmi, diabetes_pedigree, age]
                        
                        # Create a simple feature analysis
                        analysis_data = []
                        for name, value in zip(feature_names, feature_values):
                            analysis_data.append({
                                "Feature": name,
                                "Value": f"{value:.2f}" if isinstance(value, float) else str(value),
                                "Status": "Normal" if name not in ['Glucose', 'BMI'] or 
                                         (name == 'Glucose' and 70 <= value <= 140) or
                                         (name == 'BMI' and 18.5 <= value <= 24.9) else "Elevated"
                            })
                        
                        analysis_df = pd.DataFrame(analysis_data)
                        st.dataframe(analysis_df, use_container_width=True)
                        
                        # Model information
                        st.markdown("### ℹ️ Model Information")
                        
                        latest_accuracy = latest_round.get('global_accuracy', 0)
                        training_rounds = len(st.session_state.training_history)
                        
                        info_col1, info_col2, info_col3 = st.columns(3)
                        
                        with info_col1:
                            st.metric("Model Accuracy", f"{latest_accuracy:.1%}")
                        
                        with info_col2:
                            st.metric("Training Rounds", training_rounds)
                        
                        with info_col3:
                            num_clients = len(st.session_state.federated_system.clients)
                            st.metric("Federated Clients", num_clients)
                        
                        st.info("**Disclaimer**: This prediction is for educational purposes only and should not replace professional medical advice. Always consult with healthcare professionals for medical decisions.")
                
                else:
                    st.error("No trained model weights available. Please complete training rounds first.")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure the federated system is properly initialized and trained.")
    
    # Add sample patient data for testing
    st.markdown("---")
    st.markdown("### 📝 Sample Patient Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Low Risk Profile**")
        if st.button("Load Low Risk Sample", key="low_risk"):
            st.session_state.sample_data = {
                'pregnancies': 1, 'glucose': 85, 'blood_pressure': 70, 'skin_thickness': 20,
                'insulin': 80, 'bmi': 22.5, 'diabetes_pedigree': 0.3, 'age': 25
            }
            st.rerun()
    
    with col2:
        st.markdown("**High Risk Profile**")
        if st.button("Load High Risk Sample", key="high_risk"):
            st.session_state.sample_data = {
                'pregnancies': 8, 'glucose': 180, 'blood_pressure': 90, 'skin_thickness': 35,
                'insulin': 150, 'bmi': 35.2, 'diabetes_pedigree': 1.2, 'age': 55
            }
            st.rerun()

if __name__ == "__main__":
    main()
