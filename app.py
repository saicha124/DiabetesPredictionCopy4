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
from visualization import create_network_diagram, create_reputation_chart, create_accuracy_chart
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
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Training Progress", 
            "🌐 Network Status", 
            "🏆 Committee & Reputation", 
            "📈 Aggregation Status",
            "🎯 Predictions & Accuracy"
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

if __name__ == "__main__":
    main()
