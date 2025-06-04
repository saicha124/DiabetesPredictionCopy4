import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any
import networkx as nx
from sklearn.metrics import confusion_matrix

def create_network_diagram(federated_system) -> go.Figure:
    """
    Create network topology diagram showing clients, fog nodes, and leader fog
    
    Args:
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        # Create network graph
        G = nx.Graph()
        
        # Node positions
        node_positions = {}
        node_colors = []
        node_sizes = []
        node_labels = []
        
        # Add clients
        num_clients = len(federated_system.clients)
        client_radius = 3
        
        for i, client in enumerate(federated_system.clients):
            angle = 2 * np.pi * i / num_clients
            x = client_radius * np.cos(angle)
            y = client_radius * np.sin(angle)
            
            node_id = f"client_{i}"
            G.add_node(node_id)
            node_positions[node_id] = (x, y)
            node_colors.append('lightblue')
            node_sizes.append(20)
            node_labels.append(f"C{i}")
        
        # Add fog nodes
        num_fogs = len(federated_system.fog_nodes)
        fog_radius = 1.5
        
        for i, fog in enumerate(federated_system.fog_nodes):
            angle = 2 * np.pi * i / num_fogs
            x = fog_radius * np.cos(angle)
            y = fog_radius * np.sin(angle)
            
            node_id = f"fog_{i}"
            G.add_node(node_id)
            node_positions[node_id] = (x, y)
            node_colors.append('lightgreen')
            node_sizes.append(30)
            node_labels.append(f"F{i}")
        
        # Add leader fog
        leader_id = "leader_fog"
        G.add_node(leader_id)
        node_positions[leader_id] = (0, 0)
        node_colors.append('orange')
        node_sizes.append(40)
        node_labels.append("LF")
        
        # Add edges (client to assigned fog)
        for fog_idx, fog in enumerate(federated_system.fog_nodes):
            fog_id = f"fog_{fog_idx}"
            
            # Connect assigned clients to this fog
            for client_id in fog.assigned_clients:
                client_node_id = f"client_{client_id}"
                G.add_edge(client_node_id, fog_id)
            
            # Connect fog to leader
            G.add_edge(fog_id, leader_id)
        
        # Extract edge coordinates
        edge_x = []
        edge_y = []
        
        for edge in G.edges():
            x0, y0 = node_positions[edge[0]]
            x1, y1 = node_positions[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Extract node coordinates
        node_x = [node_positions[node][0] for node in G.nodes()]
        node_y = [node_positions[node][1] for node in G.nodes()]
        
        # Create figure
        fig = go.Figure()
        
        # Add edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=2, color='gray'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=2, color='black')
            ),
            text=node_labels,
            textposition="middle center",
            textfont=dict(size=10, color='black'),
            hovertemplate='<b>%{text}</b><extra></extra>',
            showlegend=False
        ))
        
        # Update layout
        fig.update_layout(
            title="Federated Learning Network Topology",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            annotations=[
                dict(
                    text="Clients (C) → Fog Nodes (F) → Leader Fog (LF)",
                    showarrow=False,
                    xref="paper", yref="paper",
                    x=0.005, y=-0.002,
                    xanchor='left', yanchor='bottom',
                    font=dict(size=12, color='gray')
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating network diagram: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating network diagram: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_reputation_chart(federated_system) -> go.Figure:
    """
    Create reputation chart for all clients
    
    Args:
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        client_ids = []
        reputations = []
        committee_status = []
        
        for i, client in enumerate(federated_system.clients):
            client_ids.append(f"Client_{i}")
            # Get noisy reputation for privacy
            noisy_reputation = federated_system.get_client_reputation(i)
            reputations.append(noisy_reputation)
            
            # Check if client is in committee
            is_committee = i in federated_system.current_committee
            committee_status.append("Committee" if is_committee else "Regular")
        
        # Create bar chart
        fig = go.Figure()
        
        # Add bars with different colors for committee members
        colors = ['gold' if status == "Committee" else 'lightblue' 
                 for status in committee_status]
        
        fig.add_trace(go.Bar(
            x=client_ids,
            y=reputations,
            marker_color=colors,
            text=[f"{rep:.3f}" for rep in reputations],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Reputation: %{y:.3f}<br>Status: %{customdata}<extra></extra>',
            customdata=committee_status
        ))
        
        # Update layout
        fig.update_layout(
            title="Client Reputation Scores (DP-Masked)",
            xaxis_title="Clients",
            yaxis_title="Reputation Score",
            yaxis=dict(range=[0, 1]),
            showlegend=False,
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating reputation chart: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating reputation chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_accuracy_chart(training_history: List[Dict]) -> go.Figure:
    """
    Create accuracy chart over training rounds
    
    Args:
        training_history: List of training round results
        
    Returns:
        Plotly figure
    """
    try:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(text="No training data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        rounds = list(range(1, len(training_history) + 1))
        accuracies = [round_info.get('global_accuracy', 0) for round_info in training_history]
        losses = [round_info.get('avg_client_loss', 0) for round_info in training_history]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Global Accuracy', 'Average Client Loss'),
            vertical_spacing=0.1
        )
        
        # Add accuracy trace
        fig.add_trace(
            go.Scatter(
                x=rounds, y=accuracies,
                mode='lines+markers',
                name='Accuracy',
                line=dict(color='blue', width=3),
                marker=dict(size=8),
                hovertemplate='Round %{x}<br>Accuracy: %{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Add loss trace
        fig.add_trace(
            go.Scatter(
                x=rounds, y=losses,
                mode='lines+markers',
                name='Loss',
                line=dict(color='red', width=3),
                marker=dict(size=8),
                hovertemplate='Round %{x}<br>Loss: %{y:.4f}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title="Training Progress Over Rounds",
            showlegend=False,
            height=500,
            plot_bgcolor='white'
        )
        
        fig.update_xaxes(title_text="Round", row=2, col=1)
        fig.update_yaxes(title_text="Accuracy", row=1, col=1)
        fig.update_yaxes(title_text="Loss", row=2, col=1)
        
        return fig
        
    except Exception as e:
        print(f"Error creating accuracy chart: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating accuracy chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_aggregation_timeline(training_history: List[Dict]) -> go.Figure:
    """
    Create timeline visualization of aggregation process
    
    Args:
        training_history: List of training round results
        
    Returns:
        Plotly figure
    """
    try:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(text="No aggregation data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        rounds = list(range(1, len(training_history) + 1))
        fog_times = [round_info.get('fog_aggregation_time', 0) for round_info in training_history]
        global_times = [round_info.get('global_aggregation_time', 0) for round_info in training_history]
        
        fig = go.Figure()
        
        # Add fog aggregation times
        fig.add_trace(go.Bar(
            x=rounds,
            y=fog_times,
            name='Fog Aggregation',
            marker_color='lightgreen',
            hovertemplate='Round %{x}<br>Fog Aggregation: %{y:.3f}s<extra></extra>'
        ))
        
        # Add global aggregation times
        fig.add_trace(go.Bar(
            x=rounds,
            y=global_times,
            name='Global Aggregation',
            marker_color='orange',
            hovertemplate='Round %{x}<br>Global Aggregation: %{y:.3f}s<extra></extra>'
        ))
        
        fig.update_layout(
            title="Aggregation Times by Round",
            xaxis_title="Round",
            yaxis_title="Time (seconds)",
            barmode='stack',
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating aggregation timeline: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating aggregation timeline: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_committee_composition_chart(federated_system) -> go.Figure:
    """
    Create committee composition visualization
    
    Args:
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        if not hasattr(federated_system, 'current_committee') or not federated_system.current_committee:
            fig = go.Figure()
            fig.add_annotation(text="No active committee", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        # Get committee members and their reputations
        committee_members = []
        member_reputations = []
        
        for member_id in federated_system.current_committee:
            committee_members.append(f"Client_{member_id}")
            reputation = federated_system.get_client_reputation(member_id)
            member_reputations.append(reputation)
        
        # Create polar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=member_reputations,
            theta=committee_members,
            fill='toself',
            name='Committee Reputation',
            line_color='blue',
            fillcolor='rgba(0,0,255,0.3)'
        ))
        
        fig.update_layout(
            title="Committee Member Reputations",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating committee composition chart: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating committee chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_privacy_metrics_chart(federated_system) -> go.Figure:
    """
    Create privacy metrics visualization
    
    Args:
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        privacy_stats = federated_system.privacy_engine.get_privacy_stats()
        
        metrics = ['Epsilon (ε)', 'Delta (δ)', 'Sensitivity']
        values = [
            privacy_stats.get('epsilon', 0),
            privacy_stats.get('delta', 0) * 1e5,  # Scale delta for visibility
            privacy_stats.get('sensitivity', 0)
        ]
        
        colors = ['red', 'orange', 'blue']
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=metrics,
            y=values,
            marker_color=colors,
            text=[f"{v:.3f}" if v >= 0.001 else f"{v:.1e}" for v in values],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Value: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Differential Privacy Parameters",
            yaxis_title="Parameter Value",
            plot_bgcolor='white',
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating privacy metrics chart: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating privacy chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_client_performance_heatmap(training_history: List[Dict], 
                                    federated_system) -> go.Figure:
    """
    Create heatmap of client performance over rounds
    
    Args:
        training_history: List of training round results
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(text="No performance data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        num_clients = len(federated_system.clients)
        num_rounds = len(training_history)
        
        # Create mock performance matrix (in practice, this would come from actual client data)
        performance_matrix = np.random.rand(num_clients, num_rounds) * 0.3 + 0.7  # Mock data
        
        client_labels = [f"Client_{i}" for i in range(num_clients)]
        round_labels = [f"Round {i+1}" for i in range(num_rounds)]
        
        fig = go.Figure(data=go.Heatmap(
            z=performance_matrix,
            x=round_labels,
            y=client_labels,
            colorscale='RdYlBu_r',
            colorbar=dict(title="Performance Score"),
            hovertemplate='<b>%{y}</b><br>%{x}<br>Performance: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Client Performance Heatmap",
            xaxis_title="Training Rounds",
            yaxis_title="Clients"
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating performance heatmap: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating heatmap: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_system_metrics_gauge(federated_system) -> go.Figure:
    """
    Create gauge charts for system metrics
    
    Args:
        federated_system: FederatedLearningSystem instance
        
    Returns:
        Plotly figure
    """
    try:
        # Create subplot with gauges
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{'type': 'indicator'}, {'type': 'indicator'}],
                   [{'type': 'indicator'}, {'type': 'indicator'}]],
            subplot_titles=('System Health', 'Privacy Level', 'Committee Efficiency', 'Network Load')
        )
        
        # System health (mock calculation)
        system_health = min(100, len(federated_system.clients) * 10)
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=system_health,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "System Health"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "darkblue"},
                   'steps': [{'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 100], 'color': "gray"}],
                   'threshold': {'line': {'color': "red", 'width': 4},
                               'thickness': 0.75, 'value': 90}}
        ), row=1, col=1)
        
        # Privacy level
        privacy_level = federated_system.privacy_epsilon * 50  # Scale for display
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=privacy_level,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Privacy Level"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "green"},
                   'steps': [{'range': [0, 30], 'color': "red"},
                            {'range': [30, 70], 'color': "yellow"},
                            {'range': [70, 100], 'color': "lightgreen"}]}
        ), row=1, col=2)
        
        # Committee efficiency
        committee_efficiency = len(federated_system.current_committee) / len(federated_system.clients) * 100
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=committee_efficiency,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Committee Efficiency"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "purple"}}
        ), row=2, col=1)
        
        # Network load (mock calculation)
        network_load = (len(federated_system.fog_nodes) + len(federated_system.clients)) * 5
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=network_load,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Network Load"},
            gauge={'axis': {'range': [None, 100]},
                   'bar': {'color': "orange"}}
        ), row=2, col=2)
        
        fig.update_layout(
            title="System Performance Metrics",
            height=500
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating system metrics gauge: {e}")
        # Return empty figure
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating metrics gauge: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_confusion_matrix_heatmap(confusion_matrix_data: Dict) -> go.Figure:
    """
    Create confusion matrix heatmap visualization
    
    Args:
        confusion_matrix_data: Dictionary containing TP, TN, FP, FN values
        
    Returns:
        Plotly figure
    """
    try:
        # Extract values
        TP = confusion_matrix_data.get('TP', 0)
        TN = confusion_matrix_data.get('TN', 0)
        FP = confusion_matrix_data.get('FP', 0)
        FN = confusion_matrix_data.get('FN', 0)
        
        # Create confusion matrix array
        cm_array = np.array([[TN, FP], [FN, TP]])
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=cm_array,
            x=['Predicted: No Diabetes', 'Predicted: Diabetes'],
            y=['Actual: No Diabetes', 'Actual: Diabetes'],
            colorscale='Blues',
            text=cm_array,
            texttemplate="%{text}",
            textfont={"size": 20},
            hovertemplate='<b>%{y}</b><br><b>%{x}</b><br>Count: %{z}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Confusion Matrix - Global Model Performance",
            xaxis_title="Predicted Label",
            yaxis_title="Actual Label",
            font=dict(size=12)
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating confusion matrix: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating confusion matrix: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_timing_analysis_chart(training_history: List[Dict]) -> go.Figure:
    """
    Create comprehensive timing analysis chart
    
    Args:
        training_history: List of training round results
        
    Returns:
        Plotly figure
    """
    try:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(text="No timing data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        rounds = list(range(1, len(training_history) + 1))
        
        # Extract timing data
        client_times = [round_info.get('avg_client_training_time', 0) for round_info in training_history]
        fog_times = [round_info.get('avg_fog_execution_time', 0) for round_info in training_history]
        global_times = [round_info.get('global_aggregation_time', 0) for round_info in training_history]
        comm_times = [round_info.get('avg_communication_time', 0) for round_info in training_history]
        
        # Create subplot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Client Training Times', 'Fog Execution Times',
                'Global Aggregation Times', 'Communication Times'
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )
        
        # Client training times
        fig.add_trace(
            go.Scatter(x=rounds, y=client_times, mode='lines+markers',
                      name='Client Training', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Fog execution times
        fig.add_trace(
            go.Scatter(x=rounds, y=fog_times, mode='lines+markers',
                      name='Fog Execution', line=dict(color='green')),
            row=1, col=2
        )
        
        # Global aggregation times
        fig.add_trace(
            go.Scatter(x=rounds, y=global_times, mode='lines+markers',
                      name='Global Aggregation', line=dict(color='orange')),
            row=2, col=1
        )
        
        # Communication times
        fig.add_trace(
            go.Scatter(x=rounds, y=comm_times, mode='lines+markers',
                      name='Communication', line=dict(color='red')),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title="Detailed Timing Analysis Across Rounds",
            showlegend=False,
            height=600
        )
        
        # Update axes
        for i in range(1, 3):
            for j in range(1, 3):
                fig.update_xaxes(title_text="Round", row=i, col=j)
                fig.update_yaxes(title_text="Time (seconds)", row=i, col=j)
        
        return fig
        
    except Exception as e:
        print(f"Error creating timing analysis: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating timing analysis: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_communication_flow_chart(training_history: List[Dict]) -> go.Figure:
    """
    Create communication flow and latency visualization
    
    Args:
        training_history: List of training round results
        
    Returns:
        Plotly figure
    """
    try:
        if not training_history:
            fig = go.Figure()
            fig.add_annotation(text="No communication data available", 
                              xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            return fig
        
        rounds = list(range(1, len(training_history) + 1))
        
        # Extract communication timing data
        comm_times = [round_info.get('avg_communication_time', 0) for round_info in training_history]
        global_comm_times = [round_info.get('global_communication_time', 0) for round_info in training_history]
        fog_agg_times = [round_info.get('fog_aggregation_time', 0) for round_info in training_history]
        
        fig = go.Figure()
        
        # Add traces for different communication phases
        fig.add_trace(go.Scatter(
            x=rounds, y=comm_times,
            mode='lines+markers',
            name='Client-to-Fog Communication',
            line=dict(color='lightblue', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds, y=fog_agg_times,
            mode='lines+markers',
            name='Fog Aggregation Phase',
            line=dict(color='lightgreen', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=rounds, y=global_comm_times,
            mode='lines+markers',
            name='Global Distribution',
            line=dict(color='orange', width=3),
            marker=dict(size=8)
        ))
        
        # Add area plot for total communication overhead
        total_comm = [c + g + f for c, g, f in zip(comm_times, global_comm_times, fog_agg_times)]
        fig.add_trace(go.Scatter(
            x=rounds, y=total_comm,
            mode='lines',
            name='Total Communication Overhead',
            line=dict(color='red', width=2, dash='dash'),
            fill='tonexty'
        ))
        
        fig.update_layout(
            title="Communication Flow & Latency Analysis",
            xaxis_title="Training Round",
            yaxis_title="Time (seconds)",
            hovermode='x unified',
            plot_bgcolor='white'
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating communication flow chart: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating communication flow chart: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

def create_performance_metrics_dashboard(confusion_matrix_data: Dict) -> go.Figure:
    """
    Create comprehensive performance metrics dashboard
    
    Args:
        confusion_matrix_data: Dictionary containing confusion matrix and metrics
        
    Returns:
        Plotly figure
    """
    try:
        # Extract metrics
        precision = confusion_matrix_data.get('precision', 0)
        recall = confusion_matrix_data.get('recall', 0)
        specificity = confusion_matrix_data.get('specificity', 0)
        f1_score = confusion_matrix_data.get('f1_score', 0)
        
        metrics = ['Precision', 'Recall', 'Specificity', 'F1-Score']
        values = [precision, recall, specificity, f1_score]
        
        # Create radar chart
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],  # Close the polygon
            theta=metrics + [metrics[0]],
            fill='toself',
            name='Performance Metrics',
            line_color='rgb(32, 146, 230)',
            fillcolor='rgba(32, 146, 230, 0.3)'
        ))
        
        fig.update_layout(
            title="Model Performance Metrics Dashboard",
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode='linear',
                    tick0=0,
                    dtick=0.2
                )
            ),
            showlegend=False
        )
        
        return fig
        
    except Exception as e:
        print(f"Error creating performance dashboard: {e}")
        fig = go.Figure()
        fig.add_annotation(text=f"Error creating performance dashboard: {str(e)}", 
                          xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig
