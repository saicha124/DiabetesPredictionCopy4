import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedClientAnalytics:
    """Advanced analytics for federated learning client performance monitoring"""
    
    def __init__(self):
        self.client_metrics_history = {}
        self.convergence_metrics = {}
        self.anomaly_detector = None
        self.performance_thresholds = {
            'accuracy': 0.7,
            'f1_score': 0.65,
            'precision': 0.7,
            'recall': 0.65
        }
        
    def update_client_performance(self, round_num: int, client_id: int, 
                                y_true: np.ndarray, y_pred: np.ndarray, 
                                y_prob: np.ndarray = None, model_params: Dict = None):
        """Update comprehensive performance metrics for a client"""
        
        # Calculate core metrics
        accuracy = np.mean(y_true == y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division='warn')
        precision = precision_score(y_true, y_pred, average='weighted', zero_division='warn')
        recall = recall_score(y_true, y_pred, average='weighted', zero_division='warn')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Classification report
        class_report = classification_report(y_true, y_pred, output_dict=True, zero_division='warn')
        
        # Store metrics
        if client_id not in self.client_metrics_history:
            self.client_metrics_history[client_id] = []
            
        metrics = {
            'round': round_num,
            'timestamp': datetime.now(),
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'data_size': len(y_true),
            'class_distribution': np.bincount(y_true),
            'prediction_confidence': np.mean(y_prob) if y_prob is not None else None,
            'model_complexity': self._estimate_model_complexity(model_params) if model_params else None
        }
        
        self.client_metrics_history[client_id].append(metrics)
        
        # Update convergence tracking
        self._update_convergence_metrics(client_id, metrics)
        
    def _estimate_model_complexity(self, model_params: Dict) -> float:
        """Estimate model complexity based on parameters"""
        if not model_params:
            return 0.5
        
        # Simple complexity estimation based on parameter count and values
        total_params = sum(np.prod(param.shape) if hasattr(param, 'shape') else 1 
                          for param in model_params.values())
        return min(1.0, total_params / 10000)  # Normalize
        
    def _update_convergence_metrics(self, client_id: int, current_metrics: Dict):
        """Update convergence tracking for client"""
        if client_id not in self.convergence_metrics:
            self.convergence_metrics[client_id] = {
                'accuracy_trend': [],
                'f1_trend': [],
                'convergence_score': 0.0,
                'stability_score': 0.0,
                'improvement_rate': 0.0
            }
        
        history = self.client_metrics_history[client_id]
        conv_metrics = self.convergence_metrics[client_id]
        
        # Update trends
        conv_metrics['accuracy_trend'] = [m['accuracy'] for m in history[-10:]]  # Last 10 rounds
        conv_metrics['f1_trend'] = [m['f1_score'] for m in history[-10:]]
        
        # Calculate convergence score (stability of recent performance)
        if len(conv_metrics['accuracy_trend']) >= 3:
            accuracy_std = np.std(conv_metrics['accuracy_trend'][-5:])
            conv_metrics['convergence_score'] = max(0, 1 - (accuracy_std * 10))
            
        # Calculate stability score
        if len(history) >= 5:
            recent_acc = [m['accuracy'] for m in history[-5:]]
            conv_metrics['stability_score'] = 1 - np.std(recent_acc)
            
        # Calculate improvement rate
        if len(history) >= 3:
            early_acc = np.mean([m['accuracy'] for m in history[:3]])
            recent_acc = np.mean([m['accuracy'] for m in history[-3:]])
            conv_metrics['improvement_rate'] = max(0, recent_acc - early_acc)
    
    def detect_anomalies(self) -> Dict[str, Any]:
        """Detect anomalous client performance using isolation forest"""
        if len(self.client_metrics_history) < 3:
            return {'anomalous_clients': [], 'performance_outliers': []}
        
        # Prepare feature matrix for anomaly detection
        features = []
        client_ids = []
        
        for client_id, history in self.client_metrics_history.items():
            if len(history) >= 3:  # Need sufficient history
                latest_metrics = history[-1]
                recent_performance = history[-3:]
                
                # Feature vector: latest performance + trends
                feature_vector = [
                    latest_metrics['accuracy'],
                    latest_metrics['f1_score'],
                    latest_metrics['precision'],
                    latest_metrics['recall'],
                    np.mean([m['accuracy'] for m in recent_performance]),
                    np.std([m['accuracy'] for m in recent_performance]),
                    len(history),  # Number of rounds participated
                    latest_metrics['data_size'],
                    latest_metrics.get('prediction_confidence', 0.5)
                ]
                
                features.append(feature_vector)
                client_ids.append(client_id)
        
        if len(features) < 2:
            return {'anomalous_clients': [], 'performance_outliers': []}
        
        # Fit isolation forest
        features_array = np.array(features)
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_array)
        
        isolation_forest = IsolationForest(contamination='auto', random_state=42)
        anomaly_scores = isolation_forest.fit_predict(features_scaled)
        
        # Identify anomalous clients
        anomalous_clients = []
        performance_outliers = []
        
        for i, (client_id, score) in enumerate(zip(client_ids, anomaly_scores)):
            if score == -1:  # Anomaly detected
                latest_metrics = self.client_metrics_history[client_id][-1]
                
                # Categorize type of anomaly
                if latest_metrics['accuracy'] < self.performance_thresholds['accuracy']:
                    anomalous_clients.append({
                        'client_id': client_id,
                        'type': 'underperforming',
                        'accuracy': latest_metrics['accuracy'],
                        'f1_score': latest_metrics['f1_score'],
                        'severity': 'high' if latest_metrics['accuracy'] < 0.5 else 'medium'
                    })
                else:
                    performance_outliers.append({
                        'client_id': client_id,
                        'type': 'outlier',
                        'accuracy': latest_metrics['accuracy'],
                        'f1_score': latest_metrics['f1_score'],
                        'anomaly_score': features_scaled[i].tolist()
                    })
        
        return {
            'anomalous_clients': anomalous_clients,
            'performance_outliers': performance_outliers,
            'total_clients_analyzed': len(client_ids)
        }
    
    def create_medical_facility_dashboard(self):
        """Create comprehensive medical facility performance dashboard"""
        if not self.client_metrics_history:
            st.warning("No client performance data available. Please complete training first.")
            return
        
        st.subheader("🏥 Medical Facility Performance Dashboard")
        
        # Overview metrics
        self._create_facility_overview()
        
        st.markdown("---")
        
        # Performance evolution
        self._create_performance_evolution()
        
        st.markdown("---")
        
        # Confusion matrix analysis
        self._create_confusion_matrix_analysis()
        
        st.markdown("---")
        
        # Anomaly detection
        self._create_anomaly_detection_panel()
        
        st.markdown("---")
        
        # Convergence analysis
        self._create_convergence_analysis()
    
    def _create_facility_overview(self):
        """Create facility overview metrics"""
        st.subheader("📊 Facility Overview")
        
        # Calculate aggregate metrics
        total_facilities = len(self.client_metrics_history)
        latest_metrics = {}
        
        for client_id, history in self.client_metrics_history.items():
            if history:
                latest_metrics[client_id] = history[-1]
        
        if not latest_metrics:
            st.warning("No recent metrics available")
            return
        
        # Aggregate statistics
        avg_accuracy = np.mean([m['accuracy'] for m in latest_metrics.values()])
        avg_f1 = np.mean([m['f1_score'] for m in latest_metrics.values()])
        avg_precision = np.mean([m['precision'] for m in latest_metrics.values()])
        avg_recall = np.mean([m['recall'] for m in latest_metrics.values()])
        
        # Display metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Medical Facilities", total_facilities)
        with col2:
            st.metric("Avg Accuracy", f"{avg_accuracy:.3f}")
        with col3:
            st.metric("Avg F1-Score", f"{avg_f1:.3f}")
        with col4:
            st.metric("Avg Precision", f"{avg_precision:.3f}")
        with col5:
            st.metric("Avg Recall", f"{avg_recall:.3f}")
        
        # Performance distribution
        performance_data = []
        for client_id, metrics in latest_metrics.items():
            performance_data.append({
                'Facility': f"Medical Station {client_id + 1}",
                'Accuracy': metrics['accuracy'],
                'F1-Score': metrics['f1_score'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'Data Size': metrics['data_size']
            })
        
        df_performance = pd.DataFrame(performance_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_bar = px.bar(
                df_performance, 
                x='Facility', 
                y=['Accuracy', 'F1-Score', 'Precision', 'Recall'],
                title="Performance Metrics by Medical Facility",
                barmode='group'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col2:
            fig_scatter = px.scatter(
                df_performance,
                x='Accuracy',
                y='F1-Score',
                size='Data Size',
                hover_name='Facility',
                title="Accuracy vs F1-Score Distribution",
                color='Facility'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _create_performance_evolution(self):
        """Create performance evolution visualization"""
        st.subheader("📈 Performance Evolution Over Time")
        
        # Prepare evolution data
        evolution_data = []
        for client_id, history in self.client_metrics_history.items():
            for metrics in history:
                evolution_data.append({
                    'Round': metrics['round'],
                    'Facility': f"Medical Station {client_id + 1}",
                    'Client_ID': client_id,
                    'Accuracy': metrics['accuracy'],
                    'F1-Score': metrics['f1_score'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'Data_Size': metrics['data_size']
                })
        
        if not evolution_data:
            st.warning("No evolution data available")
            return
        
        df_evolution = pd.DataFrame(evolution_data)
        
        # Metric selector
        metric_options = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        selected_metric = st.selectbox("Select Metric to Track", metric_options, index=0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Line plot for evolution
            fig_evolution = px.line(
                df_evolution,
                x='Round',
                y=selected_metric,
                color='Facility',
                title=f"{selected_metric} Evolution Across Training Rounds",
                markers=True
            )
            fig_evolution.update_layout(height=400)
            st.plotly_chart(fig_evolution, use_container_width=True)
        
        with col2:
            # Heatmap for performance matrix
            pivot_data = df_evolution.pivot_table(
                values=selected_metric,
                index='Facility',
                columns='Round',
                aggfunc='mean'
            ).fillna(0)
            
            fig_heatmap = px.imshow(
                pivot_data,
                title=f"{selected_metric} Heatmap by Round",
                color_continuous_scale='RdYlBu_r',
                aspect='auto'
            )
            fig_heatmap.update_layout(height=400)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # Performance improvement analysis
        st.subheader("📊 Performance Improvement Analysis")
        
        improvement_data = []
        for client_id, history in self.client_metrics_history.items():
            if len(history) >= 2:
                initial_acc = history[0]['accuracy']
                final_acc = history[-1]['accuracy']
                improvement = final_acc - initial_acc
                
                initial_f1 = history[0]['f1_score']
                final_f1 = history[-1]['f1_score']
                f1_improvement = final_f1 - initial_f1
                
                improvement_data.append({
                    'Facility': f"Medical Station {client_id + 1}",
                    'Accuracy_Improvement': improvement,
                    'F1_Improvement': f1_improvement,
                    'Initial_Accuracy': initial_acc,
                    'Final_Accuracy': final_acc,
                    'Rounds_Participated': len(history)
                })
        
        if improvement_data:
            df_improvement = pd.DataFrame(improvement_data)
            
            fig_improvement = px.scatter(
                df_improvement,
                x='Accuracy_Improvement',
                y='F1_Improvement',
                size='Rounds_Participated',
                hover_name='Facility',
                title="Performance Improvement Analysis",
                color='Final_Accuracy',
                color_continuous_scale='Viridis'
            )
            fig_improvement.add_hline(y=0, line_dash="dash", line_color="gray")
            fig_improvement.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig_improvement, use_container_width=True)
    
    def _create_confusion_matrix_analysis(self):
        """Create comprehensive confusion matrix analysis"""
        st.subheader("🔍 Confusion Matrix Analysis")
        
        # Client selector for detailed analysis
        client_options = [f"Medical Station {i + 1}" for i in self.client_metrics_history.keys()]
        if not client_options:
            st.warning("No client data available for confusion matrix analysis")
            return
        
        selected_client_name = st.selectbox("Select Medical Facility for Detailed Analysis", client_options)
        client_id = int(selected_client_name.split()[-1]) - 1
        
        if client_id not in self.client_metrics_history:
            st.warning("No data available for selected facility")
            return
        
        history = self.client_metrics_history[client_id]
        if not history:
            st.warning("No history available for selected facility")
            return
        
        # Round selector
        round_options = [f"Round {m['round']}" for m in history]
        if not round_options:
            st.warning("No training rounds available for analysis")
            return
            
        default_index = len(round_options) - 1 if round_options else 0
        selected_round_name = st.selectbox("Select Training Round", round_options, index=default_index)
        round_idx = int(selected_round_name.split()[-1])
        
        # Find corresponding metrics
        selected_metrics = None
        for metrics in history:
            if metrics['round'] == round_idx:
                selected_metrics = metrics
                break
        
        if selected_metrics is None:
            st.warning("No metrics found for selected round")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confusion matrix heatmap
            cm = selected_metrics['confusion_matrix']
            
            fig_cm = px.imshow(
                cm,
                text_auto=True,
                title=f"Confusion Matrix - {selected_client_name}, {selected_round_name}",
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=['No Diabetes', 'Diabetes'],
                y=['No Diabetes', 'Diabetes'],
                color_continuous_scale='Blues'
            )
            fig_cm.update_layout(height=400)
            st.plotly_chart(fig_cm, use_container_width=True)
        
        with col2:
            # Classification metrics
            st.subheader("Classification Metrics")
            
            class_report = selected_metrics['classification_report']
            
            # Display per-class metrics
            for class_label in ['0', '1']:  # No diabetes, Diabetes
                if class_label in class_report:
                    class_name = 'No Diabetes' if class_label == '0' else 'Diabetes'
                    st.markdown(f"**{class_name}:**")
                    
                    metrics_data = class_report[class_label]
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        st.metric("Precision", f"{metrics_data['precision']:.3f}")
                    with col_b:
                        st.metric("Recall", f"{metrics_data['recall']:.3f}")
                    with col_c:
                        st.metric("F1-Score", f"{metrics_data['f1-score']:.3f}")
        
        # Additional analysis
        st.subheader("Performance Insights")
        
        # Calculate derived metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sensitivity (Recall)", f"{sensitivity:.3f}")
        with col2:
            st.metric("Specificity", f"{specificity:.3f}")
        with col3:
            st.metric("PPV (Precision)", f"{ppv:.3f}")
        with col4:
            st.metric("NPV", f"{npv:.3f}")
        
        # Performance interpretation
        performance_level = "Excellent" if selected_metrics['accuracy'] > 0.9 else \
                          "Good" if selected_metrics['accuracy'] > 0.8 else \
                          "Fair" if selected_metrics['accuracy'] > 0.7 else "Needs Improvement"
        
        st.info(f"**Performance Level:** {performance_level} (Accuracy: {selected_metrics['accuracy']:.3f})")
    
    def _create_anomaly_detection_panel(self):
        """Create anomaly detection and outlier analysis panel"""
        st.subheader("🚨 Anomaly Detection & Outlier Analysis")
        
        # Run anomaly detection
        anomaly_results = self.detect_anomalies()
        
        if not anomaly_results['anomalous_clients'] and not anomaly_results['performance_outliers']:
            st.success("✅ No anomalies or performance outliers detected")
            st.info(f"Analyzed {anomaly_results.get('total_clients_analyzed', 0)} medical facilities")
            return
        
        # Display anomalous clients
        if anomaly_results['anomalous_clients']:
            st.subheader("⚠️ Underperforming Medical Facilities")
            
            for anomaly in anomaly_results['anomalous_clients']:
                severity_color = "🔴" if anomaly['severity'] == 'high' else "🟡"
                
                with st.expander(f"{severity_color} Medical Station {anomaly['client_id'] + 1} - {anomaly['severity'].title()} Risk"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Accuracy", f"{anomaly['accuracy']:.3f}")
                    with col2:
                        st.metric("F1-Score", f"{anomaly['f1_score']:.3f}")
                    with col3:
                        st.metric("Risk Level", anomaly['severity'].title())
                    
                    # Recommendations
                    st.markdown("**Recommendations:**")
                    if anomaly['accuracy'] < 0.5:
                        st.write("• Review data quality and preprocessing")
                        st.write("• Check for data distribution issues")
                        st.write("• Consider additional training rounds")
                    elif anomaly['accuracy'] < 0.7:
                        st.write("• Monitor training stability")
                        st.write("• Evaluate hyperparameter tuning")
                        st.write("• Check for overfitting")
        
        # Display performance outliers
        if anomaly_results['performance_outliers']:
            st.subheader("📊 Performance Outliers")
            
            outlier_data = []
            for outlier in anomaly_results['performance_outliers']:
                outlier_data.append({
                    'Facility': f"Medical Station {outlier['client_id'] + 1}",
                    'Accuracy': outlier['accuracy'],
                    'F1-Score': outlier['f1_score'],
                    'Type': outlier['type']
                })
            
            df_outliers = pd.DataFrame(outlier_data)
            st.dataframe(df_outliers, use_container_width=True)
        
        # Facility performance ranking
        st.subheader("🏆 Facility Performance Ranking")
        
        ranking_data = []
        for client_id, history in self.client_metrics_history.items():
            if history:
                latest = history[-1]
                convergence = self.convergence_metrics.get(client_id, {})
                
                ranking_data.append({
                    'Rank': 0,  # Will be calculated
                    'Facility': f"Medical Station {client_id + 1}",
                    'Accuracy': latest['accuracy'],
                    'F1-Score': latest['f1_score'],
                    'Stability': convergence.get('stability_score', 0),
                    'Improvement': convergence.get('improvement_rate', 0),
                    'Overall_Score': (latest['accuracy'] + latest['f1_score'] + 
                                    convergence.get('stability_score', 0) + 
                                    convergence.get('improvement_rate', 0)) / 4
                })
        
        if ranking_data:
            df_ranking = pd.DataFrame(ranking_data)
            df_ranking = df_ranking.sort_values('Overall_Score', ascending=False)
            df_ranking['Rank'] = range(1, len(df_ranking) + 1)
            
            # Display ranking table
            st.dataframe(
                df_ranking[['Rank', 'Facility', 'Accuracy', 'F1-Score', 'Stability', 'Improvement', 'Overall_Score']].round(3),
                use_container_width=True
            )
    
    def _create_convergence_analysis(self):
        """Create convergence analysis dashboard"""
        st.subheader("🎯 Convergence Analysis")
        
        if not self.convergence_metrics:
            st.warning("No convergence data available")
            return
        
        # Convergence overview
        convergence_data = []
        for client_id, conv_metrics in self.convergence_metrics.items():
            convergence_data.append({
                'Facility': f"Medical Station {client_id + 1}",
                'Convergence_Score': conv_metrics.get('convergence_score', 0),
                'Stability_Score': conv_metrics.get('stability_score', 0),
                'Improvement_Rate': conv_metrics.get('improvement_rate', 0),
                'Rounds_History': len(self.client_metrics_history[client_id])
            })
        
        df_convergence = pd.DataFrame(convergence_data)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Convergence scores
            fig_conv = px.bar(
                df_convergence,
                x='Facility',
                y=['Convergence_Score', 'Stability_Score'],
                title="Convergence & Stability Scores",
                barmode='group'
            )
            fig_conv.update_layout(height=400)
            st.plotly_chart(fig_conv, use_container_width=True)
        
        with col2:
            # Improvement vs stability
            fig_scatter = px.scatter(
                df_convergence,
                x='Stability_Score',
                y='Improvement_Rate',
                size='Rounds_History',
                hover_name='Facility',
                title="Stability vs Improvement Rate",
                color='Convergence_Score',
                color_continuous_scale='Viridis'
            )
            fig_scatter.update_layout(height=400)
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Convergence trends
        st.subheader("📈 Convergence Trends")
        
        # Select facility for detailed trend analysis
        facility_options = [f"Medical Station {i + 1}" for i in self.convergence_metrics.keys()]
        selected_facility = st.selectbox("Select Facility for Trend Analysis", facility_options)
        client_id = int(selected_facility.split()[-1]) - 1
        
        if client_id in self.convergence_metrics:
            conv_metrics = self.convergence_metrics[client_id]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Accuracy trend
                if conv_metrics.get('accuracy_trend'):
                    rounds = list(range(len(conv_metrics['accuracy_trend'])))
                    
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        x=rounds,
                        y=conv_metrics['accuracy_trend'],
                        mode='lines+markers',
                        name='Accuracy',
                        line=dict(color='blue')
                    ))
                    fig_trend.update_layout(
                        title=f"Accuracy Trend - {selected_facility}",
                        xaxis_title="Recent Rounds",
                        yaxis_title="Accuracy",
                        height=300
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            with col2:
                # F1 trend
                if conv_metrics.get('f1_trend'):
                    rounds = list(range(len(conv_metrics['f1_trend'])))
                    
                    fig_f1_trend = go.Figure()
                    fig_f1_trend.add_trace(go.Scatter(
                        x=rounds,
                        y=conv_metrics['f1_trend'],
                        mode='lines+markers',
                        name='F1-Score',
                        line=dict(color='green')
                    ))
                    fig_f1_trend.update_layout(
                        title=f"F1-Score Trend - {selected_facility}",
                        xaxis_title="Recent Rounds",
                        yaxis_title="F1-Score",
                        height=300
                    )
                    st.plotly_chart(fig_f1_trend, use_container_width=True)
            
            # Convergence summary
            st.subheader(f"🎯 {selected_facility} Convergence Summary")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                score = conv_metrics.get('convergence_score', 0)
                st.metric("Convergence Score", f"{score:.3f}")
                status = "Converged" if score > 0.8 else "Converging" if score > 0.5 else "Unstable"
                st.write(f"Status: **{status}**")
            
            with col2:
                stability = conv_metrics.get('stability_score', 0)
                st.metric("Stability Score", f"{stability:.3f}")
                stability_level = "High" if stability > 0.8 else "Medium" if stability > 0.6 else "Low"
                st.write(f"Stability: **{stability_level}**")
            
            with col3:
                improvement = conv_metrics.get('improvement_rate', 0)
                st.metric("Improvement Rate", f"{improvement:.3f}")
                improvement_trend = "Improving" if improvement > 0.05 else "Stable" if improvement > -0.02 else "Declining"
                st.write(f"Trend: **{improvement_trend}**")