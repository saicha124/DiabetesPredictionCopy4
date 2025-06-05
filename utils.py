import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

def calculate_metrics(y_true, y_pred, y_proba=None):
    """Calculate comprehensive classification metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    metrics['f1_score'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
    
    # ROC AUC if probabilities are provided
    if y_proba is not None:
        try:
            if len(np.unique(y_true)) == 2:  # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            else:  # Multiclass
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
        except:
            metrics['roc_auc'] = 0.0
    
    # Classification report
    metrics['classification_report'] = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    return metrics

def plot_confusion_matrix(cm, class_names=None):
    """Plot confusion matrix using matplotlib"""
    if class_names is None:
        class_names = ['No Diabetes', 'Diabetes']
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    return plt.gcf()

def plot_training_history(training_history):
    """Plot training history with multiple metrics"""
    if not training_history:
        return None
    
    df = pd.DataFrame(training_history)
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Accuracy Over Rounds', 'Loss Over Rounds', 
                       'F1 Score Over Rounds', 'Training Time per Round'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Accuracy
    fig.add_trace(
        go.Scatter(x=df['round'], y=df['accuracy'], mode='lines+markers', 
                  name='Accuracy', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Loss
    fig.add_trace(
        go.Scatter(x=df['round'], y=df['loss'], mode='lines+markers', 
                  name='Loss', line=dict(color='red')),
        row=1, col=2
    )
    
    # F1 Score
    fig.add_trace(
        go.Scatter(x=df['round'], y=df['f1_score'], mode='lines+markers', 
                  name='F1 Score', line=dict(color='green')),
        row=2, col=1
    )
    
    # Training Time
    fig.add_trace(
        go.Scatter(x=df['round'], y=df['execution_time'], mode='lines+markers', 
                  name='Execution Time', line=dict(color='orange')),
        row=2, col=2
    )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Progress")
    
    return fig

def plot_client_statistics(client_stats):
    """Plot client data distribution statistics"""
    if not client_stats:
        return None
    
    # Prepare data
    client_ids = [stat['client_id'] for stat in client_stats]
    train_samples = [stat['train_samples'] for stat in client_stats]
    test_samples = [stat['test_samples'] for stat in client_stats]
    
    # Create subplot
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Training Samples per Client', 'Test Samples per Client')
    )
    
    # Training samples
    fig.add_trace(
        go.Bar(x=client_ids, y=train_samples, name='Training Samples', 
               marker_color='lightblue'),
        row=1, col=1
    )
    
    # Test samples
    fig.add_trace(
        go.Bar(x=client_ids, y=test_samples, name='Test Samples', 
               marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Client Data Distribution")
    
    return fig

def plot_roc_curve(y_true, y_proba):
    """Plot ROC curve"""
    if len(np.unique(y_true)) != 2:
        return None
    
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = roc_auc_score(y_true, y_proba)
    
    fig = go.Figure()
    
    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {auc_score:.3f})',
        line=dict(color='blue', width=2)
    ))
    
    # Diagonal line (random classifier)
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title='ROC Curve',
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        showlegend=True,
        width=600,
        height=500
    )
    
    return fig

def create_privacy_budget_chart(privacy_budget_info):
    """Create privacy budget consumption chart"""
    rounds = list(range(1, privacy_budget_info['rounds'] + 1))
    cumulative_epsilon = [i * privacy_budget_info['per_round_epsilon'] for i in rounds]
    cumulative_delta = [i * privacy_budget_info['per_round_delta'] for i in rounds]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Cumulative Epsilon (ε)', 'Cumulative Delta (δ)')
    )
    
    # Epsilon consumption
    fig.add_trace(
        go.Scatter(x=rounds, y=cumulative_epsilon, mode='lines+markers',
                  name='Cumulative ε', line=dict(color='blue')),
        row=1, col=1
    )
    
    # Delta consumption
    fig.add_trace(
        go.Scatter(x=rounds, y=cumulative_delta, mode='lines+markers',
                  name='Cumulative δ', line=dict(color='red')),
        row=1, col=2
    )
    
    fig.update_layout(height=400, title_text="Privacy Budget Consumption")
    
    return fig

def format_metrics_table(metrics):
    """Format metrics into a readable table"""
    if not metrics:
        return pd.DataFrame()
    
    # Extract key metrics
    formatted_metrics = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Value': [
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('precision', 0):.4f}",
            f"{metrics.get('recall', 0):.4f}",
            f"{metrics.get('f1_score', 0):.4f}",
            f"{metrics.get('roc_auc', 0):.4f}"
        ]
    }
    
    return pd.DataFrame(formatted_metrics)

def save_results_to_json(results, filename='federated_learning_results.json'):
    """Save results to JSON file"""
    import json
    
    # Convert numpy arrays to lists for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    serializable_results = convert_numpy(results)
    
    try:
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving results: {e}")
        return False

def load_results_from_json(filename='federated_learning_results.json'):
    """Load results from JSON file"""
    import json
    
    try:
        with open(filename, 'r') as f:
            results = json.load(f)
        return results
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def calculate_statistical_significance(results1, results2, metric='accuracy'):
    """Calculate statistical significance between two sets of results"""
    from scipy import stats
    
    try:
        values1 = [r[metric] for r in results1 if metric in r]
        values2 = [r[metric] for r in results2 if metric in r]
        
        if len(values1) < 2 or len(values2) < 2:
            return None
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(values1, values2)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size': (np.mean(values1) - np.mean(values2)) / np.sqrt((np.var(values1) + np.var(values2)) / 2)
        }
    except Exception as e:
        print(f"Error calculating statistical significance: {e}")
        return None

def generate_summary_report(results):
    """Generate a comprehensive summary report"""
    if not results:
        return "No results available."
    
    report = []
    report.append("=" * 50)
    report.append("FEDERATED LEARNING SUMMARY REPORT")
    report.append("=" * 50)
    
    # Basic information
    report.append(f"Final Accuracy: {results.get('final_accuracy', 0):.4f}")
    report.append(f"Final Loss: {results.get('final_loss', 0):.4f}")
    report.append(f"Rounds Completed: {results.get('rounds_completed', 0)}")
    report.append(f"Total Training Time: {results.get('total_time', 0):.2f} seconds")
    
    # Training progress
    if 'training_history' in results and results['training_history']:
        report.append("\nTraining Progress:")
        report.append("-" * 20)
        
        history = results['training_history']
        best_accuracy = max([h.get('accuracy', 0) for h in history])
        best_round = [i for i, h in enumerate(history) if h.get('accuracy', 0) == best_accuracy][0] + 1
        
        report.append(f"Best Accuracy: {best_accuracy:.4f} (Round {best_round})")
        report.append(f"Average Round Time: {np.mean([h.get('execution_time', 0) for h in history]):.2f} seconds")
    
    report.append("\n" + "=" * 50)
    
    return "\n".join(report)
