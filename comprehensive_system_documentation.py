"""
Comprehensive PDF Documentation Generator for Hierarchical Federated Deep Learning System
Generates complete technical documentation covering requirements, installation, methodology, 
communication protocols, program description, and step-by-step usage guide.
"""

from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red, green, gray, lightgrey
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.graphics.shapes import Drawing, Rect, Circle, Line
from reportlab.graphics.charts.linecharts import HorizontalLineChart
from reportlab.graphics.charts.barcharts import VerticalBarChart
from datetime import datetime
import os

def create_comprehensive_system_documentation():
    """Generate comprehensive PDF documentation for the hierarchical federated learning system."""
    
    filename = f"Hierarchical_Federated_Learning_Complete_Documentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=blue
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=blue
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubHeading',
        parent=styles['Heading3'],
        fontSize=14,
        spaceAfter=8,
        spaceBefore=12,
        textColor=black
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY
    )
    
    code_style = ParagraphStyle(
        'CodeStyle',
        parent=styles['Code'],
        fontSize=9,
        backColor=lightgrey,
        borderColor=gray,
        borderWidth=1,
        borderPadding=10
    )
    
    # Story container
    story = []
    
    # Title Page
    story.append(Paragraph("Hierarchical Federated Deep Learning System", title_style))
    story.append(Spacer(1, 20))
    story.append(Paragraph("Advanced Diabetes Prediction with Privacy-Preserving Machine Learning", styles['Heading2']))
    story.append(Spacer(1, 30))
    
    # System overview table
    overview_data = [
        ['Project Type', 'Hierarchical Federated Learning Platform'],
        ['Primary Application', 'Diabetes Risk Prediction'],
        ['Architecture', '3-Tier Federation (Patient → Fog → Global)'],
        ['Security Features', 'Differential Privacy + Committee Validation'],
        ['Interface', 'Streamlit Web Application'],
        ['Language Support', 'English and French'],
        ['Documentation Date', datetime.now().strftime('%B %d, %Y')]
    ]
    
    overview_table = Table(overview_data, colWidths=[2*inch, 4*inch])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, black)
    ]))
    
    story.append(overview_table)
    story.append(PageBreak())
    
    # Table of Contents
    story.append(Paragraph("Table of Contents", heading_style))
    
    toc_data = [
        "1. System Requirements and Dependencies",
        "2. Installation Guide",
        "3. Methodology and Architecture",
        "4. Communication Protocols",
        "5. Program Description and Components",
        "6. Step-by-Step Usage Guide",
        "7. User Interface Tabs Explanation",
        "8. Advanced Features and Analytics",
        "9. Security and Privacy Features",
        "10. Troubleshooting and Best Practices"
    ]
    
    for item in toc_data:
        story.append(Paragraph(item, body_style))
    
    story.append(PageBreak())
    
    # 1. System Requirements and Dependencies
    story.append(Paragraph("1. System Requirements and Dependencies", heading_style))
    
    story.append(Paragraph("1.1 Hardware Requirements", subheading_style))
    hardware_reqs = [
        "• Minimum 4GB RAM (8GB recommended for optimal performance)",
        "• Multi-core processor (4+ cores recommended)",
        "• 2GB available disk space",
        "• Stable internet connection for federated communication",
        "• Browser with JavaScript support for web interface"
    ]
    for req in hardware_reqs:
        story.append(Paragraph(req, body_style))
    
    story.append(Paragraph("1.2 Software Dependencies", subheading_style))
    
    # Python dependencies table
    python_deps = [
        ['Package', 'Version', 'Purpose'],
        ['streamlit', 'Latest', 'Web interface framework'],
        ['numpy', '≥1.21.0', 'Numerical computations'],
        ['pandas', '≥1.3.0', 'Data manipulation'],
        ['scikit-learn', '≥1.0.0', 'Machine learning algorithms'],
        ['plotly', '≥5.0.0', 'Interactive visualizations'],
        ['matplotlib', '≥3.4.0', 'Static plotting'],
        ['seaborn', '≥0.11.0', 'Statistical visualizations'],
        ['reportlab', '≥3.6.0', 'PDF generation'],
        ['networkx', '≥2.6.0', 'Network graph analysis'],
        ['trafilatura', 'Latest', 'Web content extraction']
    ]
    
    deps_table = Table(python_deps, colWidths=[2*inch, 1*inch, 2.5*inch])
    deps_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, black)
    ]))
    
    story.append(deps_table)
    story.append(PageBreak())
    
    # 2. Installation Guide
    story.append(Paragraph("2. Installation Guide", heading_style))
    
    story.append(Paragraph("2.1 Environment Setup", subheading_style))
    
    install_steps = [
        "Step 1: Clone or download the project repository",
        "Step 2: Ensure Python 3.8+ is installed on your system",
        "Step 3: Install required dependencies using the package manager",
        "Step 4: Verify all dependencies are correctly installed",
        "Step 5: Configure the Streamlit server settings"
    ]
    
    for step in install_steps:
        story.append(Paragraph(step, body_style))
    
    story.append(Paragraph("2.2 Configuration Files", subheading_style))
    
    config_text = """
The system requires specific configuration for optimal performance:

Streamlit Configuration (.streamlit/config.toml):
[server]
headless = true
address = "0.0.0.0"
port = 5000

This configuration ensures the application runs properly in both development and deployment environments.
    """
    
    story.append(Paragraph(config_text, body_style))
    
    story.append(Paragraph("2.3 Running the Application", subheading_style))
    
    run_command = "streamlit run app.py --server.port 5000"
    story.append(Paragraph(f"Command: <font name='Courier'>{run_command}</font>", code_style))
    
    story.append(PageBreak())
    
    # 3. Methodology and Architecture
    story.append(Paragraph("3. Methodology and Architecture", heading_style))
    
    story.append(Paragraph("3.1 Hierarchical Federation Overview", subheading_style))
    
    architecture_text = """
The system implements a 3-tier hierarchical federated learning architecture:

TIER 1 - Medical Facilities (Edge Nodes):
• Local model training on patient data
• Privacy-preserving data processing
• Local differential privacy implementation
• Committee-based validation participation

TIER 2 - Fog Nodes (Regional Aggregators):
• Regional model aggregation
• Intermediate privacy protection
• Load balancing and coordination
• Regional performance optimization

TIER 3 - Global Server (Central Coordinator):
• Global model orchestration
• Final model aggregation using FedProx algorithm
• System-wide performance monitoring
• Security protocol enforcement
    """
    
    story.append(Paragraph(architecture_text, body_style))
    
    story.append(Paragraph("3.2 Machine Learning Methodology", subheading_style))
    
    ml_methodology = """
The system employs multiple machine learning approaches:

Primary Algorithm: Logistic Regression
• Interpretable predictions for medical applications
• Efficient training in federated environments
• Robust performance with limited data

Alternative Algorithms:
• Random Forest: For complex feature interactions
• Neural Networks: For deep learning capabilities
• Support Vector Machines: For high-dimensional data

Feature Engineering:
• Standardization and normalization
• Missing value imputation
• Feature selection and importance analysis
    """
    
    story.append(Paragraph(ml_methodology, body_style))
    
    story.append(PageBreak())
    
    # 4. Communication Protocols
    story.append(Paragraph("4. Communication Protocols", heading_style))
    
    story.append(Paragraph("4.1 Federation Communication Flow", subheading_style))
    
    comm_flow = [
        ['Phase', 'Source', 'Destination', 'Data Transmitted'],
        ['Initialization', 'Global Server', 'All Clients', 'Initial model parameters'],
        ['Local Training', 'Medical Facilities', 'Local Storage', 'Training progress'],
        ['Model Upload', 'Medical Facilities', 'Fog Nodes', 'Encrypted model updates'],
        ['Regional Aggregation', 'Fog Nodes', 'Global Server', 'Aggregated updates'],
        ['Global Update', 'Global Server', 'All Clients', 'Updated global model'],
        ['Validation', 'Committee Members', 'All Participants', 'Validation results']
    ]
    
    comm_table = Table(comm_flow, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 2*inch])
    comm_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, black)
    ]))
    
    story.append(comm_table)
    
    story.append(Paragraph("4.2 Security Protocols", subheading_style))
    
    security_protocols = """
The system implements multiple layers of security:

Differential Privacy:
• Gaussian and Laplace noise mechanisms
• Privacy budget management (ε-δ privacy)
• Adaptive noise scaling based on sensitivity

Committee-Based Validation:
• Multi-party consensus for model updates
• Anomaly detection for malicious participants
• Reputation scoring system

Secret Sharing:
• Polynomial-based secret sharing scheme
• Distributed weight reconstruction
• Protection against single points of failure

Communication Security:
• Encrypted parameter transmission
• Secure aggregation protocols
• Authentication and authorization
    """
    
    story.append(Paragraph(security_protocols, body_style))
    
    story.append(PageBreak())
    
    # 5. Program Description and Components
    story.append(Paragraph("5. Program Description and Components", heading_style))
    
    story.append(Paragraph("5.1 Core Components", subheading_style))
    
    components = [
        ['Component', 'File', 'Primary Function'],
        ['Main Application', 'app.py', 'Streamlit interface and coordination'],
        ['Federated Learning Manager', 'federated_learning.py', 'FL orchestration and training'],
        ['Client Simulator', 'client_simulator.py', 'Medical facility simulation'],
        ['Data Preprocessing', 'data_preprocessing.py', 'Data cleaning and preparation'],
        ['Aggregation Algorithms', 'aggregation_algorithms.py', 'FedAvg, FedProx implementation'],
        ['Differential Privacy', 'differential_privacy.py', 'Privacy protection mechanisms'],
        ['Advanced Analytics', 'advanced_client_analytics.py', 'Performance monitoring'],
        ['Data Distribution', 'data_distribution.py', 'IID/Non-IID data handling'],
        ['Secret Sharing', 'training_secret_sharing.py', 'Cryptographic protocols']
    ]
    
    comp_table = Table(components, colWidths=[2*inch, 2*inch, 2.5*inch])
    comp_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, black)
    ]))
    
    story.append(comp_table)
    
    story.append(Paragraph("5.2 Data Flow Architecture", subheading_style))
    
    data_flow = """
The system processes data through multiple stages:

1. Data Ingestion:
   • Diabetes dataset loading and validation
   • Missing value detection and handling
   • Feature standardization and encoding

2. Data Distribution:
   • Client data partitioning (IID/Non-IID)
   • Privacy-preserving data allocation
   • Quality assessment and validation

3. Local Training:
   • Individual client model training
   • Local performance evaluation
   • Privacy noise injection

4. Aggregation:
   • Secure parameter collection
   • Weighted model averaging
   • Global model reconstruction

5. Evaluation:
   • Performance metric calculation
   • Convergence analysis
   • Security validation
    """
    
    story.append(Paragraph(data_flow, body_style))
    
    story.append(PageBreak())
    
    # 6. Step-by-Step Usage Guide
    story.append(Paragraph("6. Step-by-Step Usage Guide", heading_style))
    
    story.append(Paragraph("6.1 Initial Setup", subheading_style))
    
    setup_steps = [
        "1. Launch the application using the streamlit run command",
        "2. Access the web interface through your browser",
        "3. Select your preferred language (English/French)",
        "4. Review the system overview and architecture",
        "5. Navigate to the Training Configuration tab"
    ]
    
    for step in setup_steps:
        story.append(Paragraph(step, body_style))
    
    story.append(Paragraph("6.2 Training Configuration", subheading_style))
    
    config_steps = [
        "1. Set the number of medical facilities (clients): 3-10 recommended",
        "2. Configure maximum training rounds: 20-50 for optimal results",
        "3. Set target accuracy threshold: 0.85 for high performance",
        "4. Choose aggregation algorithm: FedProx for robustness",
        "5. Enable differential privacy with appropriate epsilon values",
        "6. Configure committee size for security validation",
        "7. Select machine learning model type",
        "8. Set early stopping parameters for efficiency"
    ]
    
    for step in config_steps:
        story.append(Paragraph(step, body_style))
    
    story.append(Paragraph("6.3 Training Execution", subheading_style))
    
    execution_steps = [
        "1. Click 'Start Federated Training' to begin the process",
        "2. Monitor real-time progress through the progress bars",
        "3. Observe accuracy and loss metrics during training",
        "4. Watch for convergence indicators and early stopping",
        "5. Review final training results and model performance"
    ]
    
    for step in execution_steps:
        story.append(Paragraph(step, body_style))
    
    story.append(PageBreak())
    
    # 7. User Interface Tabs Explanation
    story.append(Paragraph("7. User Interface Tabs Explanation", heading_style))
    
    tabs_info = [
        {
            'name': '🏥 Training Configuration',
            'purpose': 'Configure federated learning parameters and start training',
            'features': [
                'Number of medical facilities configuration',
                'Training rounds and target accuracy settings',
                'Aggregation algorithm selection',
                'Differential privacy parameters',
                'Committee security settings',
                'Model type selection',
                'Early stopping configuration'
            ]
        },
        {
            'name': '🏥 Medical Station Monitoring',
            'purpose': 'Real-time monitoring of training progress and facility performance',
            'features': [
                'Live training progress visualization',
                'Individual facility performance metrics',
                'Real-time accuracy and loss tracking',
                'Training round completion status',
                'Performance comparison across facilities'
            ]
        },
        {
            'name': '🌐 Interactive Journey Visualization',
            'purpose': 'Visual representation of the federated learning process',
            'features': [
                'Network topology visualization',
                'Data flow diagrams',
                'Hierarchical architecture display',
                'Interactive facility selection',
                'Communication pattern analysis'
            ]
        },
        {
            'name': '📊 Performance Analysis',
            'purpose': 'Comprehensive analysis of training results and model performance',
            'features': [
                'Accuracy and loss progression charts',
                'Training metrics summary',
                'Performance improvement tracking',
                'Convergence analysis',
                'Final model evaluation'
            ]
        },
        {
            'name': '🩺 Patient Risk Prediction Explainer',
            'purpose': 'Individual patient diabetes risk assessment and prediction',
            'features': [
                'Patient information input form',
                'Real-time risk prediction',
                'Feature importance analysis',
                'Clinical interpretation',
                'Risk factor explanations'
            ]
        },
        {
            'name': '📈 Advanced Medical Analytics',
            'purpose': 'Deep dive into medical facility performance and correlation analysis',
            'features': [
                'Correlation matrix analysis',
                'Feature relationship visualization',
                'Clinical insights and recommendations',
                'Medical facility performance dashboard',
                'Advanced statistical analysis'
            ]
        },
        {
            'name': '🔗 Network Visualization',
            'purpose': 'Interactive network topology and communication visualization',
            'features': [
                'Network topology graphs',
                'Data flow visualization',
                'Hierarchical architecture display',
                'Performance-based node coloring',
                'Interactive network exploration'
            ]
        },
        {
            'name': '📊 Advanced Analytics Dashboard',
            'purpose': 'Comprehensive analytics with confusion matrices and performance comparisons',
            'features': [
                'Confusion matrix analysis',
                'Accuracy vs clients optimization',
                'Fog node performance analysis',
                'Comprehensive performance comparison',
                'Medical facility grading'
            ]
        }
    ]
    
    for tab in tabs_info:
        story.append(Paragraph(f"7.{tabs_info.index(tab)+1} {tab['name']}", subheading_style))
        story.append(Paragraph(f"Purpose: {tab['purpose']}", body_style))
        story.append(Paragraph("Key Features:", body_style))
        for feature in tab['features']:
            story.append(Paragraph(f"• {feature}", body_style))
        story.append(Spacer(1, 10))
    
    story.append(PageBreak())
    
    # 8. Advanced Features and Analytics
    story.append(Paragraph("8. Advanced Features and Analytics", heading_style))
    
    story.append(Paragraph("8.1 Performance Optimization", subheading_style))
    
    optimization_text = """
The system includes several performance optimization features:

Early Stopping:
• Monitors training progress for convergence
• Prevents overfitting and reduces training time
• Automatically restores best performing model
• Configurable patience and improvement thresholds

Adaptive Learning:
• Dynamic learning rate adjustment
• Performance-based parameter tuning
• Convergence detection algorithms
• Resource usage optimization

Model Selection:
• Multiple algorithm support
• Automatic best model selection
• Cross-validation integration
• Performance comparison tools
    """
    
    story.append(Paragraph(optimization_text, body_style))
    
    story.append(Paragraph("8.2 Analytics and Visualization", subheading_style))
    
    analytics_text = """
Comprehensive analytics capabilities include:

Real-time Monitoring:
• Live training progress tracking
• Performance metric visualization
• Resource utilization monitoring
• Error detection and reporting

Post-training Analysis:
• Confusion matrix analysis
• ROC curve generation
• Feature importance ranking
• Model interpretability tools

Comparative Analysis:
• Multi-model performance comparison
• Client performance benchmarking
• Aggregation algorithm evaluation
• Privacy-utility trade-off analysis
    """
    
    story.append(Paragraph(analytics_text, body_style))
    
    story.append(PageBreak())
    
    # 9. Security and Privacy Features
    story.append(Paragraph("9. Security and Privacy Features", heading_style))
    
    story.append(Paragraph("9.1 Differential Privacy Implementation", subheading_style))
    
    dp_text = """
The system implements state-of-the-art differential privacy:

Noise Mechanisms:
• Gaussian mechanism for numerical data
• Laplace mechanism for counting queries
• Exponential mechanism for categorical data
• Adaptive noise scaling based on sensitivity

Privacy Budget Management:
• ε-δ privacy guarantees
• Composition theorem application
• Budget allocation optimization
• Privacy accounting across rounds

Advanced Features:
• Local differential privacy options
• Privacy-utility optimization
• Moment accountant for tight bounds
• Personalized privacy levels
    """
    
    story.append(Paragraph(dp_text, body_style))
    
    story.append(Paragraph("9.2 Committee-Based Security", subheading_style))
    
    committee_text = """
Multi-party validation ensures system integrity:

Validation Process:
• Random committee selection
• Consensus-based model validation
• Anomaly detection algorithms
• Malicious participant identification

Reputation System:
• Historical performance tracking
• Trust score computation
• Weighted voting mechanisms
• Adaptive committee size

Security Measures:
• Byzantine fault tolerance
• Sybil attack prevention
• Model poisoning detection
• Gradient leakage protection
    """
    
    story.append(Paragraph(committee_text, body_style))
    
    story.append(PageBreak())
    
    # 10. Troubleshooting and Best Practices
    story.append(Paragraph("10. Troubleshooting and Best Practices", heading_style))
    
    story.append(Paragraph("10.1 Common Issues and Solutions", subheading_style))
    
    troubleshooting = [
        ['Issue', 'Possible Cause', 'Solution'],
        ['Training fails to start', 'Missing dependencies', 'Install all required packages'],
        ['Low accuracy results', 'Insufficient training data', 'Increase client data or rounds'],
        ['Slow convergence', 'High privacy noise', 'Adjust epsilon parameters'],
        ['Memory errors', 'Large dataset/many clients', 'Reduce batch size or clients'],
        ['Connection timeouts', 'Network instability', 'Check internet connection'],
        ['Analytics not showing', 'Training not completed', 'Complete training first']
    ]
    
    trouble_table = Table(troubleshooting, colWidths=[2*inch, 2*inch, 2.5*inch])
    trouble_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), blue),
        ('TEXTCOLOR', (0, 0), (-1, 0), black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, black)
    ]))
    
    story.append(trouble_table)
    
    story.append(Paragraph("10.2 Best Practices", subheading_style))
    
    best_practices = """
Recommendations for optimal system performance:

Configuration:
• Start with 5-10 medical facilities for balanced training
• Use 20-50 training rounds for convergence
• Set epsilon between 0.1-2.0 for privacy-utility balance
• Enable early stopping with patience of 5-10 rounds

Data Management:
• Ensure balanced data distribution across clients
• Validate data quality before training
• Monitor for missing or corrupted data
• Use appropriate preprocessing techniques

Performance:
• Monitor system resources during training
• Use appropriate hardware for large-scale experiments
• Consider distributed computing for very large deployments
• Regular backup of training results and models

Security:
• Regularly update privacy parameters
• Monitor for unusual participant behavior
• Validate committee consensus results
• Keep audit logs of all training activities
    """
    
    story.append(Paragraph(best_practices, body_style))
    
    # Footer
    story.append(Spacer(1, 30))
    
    footer_style = ParagraphStyle(
        'Footer',
        parent=styles['Normal'],
        fontSize=8,
        alignment=TA_CENTER
    )
    
    story.append(Paragraph("---", footer_style))
    story.append(Paragraph(f"Documentation generated on {datetime.now().strftime('%B %d, %Y at %H:%M')}", footer_style))
    
    # Build PDF
    doc.build(story)
    
    return filename

if __name__ == "__main__":
    filename = create_comprehensive_system_documentation()
    print(f"Comprehensive documentation generated: {filename}")