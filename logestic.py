# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Merlot Wine Quality Predictor",
    page_icon="🍷",
    layout="wide"
)

# Title and description
st.title("🍷 Merlot Wine Quality Prediction using Logistic Regression")
st.markdown("""
This app predicts whether a Merlot wine is of high quality (>=7) or low quality (<7) 
based on its physicochemical properties using Logistic Regression.
""")

# Function to load and prepare data
@st.cache_data
def load_data():
    # Creating synthetic Merlot wine dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic wine data
    data = {
        'fixed_acidity': np.random.uniform(4, 16, n_samples),
        'volatile_acidity': np.random.uniform(0.1, 1.5, n_samples),
        'citric_acid': np.random.uniform(0, 1, n_samples),
        'residual_sugar': np.random.uniform(0.5, 15, n_samples),
        'chlorides': np.random.uniform(0.01, 0.6, n_samples),
        'free_sulfur_dioxide': np.random.uniform(2, 70, n_samples),
        'total_sulfur_dioxide': np.random.uniform(10, 200, n_samples),
        'density': np.random.uniform(0.987, 1.005, n_samples),
        'pH': np.random.uniform(2.8, 4, n_samples),
        'sulphates': np.random.uniform(0.2, 2, n_samples),
        'alcohol': np.random.uniform(8, 15, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Generate quality scores based on features
    quality_score = (
        df['alcohol'] * 0.3 +
        (10 - df['volatile_acidity']) * 0.2 +
        df['citric_acid'] * 0.2 +
        df['sulphates'] * 0.15 +
        (4 - abs(df['pH'] - 3.3)) * 0.15
    )
    
    # Add some randomness
    quality_score += np.random.normal(0, 1, n_samples)
    
    # Normalize to 3-9 range
    quality_score = 3 + (quality_score - quality_score.min()) * 6 / (quality_score.max() - quality_score.min())
    df['quality'] = quality_score.round().astype(int)
    
    # Create binary target (1 for quality >=7, 0 for quality <7)
    df['high_quality'] = (df['quality'] >= 7).astype(int)
    
    return df

# Load data
with st.spinner("Loading wine data..."):
    df = load_data()

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Data Exploration", "Model Training", "Make Predictions"])

# Data Exploration Page
if page == "Data Exploration":
    st.header("📊 Data Exploration")
    
    # Show raw data
    with st.expander("View Raw Data"):
        st.dataframe(df.head(10))
        st.write(f"Dataset shape: {df.shape}")
    
    # Data statistics
    st.subheader("Data Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", df.shape[0])
    with col2:
        st.metric("Features", df.shape[1] - 2)  # Excluding quality and high_quality
    with col3:
        high_quality_pct = df['high_quality'].mean() * 100
        st.metric("High Quality Wines", f"{high_quality_pct:.1f}%")
    
    # Distribution of quality
    st.subheader("Quality Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Quality scores distribution
    df['quality'].hist(ax=axes[0], bins=15, edgecolor='black', color='skyblue')
    axes[0].set_xlabel('Quality Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Wine Quality Scores')
    
    # High quality vs Low quality
    df['high_quality'].value_counts().plot(kind='bar', ax=axes[1], color=['red', 'green'])
    axes[1].set_xticklabels(['Low Quality (<7)', 'High Quality (≥7)'])
    axes[1].set_title('High Quality vs Low Quality Wines')
    axes[1].set_ylabel('Count')
    
    st.pyplot(fig)
    
    # Feature correlations
    st.subheader("Feature Correlations with Quality")
    correlations = df.drop('quality', axis=1).corrwith(df['quality']).sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    correlations[1:].plot(kind='barh', ax=ax, color='coral')
    ax.set_xlabel('Correlation with Quality')
    ax.set_title('Feature Importance for Wine Quality')
    st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.header("🧠 Model Training")
    
    # Feature selection
    feature_columns = [col for col in df.columns if col not in ['quality', 'high_quality']]
    
    st.subheader("Select Features for Training")
    selected_features = st.multiselect(
        "Choose features to include in the model:",
        options=feature_columns,
        default=feature_columns[:5]  # Default to first 5 features
    )
    
    if len(selected_features) == 0:
        st.warning("Please select at least one feature!")
    else:
        # Prepare data
        X = df[selected_features]
        y = df['high_quality']
        
        # Split data
        test_size = st.slider("Test set size (%)", 10, 40, 20) / 100
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Model parameters
        st.subheader("Model Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0, 0.1)
        with col2:
            max_iter = st.slider("Maximum iterations", 100, 1000, 500, 50)
        
        # Train model
        if st.button("🚀 Train Model"):
            with st.spinner("Training logistic regression model..."):
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=42)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                
                # Display results
                st.success(f"Model trained successfully! Accuracy: {accuracy:.4f}")
                
                # Confusion Matrix
                st.subheader("Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                ax.set_title('Confusion Matrix')
                ax.set_xticklabels(['Low Quality', 'High Quality'])
                ax.set_yticklabels(['Low Quality', 'High Quality'])
                st.pyplot(fig)
                
                # Classification Report
                st.subheader("Classification Report")
                report = classification_report(y_test, y_pred, target_names=['Low Quality', 'High Quality'], output_dict=True)
                report_df = pd.DataFrame(report).transpose()
                st.dataframe(report_df)
                
                # Feature Importance
                st.subheader("Feature Importance (Coefficients)")
                importance_df = pd.DataFrame({
                    'Feature': selected_features,
                    'Coefficient': model.coef_[0],
                    'Abs_Coefficient': np.abs(model.coef_[0])
                }).sort_values('Abs_Coefficient', ascending=False)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['red' if x < 0 else 'green' for x in importance_df['Coefficient']]
                ax.barh(importance_df['Feature'], importance_df['Coefficient'], color=colors)
                ax.set_xlabel('Coefficient Value')
                ax.set_title('Feature Coefficients (Positive = Increases Quality)')
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                st.pyplot(fig)
                
                # Store model in session state
                st.session_state['model'] = model
                st.session_state['scaler'] = scaler
                st.session_state['selected_features'] = selected_features
                st.session_state['model_trained'] = True

# Make Predictions Page
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    if 'model_trained' not in st.session_state:
        st.warning("Please train a model first in the 'Model Training' page!")
    else:
        st.subheader("Enter Wine Characteristics")
        
        # Create input fields for features
        col1, col2 = st.columns(2)
        
        with col1:
            fixed_acidity = st.number_input("Fixed Acidity", min_value=4.0, max_value=16.0, value=8.5, step=0.1)
            volatile_acidity = st.number_input("Volatile Acidity", min_value=0.1, max_value=1.5, value=0.5, step=0.1)
            citric_acid = st.number_input("Citric Acid", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
            residual_sugar = st.number_input("Residual Sugar", min_value=0.5, max_value=15.0, value=5.0, step=0.5)
            chlorides = st.number_input("Chlorides", min_value=0.01, max_value=0.6, value=0.08, step=0.01)
            free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", min_value=2.0, max_value=70.0, value=20.0, step=1.0)
        
        with col2:
            total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", min_value=10.0, max_value=200.0, value=50.0, step=5.0)
            density = st.number_input("Density", min_value=0.987, max_value=1.005, value=0.996, step=0.001, format="%.3f")
            pH = st.number_input("pH", min_value=2.8, max_value=4.0, value=3.3, step=0.1)
            sulphates = st.number_input("Sulphates", min_value=0.2, max_value=2.0, value=0.6, step=0.1)
            alcohol = st.number_input("Alcohol", min_value=8.0, max_value=15.0, value=10.5, step=0.5)
        
        if st.button("Predict Wine Quality"):
            # Create input dataframe
            input_data = {
                'fixed_acidity': fixed_acidity,
                'volatile_acidity': volatile_acidity,
                'citric_acid': citric_acid,
                'residual_sugar': residual_sugar,
                'chlorides': chlorides,
                'free_sulfur_dioxide': free_sulfur_dioxide,
                'total_sulfur_dioxide': total_sulfur_dioxide,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Select only features used in training
            input_selected = input_df[st.session_state['selected_features']]
            
            # Scale input
            input_scaled = st.session_state['scaler'].transform(input_selected)
            
            # Make prediction
            prediction = st.session_state['model'].predict(input_scaled)[0]
            probability = st.session_state['model'].predict_proba(input_scaled)[0]
            
            # Display result
            st.subheader("Prediction Result")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("🍷 HIGH QUALITY WINE (≥7)")
                else:
                    st.error("🥃 LOW QUALITY WINE (<7)")
            
            with col2:
                st.metric("Confidence", f"{probability[prediction]*100:.1f}%")
            
            with col3:
                st.metric("Quality Probability", f"{probability[1]*100:.1f}%")
            
            # Show probability gauge
            fig, ax = plt.subplots(figsize=(8, 2))
            ax.barh(['Probability'], [probability[1]], color='green', alpha=0.6, label='High Quality')
            ax.barh(['Probability'], [probability[0]], left=[probability[1]], color='red', alpha=0.6, label='Low Quality')
            ax.set_xlim(0, 1)
            ax.set_xlabel('Probability')
            ax.legend(loc='upper right')
            st.pyplot(fig)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app demonstrates Logistic Regression for wine quality prediction. "
    "The dataset is synthetic but based on real wine characteristics."
)