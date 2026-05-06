import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Set up page
st.set_page_config(page_title="Insurance CLV Prediction", page_icon="📈", layout="wide")

@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load('models/preprocessor.joblib')
    model = joblib.load('models/best_model.joblib')
    feature_names = joblib.load('models/feature_names.joblib')
    num_cols, cat_cols = joblib.load('models/feature_types.joblib')
    return preprocessor, model, feature_names, num_cols, cat_cols

@st.cache_data
def get_original_data():
    df = pd.read_csv('Solven_IT_P09166_data.csv')
    if 'Effective To Date' in df.columns:
        # Suppress the parsing warning by letting dateutil handle it silently or specify format
        df['Effective To Date'] = pd.to_datetime(df['Effective To Date'], format='mixed', dayfirst=False)
        df['Effective_Month'] = df['Effective To Date'].dt.month
        df = df.drop('Effective To Date', axis=1)
    if 'Customer' in df.columns:
        df = df.drop('Customer', axis=1)
    
    # Fill missing similar to training
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

st.title("📈 Insurance Customer Lifetime Value Prediction")
st.write("""
Welcome to the Insurance CLV Prediction App. This application predicts the **Customer Lifetime Value (CLV)** 
based on customer demographics and policy details using a trained Machine Learning model.
""")

try:
    preprocessor, model, feature_names, num_cols, cat_cols = load_artifacts()
    df_original = get_original_data()
except Exception as e:
    st.error(f"Error loading models or data. Please ensure 'models/' directory exists and 'Solven_IT_P09166_data.csv' is present. Error: {e}")
    st.stop()

# Sidebar for user input
st.sidebar.header("Customer Details Input")

def user_input_features():
    user_data = {}
    st.sidebar.subheader("Numeric Features")
    for col in num_cols:
        min_val = float(df_original[col].min())
        max_val = float(df_original[col].max())
        mean_val = float(df_original[col].mean())
        user_data[col] = st.sidebar.slider(f"{col}", min_val, max_val, mean_val)
        
    st.sidebar.subheader("Categorical Features")
    for col in cat_cols:
        options = df_original[col].unique().tolist()
        user_data[col] = st.sidebar.selectbox(f"{col}", options)
        
    return pd.DataFrame(user_data, index=[0])

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Single Prediction", "📂 Batch Prediction", "📊 Exploratory Data Analysis"])

with tab1:
    st.subheader("Single Customer Prediction")
    st.write("Adjust the features in the sidebar to simulate a customer profile.")
    
    input_df = user_input_features()
    
    st.markdown("### Selected Customer Profile")
    st.dataframe(input_df)
    
    if st.button("Predict CLV", type="primary"):
        with st.spinner("Processing..."):
            try:
                processed_input = preprocessor.transform(input_df)
                prediction = model.predict(processed_input)
                
                st.success(f"### Predicted Customer Lifetime Value: **${prediction[0]:,.2f}**")
                
                st.markdown("### Feature Contributions (SHAP Explainability)")
                st.write("The plot below shows how each feature pushed the model output from the base value to the predicted CLV.")
                
                # TreeExplainer for Random Forest / XGBoost
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(processed_input)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, processed_input, feature_names=feature_names, plot_type="bar", show=False)
                st.pyplot(fig)
            except Exception as e:
                st.error(f"Prediction failed: {e}")

with tab2:
    st.subheader("Batch Prediction from CSV")
    st.write("Upload a CSV file containing multiple customers to predict their CLV simultaneously.")
    
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(batch_df.head())
        
        # Preprocessing
        process_df = batch_df.copy()
        if 'Customer' in process_df.columns:
            process_df = process_df.drop('Customer', axis=1)
        if 'Effective To Date' in process_df.columns:
            process_df['Effective To Date'] = pd.to_datetime(process_df['Effective To Date'], format='mixed', dayfirst=False)
            process_df['Effective_Month'] = process_df['Effective To Date'].dt.month
            process_df = process_df.drop('Effective To Date', axis=1)
            
        for col in process_df.columns:
            if col in df_original.columns:
                if pd.api.types.is_numeric_dtype(df_original[col]):
                    process_df[col] = process_df[col].fillna(df_original[col].median())
                else:
                    process_df[col] = process_df[col].fillna(df_original[col].mode()[0])
                
        if st.button("Run Batch Prediction"):
            with st.spinner("Predicting..."):
                try:
                    # Filter columns to match training exactly
                    cols_needed = num_cols + cat_cols
                    # Find missing columns and add them with default values, or just let transform fail and show user
                    processed_batch = preprocessor.transform(process_df[cols_needed])
                    predictions = model.predict(processed_batch)
                    batch_df['Predicted_CLV'] = predictions
                    
                    st.success("Batch prediction completed successfully!")
                    
                    # Ensure Customer column exists before selecting
                    disp_cols = ['Customer', 'Predicted_CLV'] if 'Customer' in batch_df.columns else batch_df.columns
                    st.dataframe(batch_df[disp_cols])
                    
                    csv = batch_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name='batch_predictions_clv.csv',
                        mime='text/csv',
                    )
                except Exception as e:
                    st.error(f"Batch prediction error: {e}. Ensure your CSV has all the necessary columns.")

with tab3:
    st.subheader("Exploratory Data Analysis")
    st.write("Below are some key insights discovered during the data preprocessing phase.")
    
    col1, col2 = st.columns(2)
    with col1:
        if os.path.exists("eda_plots/clv_distribution.png"):
            st.image("eda_plots/clv_distribution.png", caption="CLV Distribution", use_container_width=True)
    with col2:
        if os.path.exists("eda_plots/coverage_count.png"):
            st.image("eda_plots/coverage_count.png", caption="Coverage Types Count", use_container_width=True)
        
    if os.path.exists("eda_plots/correlation_heatmap.png"):
        st.image("eda_plots/correlation_heatmap.png", caption="Feature Correlation Heatmap", use_container_width=True)
