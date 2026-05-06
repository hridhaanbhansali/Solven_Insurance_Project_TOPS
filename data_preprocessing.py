import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    print("--- Basic Info ---")
    print(df.info())
    print("\n--- Shape ---")
    print(df.shape)
    
    # Drop Customer ID
    if 'Customer' in df.columns:
        df = df.drop('Customer', axis=1)
        
    # Handle Date
    if 'Effective To Date' in df.columns:
        df['Effective To Date'] = pd.to_datetime(df['Effective To Date'])
        df['Effective_Month'] = df['Effective To Date'].dt.month
        df = df.drop('Effective To Date', axis=1)

    # Fill missing values if any
    # forward fill or simply fill with mode/median
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df

def perform_eda(df, output_dir='eda_plots'):
    os.makedirs(output_dir, exist_ok=True)
    
    # Correlation Heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    
    # Distribution of Target
    plt.figure(figsize=(8, 6))
    sns.histplot(df['Customer Lifetime Value'], bins=50, kde=True)
    plt.title('Distribution of Customer Lifetime Value')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'clv_distribution.png'))
    plt.close()
    
    # Example categorical plot
    plt.figure(figsize=(8, 6))
    sns.countplot(y='Coverage', data=df)
    plt.title('Count of Coverage Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_count.png'))
    plt.close()

def preprocess_features(df, target_col='Customer Lifetime Value'):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Use sparse=False or sparse_output=False based on sklearn version. Using sparse_output for latest versions.
    try:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
    except TypeError:
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))
        ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return X, y, preprocessor, numeric_features, categorical_features

if __name__ == "__main__":
    filepath = 'Solven_IT_P09166_data.csv'
    print(f"Loading data from {filepath}...")
    df = load_and_clean_data(filepath)
    
    print("Performing EDA and saving plots...")
    perform_eda(df)
    
    print("Preprocessing features...")
    X, y, preprocessor, num_cols, cat_cols = preprocess_features(df)
    
    # Fit the preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Save processed arrays and preprocessor
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.joblib')
    
    # Save the split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    np.save('models/X_train.npy', X_train)
    np.save('models/X_test.npy', X_test)
    np.save('models/y_train.npy', y_train)
    np.save('models/y_test.npy', y_test)
    
    # Save feature names for SHAP
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    try:
        cat_feature_names = cat_encoder.get_feature_names_out(cat_cols)
    except AttributeError:
        cat_feature_names = cat_encoder.get_feature_names(cat_cols)
        
    all_feature_names = num_cols + list(cat_feature_names)
    joblib.dump(all_feature_names, 'models/feature_names.joblib')
    joblib.dump((num_cols, cat_cols), 'models/feature_types.joblib')
    
    print("Preprocessing completed and artifacts saved to 'models/' directory.")
