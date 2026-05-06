import numpy as np
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

def evaluate_model(model_name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {'Model': model_name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

if __name__ == "__main__":
    print("Loading preprocessed data...")
    try:
        X_train = np.load('models/X_train.npy')
        X_test = np.load('models/X_test.npy')
        y_train = np.load('models/y_train.npy')
        y_test = np.load('models/y_test.npy')
    except FileNotFoundError:
        print("Preprocessed data not found. Please run data_preprocessing.py first.")
        exit(1)
        
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42, n_estimators=100)
    }
    
    results = []
    best_model = None
    best_r2 = -float('inf')
    
    print("Training models...")
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        metrics = evaluate_model(name, y_test, y_pred)
        results.append(metrics)
        
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = model
            best_model_name = name
            
    results_df = pd.DataFrame(results)
    print("\n--- Model Comparison ---")
    print(results_df.to_string(index=False))
    
    print(f"\nBest Model: {best_model_name} with R2: {best_r2:.4f}")
    
    # Save the best model
    joblib.dump(best_model, 'models/best_model.joblib')
    print("Best model saved to 'models/best_model.joblib'.")
