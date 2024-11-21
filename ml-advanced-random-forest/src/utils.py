import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def save_model(model, scaler, output_dir='models'):
    """
    Save trained model and scaler
    
    Args:
    - model: Trained machine learning model
    - scaler: Feature scaler
    - output_dir: Directory to save model files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    joblib.dump(model, os.path.join(output_dir, 'random_forest_model.pkl'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(output_dir, 'feature_scaler.pkl'))
    
    print(f"Model and scaler saved in {output_dir}")

def load_saved_model(model_path='models/random_forest_model.pkl', 
                     scaler_path='models/feature_scaler.pkl'):
    """
    Load saved model and scaler
    
    Args:
    - model_path: Path to saved model
    - scaler_path: Path to saved scaler
    
    Returns:
    - Tuple of (model, scaler)
    """
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    except FileNotFoundError:
        print("Model or scaler files not found.")
        return None, None

def plot_predictions_vs_actual(y_true, y_pred, title='Predictions vs Actual'):
    """
    Create a scatter plot of predictions vs actual values
    
    Args:
    - y_true: Actual target values
    - y_pred: Predicted target values
    - title: Plot title
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def generate_model_report(model, X_test, y_test):
    """
    Generate a comprehensive model report
    
    Args:
    - model: Trained model
    - X_test: Test features
    - y_test: Test target values
    
    Returns:
    - DataFrame with model insights
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    y_pred = model.predict(X_test)
    
    report = pd.DataFrame({
        'Metric': [
            'Mean Squared Error', 
            'Root Mean Squared Error', 
            'Mean Absolute Error', 
            'R² Score'
        ],
        'Value': [
            mean_squared_error(y_test, y_pred),
            mean_squared_error(y_test, y_pred, squared=False),
            mean_absolute_error(y_test, y_pred),
            r2_score(y_test, y_pred)
        ]
    })
    
    return report