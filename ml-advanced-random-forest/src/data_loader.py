import pandas as pd
import numpy as np

def generate_synthetic_data(n_samples=1000):
    """
    Generate synthetic dataset for modeling
    
    Returns:
    - pandas DataFrame with features and target
    """
    np.random.seed(42)
    
    # Simulate complex relationships
    feature1 = np.random.normal(0, 1, n_samples)
    feature2 = np.random.exponential(2, n_samples)
    feature3 = np.random.poisson(3, n_samples)
    feature4 = np.random.uniform(0, 10, n_samples)
    
    # Create target with non-linear relationship
    target = (
        2 * feature1**2 + 
        0.5 * feature2 + 
        np.log(feature3 + 1) + 
        0.1 * feature4**1.5 + 
        np.random.normal(0, 2, n_samples)
    )
    
    data = pd.DataFrame({
        'feature1': feature1,
        'feature2': feature2,
        'feature3': feature3,
        'feature4': feature4,
        'target': target
    })
    
    return data

def load_data(data_path=None):
    """
    Load data from path or generate synthetic data
    
    Args:
    - data_path (str, optional): Path to CSV file
    
    Returns:
    - pandas DataFrame
    """
    if data_path:
        try:
            return pd.read_csv(data_path)
        except FileNotFoundError:
            print(f"File not found: {data_path}. Generating synthetic data.")
    
    return generate_synthetic_data()