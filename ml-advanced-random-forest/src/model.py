import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split, 
    cross_val_score, 
    GridSearchCV
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score
)
from sklearn.inspection import permutation_importance

class RandomForestModelBuilder:
    def __init__(self, data, target_column):
        """
        Initialize the Random Forest Model Builder
        
        Parameters:
        - data: pandas DataFrame with features and target
        - target_column: name of the target variable column
        """
        self.data = data
        self.target_column = target_column
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None 
        self.y_test = None
        self.model = None
        self.scaler = None
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """
        Prepare data by splitting into features and target, 
        then into train and test sets
        """
        # Separate features and target
        self.X = self.data.drop(self.target_column, axis=1)
        self.y = self.data[self.target_column]
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=random_state
        )
        
        # Scale features
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        return self
    
    def train_model(self, n_estimators=100, max_depth=None, min_samples_split=2):
        """
        Train Random Forest Model with configurable hyperparameters
        """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42,
            n_jobs=-1  # Use all available cores
        )
        
        self.model.fit(self.X_train_scaled, self.y_train)
        return self
    
    def hyperparameter_tuning(self):
        """
        Perform Grid Search for hyperparameter tuning
        """
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }
        
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=42),
            param_grid=param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        grid_search.fit(self.X_train_scaled, self.y_train)
        
        print("Best Parameters:", grid_search.best_params_)
        self.model = grid_search.best_estimator_
        return self
    
    def evaluate_model(self):
        """
        Evaluate model performance
        """
        # Predictions
        y_pred = self.model.predict(self.X_test_scaled)
        
        # Metrics
        mse = mean_squared_error(self.y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        
        print("\n📊 Model Performance Metrics:")
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"Root Mean Squared Error: {rmse:.2f}")
        print(f"Mean Absolute Error: {mae:.2f}")
        print(f"R² Score: {r2:.2f}")
        
        # Cross-validation
        cv_scores = cross_val_score(
            self.model, 
            self.X_train_scaled, 
            self.y_train, 
            cv=5, 
            scoring='neg_mean_squared_error'
        )
        print(f"\n🔄 Cross-Validation Scores: {-cv_scores}")
        print(f"Average CV Score: {-cv_scores.mean():.2f}")
        
        return self
    
    def feature_importance(self):
        """
        Analyze and visualize feature importance
        """
        # Permutation importance for more robust feature importance
        perm_importance = permutation_importance(
            self.model, 
            self.X_test_scaled, 
            self.y_test, 
            n_repeats=10
        )
        
        feature_importance = pd.DataFrame({
            'feature': self.X.columns,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance in Random Forest Model')
        plt.tight_layout()
        plt.show()
        
        return feature_importance
    
    def predict(self, new_data):
        """
        Make predictions on new data
        """
        # Scale new data
        new_data_scaled = self.scaler.transform(new_data)
        predictions = self.model.predict(new_data_scaled)
        return predictions