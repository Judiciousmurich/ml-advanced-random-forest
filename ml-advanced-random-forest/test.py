from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Load the ARFF file
data, meta = arff.loadarff('C:/Users/ADMIN/Downloads/dataset_31_credit-g.arff')

# Convert to pandas DataFrame
df = pd.DataFrame(data)

# Display the first few rows of the data
df.head()

# Step 1: Check the data types of the columns
print(df.dtypes)

# Step 2: Handle any categorical columns:
label_encoder = LabelEncoder()

df['class'] = label_encoder.fit_transform(df['class'])

# Step 3: Check for any missing data and handle it
print(df.isnull().sum())  

# Step 4: Handle non-numeric columns

non_numeric_columns = df.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)
for column in non_numeric_columns:
    df[column] = df[column].astype(str).str.replace(b'<0', '0')  
    df[column] = label_encoder.fit_transform(df[column]) 

# Step 5: Fill missing values with the median for numeric columns only
df = df.apply(pd.to_numeric, errors='coerce')  
df = df.fillna(df.median())  

# Step 6: Split the dataset into features and target
X = df.drop('class', axis=1) 
y = df['class']              

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# TRAIN THE MODEL
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')

# EVALUATE THE MODEL
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# MODEL TUNING
from sklearn.model_selection import GridSearchCV
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print(f"Best Parameters: {grid_search.best_params_}")

