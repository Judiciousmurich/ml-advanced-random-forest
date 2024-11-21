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

# Encode the target column 'class'
df['class'] = label_encoder.fit_transform(df['class'])

# Step 3: Check for any missing data and handle it
print(df.isnull().sum())  

# Step 4: Handle non-numeric columns

non_numeric_columns = df.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Convert all non-numeric columns to categorical if appropriate
for column in non_numeric_columns:
    df[column] = df[column].astype(str).str.replace(b'<0', '0')  
    df[column] = label_encoder.fit_transform(df[column]) 

# Step 5: Fill missing values with the median for numeric columns only
df = df.apply(pd.to_numeric, errors='coerce')  
df = df.fillna(df.median())  

# Step 6: Split the dataset into features and target
X = df.drop('class', axis=1)  # Features
y = df['class']               # Target

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


