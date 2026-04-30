import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# Set up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'Data', 'forecast_sample.csv')

# 1. Load the Dataset
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Feature Engineering
# Extract month to capture seasonality
df['Month'] = df['Date'].dt.month
# Use the previous day's temperature as a 'Lag' feature to help the model learn patterns
df['Prev_Actual'] = df['Actual_Temp'].shift(1)
# Drop rows with NaN values created by the shift operation
df = df.dropna()

# Define features (X) and target variable (y)
X = df[['Month', 'Prev_Actual']] 
y = df['Actual_Temp']

# 3. Split the Data
# Using 80% for training and 20% for testing to evaluate performance later
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize and Train the Model
# RandomForest is used for its robustness in handling non-linear climate data
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Save the Trained Model
# Serializing the model to a .pkl file for use in evaluation or deployment
joblib.dump(model, os.path.join(current_dir, 'trained_model.pkl'))
print("Model trained and saved successfully!")