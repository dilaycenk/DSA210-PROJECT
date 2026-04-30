import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, r2_score
import os
import matplotlib.pyplot as plt

# Set up file paths
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'trained_model.pkl')
data_path = os.path.join(current_dir, '..', 'Data', 'forecast_sample.csv')
figures_dir = os.path.join(current_dir, 'figures')

# Create figures directory if it doesn't exist
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)

# 1. Load the Trained Model and Dataset
model = joblib.load(model_path)
df = pd.read_csv(data_path)
df['Date'] = pd.to_datetime(df['Date'])

# 2. Recreate the same features used during training
df['Month'] = df['Date'].dt.month
df['Prev_Actual'] = df['Actual_Temp'].shift(1)
df = df.dropna()

X = df[['Month', 'Prev_Actual']]
y_actual = df['Actual_Temp']

# 3. Generate Predictions
y_pred = model.predict(X)

# 4. Calculate and Print Performance Metrics
mae = mean_absolute_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

print(f"--- Model Evaluation Results ---")
print(f"Mean Absolute Error (MAE): {mae:.2f} °C")
print(f"R-squared (R2) Score: {r2:.2f}")

# 5. Export Results to CSV
results = pd.DataFrame({
    'Date': df['Date'], 
    'Actual_Temp': y_actual, 
    'Predicted_Temp': y_pred
})
results.to_csv(os.path.join(current_dir, 'predictions_vs_actual.csv'), index=False)

# --- VISUALIZATION SECTION ---

# Figure 1: Feature Importance Plot
plt.figure(figsize=(10, 6))
features = ['Month (Seasonality)', 'Previous Day Temp']
importances = model.feature_importances_
plt.barh(features, importances, color='teal', edgecolor='black')
plt.title('Feature Importance: What drives the model?')
plt.xlabel('Importance Score')
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'feature_importance.png'))
plt.close() # Close to free up memory

# Figure 2: Actual vs Predicted Plot
plt.figure(figsize=(12, 6))
plt.plot(df['Date'], y_actual.values, label='Actual Temperature', color='blue', alpha=0.6)
plt.plot(df['Date'], y_pred, label='Model Prediction', color='red', linestyle='--', alpha=0.8)
plt.title('Performance: Actual vs Predicted Temperatures')
plt.xlabel('Date')
plt.ylabel('Temperature (°C)')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.tight_layout()
plt.savefig(os.path.join(figures_dir, 'actual_vs_predicted.png'))
plt.close()

print(f"All figures have been saved to: {figures_dir}")