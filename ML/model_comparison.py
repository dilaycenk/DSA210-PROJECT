import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Set up file paths to match your existing project structure
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, '..', 'Data', 'forecast_sample.csv')
figures_dir = os.path.join(current_dir, 'figures') 

def run_model_comparison():
    # 1. Load and Preprocess Data
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df['Month'] = df['Date'].dt.month
    df['Prev_Actual'] = df['Actual_Temp'].shift(1)
    df = df.dropna()

    X = df[['Month', 'Prev_Actual']]
    y = df['Actual_Temp']

    # Split data (80% Train, 20% Test)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 2. Define models to compare
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(max_depth=5, random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
    }

    results = []

    # 3. Train and evaluate each model
    print("Starting model comparison...")
    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        results.append({'Model': name, 'MAE': mae, 'R2': r2})
        print(f"{name} evaluation completed.")

    # 4. Detailed Visualization: R2 and MAE Comparison
    comparison_df = pd.DataFrame(results)
    sns.set_style("whitegrid")
    
    # Create two plots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Plot R2 Score (Accuracy measure)
    sns.barplot(x='Model', y='R2', data=comparison_df, palette='magma', ax=ax1)
    ax1.set_title('Model Performance Comparison (R2 Score)')
    ax1.set_ylim(0, 1.1)
    
    # Add labels to R2 bars
    for p in ax1.patches:
        ax1.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    # Plot MAE (Actual Error measure - Lower is better)
    sns.barplot(x='Model', y='MAE', data=comparison_df, palette='viridis', ax=ax2)
    ax2.set_title('Model Error Comparison (MAE)')
    ax2.set_ylabel('Error Value (Lower is better)')
    
    # Add labels to MAE bars
    for p in ax2.patches:
        ax2.annotate(format(p.get_height(), '.2f'), 
                     (p.get_x() + p.get_width() / 2., p.get_height()), 
                     ha='center', va='center', xytext=(0, 9), textcoords='offset points')

    plt.tight_layout()
    
    # Save the detailed comparison chart
    save_path = os.path.join(figures_dir, 'model_comparison_detailed.png')
    plt.savefig(save_path)
    print(f"Detailed comparison plot saved to: {save_path}")
    plt.show()

if __name__ == "__main__":
    run_model_comparison()