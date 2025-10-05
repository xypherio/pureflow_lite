import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
from joblib import dump
import os

print("Libraries imported successfully!")

def load_and_explore_data(file_path):
    """Load dataset and perform initial exploration"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def preprocess_data(df):
    """Clean and preprocess the dataset"""
    df_processed = df.copy()
    
    # Convert datetime
    df_processed['datetime'] = pd.to_datetime(df_processed['datetime'])
    
    # Map rain values
    rain_map = {
        'Not Raining': 0, 'Shower': 1, 'Light': 1, 'Light Rain': 1,
        'Moderate': 2, 'Moderate Rain': 2, 'Heavy': 3, 'Heavy Rain': 3
    }
    df_processed['isRaining'] = df_processed['isRaining'].map(rain_map).fillna(0).astype(int)
    
    # Extract time features
    df_processed['month'] = df_processed['datetime'].dt.month
    df_processed['hour'] = df_processed['datetime'].dt.hour
    
    # Philippine season: 0 = Dry (Dec-May), 1 = Rainy (Jun-Nov)
    df_processed['season'] = df_processed['month'].apply(lambda m: 0 if 12 <= m or m <= 5 else 1)
    
    # Cyclical features
    df_processed['month_sin'] = np.sin(2 * np.pi * df_processed['month'] / 12)
    df_processed['month_cos'] = np.cos(2 * np.pi * df_processed['month'] / 12)
    df_processed['hour_sin'] = np.sin(2 * np.pi * df_processed['hour'] / 24)
    df_processed['hour_cos'] = np.cos(2 * np.pi * df_processed['hour'] / 24)
    
    # Lag features (only 1 hour and 24 hour lag)
    water_params = ['pH', 'temperature', 'salinity', 'turbidity']
    for param in water_params:
        df_processed[f'{param}_lag1'] = df_processed[param].shift(1)
        df_processed[f'{param}_lag24'] = df_processed[param].shift(24)
    
    # Rain features
    df_processed['rain_binary'] = (df_processed['isRaining'] > 0).astype(int)
    df_processed['rain_intensity'] = df_processed['isRaining']
    
    # Rolling rain features
    df_processed['rain_last_6h'] = df_processed['rain_binary'].rolling(window=6, min_periods=1).sum()
    df_processed['rain_last_24h'] = df_processed['rain_binary'].rolling(window=24, min_periods=1).sum()
    
    # Remove NaN rows
    df_processed = df_processed.dropna()
    
    print(f"Processed dataset shape: {df_processed.shape}")
    return df_processed

def prepare_model_data(df):
    """Prepare features and targets"""
    feature_cols = [
        'month_sin', 'month_cos', 'hour_sin', 'hour_cos', 'season',
        'rain_binary', 'rain_intensity', 'rain_last_6h', 'rain_last_24h',
        'pH_lag1', 'temperature_lag1', 'salinity_lag1', 'turbidity_lag1',
        'pH_lag24', 'temperature_lag24', 'salinity_lag24', 'turbidity_lag24'
    ]
    
    target_cols = ['pH', 'temperature', 'salinity', 'turbidity']
    
    X = df[feature_cols].copy()
    y = df[target_cols].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Targets shape: {y.shape}")
    
    return X, y, feature_cols, target_cols

def train_lite_model(X, y):
    """Train a lightweight Random Forest model"""
    
    # Split data (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining Lightweight Random Forest Model...")
    print("Model parameters: n_estimators=30, max_depth=10, min_samples_split=20")
    
    # Ultra-lightweight Random Forest for <500MB
    model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=30,          # Reduced trees
            max_depth=10,             # Shallow trees
            min_samples_split=20,     # Higher threshold
            min_samples_leaf=10,      # Higher minimum
            max_features='sqrt',      # Reduced features per split
            random_state=42,
            n_jobs=-1
        )
    )
    
    # Train model (using original features, not scaled)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate
    print("\nModel Performance:")
    results = {}
    for i, target in enumerate(y.columns):
        r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
        mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
        
        results[target] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
        print(f"{target}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")
    
    return model, scaler, results, (X_train, X_test, y_train, y_test)

def save_model_artifacts(model, scaler, feature_cols, target_cols):
    """Save model artifacts with maximum compression"""
    
    # Save with maximum compression
    dump(model, 'pureflow_lite_model.joblib', compress=9)
    dump(scaler, 'feature_scaler.joblib', compress=9)
    
    model_info = {
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'model_type': 'MultiOutputRegressor(RandomForestRegressor)',
        'model_params': {
            'n_estimators': 30,
            'max_depth': 10,
            'min_samples_split': 20
        }
    }
    dump(model_info, 'model_info.joblib', compress=9)
    
    print("\n✓ Model artifacts saved successfully!")
    print("Files created:")
    print("- water_quality_model.joblib")
    print("- feature_scaler.joblib")
    print("- model_info.pkl")
    
    # Check file sizes
    model_size = os.path.getsize('pureflow_lite_model.joblib') / (1024 * 1024)
    scaler_size = os.path.getsize('feature_scaler.joblib') / (1024 * 1024)
    info_size = os.path.getsize('model_info.joblib') / (1024 * 1024)
    total_size = model_size + scaler_size + info_size
    
    print(f"\nFile sizes:")
    print(f"- Model: {model_size:.2f} MB")
    print(f"- Scaler: {scaler_size:.2f} MB")
    print(f"- Info: {info_size:.2f} MB")
    print(f"- Total: {total_size:.2f} MB")
    
    if total_size < 500:
        print(f"\n✓ Total size ({total_size:.2f} MB) is under 500 MB!")
    else:
        print(f"\n⚠ Warning: Total size ({total_size:.2f} MB) exceeds 500 MB")
        print("Consider reducing n_estimators or max_depth further")

# Main execution
if __name__ == "__main__":
    print("="*60)
    print("PUREFLOW AI - LIGHTWEIGHT MODEL TRAINING")
    print("="*60)
    
    # Load data
    df = load_and_explore_data('water_quality_dataset.csv')
    
    if df is not None:
        # Preprocess
        df_processed = preprocess_data(df)
        
        # Prepare data
        X, y, feature_cols, target_cols = prepare_model_data(df_processed)
        
        # Train model
        model, scaler, results, _ = train_lite_model(X, y)
        
        # Save artifacts
        save_model_artifacts(model, scaler, feature_cols, target_cols)
        
        print("\n" + "="*60)
        print("TRAINING COMPLETE!")
        print("="*60)