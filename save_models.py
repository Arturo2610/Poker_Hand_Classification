import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add src to path
sys.path.append('.')
from src.feature_engineering import PokerFeatureEngine

# Create feature engine
engine = PokerFeatureEngine()

# Load raw data
print("Loading data...")
train_df = pd.read_csv('data/poker-hand-training.csv')
test_df = pd.read_csv('data/poker-hand-testing.csv')

# Rename columns
train_df.columns = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'hand']
test_df.columns = ['S1', 'R1', 'S2', 'R2', 'S3', 'R3', 'S4', 'R4', 'S5', 'R5', 'hand']

# Generate engineered features
print("Generating features...")
train_features = engine.transform(train_df)
train_features['hand'] = train_df['hand'].values

test_features = engine.transform(test_df)
test_features['hand'] = test_df['hand'].values

# Save engineered features
print("Saving engineered features...")
os.makedirs('data', exist_ok=True)
train_features.to_csv('data/train_engineered.csv', index=False)
test_features.to_csv('data/test_engineered.csv', index=False)

# Prepare data for training
X_train_eng = train_features.drop('hand', axis=1).values
y_train = train_features['hand'].values

X_test_eng = test_features.drop('hand', axis=1).values
y_test = test_features['hand'].values

print(f"\nFeature shape: {X_train_eng.shape}")
print(f"Features: {list(train_features.drop('hand', axis=1).columns)}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_eng)
X_test_scaled = scaler.transform(X_test_eng)

# Train model
print("Training model...")
model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced',
    multi_class='multinomial',
    verbose=1
)
model.fit(X_train_scaled, y_train)

# Evaluate
print("\nEvaluating...")
from sklearn.metrics import f1_score, balanced_accuracy_score

y_pred = model.predict(X_test_scaled)
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_test, y_pred)

print(f"F1 (weighted): {f1_weighted:.4f}")
print(f"F1 (macro): {f1_macro:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")

# Save models
print("\nSaving models...")
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

print("Models saved successfully!")
print(f"Model file exists: {os.path.exists('models/best_model.pkl')}")
print(f"Scaler file exists: {os.path.exists('models/scaler.pkl')}")
print(f"Model expects {X_train_eng.shape[1]} features")