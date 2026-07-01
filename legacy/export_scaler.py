import joblib
import json
import numpy as np
import os

# Path to your current scaler (as seen in your screenshot)
scaler_path = "dataset/val/scaler.joblib"
# New "Safe" file we will create
output_path = "dataset/val/scaler_params.json"

print(f"[INFO] Loading scaler from {scaler_path}...")

if not os.path.exists(scaler_path):
    print(f"[ERROR] Could not find {scaler_path}. Are you in the project root?")
    exit(1)

try:
    # Load the scaler object
    scaler = joblib.load(scaler_path)
    
    # Extract the raw math values needed for normalization
    # Formula: X_scaled = X * scale_ + min_
    params = {
        "scale_": scaler.scale_.tolist(),
        "min_": scaler.min_.tolist()
    }
    
    # Save to a universal JSON format
    with open(output_path, 'w') as f:
        json.dump(params, f)
        
    print(f"✅ Success! Converted scaler to JSON: {output_path}")
    print("   Now the Docker container can read it without version errors.")

except Exception as e:
    print(f"[ERROR] Conversion failed: {e}")