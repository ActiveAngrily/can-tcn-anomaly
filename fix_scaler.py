import joblib
import os

# Define the path to the file seen in your screenshot
scaler_path = "dataset/val/scaler.joblib"

print(f"[INFO] Attempting to downgrade pickle protocol for: {scaler_path}")

if os.path.exists(scaler_path):
    try:
        # 1. Load the scaler using your Mac's current Python
        scaler = joblib.load(scaler_path)
        
        # 2. Save it back using Protocol 3 (Compatible with Python 3.6 in Docker)
        joblib.dump(scaler, scaler_path, protocol=3)
        
        print(f"[SUCCESS] Fixed '{scaler_path}'! It is now compatible with Docker.")
    except Exception as e:
        print(f"[ERROR] Could not process file: {e}")
else:
    print(f"[ERROR] File not found at {scaler_path}. Check your folders.")