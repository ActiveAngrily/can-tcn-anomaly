import torch
import numpy as np
import rospy
import json
import os
from std_msgs.msg import String 

# Import local class
from live_preprocessor import LivePreprocessor

# --- CONFIGURATION ---
THRESHOLD = 0.3101 
# UPDATED PATH: Points to 'models/' instead of 'ML model/'
MODEL_FILE = "models/anomaly_detector_optimized.pt" 
ROS_TOPIC = "/carla/ego_vehicle/can_data" 

# --- SYSTEM STARTUP ---
print("==================================================")
print("   VEHICULAR ANOMALY DETECTION DASHBOARD")
print("==================================================")

# A. Load Model
device = torch.device("cpu")
print(f"[STATUS] Attempting to load model from: {MODEL_FILE}")

if not os.path.exists(MODEL_FILE):
    print(f"[FATAL] Model file not found at {MODEL_FILE}")
    print("       Please ensure 'convert_coreml.py' has been run to generate the optimized model.")
    exit(1)

try:
    model = torch.jit.load(MODEL_FILE, map_location=device)
    model.eval()
    print(f"[STATUS] Model loaded successfully on device: {device}")
except Exception as e:
    print(f"[FATAL] Failed to load the TorchScript model. Exception: {e}")
    exit(1)

# B. Initialize Preprocessor
print("[STATUS] Initializing Data Preprocessor...")
try:
    # This uses the updated defaults in live_preprocessor.py
    processor = LivePreprocessor()
    print("[STATUS] Preprocessor initialized and connected.")
except Exception as e:
    print(f"[FATAL] Could not initialize preprocessor: {e}")
    print("       Check that your dataset files (scaler.joblib, final_columns.txt) are in the 'dataset/' folder.")
    exit(1)

# --- CORE LOGIC ---
def data_callback(msg):
    """
    Runs every time CARLA sends a message via ROS.
    """
    try:
        # 1. Convert ROS String to Python Dictionary
        raw_dict = json.loads(msg.data)
        
        # 2. Clean and Scale Data
        input_tensor = processor.process_new_data(raw_dict)

        # 3. Predict & Check (Only runs if buffer has 50 items)
        if input_tensor is not None:
            with torch.no_grad():
                # Run Inference
                prediction = model(input_tensor)

                # Compare Prediction vs Reality (Last value in the window)
                actual_data = input_tensor[:, -1, :] 
                
                # Calculate Reconstruction Error (MSE)
                loss = torch.nn.functional.mse_loss(prediction, actual_data)
                error_score = loss.item()

                # 4. Decision Logic
                if error_score > THRESHOLD:
                    # Red text for anomaly
                    print(f"\033[91m[ANOMALY DETECTED] Score: {error_score:.5f} (Threshold: {THRESHOLD})\033[0m")
                else:
                    # Green text for normal
                    print(f"\033[92m[NORMAL] Status OK. Score: {error_score:.5f}\033[0m")
                    
    except json.JSONDecodeError:
        print("[WARN] Received malformed JSON data from ROS bridge.")
    except Exception as e:
        print(f"[WARN] Runtime error in data loop: {e}")

# --- EXECUTION LOOP ---
if __name__ == '__main__':
    try:
        rospy.init_node('anomaly_dashboard', anonymous=True)
        rospy.Subscriber(ROS_TOPIC, String, data_callback)
        
        print(f"[INFO] Dashboard is live and listening to: {ROS_TOPIC}")
        print(f"[INFO] Anomaly Threshold set to: {THRESHOLD}")
        print("--------------------------------------------------")
        rospy.spin()
    except rospy.ROSInterruptException:
        print("[INFO] Dashboard stopped by user.")