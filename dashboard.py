import torch
import numpy as np
import rospy
import json
import os
import sys
from std_msgs.msg import String 

# Import local class
from live_preprocessor import LivePreprocessor

# --- CONFIGURATION ---
THRESHOLD = 0.3101 
MODEL_FILE = "models/anomaly_detector_optimized.pt" 
ROS_TOPIC = "/carla/ego_vehicle/can_data" 

# --- VISUALIZATION HELPERS ---
def get_risk_bar(current_score, threshold, width=30):
    ratio = current_score / threshold
    percent = ratio * 100
    
    if percent < 70:
        color = "\033[92m" # Green
        status = "NORMAL  "
        icon = "✅"
    elif percent < 100:
        color = "\033[93m" # Yellow
        status = "WARNING "
        icon = "⚠️ "
    else:
        color = "\033[91m" # Red
        status = "ANOMALY "
        icon = "🚨"

    fill_count = int(min(ratio, 1.0) * width)
    bar = "█" * fill_count + "-" * (width - fill_count)
    
    return f"{color}[{status}] {icon} |{bar}| {percent:6.1f}% Risk (Err: {current_score:.4f})\033[0m"

def get_buffer_bar(current, total, width=30):
    ratio = current / total
    fill_count = int(ratio * width)
    bar = "█" * fill_count + "-" * (width - fill_count)
    return f"\033[96m[BUFFERING] ⏳ |{bar}| {current}/{total} Samples\033[0m"

# --- SYSTEM STARTUP ---
print("==================================================")
print("   VEHICULAR ANOMALY DETECTION DASHBOARD (FIXED)")
print("==================================================")

# Load Model & Preprocessor (Same as before)
device = torch.device("cpu")
if not os.path.exists(MODEL_FILE):
    print(f"[FATAL] Model file not found at {MODEL_FILE}")
    sys.exit(1)

try:
    model = torch.jit.load(MODEL_FILE, map_location=device)
    model.eval()
    print(f"[STATUS] Model loaded successfully.")
except Exception as e:
    print(f"[FATAL] Failed to load model: {e}")
    sys.exit(1)

try:
    if os.path.exists("dataset/val/scaler_params.json"):
         processor = LivePreprocessor(scaler_path="dataset/val/scaler_params.json", columns_path="dataset/train/final_columns.txt")
    else:
         processor = LivePreprocessor(scaler_path="dataset/val/scaler.joblib", columns_path="dataset/train/final_columns.txt")
    print("[STATUS] Preprocessor connected.")
except Exception as e:
    print(f"[FATAL] Preprocessor Error: {e}")
    sys.exit(1)

# --- CORE LOGIC ---
def data_callback(msg):
    try:
        raw_dict = json.loads(msg.data)
        input_tensor = processor.process_new_data(raw_dict)

        # BUFFERING PHASE
        if input_tensor is None:
            curr = len(processor.buffer)
            req = processor.seq_length
            # FIX: Added flush=True to force the blue bar to appear
            print(get_buffer_bar(curr, req), end='\r', flush=True)
            
        # INFERENCE PHASE
        else:
            with torch.no_grad():
                prediction = model(input_tensor)
                actual_data = input_tensor[:, -1, :] 
                loss = torch.nn.functional.mse_loss(prediction, actual_data)
                error_score = loss.item()

                # FIX: Added flush=True here as well
                # Added spaces at the end to clear any leftover characters
                print(get_risk_bar(error_score, THRESHOLD) + "          ", end='\r', flush=True)
                    
    except Exception as e:
        print(f"\n[WARN] Processing Error: {e}")

if __name__ == '__main__':
    try:
        rospy.init_node('anomaly_dashboard', anonymous=True)
        rospy.Subscriber(ROS_TOPIC, String, data_callback)
        
        print(f"[INFO] Monitoring Stream: {ROS_TOPIC}")
        print("-" * 60)
        
        # Check if we are actually receiving data
        print("[DEBUG] Waiting for first message from car...", flush=True)
        
        rospy.spin()
    except rospy.ROSInterruptException:
        print("\n[INFO] Shutting down.")