import torch
import numpy as np
import rospy
import json
import os
import sys
from std_msgs.msg import String 
from live_preprocessor import LivePreprocessor

MODEL_FILE = "models/anomaly_detector_optimized.pt" 
ROS_TOPIC = "/carla/ego_vehicle/can_data" 

# Initialize
device = torch.device("cpu")
model = torch.jit.load(MODEL_FILE, map_location=device)
model.eval()

# Helper to inspect tensor values
def inspect_tensor(tensor, name):
    data = tensor.detach().numpy().flatten()
    print(f"   [{name}] Min: {data.min():.4f} | Max: {data.max():.4f} | Mean: {data.mean():.4f}")
    if data.max() > 1000 or data.min() < -1000:
        print(f"   ⚠️  WARNING: {name} VALUES ARE HUGE! SCALER FAILED.")

try:
    if os.path.exists("dataset/val/scaler_params.json"):
         print("[INFO] Loading JSON Scaler...")
         processor = LivePreprocessor(scaler_path="dataset/val/scaler_params.json", columns_path="dataset/train/final_columns.txt")
    else:
         print("[INFO] Loading Joblib Scaler...")
         processor = LivePreprocessor(scaler_path="dataset/val/scaler.joblib", columns_path="dataset/train/final_columns.txt")
except Exception as e:
    print(f"[FATAL] Preprocessor died: {e}")
    sys.exit(1)

def data_callback(msg):
    try:
        raw_dict = json.loads(msg.data)
        
        # 1. Print what we received
        print(f"\n[RECV] Time: {raw_dict.get('Timestamp')} | ID: {raw_dict.get('CAN_ID')}")
        
        # 2. Process
        input_tensor = processor.process_new_data(raw_dict)

        if input_tensor is None:
            print(f"   [Buffer] Filling... {len(processor.buffer)}/50")
        else:
            # 3. Inspect the Input to the AI
            inspect_tensor(input_tensor, "AI INPUT")

            with torch.no_grad():
                prediction = model(input_tensor)
                
                # 4. Inspect the Output from the AI
                inspect_tensor(prediction, "AI OUTPUT")
                
                actual_data = input_tensor[:, -1, :] 
                loss = torch.nn.functional.mse_loss(prediction, actual_data)
                error_score = loss.item()

                print(f"   [RESULT] Error Score: {error_score:.8f}")
                if error_score > 100:
                    print("   🚨 CRITICAL: ERROR SCORE IS IMPOSSIBLY HIGH")

    except Exception as e:
        print(f"[WARN] Crash: {e}")

if __name__ == '__main__':
    rospy.init_node('debug_dashboard', anonymous=True)
    rospy.Subscriber(ROS_TOPIC, String, data_callback)
    print(f"[*] Debug Dashboard Listening on {ROS_TOPIC}...")
    rospy.spin()