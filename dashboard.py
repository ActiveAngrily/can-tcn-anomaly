import torch
import numpy as np
import rospy
import json
from std_msgs.msg import String  # We use String because Sarisha needs a dictionary

# --- 1. IMPORT SARISHA'S CLASS ---
# We import the exact class name found in her file
from live_preprocessor import LivePreprocessor

# --- 2. CONFIGURATION ---
THRESHOLD = 0.3101  # Your calculated value
MODEL_FILE = "ML model/anomaly_detector_optimized.pt"
ROS_TOPIC = "/carla/ego_vehicle/can_data" # Standard name for CAN data strings

# --- 3. SETUP SYSTEM ---
print("--------------------------------------------------")
print(">>> SYSTEM STARTUP")

# A. Load Model
device = torch.device("cpu")
print(f">>> Loading Model: {MODEL_FILE}")
try:
    model = torch.jit.load(MODEL_FILE, map_location=device)
    model.eval()
    print("   ✅ Model Loaded.")
except Exception as e:
    print(f"   ❌ ERROR: Model not found! {e}")
    exit()

# B. Initialize Preprocessor
# We don't need to pass arguments because you fixed the 'data/' folder in Step 1
print(">>> Initializing Preprocessor...")
try:
    processor = LivePreprocessor()
    print("   ✅ Preprocessor Ready.")
except Exception as e:
    print(f"   ❌ ERROR: Missing 'data/scaler.joblib' or 'final_columns.txt'!")
    print(f"   Details: {e}")
    exit()

# --- 4. THE CORE LOGIC ---
def data_callback(msg):
    """
    Runs every time CARLA sends a message.
    """
    try:
        # 1. Convert ROS String to Python Dictionary
        # We assume the bridge sends data like: '{"CAN_ID": "0350", "D0": "FF", ...}'
        raw_dict = json.loads(msg.data)
        
        # 2. Clean Data (Sarisha's Code)
        # Returns a Tensor if buffer is full (50 items), or None
        input_tensor = processor.process_new_data(raw_dict)

        # 3. Predict & Check (Only if we have a full tensor)
        if input_tensor is not None:
            with torch.no_grad():
                # Run Anant's Model
                prediction = model(input_tensor)

                # Compare Prediction vs Reality (Last value in the window)
                # Sarisha's tensor shape is likely (1, 50, 11)
                actual_data = input_tensor[:, -1, :] 
                
                # Calculate Error (MSE)
                loss = torch.nn.functional.mse_loss(prediction, actual_data)
                error_score = loss.item()

                # 4. DECIDE
                if error_score > THRESHOLD:
                    print(f"\033[91m🚨 ANOMALY DETECTED! Score: {error_score:.5f}\033[0m")
                else:
                    print(f"\033[92m✅ Normal. Score: {error_score:.5f}\033[0m")
                    
    except json.JSONDecodeError:
        print("⚠️ Received non-JSON data. Check ROS bridge format.")
    except Exception as e:
        print(f"⚠️ Error in loop: {e}")

# --- 5. EXECUTION LOOP ---
if __name__ == '__main__':
    try:
        rospy.init_node('anomaly_dashboard', anonymous=True)
        rospy.Subscriber(ROS_TOPIC, String, data_callback)
        
        print(f">>> DASHBOARD LIVE.")
        print(f"    Listening to: {ROS_TOPIC}")
        print(f"    Threshold: {THRESHOLD}")
        print("--------------------------------------------------")
        rospy.spin()
    except rospy.ROSInterruptException:
        pass