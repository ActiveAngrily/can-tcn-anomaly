import numpy as np
import torch
import joblib
from collections import deque
import os

class LivePreprocessor:
    def __init__(self, scaler_path='data/scaler.joblib', columns_path='data/final_columns.txt', seq_length=50):
        """
        This class handles live data coming from the car simulator.
        It keeps a 'buffer' of the last 50 messages.
        """
        print("Initializing Live Preprocessor...")
        
        # 1. Load the Scaler (Recipe from Phase 1)
        if not os.path.exists(scaler_path):
            raise Exception(f"ERROR: Could not find {scaler_path}. Did you check the path?")
        self.scaler = joblib.load(scaler_path)
        print(f"--> Loaded scaler.")

        # 2. Load Column Order (Must match Phase 1 exactly)
        if not os.path.exists(columns_path):
            raise Exception(f"ERROR: Could not find {columns_path}.")
            
        with open(columns_path, 'r') as f:
            # Reads the file and creates a list ['Timestamp', 'CAN_ID', ...]
            content = f.read().strip()
            self.column_order = content.split(',')
        print(f"--> Loaded {len(self.column_order)} columns.")

        # 3. Define which columns are Hexadecimal (0x...)
        self.hex_columns = ['CAN_ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

        # 4. Create the 'Buffer' (The Sliding Window)
        # deque is a special list that automatically pops the old items when full
        self.seq_length = seq_length
        self.buffer = deque(maxlen=seq_length)
        
        print("Live Preprocessor Ready.")

    def process_new_data(self, ros_data_dict):
        """
        Input: A dictionary of ONE row of data from the car.
        Output: A PyTorch Tensor (if we have 50 items), or None (if buffering).
        """
        try:
            # --- STEP A: Clean and Order the Row ---
            ordered_row = []
            
            for col_name in self.column_order:
                # Get the value from the dictionary. Default to 0 if missing.
                raw_value = ros_data_dict.get(col_name, 0)
                
                # Convert Data Types
                if col_name in self.hex_columns:
                    # It's Hex! Convert '04a0' -> 1184
                    try:
                        # We convert to string first just in case, then to int
                        clean_val = int(str(raw_value), 16)
                    except ValueError:
                        clean_val = 0
                else:
                    # It's Decimal (Timestamp or DLC)
                    try:
                        clean_val = float(raw_value)
                    except ValueError:
                        clean_val = 0.0
                
                ordered_row.append(clean_val)

            # --- STEP B: Scale the Row ---
            # The scaler expects a list of lists: [[val1, val2...]]
            # We use .transform(), NEVER .fit() here.
            scaled_row = self.scaler.transform([ordered_row])[0]

            # --- STEP C: Add to Buffer ---
            self.buffer.append(scaled_row)

            # --- STEP D: Check if ready ---
            # If we don't have 50 items yet, we can't make a prediction.
            if len(self.buffer) < self.seq_length:
                return None
            
            # --- STEP E: Create Tensor ---
            # Convert buffer to numpy array
            window_array = np.array(self.buffer)
            
            # Convert to PyTorch Tensor and add "Batch" dimension
            # Shape: (1, 50, 11)
            tensor_out = torch.tensor(window_array).float().unsqueeze(0)
            
            return tensor_out

        except Exception as e:
            print(f"PREPROCESSING ERROR: {e}")
            return None

# --- TEST BLOCK (Runs only when you run this file directly) ---
if __name__ == "__main__":
    print("--- STARTING TEST ---")
    # Initialize
    lp = LivePreprocessor()
    
    # Create a fake data packet (what ROS looks like)
    dummy_data = {
        'Timestamp': 1479121434.85,
        'CAN_ID': '0350', 
        'DLC': '8',
        'D0': '00', 'D1': 'ff', 'D2': '00', 'D3': '00', 
        'D4': '1a', 'D5': '00', 'D6': '00', 'D7': '00'
    }
    
    print("Feeding 55 dummy packets...")
    for i in range(55):
        result = lp.process_new_data(dummy_data)
        if result is not None:
            print(f"Packet {i}: SUCCESS! Tensor shape: {result.shape}")
        else:
            # It stays silent while buffering (0-49)
            pass
    print("--- TEST COMPLETE ---")