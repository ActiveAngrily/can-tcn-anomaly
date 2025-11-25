import numpy as np
import torch
import joblib
from collections import deque
import os

class LivePreprocessor:
    def __init__(self, scaler_path='dataset/train/scaler.joblib', columns_path='dataset/final_columns.txt', seq_length=50):
        """
        This class handles live data coming from the car simulator.
        It keeps a 'buffer' of the last 50 messages.
        """
        print(f"[INFO] Initializing LivePreprocessor...")
        print(f"[INFO] Configuration: Sequence Length = {seq_length}")
        
        # 1. Load the Scaler
        if not os.path.exists(scaler_path):
            # Fallback check in case running from dataset dir
            if os.path.exists('../' + scaler_path):
                scaler_path = '../' + scaler_path
            else:
                raise Exception(f"[ERROR] Could not find scaler at {scaler_path}. Ensure you are running from the repo root.")
                
        try:
            self.scaler = joblib.load(scaler_path)
            print(f"[INFO] Successfully loaded scaler from: {scaler_path}")
        except Exception as e:
            print(f"[ERROR] Failed to load scaler: {e}")
            raise

        # 2. Load Column Order
        if not os.path.exists(columns_path):
             raise Exception(f"[ERROR] Could not find columns file at {columns_path}.")
            
        with open(columns_path, 'r') as f:
            content = f.read().strip()
            self.column_order = content.split(',')
        print(f"[INFO] Loaded column definitions. Total columns: {len(self.column_order)}")

        # 3. Define which columns are Hexadecimal (0x...)
        self.hex_columns = ['CAN_ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']

        # 4. Create the 'Buffer' (The Sliding Window)
        self.seq_length = seq_length
        self.buffer = deque(maxlen=seq_length)
        
        print("[INFO] LivePreprocessor is ready to process stream.")

    def process_new_data(self, ros_data_dict):
        """
        Input: A dictionary of ONE row of data from the car.
        Output: A PyTorch Tensor (if we have 50 items), or None (if buffering).
        """
        try:
            ordered_row = []
            
            for col_name in self.column_order:
                raw_value = ros_data_dict.get(col_name, 0)
                
                # Convert Data Types
                if col_name in self.hex_columns:
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

            # Scale the Row
            # The scaler expects a list of lists: [[val1, val2...]]
            scaled_row = self.scaler.transform([ordered_row])[0]

            # Add to Buffer
            self.buffer.append(scaled_row)

            # Check if ready
            # If we don't have 50 items yet, we can't make a prediction.
            if len(self.buffer) < self.seq_length:
                return None
            
            # Create Tensor
            window_array = np.array(self.buffer)
            # Convert to PyTorch Tensor and add "Batch" dimension: Shape (1, 50, 11)
            tensor_out = torch.tensor(window_array).float().unsqueeze(0)
            
            return tensor_out

        except Exception as e:
            print(f"[WARN] Preprocessing error encountered: {e}")
            return None