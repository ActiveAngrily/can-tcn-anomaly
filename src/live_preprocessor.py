"""
Live Preprocessor Module

This module handles the live normalization and processing of streaming CAN bus data.
It manages a sliding window buffer and transforms incoming messages into a tensor
shape suitable for the TCN model.
"""
import sys
import numpy as np
import torch
import joblib
import json
from collections import deque
import os
import time

# --- NUMPY COMPATIBILITY FIX ---
try:
    import numpy.core.multiarray
    sys.modules['numpy._core'] = numpy.core
    sys.modules['numpy._core.multiarray'] = numpy.core.multiarray
except ImportError:
    pass

class LivePreprocessor:
    """
    Maintains a sliding window of recent CAN messages and normalizes them
    using pre-computed scaling parameters from the training dataset.
    """
    def __init__(self, scaler_path='../dataset/val/scaler_params.json', columns_path='../dataset/train/final_columns.txt', seq_length=50):
        print(f"[INFO] Initializing LivePreprocessor (Hybrid Fix)...")
        
        # 1. Load Scaler
        self.use_json = False
        if scaler_path.endswith('.json'):
            self.use_json = True
            with open(scaler_path, 'r') as f:
                params = json.load(f)
                self.scale_ = np.array(params["scale_"])
                self.min_ = np.array(params["min_"])
        else:
            self.scaler = joblib.load(scaler_path)

        # 2. Load Columns
        with open(columns_path, 'r') as f:
            self.column_order = f.read().strip().split(',')

        self.hex_columns = ['CAN_ID', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7']
        self.seq_length = seq_length
        self.buffer = deque(maxlen=seq_length)
        
        # 3. TIMESTAMP BASELINE
        self.start_time = time.time()

    def process_new_data(self, ros_data_dict):
        try:
            ordered_row = []
            for col_name in self.column_order:
                
                # --- FIX 1: TIMESTAMP (Keep it relative/small) ---
                if col_name == 'Timestamp':
                    # The scaler likely expects 0.0 - 1000.0 range
                    clean_val = time.time() - self.start_time
                    # Prevent drift over long sessions
                    if clean_val > 10000: self.start_time = time.time()
                
                # --- FIX 2: CAN_ID (Must be the huge integer) ---
                elif col_name == 'CAN_ID':
                    raw_val = ros_data_dict.get(col_name, 0)
                    # Convert "0x18F02F01" -> 418393857
                    # The scaler needs this specific huge number to output 0.0-1.0
                    try:
                        clean_val = float(int(str(raw_val), 16))
                    except:
                        clean_val = 418393857.0 # Fallback to engine ID
                # -----------------------------------------------

                else:
                    # Process Sensors (D0-D7)
                    raw_value = ros_data_dict.get(col_name, 0)
                    if col_name in self.hex_columns:
                        try:
                            clean_val = float(int(str(raw_value), 16))
                        except ValueError:
                            clean_val = 0.0
                    else:
                        try:
                            clean_val = float(raw_value)
                        except ValueError:
                            clean_val = 0.0
                            
                ordered_row.append(clean_val)

            # Scale
            row_array = np.array(ordered_row)
            
            if self.use_json:
                scaled_row = (row_array * self.scale_) + self.min_
            else:
                scaled_row = self.scaler.transform([ordered_row])[0]

            self.buffer.append(scaled_row)

            if len(self.buffer) < self.seq_length:
                return None
            
            return torch.tensor(np.array(self.buffer)).float().unsqueeze(0)

        except Exception as e:
            print(f"[WARN] Preprocessing error: {e}")
            return None