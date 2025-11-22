import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import os

# --- STEP 1: DEFINE THE MODEL ARCHITECTURE (CRITICAL) ---
class TCN_Forecaster(nn.Module):
    def __init__(self, num_inputs, output_dim):
        super(TCN_Forecaster, self).__init__()
        self.network = nn.Sequential(
            nn.Conv1d(num_inputs, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1) 
        )
        self.fc = nn.Linear(128, output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        features = self.network(x)
        features = features.squeeze(-1)
        return self.fc(features)

# --- STEP 2: DEFINE PATHS AND SHAPES ---
TORCHSCRIPT_PATH = 'anomaly_detector_optimized.pt'
# *** CORRECTED: Output extension must be .mlpackage ***
COREML_PATH = 'anomaly_detector_final.mlpackage' 

# Input dimensions based on your data (50 sequence length, 11 features)
SEQUENCE_LENGTH = 50
FEATURES = 11

# --- STEP 3: DEFINE CORE ML INPUT SPECIFICATION ---
input_name = "input_data"

mlmodel_input = ct.TensorType(
    name=input_name, 
    shape=(1, SEQUENCE_LENGTH, FEATURES),
    dtype=np.float32
)

# --- STEP 4: CONVERT THE TORCHSCRIPT MODEL TO CORE ML ---
print(f"⚙️ Starting TorchScript (.pt) to Core ML Conversion on Mac...")

try:
    # Load a dummy PyTorch model to pass to the converter
    # The converter is smart enough to handle the .pt file directly.
    dummy_model = TCN_Forecaster(FEATURES, 11)

    mlmodel = ct.convert(
        model=TORCHSCRIPT_PATH,
        source='pytorch', 
        inputs=[mlmodel_input],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS12,
    )

    # --- SAVE THE CORE ML MODEL ---
    # This will now create a folder with the .mlpackage extension.
    mlmodel.save(COREML_PATH)

    print(f"✅ FINAL SUCCESS! Core ML model saved to '{COREML_PATH}'")
    print("\nThe file is optimized and ready for Xcode integration.")

except Exception as e:
    print(f"❌ Conversion failed with a critical error: {e}")