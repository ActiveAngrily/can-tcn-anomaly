import torch
import torch.nn as nn
import coremltools as ct
import numpy as np
import os

# --- STEP 1: DEFINE MODEL ARCHITECTURE ---
# We redefine this to ensure we can load the weights into a structure
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

# --- STEP 2: DEFINE PATHS ---
# Updated to reflect new directory structure
WEIGHTS_PATH = 'models/anomaly_detector.pth'
TORCHSCRIPT_PATH = 'models/anomaly_detector_optimized.pt'
COREML_PATH = 'models/anomaly_detector_final.mlpackage' 

# Model Dimensions
SEQUENCE_LENGTH = 50
FEATURES = 11

# --- STEP 3: LOAD AND TRACE MODEL ---
print(f"[1/3] Checking for weights at {WEIGHTS_PATH}...")

if not os.path.exists(WEIGHTS_PATH):
    print(f"[ERROR] Weights file not found. Please ensure 'anomaly_detector.pth' is in the 'models/' folder.")
    exit(1)

# Instantiate and Load
print("[1/3] Loading PyTorch model...")
model = TCN_Forecaster(FEATURES, 11)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=torch.device('cpu')))
model.eval()

# Create Dummy Input for Tracing
dummy_input = torch.randn(1, SEQUENCE_LENGTH, FEATURES)

# Trace and Save TorchScript
print(f"[2/3] Tracing model to TorchScript format...")
try:
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(TORCHSCRIPT_PATH)
    print(f"[2/3] Saved TorchScript model to: {TORCHSCRIPT_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to trace model: {e}")
    exit(1)

# --- STEP 4: CONVERT TO CORE ML ---
print(f"[3/3] Converting to Apple Core ML format...")

try:
    mlmodel_input = ct.TensorType(
        name="input_data", 
        shape=(1, SEQUENCE_LENGTH, FEATURES),
        dtype=np.float32
    )

    mlmodel = ct.convert(
        model=traced_model,
        source='pytorch', 
        inputs=[mlmodel_input],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS12,
    )

    mlmodel.save(COREML_PATH)
    print(f"[SUCCESS] Core ML package saved to: {COREML_PATH}")
    print("[INFO] The model is now ready for Xcode integration.")

except Exception as e:
    print(f"[ERROR] Core ML conversion failed: {e}")
    exit(1)