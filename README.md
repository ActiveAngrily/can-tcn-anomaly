# CAN-TCN-Anomaly: Real-Time Vehicular Intrusion Detection

**Platform:** Apple Silicon (M1/M2/M3) macOS & Docker (Ubuntu 18.04 / ROS Melodic)  
**Simulator:** CARLA v0.9.11  
**Model:** Temporal Convolutional Network (TCN) with Core ML Optimization

---

## 1. Project Overview

This repository contains a complete pipeline for detecting anomalies in vehicular Controller Area Network (CAN) data using a Temporal Convolutional Network (TCN).


The project addresses a specific infrastructure challenge: **Running the industry-standard CARLA Simulator and ROS Melodic bridge on modern Apple Silicon hardware**, which typically lacks native support for these x86/Linux-based tools.

### System Architecture

The system operates on a **Hybrid Split-Architecture**:

1.  **Server (Host - macOS):**
      * Runs the Windows binary of CARLA Simulator via a Wine compatibility layer (Winery).
      * Leverages Metal/MoltenVK for native graphics rendering on the Mac GPU.
2.  **Client (Guest - Docker):**
      * Runs Ubuntu 18.04 (bionic) via `linux/amd64` emulation.
      * Hosts the ROS Melodic ecosystem and the CARLA ROS Bridge.
      * Executes the AI Inference Engine (TCN).

### The AI Pipeline

The core of this project is an Anomaly Detector that monitors the CAN bus (Throttle, Brake, Steer, Speed) in real-time.

1.  **Data Ingestion:** ROS Bridge captures vehicle telemetry.
2.  **Preprocessing:** A sliding window mechanism (50 time-steps) normalizes data using a pre-calculated scaler.
3.  **Inference:** A PyTorch TCN model predicts the *expected* next state of the vehicle.
4.  **Detection:** If the reconstruction error (MSE) between the prediction and the actual sensor reading exceeds a calculated threshold, an anomaly alert is triggered.

-----

## 2\. Repository Structure

The codebase has been reorganized for modularity. Below is the file tree and the purpose of each component.

```text
.
├── src/                       # Core runtime applications
│   ├── dashboard.py           # The main runtime application (ROS Listener + AI Inference)
│   └── live_preprocessor.py   # Helper class for sliding windows and data scaling
│
├── tools/                     # Utilities and tests
│   ├── anomaly_injector.py    # "Chaos Monkey" script to force sudden vehicle errors
│   └── convert_coreml.py      # Utility to convert PyTorch models to TorchScript/CoreML
│
├── notebooks/                 # Jupyter Notebooks
│   └── modeltrain.ipynb       # Jupyter Notebook for training the TCN from scratch
│
├── legacy/                    # Older debug scripts and tools
│   ├── can_generator.py
│   ├── debug_connection.py
│   ├── debug_dashboard.py
│   ├── debug_generator.py
│   ├── export_scaler.py
│   └── fix_scaler.py
│
├── dataset/                   # Data definition and scaling artifacts
│   ├── final_columns.txt      # The schema of the CAN bus data
│   ├── train/
│   │   └── scaler.joblib      # Pre-fitted Scikit-Learn scaler
│   └── val/
│       └── detected_anomalies.csv # Log of anomalies found during validation
│
├── docker/                    # Infrastructure and virtualization
│   ├── Dockerfile.melodic     # Definition for the ROS/Ubuntu environment
│   └── launch_container.sh    # Script to spin up the environment with X11 forwarding
│
├── models/                    # Serialized AI models
│   ├── anomaly_detector.pth          # Raw PyTorch weights
│   ├── anomaly_detector_optimized.pt # JIT-compiled TorchScript model
│   └── anomaly_detector_final.mlpackage/ # Apple Core ML format
│
└── .gitignore                 # Git configuration
```

-----

## 3\. Prerequisites

Before cloning this repository, ensure your host environment is configured correctly.

### A. Host Software (macOS)

1.  **XQuartz:** Required for GUI forwarding (viewing the PyGame window from Docker).
      * Install via brew: `brew install --cask xquartz`
      * *Settings:* Enable "Allow connections from network clients" in XQuartz preferences.
2.  **Winery (or similar Wine wrapper):**
      * You must have the **Windows** version of CARLA 0.9.11 installed.
      * It must be running on port `2000` on `localhost`.
3.  **Docker Desktop:**
      * Ensure "Use Rosetta for x86/amd64 emulation on Apple Silicon" is enabled in settings for performance.

### B. External Files

You must place the CARLA Python API installer inside the project root (or accessible path) to build the Docker image.

  * File required: `CARLA_0.9.11.tar.gz` (Linux version) - *Note: This is needed inside the Docker container for the Python API, even though the Host runs the Windows simulator.*

-----

## 4\. Installation & Setup

### Step 1: Build the Docker Image

The Dockerfile is located in the `docker/` directory. It sets up Ubuntu 18.04, ROS Melodic, and all Python dependencies.

**Note:** This build process forces `linux/amd64` architecture. It may take 15-20 minutes to complete due to emulation overhead.

```bash
# Run from the repository root
# Make sure CARLA_0.9.11.tar.gz is in the same directory you build from
cp /path/to/CARLA_0.9.11.tar.gz . 

docker build --platform linux/amd64 -f docker/Dockerfile.melodic -t carla:0.9.11 .
```

### Step 2: Prepare the Network

To allow the Docker container to display graphics (PyGame) on your Mac screen, you must allow X11 connections.

```bash
# Run this every time you restart your computer
xhost +
```

### Step 3: Launch the Environment

We provide a helper script in `docker/launch_container.sh` that handles volume mounting, display environment variables, and port mapping.

```bash
cd docker
chmod +x launch_container.sh
./launch_container.sh
```

You will be dropped into a bash shell inside the container: `melodic@<container_id>:/home/melodic$`.

-----

## 5\. Running the Simulation

This process requires three terminal windows.

### Terminal 1: The Simulator (Host Mac)

1.  Open your Winery wrapper.
2.  Launch `CarlaUE4.exe`.
3.  Wait for the map to load (black screen usually resolves into a city map).

### Terminal 2: The ROS Bridge (Docker)

In the shell started by `launch_container.sh`, run the bridge. This connects the Linux ROS environment to the Mac Host Simulator.

```bash
# Inside Docker
roslaunch carla_ros_bridge carla_ros_bridge_with_example_ego_vehicle.launch host:=host.docker.internal port:=2000
```

  * **Expected Output:** A PyGame window labeled "Manual Control" should appear on your Mac desktop.
  * **Controls:** Use `W/A/S/D` to drive the vehicle. Press `B` to toggle ignition.

### Terminal 3: The AI Dashboard (Docker)

Open a new terminal tab on your Mac, connect to the running container, and start the anomaly detector.

```bash
# 1. Find Container ID
docker ps

# 2. Enter Container
docker exec -it carla_client bash

# 3. Navigate to Source Code
cd /home/melodic/can-tcn-anomaly/src

# 4. Run Dashboard
python3 dashboard.py
```

  * **Operational Status:** The dashboard will load `models/anomaly_detector_optimized.pt` and begin buffering data.
  * **Normal State:** You will see green text: `[NORMAL] Status OK. Score: 0.0024...`

-----

## 6\. Anomaly Injection (Testing)

To verify the AI is working, we use a script that hijacks the vehicle controls to create erratic behavior that the TCN model has not predicted.

1.  Keep `dashboard.py` running in Terminal 3.
2.  Open a **Terminal 4** (exec into Docker).
3.  Run the injector:

<!-- end list -->

```bash
cd tools
python3 anomaly_injector.py
```

**The Sequence of Events:**

1.  The script connects to CARLA via PythonAPI.
2.  It identifies your "Ego Vehicle".
3.  It waits 3 seconds, then forces the throttle to 100% and steering to 100% (Hard Right).
4.  **Dashboard Reaction:** The prediction error (MSE) in Terminal 3 will spike.
5.  **Alert:** The dashboard text will turn **RED**: `[ANOMALY DETECTED] Score: 0.4512 (Threshold: 0.3101)`.

-----

## 7\. Development & Model Training

### Training the Model

The model is trained using `modeltrain.ipynb`. This notebook:

1.  Loads `.pt` data files (PyTorch tensors).
2.  Defines a TCN (Temporal Convolutional Network) with dilated convolutions.
3.  Trains for 5 epochs.
4.  Saves the weights to `models/anomaly_detector.pth`.

### Optimization for Mac

Standard PyTorch models can be slow in emulated Docker environments. We use `convert_coreml.py` to trace the model into **TorchScript**, which is significantly faster and removes Python overhead during inference.

To optimize a newly trained model:

1.  Place the new `.pth` file in `models/`.
2.  Run the converter:

<!-- end list -->

```bash
cd tools
python3 convert_coreml.py
```

This will generate `models/anomaly_detector_optimized.pt` (used by the dashboard) and `models/anomaly_detector_final.mlpackage` (for Xcode/Swift apps).

-----

## 8\. Codebase Details

### `src/dashboard.py`

The central orchestrator.

  * **Inputs:** Subscribes to ROS topic `/carla/ego_vehicle/can_data`.
  * **Processing:** deserializes JSON, sends data to `LivePreprocessor`.
  * **Inference:** Passes tensor buffer to loaded JIT model.
  * **Logic:** Calculates MSE loss vs. Threshold. Prints Alerts.

### `src/live_preprocessor.py`

Handles the data engineering.

  * **Scaler:** Loads `dataset/train/scaler.joblib` to normalize incoming CAN data to 0-1 range.
  * **Buffer:** Maintains a `deque` of the last 50 time steps (required sequence length for the TCN).
  * **Hex Conversion:** Automatically parses hexadecimal CAN IDs into integers.

### `docker/Dockerfile.melodic`

A specialized build file for Apple Silicon.

  * **Base:** `ubuntu:18.04` (Replaces nvidia-cuda bases which crash on Mac).
  * **Fixes:** Installs `gnupg2` and `curl` manually to fix ROS key authentication errors.
  * **User:** Sets up a non-root user `melodic` to avoid permission issues with X11 forwarding.

-----

## 9\. Troubleshooting

**Issue: "ALSA lib... No such file or directory"**

  * *Cause:* PyGame trying to initialize audio drivers that don't exist in Docker.
  * *Fix:* The launch script sets `SDL_AUDIODRIVER=dummy`.

**Issue: "OGRE ... RenderSystem\_GL" crashes**

  * *Cause:* Mac M1/M2 graphics do not support the OpenGL calls made by RViz/CARLA in Docker.
  * *Fix:* The launch script enforces software rendering via `LIBGL_ALWAYS_SOFTWARE=1`.

**Issue: "Connection Refused: 127.0.0.1:2000"**

  * *Cause:* Docker container cannot see the Mac's localhost.
  * *Fix:* Use the DNS alias `host.docker.internal` in your launch commands.

**Issue: "Model file not found" in Dashboard**

  * *Cause:* The file paths were updated in the recent cleanup.
  * *Fix:* Ensure you run `dashboard.py` from the `src/` directory, so it can find `../models/` and `../dataset/` correctly.

-----

**License:** MIT
**Author:** ActiveAngrily, sanjanaa2102, sarisha06 