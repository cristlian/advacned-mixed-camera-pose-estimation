# Advanced Mixed-Camera 3D Pose Estimation (Offline Pipeline)

## 📌 Project Scope
This project implements an **offline multi-camera 3D pose estimation pipeline** designed to process pre-recorded video footage. The system utilizes a mixed-camera setup consisting of **one fisheye camera and two standard rectilinear cameras**.

The primary goal is to demonstrate robust backend engineering, computer vision, and machine learning engineering (MLE) skills by prioritizing accuracy and architectural cleanliness over real-time performance.

## 📂 Module Status

| Module | Status | Description |
| :--- | :--- | :--- |
| `main.py` | ✅ Active | Pipeline orchestrator and entry point. |
| `calibration_loader.py` | ✅ Active | Manages camera intrinsics and extrinsics loading. |
| `capture.py` | ✅ Active | Handles synchronized frame ingestion from video files. |
| `detection.py` | ✅ Active | Performs 2D pose estimation using YOLOv8. |
| `triangulation.py` | ✅ Active | Reconstructs 3D points from multi-view 2D detections. |
| `filtering.py` | ✅ Active | Applies temporal filtering (OneEuro/Kalman) for smoothing. |
| `tools/visualize_3d_output.py` | ✅ Active | Generates 3D visualizations of the output data. |
| `streaming.py` | ⏸️ Inactive | Real-time WebSocket server (Moved to Future Work). |

## 📚 Documentation

Detailed documentation for the system architecture and specific strategies can be found in the `docs/` directory:

- [**System Architecture**](docs/architecture.md): Overview of the data flow and pipeline stages.
- [**Fisheye Integration Strategy**](docs/fisheye_strategy.md): Details on how the mixed-camera setup and fisheye distortion are handled.

## ⚙️ Setup & Usage

### Prerequisites
- Python 3.10+
- Conda (recommended)

### Environment Setup

1.  **Create the environment:**
    ```bash
    conda env create -f environment.yml
    conda activate vision3d
    ```

2.  **Install dependencies (if not using environment.yml):**
    ```bash
    pip install -r requirements.txt
    pip install -r requirements-dev.txt  # For development tools
    ```

### Running the Pipeline

To run the full pipeline on a set of recorded videos:

```bash
python main.py --config config.toml --output results/
```

*(Note: Ensure your `config.toml` points to the correct video paths and calibration files.)*

## 🔮 Future Work / Long-Term Roadmap

The following features are part of the long-term vision but are currently out of scope for the offline pipeline:

- **Real-Time Processing**: Optimizing the pipeline for live streaming and low-latency inference.
- **Live Streaming**: Re-enabling `streaming.py` to serve pose data to WebSocket clients.
- **Advanced Fisheye Models**: Training custom pose estimation models specifically for fisheye distortion.
- **Dynamic Calibration**: Implementing runtime calibration refinement.
