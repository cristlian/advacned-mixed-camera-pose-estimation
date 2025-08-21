# CoachCube Vision - Modular Pipeline v2.0

## 🏗️ Architecture Overview

This is the new modular implementation of the CoachCube Vision system, featuring a clean separation of concerns and support for fisheye cameras.

### Module Structure

```
vision-new/
├── main.py                 # Main entry point and pipeline orchestrator
├── calibration_loader.py   # Camera calibration management ✅
├── capture.py             # Multi-camera synchronized capture (TODO)
├── detection.py           # YOLO pose detection with fisheye support (TODO)
├── triangulation.py       # Multi-view 3D reconstruction (TODO)
├── filtering.py           # Kalman filtering for temporal smoothing (TODO)
├── streaming.py           # WebSocket server for Unity (TODO)
└── tools/
    └── visualize_3d_output.py  # Offline 3D visualization tool (TODO)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create conda environment
conda create -n coachcube_vision python=3.9
conda activate coachcube_vision

# Install dependencies
pip install numpy tomli
```

### 2. Test Calibration Loading

```bash
# Generate sample calibration files and test
python main.py --generate-samples --test-projection

# Use your own calibration files
python main.py --intrinsics path/to/intrinsics.toml --extrinsics path/to/extrinsics.json

# Load specific cameras only
python main.py --intrinsics cal.toml --extrinsics ext.json --cameras 0 2
```

## 📋 Calibration File Formats

### Intrinsics (TOML)

```toml
[camera_0]
image_size = [1920, 1080]
fisheye = false  # NEW: Camera type flag
camera_matrix = [
    [fx, 0.0, cx],
    [0.0, fy, cy],
    [0.0, 0.0, 1.0]
]
distortion_coeffs = [k1, k2, p1, p2, k3]  # 5 params for standard, 4 for fisheye
```

### Extrinsics (JSON)

```json
{
  "camera_0": {
    "rotation_matrix": [
      [r11, r12, r13],
      [r21, r22, r23],
      [r31, r32, r33]
    ],
    "translation_vector": [tx, ty, tz]
  }
}
```

## 🐟 Fisheye Camera Support

The pipeline now supports fisheye cameras through the `is_fisheye` flag in the calibration:

- **Phase 0 (Current)**: Detect on distorted image, correct keypoints only
- **Phase 1 (Planned)**: Dynamic ROI undistortion
- **Phase 2 (Future)**: Fisheye-aware YOLO model

## 📊 Key Features

### CameraCalibration Dataclass

The foundation of our calibration system:

```python
@dataclass
class CameraCalibration:
    camera_id: str              # Unique identifier
    camera_index: int           # Capture system index
    is_fisheye: bool           # Camera type flag
    image_size: Tuple[int, int] # Resolution
    camera_matrix: np.ndarray   # 3x3 intrinsic matrix
    distortion_coeffs: np.ndarray  # Distortion parameters
    rotation_matrix: np.ndarray    # 3x3 world rotation
    translation_vector: np.ndarray  # 3x1 world translation
    projection_matrix: Optional[np.ndarray]  # 3x4 P matrix
```

### CalibrationLoader Class

Handles all calibration loading:

```python
loader = CalibrationLoader(
    intrinsics_path=Path("intrinsics.toml"),
    extrinsics_path=Path("extrinsics.json"),
    camera_indices=[0, 1, 2]
)
calibrations = loader.load_calibrations()
```

## 🔄 Development Roadmap

### ✅ Completed
- [x] Modular architecture design
- [x] CalibrationLoader implementation
- [x] Fisheye camera flag support
- [x] Sample configuration generator
- [x] Projection matrix computation

### 🚧 In Progress
- [ ] MultiCamCapture module
- [ ] YOLO detection with fisheye support
- [ ] Multi-view triangulation
- [ ] Kalman filtering
- [ ] WebSocket streaming

### 📅 Planned
- [ ] Dynamic ROI undistortion
- [ ] Fisheye-aware YOLO training
- [ ] Docker containerization
- [ ] CI/CD pipeline

## 🧪 Testing

Run the calibration loader test:

```bash
# Basic test
python main.py --generate-samples

# Verbose mode with projection test
python main.py --generate-samples --test-projection --verbose
```

## 📝 Notes

- The system expects 3 cameras by default (indices 0, 1, 2)
- Fisheye cameras should have `fisheye = true` in the TOML config
- All coordinates are in meters for world space, pixels for image space
- The pipeline is designed for real-time performance (target: 30 FPS)

## 🤝 Contributing

Please follow the established patterns:
- Use type hints for all functions
- Include comprehensive docstrings
- Handle errors gracefully
- Log important operations
- Maintain the modular structure

---

**Author**: CoachCube Vision Team  
**Version**: 2.0 (Modular Architecture)  
**Status**: Foundation Complete, Core Modules In Development
