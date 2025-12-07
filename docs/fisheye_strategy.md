# Fisheye Integration Strategy

## Overview
This project utilizes a mixed-camera setup involving two standard rectilinear cameras and one fisheye camera. The fisheye camera provides a wider field of view but introduces significant radial distortion that must be modeled correctly for accurate 3D reconstruction.

## Camera Setup
- **Camera 1 & 2**: Standard lens (Rectilinear). Low distortion.
- **Camera 3**: Fisheye lens. High distortion, wide FOV.

## Distortion Model
We utilize the **Kannala-Brandt** model (or OpenCV's `fisheye` module) for the fisheye camera, which is superior for wide-angle lenses compared to the standard Brown-Conrady model.

### Calibration
- **Intrinsics**: Stored in `intrinsics.toml`.
    - Fisheye cameras are flagged with `is_fisheye = true`.
    - Coefficients: `[k1, k2, k3, k4]` (typical for fisheye).
- **Extrinsics**: Stored in `extrinsics.json`.
    - Rotation and Translation relative to a global origin (usually Camera 1).

## Processing Strategy

### Phase 1: 2D Detection on Distorted Frames
Current implementation performs pose detection directly on the raw, distorted fisheye frames.
- **Pros**: No loss of FOV at the edges; computationally cheaper (no image warping).
- **Cons**: Standard pose models (trained on rectilinear images) may have lower accuracy on highly distorted limb shapes.

### Phase 2: Point Undistortion
Before triangulation, the detected 2D keypoints are undistorted.
1. **Input**: Distorted 2D point $(u_d, v_d)$.
2. **Process**: Apply inverse distortion model using calibrated intrinsics.
3. **Output**: Normalized ray or undistorted point $(u_u, v_u)$ on the virtual image plane.

### Phase 3: Triangulation
The undistorted rays from the fisheye camera are combined with rays from the standard cameras to solve for the 3D point.

## Future Improvements
- **Undistortion-First**: Warping the image to a rectilinear projection before inference (sacrifices FOV).
- **Fisheye-Specific Models**: Training or fine-tuning pose estimators on fisheye datasets.
