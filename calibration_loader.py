"""
Calibration Loader Module for CoachCube Vision Pipeline

This module handles loading camera calibration data from configuration files,
including intrinsics (from TOML) and extrinsics (from JSON). It identifies
camera types (standard vs fisheye) and packages all calibration data into
structured dataclass objects for use throughout the pipeline.

Author: CoachCube Vision Team
Date: 2025
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import tomli  # Using tomli for TOML parsing (Python 3.11+ has tomllib built-in)

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class CameraCalibration:
    """
    Complete calibration data for a single camera.
    
    Attributes:
        camera_id: Unique identifier for the camera (e.g., 0, 1, 2 or 'left', 'right', 'top')
        camera_index: Integer index for the camera in the capture system
        is_fisheye: Boolean flag indicating if this is a fisheye camera model
        image_size: Tuple of (width, height) in pixels
        camera_matrix: 3x3 intrinsic camera matrix (K)
        distortion_coeffs: Distortion coefficients (4-5 for standard, 4 for fisheye)
        rotation_matrix: 3x3 rotation matrix from camera to world coordinates (R)
        translation_vector: 3x1 translation vector from camera to world coordinates (t)
        projection_matrix: Optional 3x4 projection matrix (P = K[R|t])
    """
    camera_id: str
    camera_index: int
    is_fisheye: bool
    image_size: Tuple[int, int]
    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    projection_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self) -> None:
        """Validate and compute derived properties after initialization."""
        # Ensure numpy arrays
        self.camera_matrix = np.asarray(self.camera_matrix, dtype=np.float64)
        self.distortion_coeffs = np.asarray(self.distortion_coeffs, dtype=np.float64)
        self.rotation_matrix = np.asarray(self.rotation_matrix, dtype=np.float64)
        self.translation_vector = np.asarray(self.translation_vector, dtype=np.float64)
        
        # Validate shapes
        assert self.camera_matrix.shape == (3, 3), f"Invalid camera matrix shape: {self.camera_matrix.shape}"
        assert self.rotation_matrix.shape == (3, 3), f"Invalid rotation matrix shape: {self.rotation_matrix.shape}"
        assert self.translation_vector.shape in [(3,), (3, 1)], f"Invalid translation vector shape: {self.translation_vector.shape}"
        
        # Ensure translation vector is column vector
        if self.translation_vector.shape == (3,):
            self.translation_vector = self.translation_vector.reshape(3, 1)
        
        # Compute projection matrix if not provided
        if self.projection_matrix is None:
            self.compute_projection_matrix()
    
    def compute_projection_matrix(self) -> None:
        """Compute the 3x4 projection matrix P = K[R|t]."""
        rt_matrix = np.hstack([self.rotation_matrix, self.translation_vector])
        self.projection_matrix = self.camera_matrix @ rt_matrix
    
    def project_3d_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Project 3D world points to 2D image coordinates.
        
        Args:
            points_3d: Nx3 array of 3D points in world coordinates
            
        Returns:
            Nx2 array of 2D points in pixel coordinates
        """
        points_3d = np.asarray(points_3d)
        if points_3d.ndim == 1:
            points_3d = points_3d.reshape(1, -1)
        
        # Convert to homogeneous coordinates
        points_3d_homo = np.hstack([points_3d, np.ones((points_3d.shape[0], 1))])
        
        # Project
        points_2d_homo = (self.projection_matrix @ points_3d_homo.T).T
        
        # Convert from homogeneous to 2D
        points_2d = points_2d_homo[:, :2] / points_2d_homo[:, 2:3]
        
        return points_2d


class CalibrationLoader:
    """
    Loads and manages camera calibration data from configuration files.
    
    This class handles loading intrinsic parameters from TOML files and
    extrinsic parameters from JSON files, automatically detecting camera
    types (standard vs fisheye) and organizing the data for pipeline use.
    """
    
    def __init__(
        self,
        intrinsics_path: Path,
        extrinsics_path: Path,
        camera_indices: Optional[List[int]] = None
    ):
        """
        Initialize the calibration loader.
        
        Args:
            intrinsics_path: Path to the TOML file containing intrinsic calibration
            extrinsics_path: Path to the JSON file containing extrinsic calibration
            camera_indices: Optional list of camera indices to load (default: [0, 1, 2])
        """
        self.intrinsics_path = Path(intrinsics_path)
        self.extrinsics_path = Path(extrinsics_path)
        self.camera_indices = camera_indices or [0, 1, 2]
        
        # Validate file existence
        if not self.intrinsics_path.exists():
            raise FileNotFoundError(f"Intrinsics file not found: {self.intrinsics_path}")
        if not self.extrinsics_path.exists():
            raise FileNotFoundError(f"Extrinsics file not found: {self.extrinsics_path}")
        
        logger.info(f"Initialized CalibrationLoader with {len(self.camera_indices)} cameras")
    
    def load_intrinsics(self) -> Dict[int, Dict[str, Any]]:
        """
        Load intrinsic calibration data from TOML file.
        
        Returns:
            Dictionary mapping camera indices to intrinsic parameters
        """
        logger.info(f"Loading intrinsics from: {self.intrinsics_path}")
        
        with open(self.intrinsics_path, 'rb') as f:
            config = tomli.load(f)
        
        intrinsics = {}
        
        for cam_idx in self.camera_indices:
            cam_key = f"camera_{cam_idx}"
            
            if cam_key not in config:
                raise KeyError(f"Camera {cam_idx} not found in intrinsics file")
            
            cam_data = config[cam_key]
            
            # Parse camera matrix
            camera_matrix = np.array(cam_data['camera_matrix'], dtype=np.float64)
            
            # Parse distortion coefficients
            distortion_coeffs = np.array(cam_data['distortion_coeffs'], dtype=np.float64)
            
            # Parse image size
            image_size = tuple(cam_data['image_size'])
            
            # Check for fisheye flag (default to False for backward compatibility)
            is_fisheye = cam_data.get('fisheye', False)
            
            intrinsics[cam_idx] = {
                'camera_matrix': camera_matrix,
                'distortion_coeffs': distortion_coeffs,
                'image_size': image_size,
                'is_fisheye': is_fisheye
            }
            
            logger.debug(f"Loaded intrinsics for camera {cam_idx} (fisheye={is_fisheye})")
        
        return intrinsics
    
    def load_extrinsics(self) -> Dict[int, Dict[str, Any]]:
        """
        Load extrinsic calibration data from JSON file.
        
        Returns:
            Dictionary mapping camera indices to extrinsic parameters
        """
        logger.info(f"Loading extrinsics from: {self.extrinsics_path}")
        
        with open(self.extrinsics_path, 'r') as f:
            extrinsics_data = json.load(f)
        
        extrinsics = {}
        
        for cam_idx in self.camera_indices:
            cam_key = f"camera_{cam_idx}"
            
            if cam_key not in extrinsics_data:
                raise KeyError(f"Camera {cam_idx} not found in extrinsics file")
            
            cam_data = extrinsics_data[cam_key]
            
            # Parse rotation matrix
            rotation_matrix = np.array(cam_data['rotation_matrix'], dtype=np.float64)
            
            # Parse translation vector
            translation_vector = np.array(cam_data['translation_vector'], dtype=np.float64)
            
            extrinsics[cam_idx] = {
                'rotation_matrix': rotation_matrix,
                'translation_vector': translation_vector
            }
            
            logger.debug(f"Loaded extrinsics for camera {cam_idx}")
        
        return extrinsics
    
    def load_calibrations(self) -> List[CameraCalibration]:
        """
        Load complete calibration data for all cameras.
        
        This method combines intrinsic and extrinsic parameters into
        CameraCalibration dataclass objects for each camera.
        
        Returns:
            List of CameraCalibration objects, one per camera
        """
        logger.info("Loading complete calibration data...")
        
        # Load intrinsics and extrinsics
        intrinsics = self.load_intrinsics()
        extrinsics = self.load_extrinsics()
        
        # Combine into CameraCalibration objects
        calibrations = []
        
        for cam_idx in self.camera_indices:
            if cam_idx not in intrinsics:
                raise ValueError(f"Missing intrinsics for camera {cam_idx}")
            if cam_idx not in extrinsics:
                raise ValueError(f"Missing extrinsics for camera {cam_idx}")
            
            # Create calibration object
            calibration = CameraCalibration(
                camera_id=f"camera_{cam_idx}",
                camera_index=cam_idx,
                is_fisheye=intrinsics[cam_idx]['is_fisheye'],
                image_size=intrinsics[cam_idx]['image_size'],
                camera_matrix=intrinsics[cam_idx]['camera_matrix'],
                distortion_coeffs=intrinsics[cam_idx]['distortion_coeffs'],
                rotation_matrix=extrinsics[cam_idx]['rotation_matrix'],
                translation_vector=extrinsics[cam_idx]['translation_vector']
            )
            
            calibrations.append(calibration)
            
            logger.info(
                f"Loaded calibration for {calibration.camera_id}: "
                f"{'Fisheye' if calibration.is_fisheye else 'Standard'} camera, "
                f"resolution={calibration.image_size}"
            )
        
        logger.info(f"Successfully loaded {len(calibrations)} camera calibrations")
        return calibrations


def create_sample_config_files(output_dir: Path) -> None:
    """
    Create sample calibration configuration files for testing.
    
    Args:
        output_dir: Directory to save the sample files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample intrinsics TOML
    intrinsics_toml = """
[camera_0]
image_size = [1920, 1080]
fisheye = false
camera_matrix = [
    [1421.5, 0.0, 960.0],
    [0.0, 1421.5, 540.0],
    [0.0, 0.0, 1.0]
]
distortion_coeffs = [-0.0545, 0.0653, 0.0012, -0.0004, -0.0213]

[camera_1]
image_size = [1920, 1080]
fisheye = false
camera_matrix = [
    [1418.2, 0.0, 955.3],
    [0.0, 1419.7, 542.1],
    [0.0, 0.0, 1.0]
]
distortion_coeffs = [-0.0523, 0.0612, 0.0008, -0.0002, -0.0198]

[camera_2]
image_size = [1920, 1080]
fisheye = true
camera_matrix = [
    [892.3, 0.0, 960.0],
    [0.0, 892.3, 540.0],
    [0.0, 0.0, 1.0]
]
distortion_coeffs = [0.0892, -0.0421, 0.0123, -0.0034]
"""
    
    # Sample extrinsics JSON
    extrinsics_json = {
        "camera_0": {
            "rotation_matrix": [
                [0.8660, -0.5000, 0.0000],
                [0.4330, 0.7500, -0.5000],
                [0.2500, 0.4330, 0.8660]
            ],
            "translation_vector": [-1.5, 0.0, 2.0]
        },
        "camera_1": {
            "rotation_matrix": [
                [0.8660, 0.5000, 0.0000],
                [-0.4330, 0.7500, -0.5000],
                [-0.2500, 0.4330, 0.8660]
            ],
            "translation_vector": [1.5, 0.0, 2.0]
        },
        "camera_2": {
            "rotation_matrix": [
                [1.0000, 0.0000, 0.0000],
                [0.0000, 0.0000, -1.0000],
                [0.0000, 1.0000, 0.0000]
            ],
            "translation_vector": [0.0, 2.5, 1.0]
        }
    }
    
    # Write files
    intrinsics_path = output_dir / "intrinsics.toml"
    with open(intrinsics_path, 'w') as f:
        f.write(intrinsics_toml)
    
    extrinsics_path = output_dir / "extrinsics.json"
    with open(extrinsics_path, 'w') as f:
        json.dump(extrinsics_json, f, indent=2)
    
    logger.info(f"Created sample config files in: {output_dir}")
    return intrinsics_path, extrinsics_path
