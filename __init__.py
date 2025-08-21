"""
Multi-Camera 3D Vision - Modular Motion Capture Pipeline

A production-grade system for real-time 3D human pose estimation
using multiple synchronized cameras with fisheye support.
"""

__version__ = "2.0.0"
__author__ = "Personal Vision Project"

# Import key components for easy access
from .calibration_loader import (
    CameraCalibration,
    CalibrationLoader,
    create_sample_config_files
)

__all__ = [
    "CameraCalibration",
    "CalibrationLoader", 
    "create_sample_config_files",
]
