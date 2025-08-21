"""
Test script for CalibrationLoader module

This script verifies that the calibration loading functionality works correctly
with sample data.
"""

import sys
from pathlib import Path

# Add parent directory to path if running from tests folder
sys.path.insert(0, str(Path(__file__).parent.parent))

from calibration_loader import CalibrationLoader, create_sample_config_files


def test_calibration_loader():
    """Test the calibration loader with sample data."""
    print("="*60)
    print("TESTING CALIBRATION LOADER MODULE")
    print("="*60)
    
    # Create sample files
    test_dir = Path("test_calibration")
    test_dir.mkdir(exist_ok=True)
    
    print("\n1. Creating sample calibration files...")
    intrinsics_path, extrinsics_path = create_sample_config_files(test_dir)
    print(f"   ✓ Created: {intrinsics_path}")
    print(f"   ✓ Created: {extrinsics_path}")
    
    # Load calibrations
    print("\n2. Loading calibrations...")
    loader = CalibrationLoader(
        intrinsics_path=intrinsics_path,
        extrinsics_path=extrinsics_path,
        camera_indices=[0, 1, 2]
    )
    
    calibrations = loader.load_calibrations()
    print(f"   ✓ Loaded {len(calibrations)} camera calibrations")
    
    # Verify data
    print("\n3. Verifying calibration data...")
    for i, cal in enumerate(calibrations):
        assert cal.camera_index == i, f"Camera index mismatch"
        assert cal.camera_matrix.shape == (3, 3), f"Invalid camera matrix shape"
        assert cal.rotation_matrix.shape == (3, 3), f"Invalid rotation matrix shape"
        assert cal.translation_vector.shape == (3, 1), f"Invalid translation vector shape"
        assert cal.projection_matrix is not None, f"Projection matrix not computed"
        assert cal.projection_matrix.shape == (3, 4), f"Invalid projection matrix shape"
        
        print(f"   ✓ Camera {i}: {'Fisheye' if cal.is_fisheye else 'Standard'} - Valid")
    
    # Test projection
    print("\n4. Testing 3D to 2D projection...")
    import numpy as np
    test_point = np.array([0, 0, 1])  # 1 meter in front of world origin
    
    for cal in calibrations:
        point_2d = cal.project_3d_to_2d(test_point)
        assert point_2d.shape == (1, 2), f"Invalid projection output shape"
        print(f"   ✓ Camera {cal.camera_index}: 3D {test_point} -> 2D [{point_2d[0,0]:.1f}, {point_2d[0,1]:.1f}]")
    
    # Clean up
    print("\n5. Cleaning up test files...")
    import shutil
    shutil.rmtree(test_dir)
    print("   ✓ Test files removed")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_calibration_loader()
