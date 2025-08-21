"""
Multi-Camera 3D Vision Pipeline - Main Entry Point

This is the lean entry point that orchestrates the modular pipeline components.
Currently demonstrates the calibration loading functionality as the foundation
for the complete vision pipeline.

Author: Personal Vision Project
Date: 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

# Import our modular components
from calibration_loader import CalibrationLoader, CameraCalibration, create_sample_config_files


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def print_calibration_summary(calibrations: List[CameraCalibration]) -> None:
    """
    Print a detailed summary of loaded calibration data.
    
    Args:
        calibrations: List of CameraCalibration objects to summarize
    """
    print("\n" + "="*80)
    print("MULTI-CAMERA 3D VISION - CALIBRATION SUMMARY")
    print("="*80)
    
    for cal in calibrations:
        print(f"\n📷 {cal.camera_id.upper()}")
        print("-"*40)
        print(f"  Camera Index:     {cal.camera_index}")
        print(f"  Camera Type:      {'🐟 FISHEYE' if cal.is_fisheye else '📹 STANDARD'}")
        print(f"  Image Size:       {cal.image_size[0]} × {cal.image_size[1]} pixels")
        
        # Intrinsic parameters
        print(f"\n  Intrinsic Parameters:")
        print(f"    Focal Length X:  {cal.camera_matrix[0, 0]:.2f} px")
        print(f"    Focal Length Y:  {cal.camera_matrix[1, 1]:.2f} px")
        print(f"    Principal Point: ({cal.camera_matrix[0, 2]:.2f}, {cal.camera_matrix[1, 2]:.2f}) px")
        print(f"    Distortion Coeffs: {len(cal.distortion_coeffs)} parameters")
        
        # Extrinsic parameters
        print(f"\n  Extrinsic Parameters:")
        print(f"    Translation (world): [{cal.translation_vector[0, 0]:.3f}, "
              f"{cal.translation_vector[1, 0]:.3f}, {cal.translation_vector[2, 0]:.3f}] meters")
        
        # Compute camera orientation (simplified Euler angles for display)
        # Note: This is a simplified representation for display purposes
        rotation_matrix = cal.rotation_matrix
        pitch = np.arctan2(-rotation_matrix[2, 0], 
                          np.sqrt(rotation_matrix[2, 1]**2 + rotation_matrix[2, 2]**2))
        yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
        roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        
        print(f"    Orientation (approx): Pitch={np.degrees(pitch):.1f}°, "
              f"Yaw={np.degrees(yaw):.1f}°, Roll={np.degrees(roll):.1f}°")
        
        # Projection matrix status
        print(f"\n  Projection Matrix:  {'✓ Computed' if cal.projection_matrix is not None else '✗ Not computed'}")
        print(f"    Matrix Shape:     {cal.projection_matrix.shape if cal.projection_matrix is not None else 'N/A'}")
    
    print("\n" + "="*80)
    print(f"Total Cameras Loaded: {len(calibrations)}")
    
    # Check camera configuration
    fisheye_count = sum(1 for cal in calibrations if cal.is_fisheye)
    standard_count = len(calibrations) - fisheye_count
    
    print(f"Camera Configuration: {standard_count} Standard + {fisheye_count} Fisheye")
    print("="*80 + "\n")


def test_projection(calibrations: List[CameraCalibration]) -> None:
    """
    Test the projection functionality with sample 3D points.
    
    Args:
        calibrations: List of CameraCalibration objects to test
    """
    print("\n" + "="*80)
    print("TESTING 3D TO 2D PROJECTION")
    print("="*80)
    
    # Define a test 3D point (center of capture volume)
    test_point_3d = np.array([0.0, 0.0, 0.0])  # Origin in world coordinates
    
    print(f"\nTest 3D Point (world coordinates): {test_point_3d}")
    print("-"*40)
    
    for cal in calibrations:
        # Project the 3D point to 2D
        point_2d = cal.project_3d_to_2d(test_point_3d)
        
        print(f"\n{cal.camera_id}:")
        print(f"  Projected 2D point: ({point_2d[0, 0]:.2f}, {point_2d[0, 1]:.2f}) pixels")
        
        # Check if point is within image bounds
        in_bounds = (0 <= point_2d[0, 0] < cal.image_size[0] and 
                    0 <= point_2d[0, 1] < cal.image_size[1])
        print(f"  Within image bounds: {'✓ Yes' if in_bounds else '✗ No'}")
    
    print("\n" + "="*80 + "\n")


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Multi-Camera 3D Vision Pipeline - Modular Motion Capture System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Load calibration from existing files
  python main.py --intrinsics calibration/intrinsics.toml --extrinsics calibration/extrinsics.json
  
  # Generate and use sample calibration files
  python main.py --generate-samples
  
  # Load specific cameras only
  python main.py --intrinsics cal.toml --extrinsics ext.json --cameras 0 2
  
  # Run with verbose logging
  python main.py --intrinsics cal.toml --extrinsics ext.json --verbose
        """
    )
    
    # Calibration file arguments
    parser.add_argument(
        '--intrinsics', '-i',
        type=Path,
        help='Path to intrinsics TOML file'
    )
    parser.add_argument(
        '--extrinsics', '-e',
        type=Path,
        help='Path to extrinsics JSON file'
    )
    
    # Camera selection
    parser.add_argument(
        '--cameras', '-c',
        type=int,
        nargs='+',
        default=[0, 1, 2],
        help='Camera indices to load (default: 0 1 2)'
    )
    
    # Sample generation
    parser.add_argument(
        '--generate-samples', '-g',
        action='store_true',
        help='Generate sample calibration files for testing'
    )
    parser.add_argument(
        '--samples-dir',
        type=Path,
        default=Path('calibration/samples'),
        help='Directory for sample files (default: calibration/samples)'
    )
    
    # Testing options
    parser.add_argument(
        '--test-projection', '-t',
        action='store_true',
        help='Test 3D to 2D projection with sample points'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose debug logging'
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main entry point for the CoachCube Vision pipeline.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Parse arguments
    args = parse_arguments()
    
    # Configure logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("Starting Multi-Camera 3D Vision Pipeline (Modular Architecture v2.0)")
    
    try:
        # Handle sample generation
        if args.generate_samples:
            logger.info("Generating sample calibration files...")
            intrinsics_path, extrinsics_path = create_sample_config_files(args.samples_dir)
            print(f"\n✅ Sample files created in: {args.samples_dir}")
            print(f"   - Intrinsics: {intrinsics_path}")
            print(f"   - Extrinsics: {extrinsics_path}")
            
            # Use the generated samples if no files specified
            if not args.intrinsics:
                args.intrinsics = intrinsics_path
            if not args.extrinsics:
                args.extrinsics = extrinsics_path
        
        # Validate required arguments
        if not args.intrinsics or not args.extrinsics:
            logger.error("Missing required calibration files!")
            print("\n❌ Error: Both --intrinsics and --extrinsics files are required")
            print("   Run with --generate-samples to create test files")
            return 1
        
        # Initialize calibration loader
        logger.info(f"Loading calibration for cameras: {args.cameras}")
        loader = CalibrationLoader(
            intrinsics_path=args.intrinsics,
            extrinsics_path=args.extrinsics,
            camera_indices=args.cameras
        )
        
        # Load calibrations
        calibrations = loader.load_calibrations()
        
        # Print summary
        print_calibration_summary(calibrations)
        
        # Optional: Test projection
        if args.test_projection:
            test_projection(calibrations)
        
        # Success message
        print("✅ Calibration loading successful!")
        print("\nNext steps:")
        print("  1. Implement capture.py for multi-camera frame acquisition")
        print("  2. Implement detection.py with fisheye support")
        print("  3. Connect triangulation, filtering, and streaming modules")
        print("  4. Integrate with Unity via WebSocket streaming")
        
        return 0
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"\n❌ Error: {e}")
        return 1
        
    except Exception as e:
        logger.exception("Unexpected error occurred")
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
