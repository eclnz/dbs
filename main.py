import logging
import argparse
from datetime import datetime
from analysis import setup_logging, logger
import nibabel as nib
import bids as bd
from analysis import BIDSScanCollection
import analysis
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process BIDS dataset for similarity analysis"
    )
    parser.add_argument(
        "--bids_path", "-b", type=str, required=True, help="Path to BIDS dataset"
    )
    parser.add_argument(
        "--cache_dir",
        "-c",
        type=str,
        default="cache",
        help="Directory to store cached data",
    )
    parser.add_argument(
        "--nth_voxel", type=int, default=16, help="Nth voxel sampling rate"
    )
    parser.add_argument(
        "--max_voxels", type=int, default=200000, help="Maximum voxels per subject"
    )
    parser.add_argument("--depth", type=int, default=4, help="Maximum recursion depth")
    parser.add_argument(
        "--scan_pattern",
        type=str,
        default="*MNI152_motion",
        help="Pattern to match scan names",
    )
    parser.add_argument(
        "--mask_pattern",
        type=str,
        default="*MNI152_brain_mask",
        help="Pattern to match mask names",
    )
    parser.add_argument(
        "--affine_pattern", "-ap",
        type=str,
        default="*matrix",
        help="Pattern to match affine names",
    )
    parser.add_argument(
        "--template",
        type=str,
        help="Path to MNI152 template providing affine matrix for saved images"
    )
    args = parser.parse_args()
    
    # Validate bids path
    if not os.path.exists(args.bids_path):
        parser.error(f"BIDS path {args.bids_path} does not exist")

    # Validate that template is provided if affine_pattern is specified
    if args.affine_pattern and not args.template:
        parser.error("--template is required when --affine_pattern is specified")
    if args.affine_pattern and not os.path.exists(args.template):
        parser.error(f"Template file {args.template} does not exist")

    # Create cache directory if it doesn't exist
    os.makedirs(args.cache_dir, exist_ok=True)

    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)

    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join("logs", f"analysis-{timestamp}.log")

    # Setup logging with DEBUG level instead of INFO
    setup_logging(log_level=logging.INFO, log_file=log_file)
    logger.info(f"Starting analysis with INFO logging")
    logger.info(f"BIDS path: {args.bids_path}")
    logger.info(f"Cache directory: {args.cache_dir}")

    # Load the BIDS dataset
    logger.info(f"Loading BIDS dataset from {args.bids_path}")
    bids = bd.BIDS(args.bids_path)

    logger.info("Creating scan collection from BIDS dataset")

    # Create a scan collection from the BIDS dataset
    scans = BIDSScanCollection(
        bids_instance=bids,
        scan_pattern=args.scan_pattern,
        mask_pattern=args.mask_pattern,
        affine_pattern=args.affine_pattern,
        subject_filter=None,
        session_filter=None,
    )

    logger.info(f"Successfully loaded {len(scans.scans)} scans")

    # Proceed with grid construction and processing
    grid = scans.construct_grid(args.nth_voxel, args.max_voxels)
    scans.set_mask()

    # Visualize mask
    if scans.mask is not None:
        analysis.visualize_volume(
            scans.mask, os.path.join(args.cache_dir, "mask.png"), title="Mask"
        )

    # Process scans
    scans.process_scans(grid, max_depth=args.depth, cache_dir=args.cache_dir)

    # Visualize results
    logger.info("Generating visualizations")
    scans.visualize_indices(os.path.join(args.cache_dir, "final_indices.png"))
    scans.visualize_extreme_indices(os.path.join(args.cache_dir, "extreme_indices.png"))

    # Calculate the similarity image (sparse representation)
    logger.info("Calculating similarity image")
    similarity_volume = scans.calculate_similarity_image()
    scans.visualize_similarity_image(
        os.path.join(args.cache_dir, "sparse_similarity_image.png")
    )

    # Interpolate the similarity volume to get a smooth, full-resolution map
    logger.info("Interpolating similarity volume")
    interpolated_similarity = analysis.interpolate_volume(similarity_volume, scans.mask)

    # Visualize the interpolated similarity volume
    logger.info("Visualizing interpolated similarity volume")
    analysis.visualize_volume(
        interpolated_similarity,
        os.path.join(args.cache_dir, "interpolated_similarity.png"),
        colormap="hot",
        title="Interpolated Similarity Map",
    )

    # Create a detailed multi-slice view
    analysis.create_detailed_volume_view(
        interpolated_similarity,
        os.path.join(args.cache_dir, "interpolated_similarity_detailed.png"),
        colormap="hot",
        title="Detailed Interpolated Similarity Map",
    )

    # Save as NIfTI for potential further analysis
    try:
        logger.info("Saving interpolated similarity map as NIfTI")
        if args.template:
            try:
                # Load the template to get its affine matrix
                template_img = nib.load(args.template)
                affine = template_img.affine
                logger.info(f"Using affine matrix from template: {args.template}")
            except Exception as e:
                logger.error(f"Failed to load template: {str(e)}")
                affine = np.eye(4)
                logger.warning("Failed to load template, using identity matrix as affine")
        else:
            # Fallback to identity matrix if no template is provided
            affine = np.eye(4)
            logger.warning("No template provided, using identity matrix as affine")
            
        nifti_img = nib.Nifti1Image(interpolated_similarity, affine)
        nib.save(
            nifti_img, os.path.join(args.cache_dir, "interpolated_similarity.nii.gz")
        )
        logger.info(
            f"Saved interpolated similarity map to {os.path.join(args.cache_dir, 'interpolated_similarity.nii.gz')}"
        )
    except Exception as e:
        logger.error(f"Failed to save NIfTI: {str(e)}")

    logger.info("Analysis completed successfully")


if __name__ == "__main__":
    main()
