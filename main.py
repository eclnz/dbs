import logging
from datetime import datetime
from analysis import setup_logging, logger
import bids as bd
from analysis import BIDSScanCollection
import analysis
import pickle
import matplotlib.pyplot as plt
import os
import numpy as np

if __name__ == "__main__":
    # Create timestamp for log file
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = f"logs/analysis-{timestamp}.log"

    # Setup logging with DEBUG level instead of INFO
    setup_logging(log_level=logging.DEBUG, log_file=log_file)
    logger.info("Starting analysis with DEBUG logging")

    NTH_VOXEL_SAMPLING = 16
    MAX_VOXELS_PER_SUBJECT = 200_000
    DEPTH = 4

    # Load the BIDS dataset
    bids_path = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4"
    logger.info(f"Loading BIDS dataset from {bids_path}")
    bids = bd.BIDS(bids_path)

    logger.info("Creating scan collection from BIDS dataset")

    # Create a scan collection from the BIDS dataset
    # Filter for displacement maps in the derivatives folder
    scans = BIDSScanCollection(
        bids_instance=bids,
        scan_pattern="*MNI152_motion",
        mask_pattern="*MNI152_brain_mask",
        subject_filter=None,
        session_filter=None,
    )

    logger.info(f"Successfully loaded {len(scans.scans)} scans")

    # Proceed with grid construction and processing
    grid = scans.construct_grid(NTH_VOXEL_SAMPLING, MAX_VOXELS_PER_SUBJECT)
    scans.set_mask()
    scans.process_scans(grid, max_depth=DEPTH)

    # Visualize results
    logger.info("Generating visualizations")
    scans.visualize_indices()
    scans.visualize_extreme_indices()

    # Calculate the similarity image (sparse representation)
    logger.info("Calculating similarity image")
    similarity_volume = scans.calculate_similarity_image()
    scans.visualize_similarity_image("sparse_similarity_image.png")

    # Interpolate the similarity volume to get a smooth, full-resolution map
    logger.info("Interpolating similarity volume")
    interpolated_similarity = analysis.interpolate_volume(similarity_volume, scans.mask)

    # Visualize the interpolated similarity volume
    logger.info("Visualizing interpolated similarity volume")
    analysis.visualize_volume(
        interpolated_similarity,
        "interpolated_similarity.png",
        colormap="hot",
        title="Interpolated Similarity Map",
    )

    # Create a detailed multi-slice view
    analysis.create_detailed_volume_view(
        interpolated_similarity,
        "interpolated_similarity_detailed.png",
        colormap="hot",
        title="Detailed Interpolated Similarity Map",
    )
    # Create a blank affine matrix (identity matrix)
    affine = np.eye(4)
    # Save as NIfTI for potential further analysis
    try:
        import nibabel as nib

        logger.info("Saving interpolated similarity map as NIfTI")
        nifti_img = nib.Nifti1Image(interpolated_similarity, affine)
        nib.save(nifti_img, "interpolated_similarity.nii.gz")
        logger.info(
            "Saved interpolated similarity map to interpolated_similarity.nii.gz"
        )
    except Exception as e:
        logger.error(f"Failed to save NIfTI: {str(e)}")

    logger.info("Analysis completed successfully")
