import logging
from datetime import datetime
from analysis import setup_logging, logger
import bids as bd
from analysis import BIDSScanCollection

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
    mask = scans.compute_mask()
    import matplotlib.pyplot as plt
    import os
    
    # Create output directory if it doesn't exist
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure with 3 subplots for axial, sagittal and coronal views
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Calculate middle slices for each view
    mid_axial = mask.shape[2] // 2
    mid_sagittal = mask.shape[0] // 2
    mid_coronal = mask.shape[1] // 2
    
    # Plot axial slice
    axes[0].imshow(mask[:, :, mid_axial], cmap='gray', origin='lower')
    axes[0].set_title('Axial View')
    axes[0].axis('off')
    
    # Plot sagittal slice
    axes[1].imshow(mask[mid_sagittal, :, :], cmap='gray', origin='lower')
    axes[1].set_title('Sagittal View')
    axes[1].axis('off')
    
    # Plot coronal slice
    axes[2].imshow(mask[:, mid_coronal, :], cmap='gray', origin='lower')
    axes[2].set_title('Coronal View')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, f"mask_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    # scans.process_scans(grid, max_depth=DEPTH)
    
    # # Visualize results
    # logger.info("Generating visualizations")
    # scans.visualize_indices()
    # scans.visualize_most_similar_indices()
    
    # logger.info("Analysis completed successfully")
