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

    try:
        # Load the BIDS dataset
        bids_path = "/Users/edwardclarkson/git/qaMRI-clone/testData/BIDS4"
        logger.info(f"Loading BIDS dataset from {bids_path}")
        bids = bd.BIDS(bids_path)
        
        logger.info("Creating scan collection from BIDS dataset")
        
        # Create a scan collection from the BIDS dataset
        # Filter for displacement maps in the derivatives folder
        scans = BIDSScanCollection(
            bids_instance=bids,
            scan_pattern="*MNI152_motion",  # Pattern to match displacement maps
            subject_filter=None,  # All subjects (or specify like ["01", "02"])
            session_filter=None,  # All sessions (or specify like ["pre", "post"])
        )
        
        logger.info(f"Successfully loaded {len(scans.scans)} scans")
        
        # Proceed with grid construction and processing
        grid = scans.construct_grid(NTH_VOXEL_SAMPLING, MAX_VOXELS_PER_SUBJECT)
        scans.process_scans(grid, max_depth=DEPTH)
        
        # Visualize results
        logger.info("Generating visualizations")
        scans.visualize_indices()
        scans.visualize_most_similar_indices()
        
        logger.info("Analysis completed successfully")
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}", exc_info=True)
        raise
