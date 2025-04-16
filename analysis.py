import os
import numpy as np
import pickle
from numba import jit  # type: ignore
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
import numpy.typing as npt
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import nibabel as nib
from fnmatch import fnmatch
import bids as bd
import logging
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RBFInterpolator


# Configure logging
def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration with optional file output"""

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)

    # Add file handler if requested
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        logging.info(f"Logging to file: {log_file}")

    return root_logger


# Create module-level logger
logger = logging.getLogger(__name__)


def validate_file_path(file_path: str):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")


class Index:
    def __init__(self, x: int, y: int, z: int):
        self.x = int(x)
        self.y = int(y)
        self.z = int(z)

    def __repr__(self) -> str:
        return f"({self.x},{self.y},{self.z})"

    def __eq__(self, other):
        if not isinstance(other, Index):
            return False
        return self.x == other.x and self.y == other.y and self.z == other.z

    def __hash__(self):
        return hash((self.x, self.y, self.z))


class Grid:
    def __init__(
        self,
        reference_shape: Tuple[int, int, int],
        nth_voxel: int,
        max_voxels: Optional[int] = None,
    ):
        self.reference_shape = reference_shape
        self.nth_voxel = nth_voxel
        self.max_voxels = max_voxels

        # Create x y z coordinates
        x_coords = np.arange(0, reference_shape[0], nth_voxel)
        y_coords = np.arange(0, reference_shape[1], nth_voxel)
        z_coords = np.arange(0, reference_shape[2], nth_voxel)

        # Create grid using meshgrid and stack coordinates
        xx, yy, zz = np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

        indices = list(zip(xx.ravel(), yy.ravel(), zz.ravel()))
        self.indices = [Index(x, y, z) for x, y, z in indices]

        if max_voxels is not None and len(self.indices) > max_voxels:
            raise ValueError(
                f"Number of voxels in grid exceeds max_voxels: {len(self.indices)} > {max_voxels}"
            )

    def get_indices(self) -> List[Index]:
        return self.indices

    def fine_tune_grid(self, similar_voxel_indices: List[Index]):

        new_nth_voxel = max(1, int(np.ceil(self.nth_voxel / 2)))  # Ensure integer step

        # Generate fine grid coordinates around each center voxel
        fine_indices_set = set()  # Use a set for efficient duplicate handling

        for voxel in similar_voxel_indices:
            # Define coordinate ranges symmetrically around the voxel
            # using the new_nth_voxel step size. Add 1 to the end point for np.arange
            # to potentially include the upper bound if it falls on a step.
            x_range = np.arange(
                max(0, voxel.x - new_nth_voxel),
                min(self.reference_shape[0], voxel.x + new_nth_voxel + 1),
                new_nth_voxel,
            )
            y_range = np.arange(
                max(0, voxel.y - new_nth_voxel),
                min(self.reference_shape[1], voxel.y + new_nth_voxel + 1),
                new_nth_voxel,
            )
            z_range = np.arange(
                max(0, voxel.z - new_nth_voxel),
                min(self.reference_shape[2], voxel.z + new_nth_voxel + 1),
                new_nth_voxel,
            )

            # Create grid points within the defined ranges
            # Check if any range is empty before meshing
            if x_range.size > 0 and y_range.size > 0 and z_range.size > 0:
                grid = np.stack(
                    np.meshgrid(x_range, y_range, z_range, indexing="ij"), axis=-1
                )
                # Add valid Index objects to the set
                for point in grid.reshape(-1, 3):
                    # Ensure points are within the original reference shape strictly
                    if (
                        0 <= point[0] < self.reference_shape[0]
                        and 0 <= point[1] < self.reference_shape[1]
                        and 0 <= point[2] < self.reference_shape[2]
                    ):
                        fine_indices_set.add(
                            Index(int(point[0]), int(point[1]), int(point[2]))
                        )

        # Convert the set of unique indices to a list
        unique_indices = list(fine_indices_set)

        # Create and return a new grid with the refined indices and step size
        fine_grid = Grid(self.reference_shape, new_nth_voxel)  # Use new_nth_voxel
        fine_grid.indices = unique_indices

        return fine_grid


class IncludedIndices:
    def __init__(self, indices: List[Index]):
        self.indices = indices

    def reject_index(self, index: Index):
        if index in self.indices:
            self.indices.remove(index)
            return self
        else:
            raise ValueError(f"Index {index} not found in included indices")

    def add_indices(self, indices: List[Index]):
        current_set = set(self.indices)
        new_set = set(indices)
        self.indices.extend(new_set - current_set)
        return self

    def get_indices(self) -> List[Index]:
        return self.indices


class Similarity:
    def __init__(self):
        self.similarities: Dict[Index, np.float32] = {}

    def add_similarity(self, index: Index, similarity: np.float32):
        self.similarities[index] = similarity

    def get_similarity(self, index: Index) -> np.float32:
        return self.similarities[index]

    def add_similarities(
        self, indices: List[Index], similarities: npt.NDArray[np.float32]
    ):
        for index, similarity in zip(indices, similarities):
            self.add_similarity(index, similarity)

    def get_indices(self) -> List[Index]:
        return list(self.similarities.keys())

    def find_extreme_voxels(self, proportion: float = 0.1) -> List[Index]:
        # Calculate number of voxels to return for each extreme based on proportion
        n_voxels = int(len(self.get_indices()) * proportion)
        if n_voxels < 1:
            n_voxels = 1

        # Get sorted indices based on similarity values
        sorted_indices = np.argsort(list(self.similarities.values()))

        # Get indices of the least similar voxels (first n_voxels)
        least_similar_indices = sorted_indices[:n_voxels]
        least_similar_coords = [self.get_indices()[i] for i in least_similar_indices]

        # Get indices of the most similar voxels (last n_voxels)
        extreme_indices = sorted_indices[-n_voxels:]
        most_similar_coords = [self.get_indices()[i] for i in extreme_indices]

        # Combine both lists and return
        return least_similar_coords + most_similar_coords


class MaskScan(bd.Scan):
    def __init__(self, path: str, mask_path: Optional[str] = None):
        super().__init__(path)
        self.image: Union[np.ndarray, None] = None
        self.shape: Union[Tuple, None] = nib.load(path).shape  # type: ignore
        self.mask_path = mask_path

    def load_data(self):
        try:
            logger.debug(f"Loading data from {self.path}")
            self.img = np.asarray(nib.load(self.path).get_fdata()).astype(np.bool_)
            logger.debug(f"Data loaded successfully with shape {self.shape}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.path}: {str(e)}")
            raise

    def unload_data(self):
        self.img = None


class DisplacementScan(bd.Scan):
    """Extension of BIDS Scan that adds displacement data handling capabilities"""

    def __init__(self, path: str):
        super().__init__(path)
        self.image: Union[np.ndarray, None] = None
        self.shape: Union[Tuple, None] = nib.load(path).shape  # type: ignore
        self.displacements: Dict[Index, npt.NDArray[np.float32]] = {}
        logger.debug(f"Initialized DisplacementScan for {path}")

    def __del__(self):
        """Clean up resources when object is deleted."""
        if hasattr(self, "img") and self.img is not None:
            self.img.uncache()
            logger.debug(f"Resources released for {self.path}")

    def _validate_indices(self, indices: List[Index]):
        if self.shape is None:
            raise ValueError("Shape is not set. Call load_data first.")

        invalid_indices = []
        for index in indices:
            if index.x < 0 or index.x >= self.shape[0]:
                invalid_indices.append(f"Index x out of bounds: {index.x}")
            if index.y < 0 or index.y >= self.shape[1]:
                invalid_indices.append(f"Index y out of bounds: {index.y}")
            if index.z < 0 or index.z >= self.shape[2]:
                invalid_indices.append(f"Index z out of bounds: {index.z}")

        if invalid_indices:
            error_msg = "; ".join(invalid_indices)
            logger.error(f"Invalid indices detected: {error_msg}")
            raise ValueError(f"Invalid indices: {error_msg}")

    def load_data(self):
        try:
            logger.debug(f"Loading data from {self.path}")
            self.img = np.asarray(nib.load(self.path).get_fdata()).astype(np.float32)
            logger.debug(f"Data loaded successfully with shape {self.shape}")
        except Exception as e:
            logger.error(f"Failed to load data from {self.path}: {str(e)}")
            raise

    def apply_mask(self, mask: np.ndarray):
        if len(self.img.shape) == 3:
            self.img = self.img * mask
        elif len(self.img.shape) == 4:
            self.img = self.img * mask[:, :, :, np.newaxis]
        elif len(self.img.shape) == 5:
            self.img = self.img * mask[:, :, :, np.newaxis, np.newaxis]
        else:
            raise ValueError(f"Invalid shape: {self.img.shape}")

    def unload_data(self):
        self.img = None

    def sample_voxels(self, included_indices: IncludedIndices):
        logger.debug(
            f"Sampling {len(included_indices.get_indices())} voxels from {self.path}"
        )
        displacements_dict: Dict[Index, npt.NDArray[np.float32]] = {}

        indices_to_sample = included_indices.get_indices()
        indices_to_reject = []

        for index in indices_to_sample:
            try:
                voxel_data = self.img[index.x, index.y, index.z, :, :].astype(
                    np.float32
                )

                if np.all(voxel_data == 0):
                    indices_to_reject.append(index)
                else:
                    displacements_dict[index] = voxel_data
            except Exception as e:
                logger.error(f"Error sampling voxel at {index}: {str(e)}")
                indices_to_reject.append(index)

        logger.debug(
            f"Rejecting {len(indices_to_reject)} voxels with zero displacement"
        )
        for index in indices_to_reject:
            included_indices.reject_index(index)

        self.displacements.update(displacements_dict)
        logger.debug(
            f"Added {len(displacements_dict)} voxels to displacement dictionary"
        )

    def reject_indices(self, included_indices: IncludedIndices):
        included_set = set(included_indices.get_indices())
        current_set = set(self.displacements.keys())
        indices_to_remove = current_set - included_set

        logger.debug(
            f"Rejecting {len(indices_to_remove)} indices from displacement dictionary"
        )
        for index in indices_to_remove:
            del self.displacements[index]

    def get_displacements(self) -> npt.NDArray[np.float32]:
        logger.debug(f"Getting {len(self.displacements)} displacement vectors")
        return np.array(list(self.displacements.values()), dtype=np.float32)

    def get_specific_displacements(
        self, indices: List[Index]
    ) -> npt.NDArray[np.float32]:
        available_indices = [idx for idx in indices if idx in self.displacements]
        logger.debug(
            f"Getting displacement vectors for {len(available_indices)} out of {len(indices)} requested indices"
        )
        return np.array(
            [
                self.displacements[index]
                for index in indices
                if index in self.displacements
            ],
            dtype=np.float32,
        )


class BIDSScanCollection:
    def __init__(
        self,
        bids_instance: bd.BIDS,
        scan_pattern: str = "*",
        mask_pattern: Optional[str] = None,
        subject_filter: Optional[List[str]] = None,
        session_filter: Optional[List[str]] = None,
    ):
        self.bids = bids_instance
        self.scans: List[DisplacementScan] = []
        self.masks: List[MaskScan] = []
        self.similarity = Similarity()
        self.mask: Optional[np.ndarray] = None

        # Load matching scans from the BIDS structure
        self._load_matching_scans(
            scan_pattern, mask_pattern, subject_filter, session_filter
        )

        if not self.scans:
            raise ValueError(f"No matching scans found with pattern '{scan_pattern}'")

        # Get dimensions from the first scan
        self.dims = self._get_unique_dims()

    def _load_matching_scans(
        self,
        scan_pattern: str,
        mask_pattern: Optional[str],
        subject_filter: Optional[List[str]],
        session_filter: Optional[List[str]],
    ) -> None:

        for subject in self.bids.subjects:
            # Skip if subject doesn't match filter
            if subject_filter and subject.subject_id not in subject_filter:
                continue

            for session in subject.sessions:
                # Skip if session doesn't match filter
                if session_filter and session.session_id not in session_filter:
                    continue

                for scan in session.scans:
                    # Check if scan name matches the pattern
                    if mask_pattern and fnmatch(scan.scan_name, mask_pattern):
                        mask_scan = MaskScan(scan.path)
                        self.masks.append(mask_scan)
                        logger.info(
                            f"Added mask scan: {scan.scan_name} from {subject.get_name()}/{session.get_name()}"
                        )
                        continue

                    if fnmatch(scan.scan_name, scan_pattern):
                        try:
                            # Create a DisplacementScan from the regular BIDS Scan
                            displacement_scan = DisplacementScan(scan.path)
                            self.scans.append(displacement_scan)
                            logger.info(
                                f"Added scan: {scan.scan_name} from {subject.get_name()}/{session.get_name()}"
                            )
                        except Exception as e:
                            logger.error(
                                f"Error creating DisplacementScan for {scan.path}: {str(e)}"
                            )

    def _get_unique_dims(self) -> Tuple[int, int, int]:
        unique_dims = set()
        for scan in self.scans:
            if scan.shape is not None:
                unique_dims.add(scan.shape)
            else:
                logger.warning(f"Scan {scan.path} has no shape information")

        if not unique_dims:
            raise ValueError("No valid shape information found in any scan")

        if len(unique_dims) != 1:
            shapes_str = ", ".join(str(shape) for shape in unique_dims)
            logger.error(f"Inconsistent shapes across scans: {shapes_str}")
            raise ValueError("All scans must have the same shape.")

        shape = unique_dims.pop()
        if not isinstance(shape, tuple) or len(shape) < 3:
            raise ValueError(
                f"Invalid shape: {shape}. Expected tuple with at least 3 elements"
            )

        return (int(shape[0]), int(shape[1]), int(shape[2]))

    def _make_grid(self, nth_voxel: int, max_voxels: int) -> Grid:
        return Grid(self.dims, nth_voxel, max_voxels)

    def _validate_scans(self, indices: List[Index]):
        for scan in self.scans:
            scan._validate_indices(indices)

    def construct_grid(self, nth_voxel: int, max_voxels: int) -> Grid:
        self.grid = self._make_grid(nth_voxel, max_voxels)
        self._validate_scans(self.grid.get_indices())
        return self.grid

    def reject_indices(self, included_indices: IncludedIndices):
        for scan in self.scans:
            scan.reject_indices(included_indices)

    def set_mask(self):
        # Initialize an array to accumulate mask values
        mask_sum = np.zeros(self.dims, dtype=np.float32)
        mask_count = 0

        for mask in self.masks:
            mask.load_data()
            if np.all(mask.img == 0):
                logger.warning(f"Mask {mask.path} is empty")
                continue
            # Add mask values to the sum
            mask_sum += mask.img.astype(np.float32)
            mask_count += 1
            mask.unload_data()

        # Calculate mean mask if we have any valid masks
        if mask_count > 0:
            mean_mask = mask_sum / mask_count
            # Return binary mask where values are > 0.5
            self.mask = (mean_mask > 0.5).astype(np.float32)
        else:
            logger.warning("No valid masks found, returning empty mask")
            self.mask = np.zeros(self.dims, dtype=np.float32)

    def process_scans(
        self,
        grid: Grid,
        depth: int = 0,
        max_depth: int = 3,
        mask: Optional[np.ndarray] = None,
        cache_dir: str = ".",
    ):
        logger.info(f"Processing at depth {depth}/{max_depth}")

        # Ensure cache directory exists
        os.makedirs(cache_dir, exist_ok=True)

        scan_file = os.path.join(cache_dir, f"scan_depth_{depth}.pkl")
        indices_file = os.path.join(cache_dir, f"indices_depth_{depth}.pkl")

        # Initialize or add to included_indices based on grid
        if depth == 0:
            self.included_indices = IncludedIndices(grid.get_indices())
        else:
            # If refining, add new grid points to existing included_indices
            self.included_indices.add_indices(grid.get_indices())

        logger.info(
            f"Depth {depth}: {len(self.included_indices.get_indices())} grid points"
        )

        # Load or process the data for this level
        if not os.path.exists(scan_file) or not os.path.exists(indices_file):
            logger.info(f"Processing new data for depth {depth}")

            for scan in self.scans:
                scan.load_data()
                if self.mask is not None:
                    scan.apply_mask(self.mask)
                scan.sample_voxels(self.included_indices)
                scan.unload_data()

            # Save results for this level
            with open(scan_file, "wb") as f:
                pickle.dump(self.scans, f)
            with open(indices_file, "wb") as f:
                pickle.dump(self.included_indices, f)
        else:
            logger.info(f"Loading cached data for depth {depth}")
            try:
                with open(scan_file, "rb") as f:
                    self.scans = pickle.load(f)
                with open(indices_file, "rb") as f:
                    self.included_indices = pickle.load(f)
            except Exception as e:
                logger.error(f"Failed to load cached data: {str(e)}")
                raise

        # Make sure all scans only have valid indices
        self.reject_indices(self.included_indices)

        # Calculate similarities for new indices
        existing_similarity_indices = set(self.similarity.get_indices())
        current_indices = set(self.included_indices.get_indices())
        new_indices = list(current_indices - existing_similarity_indices)

        if new_indices:
            logger.info(f"Calculating similarity for {len(new_indices)} new indices")
            new_similarities = self.calculate_similarity(new_indices)
            self.similarity.add_similarities(new_indices, new_similarities)

        # Find most similar voxels
        self.extreme_indices = self.similarity.find_extreme_voxels(proportion=0.1)
        logger.info(f"Found {len(self.extreme_indices)} most similar voxels")

        # Base case: reached maximum depth
        if depth >= max_depth:
            logger.info(f"Reached maximum depth {max_depth}. Finalizing.")
            return

        # Create a finer grid focused around the most similar voxels
        fine_grid = grid.fine_tune_grid(self.extreme_indices)
        logger.info(
            f"Created fine grid with {len(fine_grid.get_indices())} points for depth {depth + 1}"
        )

        # Add the fine grid indices to the included indices for the next level
        self.included_indices.add_indices(fine_grid.get_indices())

        # Recursive call with increased depth - PASS THE CACHE_DIR PARAMETER
        self.process_scans(fine_grid, depth + 1, max_depth, mask, cache_dir)

    def get_scan_arrays(self) -> List[npt.NDArray[np.float32]]:
        return [scan.get_displacements() for scan in self.scans]

    def get_scan(self, index: int) -> DisplacementScan:
        return self.scans[index]

    def get_specific_displacements_array(
        self, indices: List[Index]
    ) -> List[npt.NDArray[np.float32]]:
        displacements_list = []
        for scan in self.scans:
            displacements = scan.get_specific_displacements(indices)
            displacements_list.append(displacements)
        return displacements_list

    def calculate_similarity(
        self, indices: Optional[List[Index]] = None
    ) -> npt.NDArray[np.float32]:
        # Construct list of displacements arrays for the given indices
        if indices is None:
            indices = self.included_indices.get_indices()

        scans_list = self.get_specific_displacements_array(indices)
        # Calculate similarity matrix for the list of displacements arrays
        similarity_matrix = calculate_similarity_matrix(scans_list)
        # Extract lower triangle (excluding diagonal) of similarity matrix for each voxel
        lower_tri_indices = np.tril_indices(similarity_matrix.shape[0], k=-1)
        voxel_similarities = similarity_matrix[
            lower_tri_indices[0], lower_tri_indices[1], :
        ]
        # Calculate mean similarity for each index across unique subject pairs
        mean_similarities = np.mean(voxel_similarities, axis=0)
        return mean_similarities

    def visualize_indices(self, output_path: str = "final_indices.png"):
        """
        Visualize the included indices in three anatomical planes.

        Args:
            output_path: Path to save the visualization image
        """
        logger.info(
            f"Visualizing {len(self.included_indices.get_indices())} indices..."
        )

        if self.scans is None or len(self.scans) == 0:
            raise ValueError("No scans found. Call load_data first.")

        # Get the dimensions from the first scan to ensure correct shape
        img_shape = self.scans[0].shape[:3]  # type: ignore
        logger.info(f"Creating visualization volume with shape {img_shape}")

        # Create a binary volume with 1s at the included indices
        volume = np.zeros(img_shape, dtype=np.float32)

        # Set voxels to 1 where indices exist
        indices = self.included_indices.get_indices()
        for index in indices:
            # Make sure indices are within bounds
            if (
                0 <= index.x < img_shape[0]
                and 0 <= index.y < img_shape[1]
                and 0 <= index.z < img_shape[2]
            ):
                volume[index.x, index.y, index.z] = 1.0

        logger.info(f"Created binary volume with {np.sum(volume)} active voxels")

        # Create a figure with 3 subplots for three planes
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Get the middle slices for each plane
        mid_x = img_shape[0] // 2
        mid_y = img_shape[1] // 2
        mid_z = img_shape[2] // 2

        # Create maximum intensity projections (MIPs) along each axis
        mip_z = np.mean(volume, axis=2)  # Axial MIP
        mip_y = np.mean(volume, axis=1)  # Coronal MIP
        mip_x = np.mean(volume, axis=0)  # Sagittal MIP

        # Plot axial (transverse) view
        axes[0].imshow(mip_z, cmap="viridis", origin="lower")
        axes[0].set_title(f"Axial View (MIP)")
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("X")

        # Plot coronal view
        axes[1].imshow(mip_y, cmap="viridis", origin="lower")
        axes[1].set_title(f"Coronal View (MIP)")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")

        # Plot sagittal view
        axes[2].imshow(mip_x, cmap="viridis", origin="lower")
        axes[2].set_title(f"Sagittal View (MIP)")
        axes[2].set_xlabel("Z")
        axes[2].set_ylabel("Y")

        # Add a colorbar
        cbar = fig.colorbar(
            axes[0].images[0], ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
        )
        cbar.set_label("Voxel Presence")

        # Add information about the number of indices
        plt.suptitle(f"Visualization of {len(indices)} Selected Indices", fontsize=16)

        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualization saved to {output_path}")

        # Additionally, create separate slices for each plane
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))

        # Calculate slice positions (middle, middle±quarter)
        x_slices = [mid_x - mid_x // 2, mid_x, mid_x + mid_x // 2]
        y_slices = [mid_y - mid_y // 2, mid_y, mid_y + mid_y // 2]
        z_slices = [mid_z - mid_z // 2, mid_z, mid_z + mid_z // 2]

        # Plot axial slices (top row)
        for i, z in enumerate(z_slices):
            if z < img_shape[2]:
                axes[0, i].imshow(volume[:, :, z], cmap="viridis", origin="lower")
                axes[0, i].set_title(f"Axial Slice z={z}")
                axes[0, i].set_xlabel("Y")
                axes[0, i].set_ylabel("X")

        # Plot coronal slices (middle row)
        for i, y in enumerate(y_slices):
            if y < img_shape[1]:
                axes[1, i].imshow(volume[:, y, :], cmap="viridis", origin="lower")
                axes[1, i].set_title(f"Coronal Slice y={y}")
                axes[1, i].set_xlabel("Z")
                axes[1, i].set_ylabel("X")

        # Plot sagittal slices (bottom row)
        for i, x in enumerate(x_slices):
            if x < img_shape[0]:
                axes[2, i].imshow(volume[x, :, :], cmap="viridis", origin="lower")
                axes[2, i].set_title(f"Sagittal Slice x={x}")
                axes[2, i].set_xlabel("Z")
                axes[2, i].set_ylabel("Y")

        plt.suptitle(f"Detailed Slices of {len(indices)} Selected Indices", fontsize=16)

        # Save detailed figure
        plt.tight_layout()
        plt.savefig(
            output_path.replace(".png", "_detailed.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

        logger.info(
            f"Detailed visualization saved to {output_path.replace('.png', '_detailed.png')}"
        )

    def visualize_extreme_indices(self, output_path: str = "extreme_indices.png"):
        """
        Visualize the MOST SIMILAR included indices from the final round
        in three anatomical planes.

        Args:
            output_path: Path to save the visualization image
        """
        if not hasattr(self, "extreme_indices") or not self.extreme_indices:
            logger.info(
                "No 'extreme_indices' found. Run process_scans to completion first."
            )
            return

        num_similar = len(self.extreme_indices)
        logger.info(f"Visualizing {num_similar} MOST SIMILAR indices...")

        # Get the dimensions from the first scan
        img_shape = self.scans[0].shape[:3]
        logger.info(f"Creating visualization volume with shape {img_shape}")

        # Create a binary volume with 1s ONLY at the most similar indices
        volume = np.zeros(img_shape, dtype=np.float32)

        for index in self.extreme_indices:
            if (
                0 <= index.x < img_shape[0]
                and 0 <= index.y < img_shape[1]
                and 0 <= index.z < img_shape[2]
            ):
                volume[index.x, index.y, index.z] = 1.0

        logger.info(
            f"Created binary volume with {np.sum(volume)} active 'most similar' voxels"
        )

        # --- Plotting logic (similar to visualize_indices) ---

        # Create a figure with 3 subplots for MIPs
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        mip_z = np.max(volume, axis=2)
        mip_y = np.max(volume, axis=1)
        mip_x = np.max(volume, axis=0)

        # Plot MIPs
        im0 = axes[0].imshow(mip_z.T, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[0].set_title("Axial View (MIP)")
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("X")
        im1 = axes[1].imshow(mip_y.T, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[1].set_title("Coronal View (MIP)")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")
        im2 = axes[2].imshow(mip_x.T, cmap="hot", origin="lower", vmin=0, vmax=1)
        axes[2].set_title("Sagittal View (MIP)")
        axes[2].set_xlabel("Z")
        axes[2].set_ylabel("Y")

        # Add a simple colorbar (just indicates presence/absence)
        cbar = fig.colorbar(
            im0, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, ticks=[0, 1]
        )
        cbar.set_ticklabels(["Absent", "Present"])
        cbar.set_label("Most Similar Voxel Presence")

        plt.suptitle(
            f"Visualization of {num_similar} MOST SIMILAR Indices (Final Round)",
            fontsize=16,
        )
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # type:ignore
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Most similar indices visualization saved to {output_path}")

        # --- Detailed Slices (Optional, but recommended) ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        mid_x, mid_y, mid_z = img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2
        x_slices = [mid_x - mid_x // 2, mid_x, mid_x + mid_x // 2]
        y_slices = [mid_y - mid_y // 2, mid_y, mid_y + mid_y // 2]
        z_slices = [mid_z - mid_z // 2, mid_z, mid_z + mid_z // 2]

        # Plot axial slices
        for i, z in enumerate(z_slices):
            if 0 <= z < img_shape[2]:
                axes[0, i].imshow(
                    volume[:, :, z].T, cmap="hot", origin="lower", vmin=0, vmax=1
                )
                axes[0, i].set_title(f"Axial Slice z={z}")
                axes[0, i].set_xlabel("Y")
                axes[0, i].set_ylabel("X")
        # Plot coronal slices
        for i, y in enumerate(y_slices):
            if 0 <= y < img_shape[1]:
                axes[1, i].imshow(
                    volume[:, y, :].T, cmap="hot", origin="lower", vmin=0, vmax=1
                )
                axes[1, i].set_title(f"Coronal Slice y={y}")
                axes[1, i].set_xlabel("Z")
                axes[1, i].set_ylabel("X")
        # Plot sagittal slices
        for i, x in enumerate(x_slices):
            if 0 <= x < img_shape[0]:
                axes[2, i].imshow(
                    volume[x, :, :].T, cmap="hot", origin="lower", vmin=0, vmax=1
                )
                axes[2, i].set_title(f"Sagittal Slice x={x}")
                axes[2, i].set_xlabel("Z")
                axes[2, i].set_ylabel("Y")

        plt.suptitle(
            f"Detailed Slices of {num_similar} MOST SIMILAR Indices", fontsize=16
        )
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # type: ignore
        plt.savefig(
            output_path.replace(".png", "_detailed.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info(
            f"Detailed most similar indices visualization saved to {output_path.replace('.png', '_detailed.png')}"
        )

    def visualize_similarity_image(self, output_path: str = "similarity_image.png"):
        similarity_volume = self.calculate_similarity_image()
        img_shape = similarity_volume.shape[:3]
        logger.info(f"Creating similarity visualization volume with shape {img_shape}")

        # Create a figure with 3 subplots for three planes
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Create maximum intensity projections (MIPs) along each axis
        mip_z = np.mean(similarity_volume, axis=2)  # Axial MIP
        mip_y = np.mean(similarity_volume, axis=1)  # Coronal MIP
        mip_x = np.mean(similarity_volume, axis=0)  # Sagittal MIP

        # Plot axial (transverse) view
        im = axes[0].imshow(mip_z, cmap="hot", origin="lower")
        axes[0].set_title(f"Axial View (MIP)")
        axes[0].set_xlabel("Y")
        axes[0].set_ylabel("X")

        # Plot coronal view
        axes[1].imshow(mip_y, cmap="hot", origin="lower")
        axes[1].set_title(f"Coronal View (MIP)")
        axes[1].set_xlabel("Z")
        axes[1].set_ylabel("X")

        # Plot sagittal view
        axes[2].imshow(mip_x, cmap="hot", origin="lower")
        axes[2].set_title(f"Sagittal View (MIP)")
        axes[2].set_xlabel("Z")
        axes[2].set_ylabel("Y")

        # Add a single colorbar
        cbar = fig.colorbar(
            im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1
        )
        cbar.set_label("Similarity Score")

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        logger.info(f"Similarity image visualization saved to {output_path}")

    def calculate_similarity_image(
        self,
        scans: Optional[List[DisplacementScan]] = None,
        indices: Optional[IncludedIndices] = None,
    ) -> np.ndarray:
        # Create empty similarity volume with the same dimensions as our reference
        similarity_volume = np.zeros(self.dims, dtype=np.float32)

        # Use provided scans or default to self.scans
        if scans is None:
            scans = self.scans

        # Get all unique indices from all scans
        if indices is None:
            indices = self.included_indices

        # Convert to list for similarity calculation
        all_indices_list = list(indices.get_indices())

        logger.info(f"Calculating similarity image for {len(all_indices_list)} voxels")

        # Calculate similarity using the existing method
        similarities = self.calculate_similarity(all_indices_list)

        # Map similarities back to the 3D volume
        for idx, index in enumerate(all_indices_list):
            similarity_volume[index.x, index.y, index.z] = similarities[idx]

        logger.info("Similarity image calculation complete")
        return similarity_volume


@jit(nopython=True)
def cosine_similarity_3d_timeseries(vec1: np.ndarray, vec2: np.ndarray) -> np.float32:
    # Dot product across all timepoints
    dot_product = np.sum(
        vec1[0, :] * vec2[0, :] + vec1[1, :] * vec2[1, :] + vec1[2, :] * vec2[2, :]
    )

    # Magnitudes
    mag1 = np.sqrt(np.sum(vec1[0, :] ** 2 + vec1[1, :] ** 2 + vec1[2, :] ** 2))
    mag2 = np.sqrt(np.sum(vec2[0, :] ** 2 + vec2[1, :] ** 2 + vec2[2, :] ** 2))

    return dot_product / (mag1 * mag2)


@jit(nopython=True)
def compare_2_subjects(
    subject1_displacements: npt.NDArray[np.float32],
    subject2_displacements: npt.NDArray[np.float32],
    n_voxels: int,
) -> npt.NDArray[np.float32]:
    """Compare all voxels between two subjects"""
    similarities: npt.NDArray[np.float32] = np.zeros(n_voxels, dtype=np.float32)

    for i in range(n_voxels):
        similarity = cosine_similarity_3d_timeseries(
            subject1_displacements[i, :, :], subject2_displacements[i, :, :]
        )
        similarities[i] = similarity
    return similarities


@jit(nopython=True)
def calculate_similarity_matrix(
    scans: List[npt.NDArray[np.float32]],
) -> npt.NDArray[np.float32]:
    num_voxels = len(scans[0])
    similarities: npt.NDArray[np.float32] = np.zeros(
        (len(scans), len(scans), num_voxels), dtype=np.float32
    )

    for i, subject1 in enumerate(scans):
        for j, subject2 in enumerate(scans):
            similarities[i, j, :] = compare_2_subjects(subject1, subject2, num_voxels)

    return similarities


def interpolate_volume(volume, mask=None):
    logger.info(f"Interpolating volume with shape {volume.shape} using RBFInterpolator")

    # Find non-zero voxels (valid data points)
    non_zero_coords = np.argwhere(volume > 0)
    values = volume[volume > 0]

    # Build interpolator using only non-zero points
    rbf = RBFInterpolator(non_zero_coords, values, neighbors=20, smoothing=0.1, degree=0)

    # Interpolate the entire volume grid
    grid_coords = (
        np.array(
            np.meshgrid(
                np.arange(volume.shape[0]),
                np.arange(volume.shape[1]),
                np.arange(volume.shape[2]),
                indexing="ij",
            )
        )
        .reshape(3, -1)
        .T
    )

    interpolated_flat = rbf(grid_coords)
    interpolated_volume = interpolated_flat.reshape(volume.shape)

    # Smooth the result to reduce artifacts
    smoothed_volume = gaussian_filter(interpolated_volume, sigma=1.0)

    # Apply mask if provided
    if mask is not None:
        smoothed_volume *= mask

    logger.info("Interpolation complete")
    return smoothed_volume


def save_similarity_nifti(self, output_path: str) -> None:
    """
    Generate and save a full-resolution similarity map as a NIfTI file.

    Args:
        output_path: Path where to save the NIfTI file
    """
    interpolated_volume = self.interpolate_similarity_volume()

    # Save as NIfTI
    import nibabel as nib

    logger.info(f"Saving interpolated similarity map to {output_path}")

    # Create NIfTI image from similarity volume using a default affine
    affine = np.eye(4)  # Create identity matrix as default affine
    nifti_img = nib.Nifti1Image(interpolated_volume, affine)
    nib.save(nifti_img, output_path)

    logger.info(f"Saved similarity map to {output_path}")


def visualize_volume(
    volume: np.ndarray,
    output_path: str,
    colormap: str = "viridis",
    title: str = "Volume Visualization",
):
    """Static helper method to visualize a 3D volume with orthogonal slices

    Args:
        volume: 3D numpy array containing the volume data
        output_path: Path to save the visualization image
        colormap: Matplotlib colormap name (default: 'viridis')
        title: Title for the visualization (default: 'Volume Visualization')
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    if not isinstance(volume, np.ndarray) or volume.ndim != 3:
        raise ValueError("Input volume must be a 3D numpy array")

    img_shape = volume.shape
    logger.info(f"Visualizing volume with shape {img_shape}")

    # Create figure with 3 subplots for three planes
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(title, fontsize=16)

    # Get middle slices
    mid_x, mid_y, mid_z = [dim // 2 for dim in img_shape]

    # Calculate colormap range based on non-zero values
    non_zero = volume[volume > 0]
    vmin, vmax = np.percentile(non_zero, [5, 95]) if len(non_zero) > 0 else (0, 1)
    cmap = cm.get_cmap(colormap)

    # Plot slices with consistent colormap range
    slices = [
        (volume[:, :, mid_z], "Axial Slice", "X", "Y", f"z={mid_z}"),
        (volume[:, mid_y, :], "Coronal Slice", "X", "Z", f"y={mid_y}"),
        (volume[mid_x, :, :], "Sagittal Slice", "Y", "Z", f"x={mid_x}"),
    ]

    for ax, (data, slice_type, xlabel, ylabel, pos) in zip(axes, slices):
        data = np.rot90(data, 3)
        im = ax.imshow(data, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(f"{slice_type} ({pos})")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1)
    cbar.set_label("Value")

    # Save and close figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Volume visualization saved to {output_path}")


def create_detailed_volume_view(
    volume: np.ndarray,
    output_path: str,
    colormap: str = "viridis",
    title: str = "Detailed Volume View",
):
    """Create a more detailed view with multiple slices along each dimension"""
    import matplotlib.pyplot as plt

    img_shape = volume.shape

    # Create a 3x3 grid of slices
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))

    # Calculate slice positions (25%, 50%, 75% through the volume)
    slice_positions = [
        [int(img_shape[0] * 0.25), int(img_shape[0] * 0.5), int(img_shape[0] * 0.75)],
        [int(img_shape[1] * 0.25), int(img_shape[1] * 0.5), int(img_shape[1] * 0.75)],
        [int(img_shape[2] * 0.25), int(img_shape[2] * 0.5), int(img_shape[2] * 0.75)],
    ]

    # Get color scaling for consistency
    non_zero = volume[volume > 0]
    if len(non_zero) > 0:
        vmin, vmax = np.percentile(non_zero, [5, 95])
    else:
        vmin, vmax = 0, 1
    cmap = plt.cm.get_cmap(colormap)

    # Plot axial slices (top row)
    for i, z in enumerate(slice_positions[2]):
        axes[0, i].imshow(
            volume[:, :, z], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        axes[0, i].set_title(f"Axial (z={z})")

    # Plot coronal slices (middle row)
    for i, y in enumerate(slice_positions[1]):
        axes[1, i].imshow(
            volume[:, y, :], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        axes[1, i].set_title(f"Coronal (y={y})")

    # Plot sagittal slices (bottom row)
    for i, x in enumerate(slice_positions[0]):
        axes[2, i].imshow(
            volume[x, :, :], cmap=cmap, origin="lower", vmin=vmin, vmax=vmax
        )
        axes[2, i].set_title(f"Sagittal (x={x})")

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Detailed volume visualization saved to {output_path}")
