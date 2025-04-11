import os
import numpy as np
import pickle
from numba import jit
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy.typing as npt
import matplotlib.pyplot as plt
import nibabel as nib


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
        
        new_nth_voxel = max(1, int(np.ceil(self.nth_voxel / 2))) # Ensure integer step
        
        # Generate fine grid coordinates around each center voxel
        fine_indices_set = set() # Use a set for efficient duplicate handling
        
        for voxel in similar_voxel_indices:
            # Define coordinate ranges symmetrically around the voxel
            # using the new_nth_voxel step size. Add 1 to the end point for np.arange
            # to potentially include the upper bound if it falls on a step.
            x_range = np.arange(
                max(0, voxel.x - new_nth_voxel),
                min(self.reference_shape[0], voxel.x + new_nth_voxel + 1),
                new_nth_voxel
            )
            y_range = np.arange(
                max(0, voxel.y - new_nth_voxel),
                min(self.reference_shape[1], voxel.y + new_nth_voxel + 1),
                new_nth_voxel
            )
            z_range = np.arange(
                max(0, voxel.z - new_nth_voxel),
                min(self.reference_shape[2], voxel.z + new_nth_voxel + 1),
                new_nth_voxel
            )

            # Create grid points within the defined ranges
            # Check if any range is empty before meshing
            if x_range.size > 0 and y_range.size > 0 and z_range.size > 0:
                grid = np.stack(np.meshgrid(x_range, y_range, z_range, indexing='ij'), axis=-1)
                # Add valid Index objects to the set
                for point in grid.reshape(-1, 3):
                    # Ensure points are within the original reference shape strictly
                    if (0 <= point[0] < self.reference_shape[0] and
                        0 <= point[1] < self.reference_shape[1] and
                        0 <= point[2] < self.reference_shape[2]):
                         fine_indices_set.add(Index(int(point[0]), int(point[1]), int(point[2])))
        
        # Convert the set of unique indices to a list
        unique_indices = list(fine_indices_set)
        
        # Create and return a new grid with the refined indices and step size
        fine_grid = Grid(self.reference_shape, new_nth_voxel) # Use new_nth_voxel
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


@dataclass
class VoxelDisplacement:
    """Class to hold displacement data for a single voxel"""

    voxel_coords: Index
    displacements: npt.NDArray[np.float32]  # Shape: (3, timepoints)

    def __init__(
        self,
        voxel_coords: Index,
        x: npt.NDArray[np.float32],
        y: npt.NDArray[np.float32],
        z: npt.NDArray[np.float32],
    ):
        self.voxel_coords = voxel_coords
        # Stack x, y, z into a single array
        self.displacements = np.stack((x, y, z), axis=0)


class Scan:
    def __init__(self, file_path: str):
        self.scan_id = file_path.split("/")[-1].split(".")[0]
        self.file_path = file_path
        validate_file_path(file_path)
        self.img = nib.load(self.file_path, mmap=True)
        self.shape = self.img.shape #type: ignore
        self.displacements: Dict[Index, npt.NDArray[np.float32]] = {}

    def __del__(self):
        """Clean up resources when object is deleted."""
        if hasattr(self, "img") and self.img is not None:
            if hasattr(self.img, "uncache"):
                self.img.uncache()

    def _load_fdata(self) -> npt.NDArray[np.float32]:
        return self.img.get_fdata().astype(np.float32) #type:ignore

    def _unload_fdata(self):
        if hasattr(self.img, "uncache"):
            self.img.uncache()

    def _validate_indices(self, indices: List[Index]):
        for index in indices:
            if index.x < 0 or index.x >= self.shape[0]:
                raise ValueError(f"Index x out of bounds: {index.x}")
            if index.y < 0 or index.y >= self.shape[1]:
                raise ValueError(f"Index y out of bounds: {index.y}")
            if index.z < 0 or index.z >= self.shape[2]:
                raise ValueError(f"Index z out of bounds: {index.z}")

    def sample_voxels(self, included_indices: IncludedIndices):
        displacements_dict: Dict[Index, npt.NDArray[np.float32]] = {} 
        full_volume = self._load_fdata()

        indices_to_sample = included_indices.get_indices()
        indices_to_reject = []

        for index in indices_to_sample:
            voxel_data = full_volume[index.x, index.y, index.z, :, :].astype(np.float32)

            if np.all(voxel_data == 0):
                indices_to_reject.append(index)
            else:
                displacements_dict[index] = voxel_data

        for index in indices_to_reject:
            included_indices.reject_index(index)

        self.displacements.update(displacements_dict)
        self._unload_fdata()
        
    def reject_indices(self, included_indices: IncludedIndices):
        included_set = set(included_indices.get_indices())
        current_set = set(self.displacements.keys())
        
        for index in current_set - included_set:
            del self.displacements[index]

    def get_displacements(self) -> npt.NDArray[np.float32]:
        return np.array(list(self.displacements.values()), dtype=np.float32)


class ScanCollection:
    def __init__(self, scan_paths: List[str]):
        self.scans = [Scan(path) for path in scan_paths]
        self.dims = self._get_unique_dims()

    def _get_unique_dims(self) -> Tuple[int, int, int]:
        unique_dims = set()
        for scan in self.scans:
            unique_dims.add(scan.shape)
        if len(unique_dims) != 1:
            raise ValueError("All scans must have the same shape.")
        return unique_dims.pop()

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

    def process_scans(self, grid: Grid, depth: int = 0, max_depth: int = 3):
        """Process scans with increasingly fine grid resolutions.
        
        Args:
            grid: The current grid to use for sampling
            depth: Current recursion depth (default: 0)
            max_depth: Maximum recursion depth (default: 3)
        """
        scan_file = f"scan_depth_{depth}.pkl"
        indices_file = f"indices_depth_{depth}.pkl"
        
        # Initialize or add to included_indices based on grid
        if depth == 0:
            self.included_indices = IncludedIndices(grid.get_indices())
        else:
            # If refining, add new grid points to existing included_indices
            self.included_indices.add_indices(grid.get_indices()) 
        
        print(f"Processing depth {depth} with {len(self.included_indices.get_indices())} potential grid points")
        
        # Load or process the data for this level
        if not os.path.exists(scan_file) or not os.path.exists(indices_file):
            for scan in self.scans:
                scan.sample_voxels(self.included_indices)
        
            # Save results for this level
            with open(scan_file, "wb") as f:
                pickle.dump(self.scans, f)
            with open(indices_file, "wb") as f:
                pickle.dump(self.included_indices, f)
        else:
            with open(scan_file, "rb") as f:
                self.scans = pickle.load(f)
            with open(indices_file, "rb") as f:
                self.included_indices = pickle.load(f)
        
        # Make sure all scans only have valid indices
        self.reject_indices(self.included_indices)
        
        # Calculate similarity and find most similar voxels for this level
        analyzer = DisplacementAnalyzer(self.scans, self.included_indices)
        similar_voxel_indices = analyzer.find_extreme_voxels(0.25) # Proportion used for refinement
        
        # Base case: reached maximum depth
        if depth >= max_depth:
            print(f"Reached maximum depth {max_depth}. Finalizing.")
            # Store the final most similar indices (e.g., top 10 or a fixed proportion)
            self.most_similar_indices = analyzer.find_extreme_voxels(proportion=0.1) # Example: Store top 10%
            print(f"Stored {len(self.most_similar_indices)} most similar indices for final visualization.")
            return
        
        # --- Recursive step ---
        print(f"Found {len(similar_voxel_indices)} similar voxels for fine-tuning at depth {depth}")
        
        # Create a finer grid focused around the most similar voxels
        fine_grid = grid.fine_tune_grid(similar_voxel_indices)
        print(f"Created fine grid with {len(fine_grid.get_indices())} points for depth {depth + 1}")
        
        # Add the fine grid indices to the included indices for the next level
        # Make sure to sample these new points
        self.included_indices.add_indices(fine_grid.get_indices()) 
        
        # Recursive call with increased depth
        self.process_scans(fine_grid, depth + 1, max_depth)

    def get_scan_arrays(self) -> List[npt.NDArray[np.float32]]:
        return [scan.get_displacements() for scan in self.scans]

    def get_scan(self, index: int) -> Scan:
        return self.scans[index]

    def visualize_indices(self, output_path: str = "final_indices.png"):
        """
        Visualize the included indices in three anatomical planes.
        
        Args:
            output_path: Path to save the visualization image
        """
        print(f"Visualizing {len(self.included_indices.get_indices())} indices...")
        
        # Get the dimensions from the first scan to ensure correct shape
        img_shape = self.scans[0].shape[:3]  # Take only the spatial dimensions (x, y, z)
        print(f"Creating visualization volume with shape {img_shape}")
        
        # Create a binary volume with 1s at the included indices
        volume = np.zeros(img_shape, dtype=np.float32)
        
        # Set voxels to 1 where indices exist
        indices = self.included_indices.get_indices()
        for index in indices:
            # Make sure indices are within bounds
            if (0 <= index.x < img_shape[0] and 
                0 <= index.y < img_shape[1] and 
                0 <= index.z < img_shape[2]):
                volume[index.x, index.y, index.z] = 1.0
        
        print(f"Created binary volume with {np.sum(volume)} active voxels")
        
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
        axes[0].imshow(mip_z, cmap='viridis', origin='lower')
        axes[0].set_title(f'Axial View (MIP)')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('X')
        
        # Plot coronal view
        axes[1].imshow(mip_y, cmap='viridis', origin='lower')
        axes[1].set_title(f'Coronal View (MIP)')
        axes[1].set_xlabel('Z')
        axes[1].set_ylabel('X')
        
        # Plot sagittal view
        axes[2].imshow(mip_x, cmap='viridis', origin='lower')
        axes[2].set_title(f'Sagittal View (MIP)')
        axes[2].set_xlabel('Z')
        axes[2].set_ylabel('Y')
        
        # Add a colorbar
        cbar = fig.colorbar(axes[0].images[0], ax=axes, orientation='horizontal', fraction=0.05, pad=0.1)
        cbar.set_label('Voxel Presence')
        
        # Add information about the number of indices
        plt.suptitle(f'Visualization of {len(indices)} Selected Indices', fontsize=16)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualization saved to {output_path}")
        
        # Additionally, create separate slices for each plane
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        
        # Calculate slice positions (middle, middle±quarter)
        x_slices = [mid_x - mid_x//2, mid_x, mid_x + mid_x//2]
        y_slices = [mid_y - mid_y//2, mid_y, mid_y + mid_y//2]
        z_slices = [mid_z - mid_z//2, mid_z, mid_z + mid_z//2]
        
        # Plot axial slices (top row)
        for i, z in enumerate(z_slices):
            if z < img_shape[2]:
                axes[0, i].imshow(volume[:, :, z], cmap='viridis', origin='lower')
                axes[0, i].set_title(f'Axial Slice z={z}')
                axes[0, i].set_xlabel('Y')
                axes[0, i].set_ylabel('X')
        
        # Plot coronal slices (middle row)
        for i, y in enumerate(y_slices):
            if y < img_shape[1]:
                axes[1, i].imshow(volume[:, y, :], cmap='viridis', origin='lower')
                axes[1, i].set_title(f'Coronal Slice y={y}')
                axes[1, i].set_xlabel('Z')
                axes[1, i].set_ylabel('X')
        
        # Plot sagittal slices (bottom row)
        for i, x in enumerate(x_slices):
            if x < img_shape[0]:
                axes[2, i].imshow(volume[x, :, :], cmap='viridis', origin='lower')
                axes[2, i].set_title(f'Sagittal Slice x={x}')
                axes[2, i].set_xlabel('Z')
                axes[2, i].set_ylabel('Y')
        
        plt.suptitle(f'Detailed Slices of {len(indices)} Selected Indices', fontsize=16)
        
        # Save detailed figure
        plt.tight_layout()
        plt.savefig(output_path.replace('.png', '_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detailed visualization saved to {output_path.replace('.png', '_detailed.png')}")

    def visualize_most_similar_indices(self, output_path: str = "most_similar_indices.png"):
        """
        Visualize the MOST SIMILAR included indices from the final round 
        in three anatomical planes.
        
        Args:
            output_path: Path to save the visualization image
        """
        if not hasattr(self, 'most_similar_indices') or not self.most_similar_indices:
            print("No 'most_similar_indices' found. Run process_scans to completion first.")
            return
        
        num_similar = len(self.most_similar_indices)
        print(f"Visualizing {num_similar} MOST SIMILAR indices...")
        
        # Get the dimensions from the first scan
        img_shape = self.scans[0].shape[:3]
        print(f"Creating visualization volume with shape {img_shape}")
        
        # Create a binary volume with 1s ONLY at the most similar indices
        volume = np.zeros(img_shape, dtype=np.float32)
        
        for index in self.most_similar_indices:
            if (0 <= index.x < img_shape[0] and 
                0 <= index.y < img_shape[1] and 
                0 <= index.z < img_shape[2]):
                volume[index.x, index.y, index.z] = 1.0
            
        print(f"Created binary volume with {np.sum(volume)} active 'most similar' voxels")

        # --- Plotting logic (similar to visualize_indices) ---
        
        # Create a figure with 3 subplots for MIPs
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        mip_z = np.max(volume, axis=2)
        mip_y = np.max(volume, axis=1)
        mip_x = np.max(volume, axis=0)
        
        # Plot MIPs
        im0 = axes[0].imshow(mip_z.T, cmap='hot', origin='lower', vmin=0, vmax=1) 
        axes[0].set_title('Axial View (MIP)')
        axes[0].set_xlabel('Y'); axes[0].set_ylabel('X')
        im1 = axes[1].imshow(mip_y.T, cmap='hot', origin='lower', vmin=0, vmax=1)
        axes[1].set_title('Coronal View (MIP)')
        axes[1].set_xlabel('Z'); axes[1].set_ylabel('X')
        im2 = axes[2].imshow(mip_x.T, cmap='hot', origin='lower', vmin=0, vmax=1)
        axes[2].set_title('Sagittal View (MIP)')
        axes[2].set_xlabel('Z'); axes[2].set_ylabel('Y')
        
        # Add a simple colorbar (just indicates presence/absence)
        cbar = fig.colorbar(im0, ax=axes, orientation='horizontal', fraction=0.05, pad=0.1, ticks=[0, 1])
        cbar.set_ticklabels(['Absent', 'Present'])
        cbar.set_label('Most Similar Voxel Presence')
        
        plt.suptitle(f'Visualization of {num_similar} MOST SIMILAR Indices (Final Round)', fontsize=16)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # type:ignore
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Most similar indices visualization saved to {output_path}")

        # --- Detailed Slices (Optional, but recommended) ---
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        mid_x, mid_y, mid_z = img_shape[0]//2, img_shape[1]//2, img_shape[2]//2
        x_slices = [mid_x - mid_x//2, mid_x, mid_x + mid_x//2]
        y_slices = [mid_y - mid_y//2, mid_y, mid_y + mid_y//2]
        z_slices = [mid_z - mid_z//2, mid_z, mid_z + mid_z//2]

        # Plot axial slices
        for i, z in enumerate(z_slices):
            if 0 <= z < img_shape[2]:
                 axes[0, i].imshow(volume[:, :, z].T, cmap='hot', origin='lower', vmin=0, vmax=1)
                 axes[0, i].set_title(f'Axial Slice z={z}'); axes[0, i].set_xlabel('Y'); axes[0, i].set_ylabel('X')
        # Plot coronal slices
        for i, y in enumerate(y_slices):
             if 0 <= y < img_shape[1]:
                 axes[1, i].imshow(volume[:, y, :].T, cmap='hot', origin='lower', vmin=0, vmax=1)
                 axes[1, i].set_title(f'Coronal Slice y={y}'); axes[1, i].set_xlabel('Z'); axes[1, i].set_ylabel('X')
        # Plot sagittal slices
        for i, x in enumerate(x_slices):
             if 0 <= x < img_shape[0]:
                 axes[2, i].imshow(volume[x, :, :].T, cmap='hot', origin='lower', vmin=0, vmax=1)
                 axes[2, i].set_title(f'Sagittal Slice x={x}'); axes[2, i].set_xlabel('Z'); axes[2, i].set_ylabel('Y')

        plt.suptitle(f'Detailed Slices of {num_similar} MOST SIMILAR Indices', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # type: ignore
        plt.savefig(output_path.replace('.png', '_detailed.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Detailed most similar indices visualization saved to {output_path.replace('.png', '_detailed.png')}")


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
def compare_all_subjects(
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


class DisplacementAnalyzer:
    def __init__(self, scans: List[Scan], included_indices: IncludedIndices):
        self.scans = scans
        self.index = included_indices.get_indices()

    def compare_all_subjects(self) -> npt.NDArray[np.float32]:
        scans_list = [scan.get_displacements() for scan in self.scans]
        return compare_all_subjects(scans_list)

    def find_extreme_voxels(self, proportion: float) -> List[Index]:
        # Calculate similarities between all subjects
        similarities = self.compare_all_subjects()

        # Extract lower triangle (excluding diagonal) of similarity matrix for each voxel
        lower_tri_indices = np.tril_indices(similarities.shape[0], k=-1)
        voxel_similarities = similarities[lower_tri_indices[0], lower_tri_indices[1], :]
        # Calculate mean similarity for each voxel across unique subject pairs
        mean_similarities = np.mean(voxel_similarities, axis=0)

        # Calculate number of voxels to return based on proportion
        n_voxels = int(len(self.index) * proportion)
        if n_voxels < 1:
            n_voxels = 1

        # Get indices of the most similar voxels
        most_similar_indices = np.argsort(mean_similarities)[-n_voxels:]
        most_similar_coords = [self.index[i] for i in most_similar_indices]

        return most_similar_coords

if __name__ == "__main__":

    NTH_VOXEL_SAMPLING = 16 # Divide the voxel size by 2 each time
    MAX_VOXELS_PER_SUBJECT = 200_000
    DEPTH = 4

    # --- Define Subject Scan Paths (Replace with your actual paths) ---
    subject_scan_paths = [
        os.path.join("maps", f) for f in os.listdir("maps") if f.endswith(".nii.gz")
    ]

    # --- Load Data for all Subjects ---
    print("Loading subject data...")
    scans_list: List[Scan] = []

    scans = ScanCollection(subject_scan_paths)
    grid = scans.construct_grid(NTH_VOXEL_SAMPLING, MAX_VOXELS_PER_SUBJECT)
    scans.process_scans(grid, max_depth=DEPTH)
    
    # Visualize ALL final included indices
    scans.visualize_indices() 
    
    # Visualize ONLY the MOST SIMILAR final indices
    scans.visualize_most_similar_indices() 

    # for scan_path in subject_scan_paths:
    #     if not os.path.exists(scan_path):
    #         print(f"Error: Scan file not found for {scan_path}. Skipping subject.")
    #         continue
    #     try:
    #         subject_data = load_subject_data(
    #             subject_id=scan_path,
    #             scan_path=scan_path,
    #             nth_voxel=NTH_VOXEL_SAMPLING,
    #             max_voxels=MAX_VOXELS_PER_SUBJECT
    #         )
    #         subjects_list.append(subject_data)
    #     except FileNotFoundError:
    #         print(f"Error: Scan file not found for {sub_id} at {scan_path}. Skipping subject.")
    #     except Exception as e:
    #          print(f"Error loading data for {scan_path}: {e}. Skipping subject.")

    # if not subjects_list:
    #     print("No subject data loaded. Exiting.")
    #     exit()
    # elif len(subjects_list) < 2:
    #      print("Need at least two subjects with loaded data to perform analysis. Exiting.")
    #      exit()

    # print(f"\nSuccessfully loaded data for {len(subjects_list)} subjects.")

    # # --- Perform Analysis ---
    # analyzer = DisplacementAnalyzer(subjects_list)

    # # Get extreme cases (comparing across all loaded subjects)
    # print("\nCalculating similarities and finding extreme cases...")
    # results = analyzer.get_extreme_cases(n_cases=3) # Find top/bottom 3

    # # --- Plot Results ---
    # print("\nPlotting results...")
    # # Ensure matplotlib is imported
    # import matplotlib.pyplot as plt

    # # Plot individual traces for the top/bottom voxels
    # # plot_extreme_cases(results) # This function needs adjustment for coordinate IDs

    # # Plot average traces with error bars for the top/bottom voxels
    # # plot_average_traces(results) # This function also needs adjustment

    # print("\nAnalysis complete.")

    # TODO: Update plotting functions to handle coordinate tuples as IDs if needed
    #       or modify `get_extreme_cases` to return string IDs along with coordinates.
