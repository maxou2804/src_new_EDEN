"""
COMPETITIVE DISTANCE SIMULATION - Consolidated Implementation

This module provides a complete simulation framework for multi-seed competitive growth
with competitive distance-based probabilities.

Key Features:
- Optimized Eden growth with set-based boundaries
- Competitive distance probability calculation (with KD-tree and Numba options)
- Roughened seed initialization
- Parallel radial profile computation
- Animation support
- Cluster merging detection
"""

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree
import csv
import random
from pathlib import Path
from scipy.ndimage import gaussian_filter
import math

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@njit
def get_neighbors(i, j, rows, cols):
    """Get valid Neumann neighbors (4-connectivity)."""
    neighbors = []
    if i > 0:
        neighbors.append((i - 1, j))
    if i < rows - 1:
        neighbors.append((i + 1, j))
    if j > 0:
        neighbors.append((i, j - 1))
    if j < cols - 1:
        neighbors.append((i, j + 1))
    return neighbors


@njit
def euclidean_distance(i1, j1, i2, j2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((i1 - i2)**2 + (j1 - j2)**2)


# ============================================================================
# SEED INITIALIZATION
# ============================================================================

def generate_seed_configs(n, l, r, r_lcc, r_urb, roughness):
    """
    Generate seed configurations for urban simulation.
    
    Parameters:
    - n: number of seeds (excluding the largest component)
    - l: size of the grid
    - r: radius around the largest component where seeds should be placed
    - r_lcc: radius of the largest connected component (LCC)
    - r_urb: list of radii for the seeds (length should be n)
    - roughness: global roughness value for all seeds
    
    Returns:
    - List of seed configuration dictionaries
    """
    seed_configs = []
    
    # Add the largest connected component at the center
    center = (int(l / 2), int(l / 2))
    seed_configs.append({
        'position': center,
        'radius': r_lcc,
        'roughness': roughness
    })
    
    # Generate n seeds randomly placed on a circle around the LCC
    for i in range(n):
        print(i)
        # Random angle in radians
        angle = random.uniform(0, 2 * math.pi)

       
        # Calculate position on the circle of radius r around the center
        x =int( center[0] + (r+np.random.normal(0,20) )* math.cos(angle))
        y =int( center[1] + (r+np.random.normal(0,20) ) * math.sin(angle))
        
        # Get radius from r_urb list
        radius = r_urb[i]
        
        seed_configs.append({
            'position': (x, y),
            'radius': radius,
            'roughness': roughness
        })
    return seed_configs

def create_roughened_seed(grid, center_i, center_j, radius, roughness=0.3, seed_id=1):
    """
    Create a roughened circular seed instead of a smooth circle.
    
    Parameters:
    -----------
    grid : np.ndarray
        Grid to place seed on
    center_i, center_j : int
        Center coordinates of seed
    radius : float
        Approximate radius of seed
    roughness : float (0-1)
        Amount of roughness (0 = smooth circle, 1 = very rough)
    seed_id : int
        ID to assign to this seed's cells
    
    Returns:
    --------
    grid : np.ndarray
        Grid with roughened seed
    n_cells : int
        Number of cells in the seed
    """
    rows, cols = grid.shape
    
    # Generate random radial variations
    n_angles = max(int(2 * np.pi * radius), 20)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    np.random.seed(None)
    radial_noise = np.random.normal(0, roughness * radius, n_angles)
    varied_radii = radius + radial_noise
    varied_radii = np.clip(varied_radii, radius * 0.3, radius * 1.7)
    
    # Smooth the radial variations
    window_size = max(3, n_angles // 20)
    smoothed_radii = np.convolve(
        np.concatenate([varied_radii[-window_size:], varied_radii, varied_radii[:window_size]]),
        np.ones(window_size) / window_size,
        mode='valid'
    )
    
    # Place cells
    n_cells = 0
    for i in range(max(0, center_i - int(radius * 2)), 
                   min(rows, center_i + int(radius * 2) + 1)):
        for j in range(max(0, center_j - int(radius * 2)), 
                       min(cols, center_j + int(radius * 2) + 1)):
            di = i - center_i
            dj = j - center_j
            dist = np.sqrt(di**2 + dj**2)
            angle = np.arctan2(dj, di)
            
            angle_idx = int((angle + np.pi) / (2*np.pi) * n_angles) % n_angles
            threshold_radius = smoothed_radii[angle_idx]
            
            if dist <= threshold_radius and grid[i, j] == 0:
                grid[i, j] = seed_id
                n_cells += 1
    
    return grid, n_cells


    from scipy.ndimage import gaussian_filter

    

def create_urban_cluster(grid, center_i, center_j, radius, complexity=0.5, 
                         compactness=0.7, seed_id=1):
    """
    Create an urban cluster using noise-based organic shapes.
    
    Parameters:
    -----------
    complexity : float (0-1)
        Higher = more irregular boundaries (like sprawling suburbs)
    compactness : float (0-1)
        Higher = more compact (like dense city), lower = more sprawling
    """
    rows, cols = grid.shape
    target_area = np.pi * radius**2
    
    # Create noise field
    np.random.seed(None)
    noise_scale = radius * (0.3 + 0.4 * complexity)
    
    # Generate random noise
    search_radius = int(radius * 2)
    local_noise = np.random.randn(search_radius * 2, search_radius * 2)
    smoothed_noise = gaussian_filter(local_noise, sigma=noise_scale/2)
    
    # Calculate distance-based probability field
    candidate_cells = []
    for i in range(max(0, center_i - search_radius), 
                   min(rows, center_i + search_radius)):
        for j in range(max(0, center_j - search_radius), 
                       min(cols, center_j + search_radius)):
            if grid[i, j] == 0:
                di = i - center_i
                dj = j - center_j
                dist = np.sqrt(di**2 + dj**2)
                
                # Combine distance decay with noise
                dist_factor = np.exp(-dist / (radius * compactness))
                noise_i = min(di + search_radius, smoothed_noise.shape[0]-1)
                noise_j = min(dj + search_radius, smoothed_noise.shape[1]-1)
                noise_val = smoothed_noise[noise_i, noise_j]
                
                probability = dist_factor + complexity * noise_val
                candidate_cells.append((i, j, probability))
    
    # Select cells based on probabilities to match target area
    candidate_cells.sort(key=lambda x: x[2], reverse=True)
    n_cells_target = int(target_area)
    
    n_cells = 0
    for i, j, _ in candidate_cells[:n_cells_target]:
        grid[i, j] = seed_id
        n_cells += 1
    
    return grid, n_cells


def initialize_roughened_seeds(grid_size, seed_configs):
    """
    Initialize multiple roughened seeds.
    
    Parameters:
    -----------
    grid_size : int
        Size of square grid
    seed_configs : list of dicts
        Each dict contains:
        - 'position': (i, j) tuple
        - 'radius': float (initial radius)
        - 'roughness': float (optional, default 0.3)
    
    Returns:
    --------
    grid : np.ndarray
        Grid with all seeds initialized
    boundary_sets : list of sets
        Boundary sets for each seed
    seed_ids : np.ndarray
        Array of seed IDs
    initial_sizes : dict
        Initial size of each seed
    """
    grid = np.zeros((grid_size, grid_size), dtype=np.int32)
    boundary_sets = []
    seed_ids = []
    initial_sizes = {}
    
    for seed_idx, config in enumerate(seed_configs):
        seed_id = seed_idx + 1
        position = config['position']
        radius = config['radius']
        roughness = config.get('roughness', 0.3)
        
        grid, n_cells = create_roughened_seed(
            grid=grid, center_i=position[0], center_j=position[1], radius=radius,
           seed_id= seed_id)
        
        
        initial_sizes[seed_id] = n_cells
        seed_ids.append(seed_id)
        
        # Find boundary cells
        boundary = set()
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == seed_id:
                    neighbors = get_neighbors(i, j, grid_size, grid_size)
                    for ni, nj in neighbors:
                        if grid[ni, nj] == 0 and (ni, nj) not in boundary:
                            boundary.add((ni, nj))
        
        boundary_sets.append(boundary)
    
    return grid, boundary_sets, np.array(seed_ids), initial_sizes


# ============================================================================
# COMPETITIVE DISTANCE PROBABILITY
# ============================================================================

class CompetitiveDistanceCalculator:
    """
    Efficient calculator for competitive distance probabilities using KD-trees.
    """
    
    def __init__(self, beta=1.0):
        """
        Parameters:
        -----------
        beta : float
            Distance exponent (0-2)
            - beta=0: uniform (distance doesn't matter)
            - beta=1: linear distance penalty
            - beta=2: strong distance penalty
        """
        self.beta = beta
        self.kdtrees = {}
        self.masses = {}
        self.needs_update = True
    def calculate_probability_vectorized(self, i_array, j_array, seed_id):
    
        n_cells = len(i_array)
        probabilities = np.ones(n_cells, dtype=np.float64)
    
        cell_positions = np.column_stack([i_array, j_array])
    
        for other_id, kdtree in self.kdtrees.items():
            if other_id == seed_id:
                continue
        
        # Query all cells at once (vectorized)
            dists, _ = kdtree.query(cell_positions, k=1)
        
            if other_id in self.masses:
                mass = self.masses[other_id]
                valid = dists > 0
                probabilities[valid] += mass / (dists[valid] ** self.beta)
    
        return probabilities
    
    def update_boundaries(self, boundary_sets, seed_ids, grid):
        """Update internal data structures with current boundaries."""
        self.kdtrees.clear()
        self.masses.clear()
        
        for idx, seed_id in enumerate(seed_ids):
            seed_id = int(seed_id)
            boundary = boundary_sets[idx]
            
            if len(boundary) == 0:
                continue
            
            boundary_array = np.array(list(boundary), dtype=np.float64)
            self.kdtrees[seed_id] = cKDTree(boundary_array)
            self.masses[seed_id] = len(boundary)
        
        self.needs_update = False
    
    def calculate_probability(self, i, j, seed_id):
        """
        Calculate probability for cell (i,j) belonging to seed_id.
        
        Returns:
        --------
        probability : float
            Probability weight (higher = more likely)
        """
        if self.needs_update or len(self.kdtrees) == 0:
            return 1.0
        
        total_prob = 1.0
        cell_pos = np.array([[i, j]], dtype=np.float64)
        
        for other_id, kdtree in self.kdtrees.items():
            if other_id == seed_id:
                continue
            
            dist, _ = kdtree.query(cell_pos, k=1)
            dist = dist[0]
            
            if dist > 0 and other_id in self.masses:
                mass = self.masses[other_id]
                total_prob += mass / (dist ** self.beta)
        
        return total_prob


@njit
def calculate_competitive_probability_numba(
    i, j, seed_id, beta, gamma,
    boundary_arrays, boundary_sizes, masses, n_clusters
):
    """
    Numba-optimized competitive probability calculation.
    Faster for small boundaries but doesn't use KD-tree.
    """
    total_prob = 1.0
    
    for cluster_idx in range(n_clusters):
        if cluster_idx + 1 == seed_id:
            continue
        
        boundary_size = int(boundary_sizes[cluster_idx])
        if boundary_size == 0:
            continue
        
        min_dist_sq = 1e9
        
        for pt_idx in range(boundary_size):
            pi = boundary_arrays[cluster_idx, pt_idx, 0]
            pj = boundary_arrays[cluster_idx, pt_idx, 1]
            
            dist_sq = (i - pi) ** 2 + (j - pj) ** 2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
        
        if min_dist_sq > 0:
            min_dist = np.sqrt(min_dist_sq)
            mass = masses[cluster_idx]
            total_prob += mass / (min_dist ** beta)
    
    total_prob = total_prob ** gamma
    return total_prob


def prepare_boundary_data_for_numba(boundary_sets, seed_ids, grid):
    """Prepare boundary data in Numba-compatible format."""
    n_clusters = len(seed_ids)
    max_boundary_size = max(len(b) for b in boundary_sets) if boundary_sets else 1
    
    boundary_arrays = np.zeros((n_clusters, max_boundary_size, 2), dtype=np.int64)
    boundary_sizes = np.zeros(n_clusters, dtype=np.int64)
    masses = np.zeros(n_clusters, dtype=np.float64)
    
    for idx, seed_id in enumerate(seed_ids):
        boundary = boundary_sets[idx]
        boundary_sizes[idx] = len(boundary)
        
        for pt_idx, (i, j) in enumerate(boundary):
            if pt_idx >= max_boundary_size:
                break
            boundary_arrays[idx, pt_idx, 0] = i
            boundary_arrays[idx, pt_idx, 1] = j
        
        masses[idx] = np.sum(grid == seed_id)
    
    return boundary_arrays, boundary_sizes, masses, n_clusters


def create_competitive_prob_params(boundary_sets, seed_ids, grid, beta=1.0, use_kdtree=True):
    """
    Create probability parameters for competitive distance calculation.
    
    Parameters:
    -----------
    boundary_sets : list of sets
        Current boundary sets
    seed_ids : np.ndarray
        Seed IDs
    grid : np.ndarray
        Current grid
    beta : float
        Distance exponent
    use_kdtree : bool
        If True, use KD-tree (fast for large boundaries)
        If False, use Numba (fast for small boundaries)
    
    Returns:
    --------
    prob_params_list : list
        List of prob_params for each seed
    """
    if use_kdtree:
        calculator = CompetitiveDistanceCalculator(beta=beta)
        calculator.update_boundaries(boundary_sets, seed_ids, grid)
        return [calculator] * len(seed_ids)
    else:
        boundary_arrays, boundary_sizes, masses, n_clusters = \
            prepare_boundary_data_for_numba(boundary_sets, seed_ids, grid)
        params = (beta, boundary_arrays, boundary_sizes, masses, n_clusters)
        return [params] * len(seed_ids)


# ============================================================================
# GROWTH FUNCTIONS
# ============================================================================

def eden_growth_step_competitive(
    grid, boundary_set, seed_id, beta, gamma,
    boundary_sets, seed_ids, use_kdtree=True, prob_calculator=None
):
    """
    Eden growth step with competitive distance probability.
    """
    if len(boundary_set) == 0:
        return grid, boundary_set
    
    boundary_array = np.array(list(boundary_set))
    n_boundary = len(boundary_array)
    probabilities = np.zeros(n_boundary, dtype=np.float64)
    
    if use_kdtree and isinstance(prob_calculator, CompetitiveDistanceCalculator):
        probabilities = prob_calculator.calculate_probability_vectorized(
    boundary_array[:, 0], boundary_array[:, 1], seed_id
)
    else:
        beta, boundary_arrays, boundary_sizes, masses, n_clusters = prob_calculator
        for idx in range(n_boundary):
            i, j = boundary_array[idx]
            probabilities[idx] = calculate_competitive_probability_numba(
                i, j, seed_id, beta, gamma,
                boundary_arrays, boundary_sizes, masses, n_clusters
            )
    
    # Normalize and select
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities /= prob_sum
    else:
        probabilities[:] = 1.0 / n_boundary
    
    cumsum = np.cumsum(probabilities)
    rand_val = np.random.random()
    selected_idx = np.searchsorted(cumsum, rand_val)
    
    if selected_idx >= n_boundary:
        selected_idx = n_boundary - 1
    
    new_i, new_j = boundary_array[selected_idx]
    
    # Urbanize cell
    grid[new_i, new_j] = seed_id
    boundary_set.discard((new_i, new_j))
    
    # Add new neighbors
    rows, cols = grid.shape
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    for ni, nj in neighbors:
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return grid, boundary_set


# ============================================================================
# RADIAL PROFILE & WIDTH ANALYSIS
# ============================================================================

@njit(parallel=True)
def radial_profile_from_grid_parallel(grid, n_theta=10000, ray_step=0.5):
    """
    Compute r(θ, t) by ray marching from the center.
    Returns sorted thetas for efficient width analysis.
    """
    h, w = grid.shape
    cx = (h - 1) / 2.0
    cy = (w - 1) / 2.0
    
    thetas = np.empty(n_theta, dtype=np.float64)
    cos_t = np.empty(n_theta, dtype=np.float64)
    sin_t = np.empty(n_theta, dtype=np.float64)
    
    for k in range(n_theta):
        theta = 2.0 * np.pi * k / n_theta
        thetas[k] = theta
        cos_t[k] = np.cos(theta)
        sin_t[k] = np.sin(theta)
    
    rmax = min(cx, cy) - 1.0
    r_vals = np.zeros(n_theta, dtype=np.float64)
    
    for k in prange(n_theta):
        ct = cos_t[k]
        st = sin_t[k]
        r = 0.0
        r_last = 0.0
        
        while r <= rmax:
            x = cx + r * ct
            y = cy + r * st
            
            ix = int(round(x))
            iy = int(round(y))
            
            if ix < 0 or ix >= h or iy < 0 or iy >= w:
                break
            
            if grid[ix, iy]:
                r_last = r
            
            r += ray_step
        
        r_vals[k] = r_last
    
    return thetas, r_vals


@njit
def _compute_sector_stats(r_sorted, N):
    """Compute sector statistics for a given N."""
    n_samples = len(r_sorted)
    base_size = n_samples // N
    remainder = n_samples % N
    
    means = np.empty(N, dtype=np.float64)
    vars_ = np.empty(N, dtype=np.float64)
    
    idx = 0
    for i in range(N):
        sector_size = base_size + (1 if i < remainder else 0)
        
        if sector_size == 0:
            means[i] = np.nan
            vars_[i] = np.nan
            continue
        
        sector_end = idx + sector_size
        
        sector_sum = 0.0
        for j in range(idx, sector_end):
            sector_sum += r_sorted[j]
        mean_val = sector_sum / sector_size
        means[i] = mean_val
        
        var_sum = 0.0
        for j in range(idx, sector_end):
            diff = r_sorted[j] - mean_val
            var_sum += diff * diff
        vars_[i] = var_sum / sector_size
        
        idx = sector_end
    
    return means, vars_


@njit(parallel=True)
def width_vs_arc_length_optimized(theta, r, N_list):
    """Compute width vs arc length for multiple N values (optimized)."""
    r_sorted = r  # Already sorted from radial_profile
    
    n_samples = len(r_sorted)
    n_N = len(N_list)
    
    ell_list = np.empty(n_N, dtype=np.float64)
    w_list = np.empty(n_N, dtype=np.float64)
    
    for idx in prange(n_N):
        N = int(N_list[idx])
        N = max(2, min(n_samples, N))
        
        means, vars_ = _compute_sector_stats(r_sorted, N)
        
        valid_count = 0
        mean_sum = 0.0
        var_sum = 0.0
        
        for i in range(N):
            if not np.isnan(means[i]) and not np.isnan(vars_[i]):
                mean_sum += means[i]
                var_sum += vars_[i]
                valid_count += 1
        
        if valid_count > 0:
            delta_theta = 2.0 * np.pi / N
            ell_bar = delta_theta * (mean_sum / valid_count)
            w_sq = var_sum / valid_count
            w = np.sqrt(w_sq)
        else:
            ell_bar = np.nan
            w = np.nan
        
        ell_list[idx] = ell_bar
        w_list[idx] = w
    
    return ell_list, w_list


# ============================================================================
# MERGE DETECTION
# ============================================================================

@njit
def check_boundary_collision_fast(boundary_array_i, boundary_array_j, size_i, size_j):
    """
    Fast collision detection using sorted merge approach.
    O(n log n) instead of O(n*m) for large boundaries.
    """
    if size_i == 0 or size_j == 0:
        return False
    
    # For small boundaries, use simple nested loop
    if size_i * size_j < 1000:
        for idx_i in range(size_i):
            cell_i_0 = boundary_array_i[idx_i, 0]
            cell_i_1 = boundary_array_i[idx_i, 1]
            
            for idx_j in range(size_j):
                cell_j_0 = boundary_array_j[idx_j, 0]
                cell_j_1 = boundary_array_j[idx_j, 1]
                
                if cell_i_0 == cell_j_0 and cell_i_1 == cell_j_1:
                    return True
        return False
    
    # For larger boundaries, use hash-like comparison
    # Convert coordinates to single integers for faster comparison
    hash_i = np.empty(size_i, dtype=np.int64)
    for idx in range(size_i):
        hash_i[idx] = boundary_array_i[idx, 0] * 100000 + boundary_array_i[idx, 1]
    
    hash_j = np.empty(size_j, dtype=np.int64)
    for idx in range(size_j):
        hash_j[idx] = boundary_array_j[idx, 0] * 100000 + boundary_array_j[idx, 1]
    
    # Sort both arrays
    hash_i_sorted = np.sort(hash_i)
    hash_j_sorted = np.sort(hash_j)
    
    # Two-pointer approach to find intersection
    i = 0
    j = 0
    while i < size_i and j < size_j:
        if hash_i_sorted[i] == hash_j_sorted[j]:
            return True
        elif hash_i_sorted[i] < hash_j_sorted[j]:
            i += 1
        else:
            j += 1
    
    return False


def convert_boundary_sets_to_arrays(boundary_sets, max_size=10000):
    """
    Convert list of boundary sets to Numba-compatible arrays.
    
    Returns:
    --------
    boundary_arrays : list of np.ndarray
        Each array is shape (N, 2) containing boundary coordinates
    boundary_sizes : np.ndarray
        Array of actual boundary sizes
    """
    n_seeds = len(boundary_sets)
    boundary_arrays = []
    boundary_sizes = np.zeros(n_seeds, dtype=np.int64)
    
    for idx, boundary_set in enumerate(boundary_sets):
        size = len(boundary_set)
        boundary_sizes[idx] = size
        
        if size == 0:
            # Empty boundary - create dummy array
            boundary_arrays.append(np.zeros((1, 2), dtype=np.int64))
        else:
            # Convert set to array
            boundary_array = np.array(list(boundary_set), dtype=np.int64)
            boundary_arrays.append(boundary_array)
    
    return boundary_arrays, boundary_sizes


def check_and_merge_clusters_optimized(grid, boundary_sets, seed_ids, merger_map_array, grid_size):
    """
    Check for clusters that have merged and update accordingly.
    Returns list of (smaller_id, larger_id) tuples for mergers detected.
    
    Uses Numba-optimized collision detection with array-based boundaries.
    
    Parameters:
    -----------
    merger_map_array : np.ndarray
        Array where merger_map_array[seed_id] = final_merged_id
        Size should be max(seed_ids) + 1
    """
    mergers = []
    n_seeds = len(seed_ids)
    
    # Convert boundary sets to arrays for Numba
    boundary_arrays, boundary_sizes = convert_boundary_sets_to_arrays(boundary_sets)
    
    for i in range(n_seeds):
        seed_id_i = int(seed_ids[i])
        
        if boundary_sizes[i] == 0:
            continue
        
        for j in range(i + 1, n_seeds):
            seed_id_j = int(seed_ids[j])
            
            if boundary_sizes[j] == 0:
                continue
            
            # Check for collision using optimized Numba function
            collision = check_boundary_collision_fast(
                boundary_arrays[i], boundary_arrays[j],
                boundary_sizes[i], boundary_sizes[j]
            )
            
            if collision:
                # Merge smaller into larger
                size_i = np.sum(grid == seed_id_i)
                size_j = np.sum(grid == seed_id_j)
                
                if size_i > size_j:
                    grid[grid == seed_id_j] = seed_id_i
                    boundary_sets[i] = boundary_sets[i].union(boundary_sets[j])
                    boundary_sets[j] = set()
                    merger_map_array[seed_id_j] = seed_id_i
                    mergers.append((seed_id_j, seed_id_i))
                else:
                    grid[grid == seed_id_i] = seed_id_j
                    boundary_sets[j] = boundary_sets[j].union(boundary_sets[i])
                    boundary_sets[i] = set()
                    merger_map_array[seed_id_i] = seed_id_j
                    mergers.append((seed_id_i, seed_id_j))
    
    return mergers
# ============================================================================
# SPATIAL PROBABILITY PROFILE
# ============================================================================

@njit
def calculate_gaussian_probability(i, j, center_i, center_j, variance):
    """
    Calculate 2D Gaussian probability based on distance to center.
    
    Parameters:
    -----------
    i, j : int
        Cell coordinates
    center_i, center_j : float
        Center of the grid
    variance : float
        Variance of the Gaussian (σ²)
    
    Returns:
    --------
    probability : float
        Probability value (0-1) based on Gaussian distribution
    """
    dist_sq = (i - center_i) ** 2 + (j - center_j) ** 2
    return np.exp(-dist_sq / (2 * variance))


# ============================================================================
# GROWTH FUNCTIONS (MODIFIED)
# ============================================================================

def eden_growth_step_competitive(
    grid, boundary_set, seed_id, beta, gamma,
    boundary_sets, seed_ids, use_kdtree=True, prob_calculator=None,
    use_spatial_filter=False, center_i=None, center_j=None, spatial_variance=None
):
    """
    Eden growth step with competitive distance probability.
    
    Additional Parameters:
    ----------------------
    use_spatial_filter : bool
        If True, apply Gaussian spatial filter after competitive selection
    center_i, center_j : float
        Center coordinates for Gaussian filter
    spatial_variance : float
        Variance (σ²) for Gaussian filter
    """
    if len(boundary_set) == 0:
        return grid, boundary_set
    
    boundary_array = np.array(list(boundary_set))
    n_boundary = len(boundary_array)
    probabilities = np.zeros(n_boundary, dtype=np.float64)
    
    if use_kdtree and isinstance(prob_calculator, CompetitiveDistanceCalculator):
        for idx in range(n_boundary):
            i, j = boundary_array[idx]
            probabilities[idx] = prob_calculator.calculate_probability(i, j, seed_id)
    else:
        beta, boundary_arrays, boundary_sizes, masses, n_clusters = prob_calculator
        for idx in range(n_boundary):
            i, j = boundary_array[idx]
            probabilities[idx] = calculate_competitive_probability_numba(
                i, j, seed_id, beta, gamma,
                boundary_arrays, boundary_sizes, masses, n_clusters
            )
    
    # Normalize and select
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities /= prob_sum
    else:
        probabilities[:] = 1.0 / n_boundary
    
    cumsum = np.cumsum(probabilities)
    rand_val = np.random.random()
    selected_idx = np.searchsorted(cumsum, rand_val)
    
    if selected_idx >= n_boundary:
        selected_idx = n_boundary - 1
    
    new_i, new_j = boundary_array[selected_idx]
    
    # Apply spatial Gaussian filter if enabled
    if use_spatial_filter:
        gaussian_prob = calculate_gaussian_probability(
            new_i, new_j, center_i, center_j, spatial_variance
        )
        
        # Decide whether to grow based on Gaussian probability
        if np.random.random() > gaussian_prob:
            # Don't grow this cell, return unchanged
            return grid, boundary_set
    
    # Urbanize cell
    grid[new_i, new_j] = seed_id
    boundary_set.discard((new_i, new_j))
    
    # Add new neighbors
    rows, cols = grid.shape
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    for ni, nj in neighbors:
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return grid, boundary_set

# ============================================================================
# ANIMATION
# ============================================================================

def create_growth_animation(frames, seed_ids, output_file, fps=10, title="Growth Animation"):
    """
    Create animated GIF from simulation frames.
    
    Parameters:
    -----------
    frames : list of np.ndarray
        List of grid snapshots
    seed_ids : np.ndarray
        Seed IDs for color mapping
    output_file : str
        Output GIF filename
    fps : int
        Frames per second
    title : str
        Animation title
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("Warning: matplotlib not available, skipping animation")
        return
    
    n_seeds = len(seed_ids)
    max_seed_id = int(np.max(seed_ids))
    
    # Create colormap - need colors for all seed IDs from 0 to max_seed_id
    # Background (0) will be white, seeds will get distinct colors
    if n_seeds <= 10:
        # Use tab10 colormap for up to 10 distinct colors
        colors = ['white'] + list(plt.cm.tab10.colors[:n_seeds])
    else:
        # Use hsv colormap for more than 10 seeds to generate enough distinct colors
        colors = ['white'] + [plt.cm.hsv(i/n_seeds) for i in range(n_seeds)]
    
    cmap = mcolors.ListedColormap(colors)
    # Create bounds for all possible seed IDs (0 to max_seed_id)
    bounds = np.arange(len(colors) + 1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Create animation
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame_idx):
        ax.clear()
        im = ax.imshow(frames[frame_idx], cmap=cmap, norm=norm, interpolation='nearest')
        ax.set_title(f"{title} - Frame {frame_idx + 1}/{len(frames)}")
        ax.axis('off')
        return [im]
    
    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000/fps, blit=True)
    
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer)
    plt.close(fig)


# ============================================================================
# PURE EDEN GROWTH (NO COMPETITIVE DISTANCE)
# ============================================================================

def eden_growth_step_pure(grid, boundary_set, seed_id):
    """
    Pure Eden growth: uniform random selection from boundary.
    No competitive distance calculations - much faster!
    
    Parameters:
    -----------
    grid : np.ndarray
        Current grid
    boundary_set : set
        Set of boundary cells for this seed
    seed_id : int
        ID of this seed
    
    Returns:
    --------
    grid : np.ndarray
        Updated grid
    boundary_set : set
        Updated boundary set
    """
    if len(boundary_set) == 0:
        return grid, boundary_set
    
    # Uniform random selection - all cells equal probability
    new_i, new_j = random.choice(list(boundary_set))
    
    # Urbanize cell
    grid[new_i, new_j] = seed_id
    boundary_set.discard((new_i, new_j))
    
    # Add new neighbors
    rows, cols = grid.shape
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    for ni, nj in neighbors:
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return grid, boundary_set

def load_grid_from_csv(csv_file):
    """
    Load grid from CSV file.
    
    Parameters:
    -----------
    csv_file : str
        Path to CSV file containing grid data
    
    Returns:
    --------
    grid : np.ndarray
        Loaded grid (int32)
    """
    grid = np.loadtxt(csv_file, delimiter=',', dtype=np.int32)
    print(f"✓ Grid loaded from: {csv_file}")
    print(f"  Shape: {grid.shape}")
    print(f"  Unique seeds: {np.unique(grid[grid > 0])}")
    return grid

def load_grid_from_npy(npy_file):
    """
    Load grid from .npy file.
    
    Parameters:
    -----------
    npy_file : str
        Path to .npy file containing grid data
    
    Returns:
    --------
    grid : np.ndarray
        Loaded grid
    """
    grid = np.load(npy_file)
    print(f"✓ Grid loaded from: {npy_file}")
    print(f"  Shape: {grid.shape}")
    print(f"  Unique seeds: {np.unique(grid[grid > 0])}")
    return grid



def load_grid_auto(file_path):
    """
    Automatically load grid based on file extension.
    
    Parameters:
    -----------
    file_path : str or Path
        Path to grid file (.csv or .npy)
    
    Returns:
    --------
    grid : np.ndarray
        Loaded grid
    """
    file_path = Path(file_path)
    
    if file_path.suffix == '.npy':
        return load_grid_from_npy(file_path)
    elif file_path.suffix == '.csv':
        return load_grid_from_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}. Use .npy or .csv")
def initialize_boundary_sets_from_grid(grid):
    """
    Initialize boundary sets from an existing grid.
    
    Parameters:
    -----------
    grid : np.ndarray
        Grid with seeds already placed
    
    Returns:
    --------
    boundary_sets : list of sets
        Boundary sets for each seed
    seed_ids : np.ndarray
        Array of seed IDs
    initial_sizes : dict
        Initial size of each seed
    """
    grid_size = grid.shape[0]
    seed_ids = np.unique(grid[grid > 0])
    boundary_sets = []
    initial_sizes = {}
    
    for seed_id in seed_ids:
        initial_sizes[seed_id] = np.sum(grid == seed_id)
        
        # Find boundary cells for this seed
        boundary = set()
        for i in range(grid_size):
            for j in range(grid_size):
                if grid[i, j] == seed_id:
                    neighbors = get_neighbors(i, j, grid_size, grid_size)
                    for ni, nj in neighbors:
                        if grid[ni, nj] == 0 and (ni, nj) not in boundary:
                            boundary.add((ni, nj))
        
        boundary_sets.append(boundary)
    
    return boundary_sets, seed_ids, initial_sizes

# ============================================================================
# MAIN SIMULATION (MODIFIED)
# ============================================================================

def simulate_with_competitive_distance(
    grid_size,
    seed_configs,
    timesteps,
    output_file,
    metric_timestep=100,
    beta=1.0,
    gamma=1.0,
    update_frequency=10,
    merge_check_frequency=100,
    N_sampling=10000,
    num_N=20,
    create_animation=False,
    animation_interval=1000,
    animation_file=None,
    use_spatial_filter=False,
    spatial_variance=None,
    use_competitive_distance=False,
    initial_grid_file=None  # NEW PARAMETER
):
    """
    Multi-seed simulation with optional competitive distance probability.
    
    Parameters:
    -----------
    ... (all previous parameters)
    initial_grid_file : str or Path, optional
        Path to initial grid file (.csv or .npy) to continue from.
        If provided, the grid is loaded and seed_configs is ignored.
    
    Returns:
    --------
    grid : np.ndarray
        Final grid
    largest_seed_stats : dict
        Statistics for largest seed
    all_stats : dict
        Statistics for all seeds
    """
    
    print("=" * 70)
    if use_competitive_distance:
        print("COMPETITIVE DISTANCE SIMULATION")
        print(f"Beta: {beta}, Gamma: {gamma}")
        print(f"Update frequency: every {update_frequency} iterations")
    else:
        print("PURE EDEN GROWTH SIMULATION")
        print("Mode: Uniform random selection (no competitive distance)")
    print("=" * 70)
    print(f"Grid: {grid_size}×{grid_size}")
    
    # Initialize grid
    if initial_grid_file is not None:
        # Load from file
        print(f"\n🔄 Loading initial grid from: {initial_grid_file}")
        grid = load_grid_auto(initial_grid_file)
        
        # Verify grid size
        if grid.shape[0] != grid_size or grid.shape[1] != grid_size:
            print(f"⚠ Warning: Loaded grid size {grid.shape} doesn't match requested {grid_size}×{grid_size}")
            print(f"  Using loaded grid size: {grid.shape[0]}×{grid.shape[1]}")
            grid_size = grid.shape[0]
        
        # Initialize from loaded grid
        boundary_sets, seed_ids, initial_sizes = initialize_boundary_sets_from_grid(grid)
        
        print(f"\nLoaded seeds: {len(seed_ids)}")
        for sid, size in initial_sizes.items():
            print(f"  Seed {sid}: {size} cells")
    else:
        # Initialize from seed configs
        print(f"Seeds: {seed_configs['n']}")
        
        seed_configs_id = generate_seed_configs(
            n=seed_configs['n'],
            l=seed_configs['l'],
            r=seed_configs['r'],
            r_lcc=seed_configs['r_lcc'],
            r_urb=seed_configs['r_urb'],
            roughness=seed_configs['roughness']
        )
        
        grid, boundary_sets, seed_ids, initial_sizes = initialize_roughened_seeds(
            grid_size, seed_configs_id
        )
        
        print("\nInitial sizes:")
        for sid, size in initial_sizes.items():
            print(f"  Seed {sid}: {size} cells")
    
    print(f"Steps: {timesteps:,}")
    if use_spatial_filter:
        if spatial_variance is None:
            spatial_variance = (grid_size / 4.0) ** 2
        print(f"Spatial Gaussian filter: ON (σ² = {spatial_variance:.2f})")
    if create_animation:
        print(f"Animation: ON (every {animation_interval} steps)")
    print("=" * 70)
    
    # Calculate grid center for spatial filter
    center_i = (grid_size - 1) / 2.0
    center_j = (grid_size - 1) / 2.0
    
    # Set default spatial variance if needed
    if use_spatial_filter and spatial_variance is None:
        spatial_variance = (grid_size / 4.0) ** 2
    
    # Initialize probability calculator ONLY if using competitive distance
    prob_params_list = None
    use_kdtree = False
    
    if use_competitive_distance:
        use_kdtree = max(len(b) for b in boundary_sets if len(b) > 0) > 100
        print(f"\nUsing {'KD-tree' if use_kdtree else 'Numba'} for probability calculation")
        
        prob_params_list = create_competitive_prob_params(
            boundary_sets, seed_ids, grid, beta, use_kdtree
        )
    else:
        print("\nNo probability calculation needed (pure Eden)")
    
    # Setup animation
    animation_frames = []
    if create_animation:
        if animation_file is None:
            if output_file:
                base = Path(output_file).stem
                mode_suffix = "competitive" if use_competitive_distance else "pure_eden"
                animation_file = str(Path(output_file).parent / f"{base}_{mode_suffix}_animation.gif")
            else:
                mode_suffix = "competitive" if use_competitive_distance else "pure_eden"
                animation_file = f"{mode_suffix}_growth_animation.gif"
        print(f"Animation will be saved to: {animation_file}")
        animation_frames.append(grid.copy())
    
    # Open output
    if output_file:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['timestep', 'l', 'w', 'cells', 'boundary_size', 'total_urbanized'])
        io_buffer = []
    else:
        f = None
        writer = None
        io_buffer = None
    
    # Growth loop
    print("\nGrowing...")
    t = 0
    iteration = 0
    N_list_cache = None
    prev_len = 0
    
    # Initialize merger_map as numpy array
    max_seed_id = int(np.max(seed_ids))
    merger_map_array = np.arange(max_seed_id + 1, dtype=np.int64)
    
    while t < timesteps:
        active_seeds = [i for i in range(len(boundary_sets)) if len(boundary_sets[i]) > 0]
        
        if not active_seeds:
            print("\nAll boundaries exhausted")
            break
        
        # Update probability calculator periodically (only if using competitive distance)
        if use_competitive_distance and iteration % update_frequency == 0:
            prob_params_list = create_competitive_prob_params(
                boundary_sets, seed_ids, grid, beta, use_kdtree
            )
        
        # Grow all active seeds
        for seed_idx in active_seeds:
            seed_id = int(seed_ids[seed_idx])
            
            if use_competitive_distance:
                # Competitive distance growth
                prob_calc = prob_params_list[seed_idx]
                
                grid, boundary_sets[seed_idx] = eden_growth_step_competitive(
                    grid, boundary_sets[seed_idx], seed_id, beta, gamma,
                    boundary_sets, seed_ids, use_kdtree, prob_calc,
                    use_spatial_filter, center_i, center_j, spatial_variance
                )
            else:
                # Pure Eden growth
                grid, boundary_sets[seed_idx] = eden_growth_step_pure(
                    grid, boundary_sets[seed_idx], seed_id
                )
            
        t += 1
        if t >= timesteps:
            break
        
        iteration += 1
        
        # Check merges
        if iteration % merge_check_frequency == 0:
            mergers = check_and_merge_clusters_optimized(
                grid, boundary_sets, seed_ids, merger_map_array, grid_size
            )
            if mergers:
                print(f"\n  Mergers at iteration {iteration}")
        
        # Capture animation frame
        if create_animation and t % animation_interval == 0:
            animation_frames.append(grid.copy())
        
        # Metrics
        if t % metric_timestep == 0:
            print(f"Step {t:,}/{timesteps:,}", end='\r')
            
            if writer:
                seed_sizes = {sid: np.sum(grid == sid) for sid in seed_ids}
                largest_id = max(seed_sizes, key=seed_sizes.get)
                
                # Calculate total urbanized area
                total_urbanized = np.sum(grid > 0)
                
                mask = (grid == largest_id).astype(np.int8)
                if np.sum(mask) > 10:
                    thetas, r = radial_profile_from_grid_parallel(mask, N_sampling, 0.5)
                    
                    if N_list_cache is None or len(r) > prev_len * 1.1:
                        max_N = len(r) // 5
                        if max_N >= 5:
                            N_list_cache = np.logspace(
                                np.log10(5), np.log10(max_N),
                                min(num_N, max_N - 4), dtype=np.int64
                            )
                            N_list_cache = np.unique(N_list_cache)
                            prev_len = len(r)
                    
                    ell, w = width_vs_arc_length_optimized(thetas, r, N_list_cache)
                    
                    for ii in range(len(N_list_cache)):
                        io_buffer.append([t, ell[ii], w[ii], seed_sizes[largest_id],
                                        len(boundary_sets[largest_id - 1]), total_urbanized])
                
                if len(io_buffer) >= 1000:
                    writer.writerows(io_buffer)
                    io_buffer.clear()
                    f.flush()
    
    print()
    
    if io_buffer and writer:
        writer.writerows(io_buffer)
    if f:
        f.close()
    
    # Create animation
    if create_animation and len(animation_frames) > 0:
        print("\nCreating animation...")
        mode_str = f"β={beta}, γ={gamma}" if use_competitive_distance else "Pure Eden"
        create_growth_animation(
            animation_frames, 
            seed_ids, 
            animation_file,
            fps=10,
            title=f"Growth ({mode_str})"
        )
        print(f"✓ Animation saved: {animation_file}")
    
    # Final stats
    all_stats = {}
    for sid in seed_ids:
        n_cells = np.sum(grid == sid)
        if n_cells > 0:
            absorbed_seeds = [int(k) for k in range(len(merger_map_array)) 
                            if merger_map_array[k] == sid and k != sid and k != 0]
            
            all_stats[sid] = {
                'cells': n_cells,
                'percentage': n_cells / (grid_size**2) * 100,
                'initial_size': initial_sizes[sid],
                'absorbed_seeds': absorbed_seeds
            }
    
    if len(all_stats) == 0:
        print("\n⚠ All seeds were absorbed!")
        return grid, {'cells': 0, 'percentage': 0}, {}
    
    largest_id = max(all_stats, key=lambda k: all_stats[k]['cells'])
    largest_seed_stats = all_stats[largest_id]
    
    print("\n" + "=" * 70)
    print(f"LARGEST: Seed {largest_id} ({largest_seed_stats['cells']:,} cells)")
    if largest_seed_stats['absorbed_seeds']:
        absorbed_list = ', '.join(map(str, largest_seed_stats['absorbed_seeds']))
        print(f"  Absorbed: Seeds {absorbed_list}")
    print("=" * 70)
    
    return grid, largest_seed_stats, all_stats







# =======================================================================
# PARALLEL ENSEMBLE SIMULATIONS
# ============================================================================

import multiprocessing as mp
from functools import partial
import time

def run_single_realization(
    realization_id,
    grid_size,
    seed_configs,
    timesteps,
    output_dir,
    metric_timestep,
    beta,
    gamma,
    update_frequency,
    merge_check_frequency,
    N_sampling,
    num_N,
    use_competitive_distance,
    use_spatial_filter,
    spatial_variance,
    save_individual_outputs,
    save_individual_grids,
    random_seed_offset
):
    """
    Run a single realization of the simulation.
    
    This function is designed to be called in parallel.
    """
    # Set unique random seed for this realization
    np.random.seed(random_seed_offset + realization_id)
    random.seed(random_seed_offset + realization_id)
    
    # Create output file path
    if save_individual_outputs:
        output_file = str(Path(output_dir) / f"realization_{realization_id:03d}.csv")
    else:
        output_file = None
    
    start_time = time.time()
    
    # Run simulation
    grid, largest_stats, all_stats = simulate_with_competitive_distance(
        grid_size=grid_size,
        seed_configs=seed_configs,
        timesteps=timesteps,
        output_file=output_file,
        metric_timestep=metric_timestep,
        beta=beta,
        gamma=gamma,
        update_frequency=update_frequency,
        merge_check_frequency=merge_check_frequency,
        N_sampling=N_sampling,
        num_N=num_N,
        create_animation=False,  # Don't create animations in parallel runs
        use_competitive_distance=use_competitive_distance,
        use_spatial_filter=use_spatial_filter,
        spatial_variance=spatial_variance
    )
    
    elapsed_time = time.time() - start_time
    
    # Save grid if requested
    if save_individual_grids:
        grid_file = str(Path(output_dir) / f"grid_{realization_id:03d}.npy")
        np.save(grid_file, grid)
    
    # Return summary statistics
    result = {
        'realization_id': realization_id,
        'elapsed_time': elapsed_time,
        'largest_seed_id': max(all_stats, key=lambda k: all_stats[k]['cells']) if all_stats else None,
        'largest_seed_cells': largest_stats['cells'] if largest_stats else 0,
        'largest_seed_percentage': largest_stats['percentage'] if largest_stats else 0,
        'n_surviving_seeds': len(all_stats),
        'all_stats': all_stats,
        'grid_shape': grid.shape
    }
    
    print(f"✓ Realization {realization_id} completed in {elapsed_time:.1f}s")
    
    return result


def run_parallel_ensemble(
    n_realizations,
    grid_size,
    seed_configs,
    timesteps,
    output_dir='ensemble_output',
    n_workers=None,
    metric_timestep=100,
    beta=1.0,
    gamma=1.0,
    update_frequency=10,
    merge_check_frequency=100,
    N_sampling=10000,
    num_N=20,
    use_competitive_distance=True,
    use_spatial_filter=False,
    spatial_variance=None,
    save_individual_outputs=True,
    save_individual_grids=False,
    random_seed_base=None
):
    """
    Run multiple independent realizations of the simulation in parallel.
    
    Parameters:
    -----------
    n_realizations : int
        Number of independent simulations to run
    grid_size : int
        Grid size for each simulation
    seed_configs : list of dict
        Seed configurations (same for all realizations)
    timesteps : int
        Number of growth steps per simulation
    output_dir : str
        Directory to save outputs
    n_workers : int, optional
        Number of parallel workers (default: CPU count - 1)
    metric_timestep : int
        Measurement frequency
    beta : float
        Distance exponent (if using competitive distance)
    gamma : float
        Probability exponent (if using competitive distance)
    update_frequency : int
        Update frequency for probability calculator
    merge_check_frequency : int
        Merge check frequency
    N_sampling : int
        Number of angles for radial profile
    num_N : int
        Number of N values for width analysis
    use_competitive_distance : bool
        If True, use competitive distance; if False, use pure Eden
    use_spatial_filter : bool
        If True, apply Gaussian spatial filter
    spatial_variance : float, optional
        Variance for spatial filter
    save_individual_outputs : bool
        If True, save CSV for each realization
    save_individual_grids : bool
        If True, save final grid as .npy for each realization
    random_seed_base : int, optional
        Base random seed (default: current time)
    
    Returns:
    --------
    results : list of dict
        Summary statistics for each realization
    aggregate_stats : dict
        Aggregated statistics across all realizations
    """
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set default number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    
    # Set random seed base
    if random_seed_base is None:
        random_seed_base = int(time.time() * 1000) % 1000000
    


    print("=" * 70)
    print("PARALLEL ENSEMBLE SIMULATION")
    print("=" * 70)
    print(f"Realizations: {n_realizations}")
    print(f"Workers: {n_workers}")
    print(f"Grid: {grid_size}×{grid_size}")
    print(f"Seeds: {len(seed_configs)}")
    print(f"Steps per realization: {timesteps:,}")
    if use_competitive_distance:
        print(f"Mode: Competitive Distance (β={beta}, γ={gamma})")
    else:
        print(f"Mode: Pure Eden")
    print(f"Output directory: {output_dir}")
    print(f"Random seed base: {random_seed_base}")
    print("=" * 70)
    
    start_time = time.time()
    
    # Create partial function with fixed parameters
    run_func = partial(
        run_single_realization,
        grid_size=grid_size,
        seed_configs=seed_configs,
        timesteps=timesteps,
        output_dir=output_dir,
        metric_timestep=metric_timestep,
        beta=beta,
        gamma=gamma,
        update_frequency=update_frequency,
        merge_check_frequency=merge_check_frequency,
        N_sampling=N_sampling,
        num_N=num_N,
        use_competitive_distance=use_competitive_distance,
        use_spatial_filter=use_spatial_filter,
        spatial_variance=spatial_variance,
        save_individual_outputs=save_individual_outputs,
        save_individual_grids=save_individual_grids,
        random_seed_offset=random_seed_base
    )
    
    # Run simulations in parallel
    print(f"\nStarting {n_realizations} parallel simulations...")
    
    with mp.Pool(processes=n_workers) as pool:
        results = pool.map(run_func, range(n_realizations))
    
    total_time = time.time() - start_time
    
    # Aggregate statistics
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)
    
    aggregate_stats = aggregate_ensemble_results(results, seed_configs)
    
    # Save aggregate results
    save_aggregate_results(results, aggregate_stats, output_path)
    
    print(f"\n✓ All {n_realizations} realizations completed in {total_time:.1f}s")
    print(f"  Average time per realization: {total_time/n_realizations:.1f}s")
    print(f"  Speedup vs sequential: {sum(r['elapsed_time'] for r in results)/total_time:.1f}x")
    
    return results, aggregate_stats


def aggregate_ensemble_results(results, seed_configs):
    """
    Aggregate statistics across all realizations.
    
    Returns:
    --------
    aggregate_stats : dict
        Contains mean, std, min, max for various metrics
    """
    n_realizations = len(results)
    n_seeds = len(seed_configs)
    
    # Extract metrics
    largest_cells = np.array([r['largest_seed_cells'] for r in results])
    largest_pcts = np.array([r['largest_seed_percentage'] for r in results])
    n_surviving = np.array([r['n_surviving_seeds'] for r in results])
    times = np.array([r['elapsed_time'] for r in results])
    
    # Count wins per seed
    seed_wins = {i+1: 0 for i in range(n_seeds)}
    for r in results:
        if r['largest_seed_id'] is not None:
            seed_wins[r['largest_seed_id']] += 1
    
    aggregate_stats = {
        'n_realizations': n_realizations,
        'n_seeds': n_seeds,
        
        # Largest cluster size
        'largest_cells_mean': np.mean(largest_cells),
        'largest_cells_std': np.std(largest_cells),
        'largest_cells_min': np.min(largest_cells),
        'largest_cells_max': np.max(largest_cells),
        
        # Largest cluster percentage
        'largest_pct_mean': np.mean(largest_pcts),
        'largest_pct_std': np.std(largest_pcts),
        'largest_pct_min': np.min(largest_pcts),
        'largest_pct_max': np.max(largest_pcts),
        
        # Surviving seeds
        'n_surviving_mean': np.mean(n_surviving),
        'n_surviving_std': np.std(n_surviving),
        'n_surviving_min': np.min(n_surviving),
        'n_surviving_max': np.max(n_surviving),
        
        # Computation time
        'time_mean': np.mean(times),
        'time_std': np.std(times),
        'time_min': np.min(times),
        'time_max': np.max(times),
        
        # Win statistics
        'seed_wins': seed_wins,
        'seed_win_rates': {k: v/n_realizations for k, v in seed_wins.items()}
    }
    
    # Print summary
    print(f"\nLargest cluster size: {aggregate_stats['largest_cells_mean']:.0f} ± "
          f"{aggregate_stats['largest_cells_std']:.0f} cells")
    print(f"Largest cluster %: {aggregate_stats['largest_pct_mean']:.1f} ± "
          f"{aggregate_stats['largest_pct_std']:.1f}%")
    print(f"Surviving seeds: {aggregate_stats['n_surviving_mean']:.1f} ± "
          f"{aggregate_stats['n_surviving_std']:.1f}")
    
    print("\nWin rates per seed:")
    for seed_id, win_rate in sorted(aggregate_stats['seed_win_rates'].items()):
        print(f"  Seed {seed_id}: {win_rate*100:.1f}% ({seed_wins[seed_id]}/{n_realizations} wins)")
    
    return aggregate_stats


def save_aggregate_results(results, aggregate_stats, output_path):
    """Save aggregated results to files."""
    
    # Save summary statistics
    summary_file = output_path / 'aggregate_summary.txt'
    with open(summary_file, 'w') as f:
        f.write("ENSEMBLE SIMULATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Number of realizations: {aggregate_stats['n_realizations']}\n")
        f.write(f"Number of seeds: {aggregate_stats['n_seeds']}\n\n")
        
        f.write("LARGEST CLUSTER SIZE:\n")
        f.write(f"  Mean: {aggregate_stats['largest_cells_mean']:.0f}\n")
        f.write(f"  Std:  {aggregate_stats['largest_cells_std']:.0f}\n")
        f.write(f"  Min:  {aggregate_stats['largest_cells_min']:.0f}\n")
        f.write(f"  Max:  {aggregate_stats['largest_cells_max']:.0f}\n\n")
        
        f.write("LARGEST CLUSTER PERCENTAGE:\n")
        f.write(f"  Mean: {aggregate_stats['largest_pct_mean']:.2f}%\n")
        f.write(f"  Std:  {aggregate_stats['largest_pct_std']:.2f}%\n")
        f.write(f"  Min:  {aggregate_stats['largest_pct_min']:.2f}%\n")
        f.write(f"  Max:  {aggregate_stats['largest_pct_max']:.2f}%\n\n")
        
        f.write("SURVIVING SEEDS:\n")
        f.write(f"  Mean: {aggregate_stats['n_surviving_mean']:.2f}\n")
        f.write(f"  Std:  {aggregate_stats['n_surviving_std']:.2f}\n")
        f.write(f"  Min:  {aggregate_stats['n_surviving_min']:.0f}\n")
        f.write(f"  Max:  {aggregate_stats['n_surviving_max']:.0f}\n\n")
        
        f.write("WIN RATES PER SEED:\n")
        for seed_id, win_rate in sorted(aggregate_stats['seed_win_rates'].items()):
            wins = aggregate_stats['seed_wins'][seed_id]
            f.write(f"  Seed {seed_id}: {win_rate*100:.1f}% ({wins}/{aggregate_stats['n_realizations']})\n")
    
    print(f"\n✓ Summary saved to: {summary_file}")
    
    # Save detailed results as CSV
    results_file = output_path / 'all_realizations.csv'
    with open(results_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['realization_id', 'largest_seed_id', 'largest_cells', 
                        'largest_pct', 'n_surviving', 'elapsed_time'])
        
        for r in results:
            writer.writerow([
                r['realization_id'],
                r['largest_seed_id'],
                r['largest_seed_cells'],
                r['largest_seed_percentage'],
                r['n_surviving_seeds'],
                r['elapsed_time']
            ])
    
    print(f"✓ Detailed results saved to: {results_file}")