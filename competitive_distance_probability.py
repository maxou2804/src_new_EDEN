"""
COMPETITIVE DISTANCE PROBABILITY - Optimized Implementation

P_i = sum_j (m_j / d_ij^beta)

where:
- P_i = probability for cell i
- m_j = mass (perimeter size) of competing cluster j
- d_ij = minimum distance from cell i to cluster j's perimeter
- beta = distance exponent (0-2)

Optimizations:
1. Pre-compute and cache boundary arrays
2. Use KD-tree for fast nearest neighbor queries
3. Numba JIT compilation for distance calculations
4. Only update when boundaries change significantly
"""

import numpy as np
from numba import njit
from scipy.spatial import cKDTree


class CompetitiveDistanceCalculator:
    """
    Efficient calculator for competitive distance probabilities.
    Pre-computes data structures for fast probability queries.
    """
    
    def __init__(self, beta=1.0):
        """
        Initialize calculator.
        
        Parameters:
        -----------
        beta : float
            Distance exponent (0-2)
            - beta=0: uniform (distance doesn't matter)
            - beta=1: linear distance penalty
            - beta=2: strong distance penalty
        """
        self.beta = beta
        self.kdtrees = {}  # KD-trees for each cluster
        self.masses = {}   # Mass (perimeter size) for each cluster
        self.needs_update = True
    
    def update_boundaries(self, boundary_sets, seed_ids, grid):
        """
        Update internal data structures with current boundaries.
        Call this periodically (not every step!) for efficiency.
        
        Parameters:
        -----------
        boundary_sets : list of sets
            Boundary sets for each seed
        seed_ids : np.ndarray
            Array of seed IDs
        grid : np.ndarray
            Current grid state
        """
        self.kdtrees.clear()
        self.masses.clear()
        
        for idx, seed_id in enumerate(seed_ids):
            seed_id = int(seed_id)
            boundary = boundary_sets[idx]
            
            if len(boundary) == 0:
                continue
            
            # Convert boundary to array for KD-tree
            boundary_array = np.array(list(boundary), dtype=np.float64)
            
            # Build KD-tree for fast nearest neighbor queries
            self.kdtrees[seed_id] = cKDTree(boundary_array)
            
            # Store mass (could be perimeter size or cell count)
            self.masses[seed_id] = len(boundary)
        
        self.needs_update = False
    
    def calculate_probability(self, i, j, seed_id):
        """
        Calculate probability for cell (i,j) belonging to seed_id.
        
        Uses formula: P_i = 1 + sum_j (m_j / d_ij^beta)
        where sum is over competing clusters (j != seed_id)
        
        Parameters:
        -----------
        i, j : int
            Cell coordinates
        seed_id : int
            ID of seed trying to claim this cell
        
        Returns:
        --------
        probability : float
            Probability weight (higher = more likely)
        """
        if self.needs_update or len(self.kdtrees) == 0:
            return 1.0
        
        total_prob = 1.0  # Base probability
        cell_pos = np.array([[i, j]], dtype=np.float64)
        
        # Sum contributions from all competing clusters
        for other_id, kdtree in self.kdtrees.items():
            if other_id == seed_id:
                continue  # Skip own cluster
            
            # Find minimum distance to this cluster's perimeter
            dist, _ = kdtree.query(cell_pos, k=1)
            dist = dist[0]
            
            if dist > 0 and other_id in self.masses:
                mass = self.masses[other_id]
                # Add contribution: m_j / d_ij^beta
                total_prob += mass / (dist ** self.beta)
        
        return total_prob


# ========== NUMBA-OPTIMIZED VERSION (NO SCIPY DEPENDENCY) ==========

@njit
def calculate_competitive_probability_numba(
    i, j, seed_id, beta, gamma,
    boundary_arrays, boundary_sizes, masses, n_clusters
):
    """
    Numba-optimized competitive probability calculation.
    Doesn't use KD-tree (slower for large boundaries) but faster for small ones.
    
    Parameters:
    -----------
    i, j : int
        Cell coordinates
    seed_id : int
        Current seed ID
    beta : float
        Distance exponent
    boundary_arrays : np.ndarray
        Shape (n_clusters, max_boundary_size, 2)
        Padded boundary coordinates for all clusters
    boundary_sizes : np.ndarray
        Shape (n_clusters,)
        Actual boundary size for each cluster
    masses : np.ndarray
        Shape (n_clusters,)
        Mass for each cluster
    n_clusters : int
        Number of clusters
    
    Returns:
    --------
    probability : float
        Probability weight
    """
    total_prob = 1.0
    
    for cluster_idx in range(n_clusters):
        # Skip own cluster
        if cluster_idx + 1 == seed_id:
            continue
        
        boundary_size = int(boundary_sizes[cluster_idx])
        if boundary_size == 0:
            continue
        
        # Find minimum distance to this cluster's boundary
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
            
            # Add contribution
            total_prob += mass / (min_dist ** beta)
    total_prob=total_prob**gamma
    return total_prob


def prepare_boundary_data_for_numba(boundary_sets, seed_ids, grid):
    """
    Prepare boundary data in Numba-compatible format.
    
    Returns:
    --------
    boundary_arrays : np.ndarray
        Shape (n_clusters, max_boundary_size, 2)
    boundary_sizes : np.ndarray
        Shape (n_clusters,)
    masses : np.ndarray
        Shape (n_clusters,)
    n_clusters : int
    """
    n_clusters = len(seed_ids)
    max_boundary_size = max(len(b) for b in boundary_sets)
    
    # Initialize arrays
    boundary_arrays = np.zeros((n_clusters, max_boundary_size, 2), dtype=np.int64)
    boundary_sizes = np.zeros(n_clusters, dtype=np.int64)
    masses = np.zeros(n_clusters, dtype=np.float64)
    
    # Fill arrays
    for idx, seed_id in enumerate(seed_ids):
        boundary = boundary_sets[idx]
        boundary_sizes[idx] = len(boundary)
        
        # Store boundary coordinates
        for pt_idx, (i, j) in enumerate(boundary):
            if pt_idx >= max_boundary_size:
                break
            boundary_arrays[idx, pt_idx, 0] = i
            boundary_arrays[idx, pt_idx, 1] = j
        
        # Calculate mass (use cell count)
        masses[idx] = np.sum(grid == seed_id)
    
    return boundary_arrays, boundary_sizes, masses, n_clusters


# ========== INTEGRATION WITH MAIN SIMULATION ==========

def create_competitive_prob_params(
    boundary_sets, seed_ids, grid, beta=1.0, use_kdtree=True
):
    """
    Create prob_params for type 5 (competitive distance).
    
    Parameters:
    -----------
    boundary_sets : list of sets
        Current boundary sets
    seed_ids : np.ndarray
        Seed IDs
    grid : np.ndarray
        Current grid
    beta : float
        Distance exponent (0-2)
    use_kdtree : bool
        If True, use KD-tree (fast for large boundaries)
        If False, use Numba (fast for small boundaries)
    
    Returns:
    --------
    prob_params_list : list
        List of prob_params for each seed
        Each element is either CompetitiveDistanceCalculator or tuple
    """
    if use_kdtree:
        # Use KD-tree approach (best for boundaries > 100 points)
        calculator = CompetitiveDistanceCalculator(beta=beta)
        calculator.update_boundaries(boundary_sets, seed_ids, grid)
        
        # Return same calculator for all seeds
        return [calculator] * len(seed_ids)
    
    else:
        # Use Numba approach (best for boundaries < 100 points)
        boundary_arrays, boundary_sizes, masses, n_clusters = \
            prepare_boundary_data_for_numba(boundary_sets, seed_ids, grid)
        
        # Pack into tuple for Numba function
        params = (beta, boundary_arrays, boundary_sizes, masses, n_clusters)
        
        # Return same params for all seeds
        return [params] * len(seed_ids)


# ========== MODIFIED GROWTH STEP ==========

def eden_growth_step_competitive(
    grid, boundary_set, seed_id, beta, gamma, 
    boundary_sets, seed_ids, use_kdtree=True, prob_calculator=None
):
    """
    Eden growth step with competitive distance probability.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current grid
    boundary_set : set
        Boundary for this seed
    seed_id : int
        Current seed ID
    beta : float
        Distance exponent
    boundary_sets : list of sets
        All boundary sets (for calculating competition)
    seed_ids : np.ndarray
        All seed IDs
    use_kdtree : bool
        Use KD-tree (True) or Numba (False)
    prob_calculator : CompetitiveDistanceCalculator or tuple
        Pre-computed probability calculator/data
    
    Returns:
    --------
    grid : np.ndarray
        Updated grid
    boundary_set : set
        Updated boundary
    """
    if len(boundary_set) == 0:
        return grid, boundary_set
    
    # Convert boundary to array
    boundary_array = np.array(list(boundary_set))
    n_boundary = len(boundary_array)
    
    # Calculate probabilities for all boundary cells
    probabilities = np.zeros(n_boundary, dtype=np.float64)
    
    if use_kdtree and isinstance(prob_calculator, CompetitiveDistanceCalculator):
        # Use KD-tree calculator
        for idx in range(n_boundary):
            i, j = boundary_array[idx]
            probabilities[idx] = prob_calculator.calculate_probability(i, j, seed_id)
    
    else:
        # Use Numba calculator
        beta, boundary_arrays, boundary_sizes, masses, n_clusters = prob_calculator
        
        for idx in range(n_boundary):
            i, j = boundary_array[idx]
            probabilities[idx] = calculate_competitive_probability_numba(
                i, j, seed_id, beta, gamma,
                boundary_arrays, boundary_sizes, masses, n_clusters
            )
    
    # Normalize probabilities
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities /= prob_sum
    else:
        probabilities[:] = 1.0 / n_boundary
    
    # Select cell based on probability
    cumsum = np.cumsum(probabilities)
    rand_val = np.random.random()
    selected_idx = np.searchsorted(cumsum, rand_val)
    
    if selected_idx >= n_boundary:
        selected_idx = n_boundary - 1
    
    new_i, new_j = boundary_array[selected_idx]
    
    # Urbanize cell
    grid[new_i, new_j] = seed_id
    
    # Update boundary
    boundary_set.discard((new_i, new_j))
    
    # Add new neighbors
    from simulation import get_neighbors
    rows, cols = grid.shape
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    
    for ni, nj in neighbors:
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return grid, boundary_set


