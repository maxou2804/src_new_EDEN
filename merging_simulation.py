"""
CLUSTER MERGING FUNCTIONALITY

This module provides functions to detect and merge clusters that physically touch,
so they grow as a single entity instead of maintaining separate identities.
"""

import numpy as np
from collections import defaultdict


class UnionFind:
    """
    Union-Find (Disjoint Set) data structure for tracking cluster connections.
    """
    def __init__(self, elements):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x):
        """Find the root/representative of x's set."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x, y):
        """Union the sets containing x and y."""
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x == root_y:
            return
        
        # Union by rank
        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1
    
    def get_groups(self):
        """Return a dict mapping each root to its group members."""
        groups = defaultdict(list)
        for element in self.parent:
            root = self.find(element)
            groups[root].append(element)
        return dict(groups)


def detect_touching_clusters(grid, seed_ids):
    """
    Detect which clusters are physically touching (adjacent cells).
    
    Parameters:
    -----------
    grid : np.ndarray
        Current grid state with seed_id values
    seed_ids : np.ndarray
        Array of active seed IDs
    
    Returns:
    --------
    touching_pairs : list of tuples
        List of (seed_id1, seed_id2) pairs that are touching
    """
    touching_pairs = set()
    rows, cols = grid.shape
    
    # Check all adjacent pairs in the grid
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == 0:
                continue
            
            current_id = grid[i, j]
            
            # Check 4-connected neighbors
            neighbors = []
            if i > 0:
                neighbors.append((i-1, j))
            if i < rows - 1:
                neighbors.append((i+1, j))
            if j > 0:
                neighbors.append((i, j-1))
            if j < cols - 1:
                neighbors.append((i, j+1))
            
            for ni, nj in neighbors:
                neighbor_id = grid[ni, nj]
                if neighbor_id > 0 and neighbor_id != current_id:
                    # Found touching clusters
                    pair = tuple(sorted([current_id, neighbor_id]))
                    touching_pairs.add(pair)
    
    return list(touching_pairs)


def merge_clusters(grid, boundary_sets, seed_ids, prob_params=None):
    """
    Merge clusters that are physically touching into single entities.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current grid state
    boundary_sets : list of sets
        Boundary sets for each seed
    seed_ids : np.ndarray
        Array of active seed IDs
    prob_params : list, optional
        Probability parameters for each seed (for competitive distance)
    
    Returns:
    --------
    grid : np.ndarray
        Updated grid with merged cluster IDs
    boundary_sets : list of sets
        Updated boundary sets (merged clusters combined)
    seed_ids : np.ndarray
        Updated seed IDs (absorbed seeds removed)
    prob_params : list or None
        Updated probability parameters
    merge_map : dict
        Mapping from old seed_id to new seed_id (for tracking merges)
    """
    # Detect touching clusters
    touching_pairs = detect_touching_clusters(grid, seed_ids)
    
    if not touching_pairs:
        # No merges needed
        return grid, boundary_sets, seed_ids, prob_params, {}
    
    # Use Union-Find to group connected clusters
    uf = UnionFind(seed_ids)
    for id1, id2 in touching_pairs:
        uf.union(id1, id2)
    
    # Get merged groups
    groups = uf.get_groups()
    
    # If all clusters are in separate groups, no merging needed
    if len(groups) == len(seed_ids):
        return grid, boundary_sets, seed_ids, prob_params, {}
    
    # Create mapping from old seed_id to new seed_id (representative)
    merge_map = {}
    for root, members in groups.items():
        for member in members:
            merge_map[member] = root
    
    # Update grid with new seed IDs
    new_grid = grid.copy()
    for old_id, new_id in merge_map.items():
        if old_id != new_id:
            new_grid[grid == old_id] = new_id
    
    # Merge boundary sets and other data structures
    new_boundary_sets = []
    new_seed_ids = []
    new_prob_params = [] if prob_params is not None else None
    
    # Create index mapping for old seed_ids
    old_id_to_idx = {sid: idx for idx, sid in enumerate(seed_ids)}
    
    for root, members in groups.items():
        # This root represents a merged cluster
        new_seed_ids.append(root)
        
        # Merge all boundary sets from member clusters
        merged_boundary = set()
        for member in members:
            idx = old_id_to_idx[member]
            merged_boundary.update(boundary_sets[idx])
        
        # Remove any cells that are no longer on the boundary
        # (cells that were boundaries between merged clusters)
        valid_boundary = set()
        rows, cols = new_grid.shape
        for bi, bj in merged_boundary:
            # Check if this cell is still on the boundary
            # (i.e., it's empty and adjacent to the merged cluster)
            if new_grid[bi, bj] == 0:
                # Check if adjacent to our cluster
                is_boundary = False
                neighbors = []
                if bi > 0:
                    neighbors.append((bi-1, bj))
                if bi < rows - 1:
                    neighbors.append((bi+1, bj))
                if bj > 0:
                    neighbors.append((bi, bj-1))
                if bj < cols - 1:
                    neighbors.append((bi, bj+1))
                
                for ni, nj in neighbors:
                    if new_grid[ni, nj] == root:
                        is_boundary = True
                        break
                
                if is_boundary:
                    valid_boundary.add((bi, bj))
        
        new_boundary_sets.append(valid_boundary)
        
        # For probability parameters, keep the one from the root cluster
        if prob_params is not None:
            root_idx = old_id_to_idx[root]
            new_prob_params.append(prob_params[root_idx])
    
    new_seed_ids = np.array(new_seed_ids)
    
    return new_grid, new_boundary_sets, new_seed_ids, new_prob_params, merge_map


def should_merge(merge_frequency='always'):
    """
    Determine if merging should occur at this step.
    
    Parameters:
    -----------
    merge_frequency : str or int
        - 'always': Merge every step
        - 'never': Never merge
        - int n: Merge every n steps
    
    Returns:
    --------
    should_merge : bool
    """
    if merge_frequency == 'always':
        return True
    elif merge_frequency == 'never':
        return False
    else:
        # Implement step-based merging if needed
        return True


# ========== MODIFIED SIMULATION LOOP WITH MERGING ==========

def eden_growth_with_merging(
    grid, 
    boundary_sets, 
    seed_ids,
    beta=1.0,
    gamma=1.0,
    total_steps=1000,
    merge_frequency='always',
    use_kdtree=True
):
    """
    Eden growth simulation with automatic cluster merging.
    
    Parameters:
    -----------
    grid : np.ndarray
        Initial grid
    boundary_sets : list of sets
        Initial boundary sets
    seed_ids : np.ndarray
        Initial seed IDs
    beta : float
        Distance exponent for competitive probability
    gamma : float
        Additional parameter for probability calculation
    total_steps : int
        Number of growth steps
    merge_frequency : str or int
        How often to check for merges ('always', 'never', or int)
    use_kdtree : bool
        Whether to use KD-tree for probability calculation
    
    Returns:
    --------
    grid : np.ndarray
        Final grid state
    history : dict
        History of merges and cluster counts
    """
    from competitive_distance_probability import (
        create_competitive_prob_params,
        eden_growth_step_competitive
    )
    
    history = {
        'n_clusters': [len(seed_ids)],
        'merge_events': [],
        'timesteps': [0]
    }
    
    step_count = 0
    
    for step in range(total_steps):
        # Check for merges
        if merge_frequency == 'always' or (isinstance(merge_frequency, int) and step % merge_frequency == 0):
            grid, boundary_sets, seed_ids, _, merge_map = merge_clusters(
                grid, boundary_sets, seed_ids
            )
            
            if merge_map:
                history['merge_events'].append({
                    'step': step,
                    'merge_map': merge_map,
                    'n_clusters_after': len(seed_ids)
                })
                print(f"Step {step}: Merged clusters. Now {len(seed_ids)} clusters remaining.")
        
        # Update probability calculator with current state
        prob_params = create_competitive_prob_params(
            boundary_sets, seed_ids, grid, beta, use_kdtree
        )
        
        # Grow each cluster
        for idx, seed_id in enumerate(seed_ids):
            if len(boundary_sets[idx]) == 0:
                continue
            
            grid, boundary_sets[idx] = eden_growth_step_competitive(
                grid=grid,
                boundary_set=boundary_sets[idx],
                seed_id=seed_id,
                beta=beta,
                gamma=gamma,
                boundary_sets=boundary_sets,
                seed_ids=seed_ids,
                use_kdtree=use_kdtree,
                prob_calculator=prob_params[idx]
            )
            
            step_count += 1
        
        # Record history
        if step % 100 == 0:
            history['n_clusters'].append(len(seed_ids))
            history['timesteps'].append(step)
    
    return grid, history



