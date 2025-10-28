import numpy as np
from numba import njit, prange
import random
import csv


import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path

import time
from competitive_distance_probability import (
    CompetitiveDistanceCalculator,
    create_competitive_prob_params
)

# Import optimized radial/width functions
from eden_growth_optimized import (
    radial_profile_from_grid_parallel,
    width_vs_arc_length_optimized
)

# ========== HELPER FUNCTIONS ==========

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


# ========== ROUGHENED SEED INITIALIZATION ==========

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
    roughness : float
        Amount of roughness (0 = smooth circle, 1 = very rough)
        Controls the standard deviation of radial variation
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
    
    # Generate random radial variations around the circle
    # Use angular samples to create roughness
    n_angles = max(int(2 * np.pi * radius), 20)
    angles = np.linspace(0, 2*np.pi, n_angles, endpoint=False)
    
    # Create radial variations using random noise
    np.random.seed(None)  # Ensure randomness
    radial_noise = np.random.normal(0, roughness * radius, n_angles)
    varied_radii = radius + radial_noise
    
    # Ensure radii stay positive and reasonable
    varied_radii = np.clip(varied_radii, radius * 0.3, radius * 1.7)
    
    # Smooth the radial variations to avoid single-pixel noise
    # Apply a simple moving average
    window_size = max(3, n_angles // 20)
    smoothed_radii = np.convolve(
        np.concatenate([varied_radii[-window_size:], varied_radii, varied_radii[:window_size]]),
        np.ones(window_size) / window_size,
        mode='valid'
    )
    
    # Place cells within the roughened boundary
    n_cells = 0
    for i in range(max(0, center_i - int(radius * 2)), 
                   min(rows, center_i + int(radius * 2) + 1)):
        for j in range(max(0, center_j - int(radius * 2)), 
                       min(cols, center_j + int(radius * 2) + 1)):
            
            # Calculate angle and distance from center
            di = i - center_i
            dj = j - center_j
            dist = np.sqrt(di**2 + dj**2)
            angle = np.arctan2(dj, di)
            
            # Find the corresponding varied radius for this angle
            angle_idx = int((angle + np.pi) / (2*np.pi) * n_angles) % n_angles
            threshold_radius = smoothed_radii[angle_idx]
            
            # Place cell if within the varied radius
            if dist <= threshold_radius and grid[i, j] == 0:
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
        
        # Create roughened seed
        grid, n_cells = create_roughened_seed(
            grid, position[0], position[1], radius, roughness, seed_id
        )
        
        initial_sizes[seed_id] = n_cells
        seed_ids.append(seed_id)
        
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
    
    return grid, boundary_sets, np.array(seed_ids), initial_sizes


# ========== PROBABILITY-WEIGHTED GROWTH ==========

@njit
def calculate_cell_probability(grid, i, j, seed_id, prob_function_type=0, beta=1.0,use_kdtree=False,
                                prob_params=None):
    """
    Calculate the probability weight for a boundary cell.
    
    Parameters:
    -----------
    grid : np.ndarray
        Current grid state
    i, j : int
        Cell coordinates
    seed_id : int
        ID of the seed trying to claim this cell
    prob_function_type : int
        Type of probability function:
        0 = uniform (all cells equal probability)
        1 = neighbor_count (prefer cells with more occupied neighbors)
  
    Returns:
    --------
    probability : float
        Probability weight for this cell (higher = more likely)
    """
    rows, cols = grid.shape
    
    if prob_function_type == 0:
        # Uniform probability
        return 1.0
    if prob_function_type == 1:

        
        return 1.0

    
    


@njit
def update_boundary_weighted(grid, boundary_set, new_i, new_j, seed_id):
    """Update boundary after urbanizing a cell."""
    rows, cols = grid.shape
    
    if (new_i, new_j) in boundary_set:
        boundary_set.remove((new_i, new_j))
    
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    
    for ni, nj in neighbors:
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return boundary_set


def eden_growth_step_weighted(grid, boundary_set, seed_id, 
                               prob_function_type=0, prob_params=None):

    if len(boundary_set) == 0:
        return grid, boundary_set
    
    # Convert boundary to array
    boundary_array = np.empty((len(boundary_set), 2), dtype=np.int64)
    probabilities = np.empty(len(boundary_set), dtype=np.float64)
    
    idx = 0
    for item in boundary_set:
        boundary_array[idx, 0] = item[0]
        boundary_array[idx, 1] = item[1]
        
        # Calculate probability for this cell
        probabilities[idx] = calculate_cell_probability(
            grid, item[0], item[1], seed_id, 
            prob_function_type, prob_params
        )
        idx += 1
    
    # Normalize probabilities
    prob_sum = np.sum(probabilities)
    if prob_sum > 0:
        probabilities = probabilities / prob_sum
    else:
        probabilities = np.ones(len(probabilities)) / len(probabilities)
    
    # Select cell based on probability distribution
    cumsum = np.cumsum(probabilities)
    rand_val = np.random.random()
    selected_idx = np.searchsorted(cumsum, rand_val)
    
    # Ensure valid index
    if selected_idx >= len(boundary_array):
        selected_idx = len(boundary_array) - 1
    
    new_i = boundary_array[selected_idx, 0]
    new_j = boundary_array[selected_idx, 1]
    
    # Urbanize the cell
    grid[new_i, new_j] = seed_id
    
    # Update boundary
    boundary_set = update_boundary_weighted(grid, boundary_set, new_i, new_j, seed_id)
    
    return grid, boundary_set




@njit
def get_cluster_bounding_box(grid, seed_id):
    """
    Get bounding box for a cluster.
    Returns (min_i, max_i, min_j, max_j) or None if cluster empty.
    """
    rows, cols = grid.shape
    min_i, max_i = rows, 0
    min_j, max_j = cols, 0
    found = False
    
    for i in range(rows):
        for j in range(cols):
            if grid[i, j] == seed_id:
                found = True
                if i < min_i:
                    min_i = i
                if i > max_i:
                    max_i = i
                if j < min_j:
                    min_j = j
                if j > max_j:
                    max_j = j
    
    if not found:
        return None
    
    return (min_i, max_i, min_j, max_j)


@njit
def bounding_boxes_close(bbox_a, bbox_b, margin=5):
    """
    Check if two bounding boxes are close enough to potentially touch.
    Much faster than checking all cells!
    """
    if bbox_a is None or bbox_b is None:
        return False
    
    min_i_a, max_i_a, min_j_a, max_j_a = bbox_a
    min_i_b, max_i_b, min_j_b, max_j_b = bbox_b
    
    # Check if boxes overlap or are within margin
    i_gap = max(0, min_i_b - max_i_a - 1, min_i_a - max_i_b - 1)
    j_gap = max(0, min_j_b - max_j_a - 1, min_j_a - max_j_b - 1)
    
    # If gap in either dimension is large, they're not close
    return i_gap <= margin and j_gap <= margin


@njit
def clusters_are_touching_optimized(grid, seed_a, seed_b, bbox_a, bbox_b):
    """
    Optimized touching check: only scan bounding box overlap region.
    Much faster than scanning entire grid!
    """
    if bbox_a is None or bbox_b is None:
        return False
    
    min_i_a, max_i_a, min_j_a, max_j_a = bbox_a
    min_i_b, max_i_b, min_j_b, max_j_b = bbox_b
    
    # Find overlap region to scan
    scan_min_i = max(min_i_a - 1, min_i_b - 1, 0)
    scan_max_i = min(max_i_a + 1, max_i_b + 1, grid.shape[0] - 1)
    scan_min_j = max(min_j_a - 1, min_j_b - 1, 0)
    scan_max_j = min(max_j_a + 1, max_j_b + 1, grid.shape[1] - 1)
    
    # Only scan the overlap region (much smaller than full grid!)
    for i in range(scan_min_i, scan_max_i + 1):
        for j in range(scan_min_j, scan_max_j + 1):
            if grid[i, j] == seed_a:
                # Check if any neighbor is seed_b
                neighbors = get_neighbors(i, j, grid.shape[0], grid.shape[1])
                for ni, nj in neighbors:
                    if grid[ni, nj] == seed_b:
                        return True
    
    return False


def check_and_merge_clusters_optimized(grid, boundary_sets, seed_ids, merger_map, grid_size):
    """
    OPTIMIZED merge checking:
    1. Get bounding boxes for all clusters
    2. Only check pairs whose boxes are close
    3. Scan only overlap regions (not full grid)
    
    Expected speedup: 10-50× over naive version!
    """
    mergers_occurred = False
    
    # Find active seeds and their bounding boxes
    active_seeds = []
    bboxes = {}
    
    for sid in seed_ids:
        if np.sum(grid == sid) > 0:
            active_seeds.append(int(sid))
            bboxes[int(sid)] = get_cluster_bounding_box(grid, sid)
    
    # Check pairs - but only if bounding boxes are close!
    for i in range(len(active_seeds)):
        for j in range(i + 1, len(active_seeds)):
            seed_a = active_seeds[i]
            seed_b = active_seeds[j]
            
            # Fast rejection: are bounding boxes even close?
            if not bounding_boxes_close(bboxes[seed_a], bboxes[seed_b], margin=5):
                continue  # Skip expensive touching check!
            
            # Boxes are close, check if actually touching
            if clusters_are_touching_optimized(grid, seed_a, seed_b, 
                                              bboxes[seed_a], bboxes[seed_b]):
                # Merge: larger absorbs smaller
                size_a = np.sum(grid == seed_a)
                size_b = np.sum(grid == seed_b)
                
                if size_a >= size_b:
                    dominant, absorbed = seed_a, seed_b
                else:
                    dominant, absorbed = seed_b, seed_a
                
                # Perform merge
                merge_clusters(grid, boundary_sets, seed_ids, 
                             dominant, absorbed, merger_map)
                
                mergers_occurred = True
                
                print(f"    Seed {absorbed} merged into Seed {dominant} "
                      f"(bboxes were close)")
                
                # Update bounding boxes after merge
                bboxes[dominant] = get_cluster_bounding_box(grid, dominant)
                bboxes[absorbed] = None
    
    return mergers_occurred


def merge_clusters(grid, boundary_sets, seed_ids, dominant, absorbed, merger_map):
    """Merge absorbed cluster into dominant cluster."""
    # Find indices
    dominant_idx = np.where(seed_ids == dominant)[0][0]
    absorbed_idx = np.where(seed_ids == absorbed)[0][0]
    
    # Relabel cells
    grid[grid == absorbed] = dominant
    
    # Merge boundaries
    boundary_sets[dominant_idx].update(boundary_sets[absorbed_idx])
    
    # Remove occupied cells from boundary
    cells_to_remove = set()
    for cell in boundary_sets[dominant_idx]:
        i, j = cell
        if grid[i, j] != 0:
            cells_to_remove.add(cell)
    
    for cell in cells_to_remove:
        boundary_sets[dominant_idx].discard(cell)
    
    # Clear absorbed boundary
    boundary_sets[absorbed_idx].clear()
    
    # Update merger map
    merger_map[absorbed] = dominant
    for key in merger_map:
        if merger_map[key] == absorbed:
            merger_map[key] = dominant


# ========== OPTIMIZED MAIN SIMULATION ==========

def create_growth_animation(
    frames, 
    seed_ids, 
    output_file, 
    fps=10, 
    title="Competitive Growth",
    dpi=100
):
    """
    Create animated GIF from captured frames.
    
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
    dpi : int
        Resolution
    """
    if len(frames) == 0:
        print("No frames to animate!")
        return
    
    n_seeds = len(seed_ids)
    
    # Create colormap
    if n_seeds == 1:
        colors = ['white', 'lightblue', 'steelblue']
        cmap = mcolors.LinearSegmentedColormap.from_list('custom', colors, N=3)
    else:
        base_colors = ['white', 'tab:blue', 'tab:orange', 'tab:green', 
                      'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',
                      'tab:gray', 'tab:olive', 'tab:cyan']
        colors_to_use = base_colors[:n_seeds+1]
        cmap = mcolors.ListedColormap(colors_to_use)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Initial plot
    im = ax.imshow(frames[0], cmap=cmap, interpolation='nearest', 
                   vmin=0, vmax=n_seeds)
    ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if n_seeds > 1:
        cbar.set_ticks(range(n_seeds + 1))
        labels = ['Empty'] + [f'Seed {i+1}' for i in range(n_seeds)]
        cbar.set_ticklabels(labels)
    
    # Title with step counter
    title_text = ax.text(0.5, 1.02, '', transform=ax.transAxes,
                        ha='center', fontsize=12, fontweight='bold')
    
    # Cell count text
    count_text = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                        ha='left', va='top', fontsize=10,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def update(frame_idx):
        """Update function for animation."""
        frame = frames[frame_idx]
        im.set_array(frame)
        
        # Update title
        title_text.set_text(f'{title} - Frame {frame_idx}/{len(frames)-1}')
        
        # Update cell count
        n_cells = np.sum(frame > 0)
        pct = (n_cells / frame.size) * 100
        count_text.set_text(f'{n_cells:,} cells ({pct:.1f}%)')
        
        return [im, title_text, count_text]
    
    # Create animation
    anim = FuncAnimation(fig, update, frames=len(frames), 
                        interval=1000/fps, blit=True, repeat=True)
    
    # Save as GIF
    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=dpi)
    
    plt.close(fig)



def simulate_with_competitive_distance(
    grid_size,
    seed_configs,
    timesteps,
    output_file,
    metric_timestep,
    beta=1.0,
    gamma=1.0,
    update_frequency=50,  # Update probability data every N iterations
    N_sampling=5000,
    num_N=15,
    merge_check_frequency=500,
    create_animation=False,  # NEW: Enable/disable animation
    animation_interval=100,  # NEW: Capture frame every N steps
    animation_file=None  # NEW: Output GIF filename (auto-generated if None)
):
    """
    Multi-seed simulation with competitive distance probability.
    
    Parameters:
    -----------
    grid_size : int
        Grid size
    seed_configs : list of dict
        Seed configurations
    timesteps : int
        Total growth steps
    output_file : str
        Output CSV file
    metric_timestep : int
        Measurement frequency
    beta : float
        Distance exponent (0-2)
        - 0: uniform (no distance effect)
        - 1: linear distance penalty
        - 2: strong distance penalty
    update_frequency : int
        Update probability calculator every N iterations
        Higher = faster but less accurate
        Lower = slower but more accurate
    N_sampling : int
        Angles for radial profile
    num_N : int
        N values for width analysis
    create_animation : bool
        If True, create animated GIF showing growth over time
    animation_interval : int
        Capture frame every N steps (lower = smoother but larger file)
    animation_file : str, optional
        Output GIF filename. If None, auto-generates from output_file
    
    Returns:
    --------
    grid : np.ndarray
        Final grid
    largest_seed_stats : dict
        Statistics for largest seed
    all_stats : dict
        Statistics for all seeds
    """

    from eden_growth_optimized import (
        radial_profile_from_grid_parallel,
        width_vs_arc_length_optimized
    )
  
    import csv
    
    print("="*70)
    print("COMPETITIVE DISTANCE SIMULATION")
    print("="*70)
    print(f"Grid: {grid_size}×{grid_size}")
    print(f"Seeds: {len(seed_configs)}")
    print(f"Steps: {timesteps:,}")
    print(f"Beta: {beta}")
    print(f"Update frequency: every {update_frequency} iterations")
    if create_animation:
        print(f"Animation: ON (every {animation_interval} steps)")
    print("="*70)
    
    # Initialize
    grid, boundary_sets, seed_ids, initial_sizes = initialize_roughened_seeds(
        grid_size, seed_configs
    )
    
    print("\nInitial sizes:")
    for sid, size in initial_sizes.items():
        print(f"  Seed {sid}: {size} cells")
    
    # Initialize probability calculator
    use_kdtree = max(len(b) for b in boundary_sets if len(b) > 0) > 100
    print(f"\nUsing {'KD-tree' if use_kdtree else 'Numba'} for probability calculation")
    
    prob_params_list = create_competitive_prob_params(
        boundary_sets, seed_ids, grid, beta, use_kdtree
    )
    
    # Setup animation if requested
    animation_frames = []
    if create_animation:
        # Determine output filename
        if animation_file is None:
            if output_file:
                from pathlib import Path
                base = Path(output_file).stem
                animation_file = str(Path(output_file).parent / f"{base}_animation.gif")
            else:
                animation_file = "competitive_growth_animation.gif"
        
        print(f"Animation will be saved to: {animation_file}")
        
        # Capture initial frame
        animation_frames.append(grid.copy())
    
    # Open output
    if output_file:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['timestep', 'l', 'w', 'cells', 'boundary_size'])
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
    merger_map = {int(sid): int(sid) for sid in seed_ids}
    
    while t < timesteps:
        # Find active seeds
        active_seeds = [i for i in range(len(boundary_sets)) if len(boundary_sets[i]) > 0]
        
        if not active_seeds:
            print("\nAll boundaries exhausted")
            break
        
        # Update probability calculator periodically
        if iteration % update_frequency == 0:
            prob_params_list = create_competitive_prob_params(
                boundary_sets, seed_ids, grid, beta, use_kdtree
            )
        
        # Grow all active seeds
        for seed_idx in active_seeds:
            seed_id = int(seed_ids[seed_idx])
            prob_calc = prob_params_list[seed_idx]
            
            # Grow using competitive distance
            from competitive_distance_probability import eden_growth_step_competitive
            grid, boundary_sets[seed_idx] = eden_growth_step_competitive(
                grid, boundary_sets[seed_idx], seed_id, beta, gamma,
                boundary_sets, seed_ids, use_kdtree, prob_calc
            )
            
            t += 1
            if t >= timesteps:
                break
        
        iteration += 1
        
        # Check merges
        if iteration % merge_check_frequency == 0:
            mergers = check_and_merge_clusters_optimized(
                grid, boundary_sets, seed_ids, merger_map, grid_size
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
                
                mask = (grid == largest_id).astype(np.int8)
                if np.sum(mask) > 10:
                    thetas, r = radial_profile_from_grid_parallel(mask, N_sampling, 0.5)
                    
                    if N_list_cache is None or len(r) > prev_len * 1.1:
                        max_N = len(r) // 5
                        if max_N >= 5:
                            N_list_cache = np.logspace(
                                np.log10(5), np.log10(max_N),
                                min(num_N, max_N-4), dtype=np.int64
                            )
                            N_list_cache = np.unique(N_list_cache)
                            prev_len = len(r)
                    
                    ell, w = width_vs_arc_length_optimized(thetas, r, N_list_cache)
                    
                    for ii in range(len(N_list_cache)):
                        io_buffer.append([t, ell[ii], w[ii], seed_sizes[largest_id],
                                        len(boundary_sets[largest_id-1])])
                
                if len(io_buffer) >= 1000:
                    writer.writerows(io_buffer)
                    io_buffer.clear()
                    f.flush()
    
    print()
    
    if io_buffer and writer:
        writer.writerows(io_buffer)
    if f:
        f.close()
    
    # Create animation if requested
    if create_animation and len(animation_frames) > 0:
        print("\nCreating animation...")
        create_growth_animation(
            animation_frames, 
            seed_ids, 
            animation_file,
            fps=10,
            title=f"Competitive Growth (β={beta})"
        )
        print(f"✓ Animation saved: {animation_file}")
    
    # Final stats
    all_stats = {}
    for sid in seed_ids:
        n_cells = np.sum(grid == sid)
        if n_cells > 0:
            all_stats[sid] = {
                'cells': n_cells,
                'percentage': n_cells / (grid_size**2) * 100,
                'initial_size': initial_sizes[sid],
                'absorbed_seeds': [k for k, v in merger_map.items() if v == sid and k != sid]
            }
    
    if len(all_stats) == 0:
        print("\n⚠ All seeds were absorbed!")
        # Return empty stats
        return grid, {'cells': 0, 'percentage': 0}, {}
    
    largest_id = max(all_stats, key=lambda k: all_stats[k]['cells'])
    largest_seed_stats = all_stats[largest_id]
    
    print("\n" + "="*70)
    print(f"LARGEST: Seed {largest_id} ({largest_seed_stats['cells']:,} cells)")
    if largest_seed_stats['absorbed_seeds']:
        absorbed_list = ', '.join(map(str, largest_seed_stats['absorbed_seeds']))
        print(f"  Absorbed: Seeds {absorbed_list}")
    print("="*70)
    
    return grid, largest_seed_stats, all_stats


# ========== EXAMPLES ==========
