import numpy as np
from numba import njit, prange
from numba.typed import List
import random
import csv

# ---------- EDEN GROWTH ------------------
@njit
def get_neighbors(i, j, rows, cols):
    """
    Get valid Neumann neighbors (4-connectivity) for a given cell.
    Returns a list of (row, col) tuples for valid neighbors.
    """
    neighbors = []
    # Up
    if i > 0:
        neighbors.append((i - 1, j))
    # Down
    if i < rows - 1:
        neighbors.append((i + 1, j))
    # Left
    if j > 0:
        neighbors.append((i, j - 1))
    # Right
    if j < cols - 1:
        neighbors.append((i, j + 1))
    
    return neighbors


@njit
def update_boundary(grid, boundary_set, new_i, new_j):
    """
    Update the boundary set after urbanizing a cell.
    Removes the newly urbanized cell and adds its non-urbanized neighbors.
    """
    rows, cols = grid.shape
    
    # Remove the newly urbanized point from boundary
    if (new_i, new_j) in boundary_set:
        boundary_set.remove((new_i, new_j))
    
    # Check all neighbors of the newly urbanized cell
    neighbors = get_neighbors(new_i, new_j, rows, cols)
    
    for ni, nj in neighbors:
        # If neighbor is not urbanized and not already in boundary, add it
        if grid[ni, nj] == 0 and (ni, nj) not in boundary_set:
            boundary_set.add((ni, nj))
    
    return boundary_set


@njit
def eden_growth_step(grid, boundary_list):
    """
    Perform one step of Eden growth.
    
    Parameters:
    -----------
    grid : np.ndarray (2D, dtype=np.int8 or bool)
        The urbanization grid where 1/True = urbanized, 0/False = non-urbanized
    boundary_list : List of tuples
        List of (row, col) indices representing boundary points
    
    Returns:
    --------
    grid : np.ndarray
        Updated grid
    boundary_list : List of tuples
        Updated boundary list
    """
    if len(boundary_list) == 0:
        return grid, boundary_list
    
    # Pick a random boundary point
    idx = random.randint(0, len(boundary_list) - 1)
    new_i, new_j = boundary_list[idx]
    
    # Urbanize the selected point
    grid[new_i, new_j] = 1
    
    # Convert boundary list to set for efficient operations
    boundary_set = set(boundary_list)
    
    # Update boundary
    boundary_set = update_boundary(grid, boundary_set, new_i, new_j)
    
    # Convert back to list
    boundary_list = List(boundary_set)
    
    return grid, boundary_list


@njit
def eden_growth_simulate(grid, boundary_list, n_steps):
    """
    Simulate Eden growth for n_steps.
    
    Parameters:
    -----------
    grid : np.ndarray (2D, dtype=np.int8 or bool)
        The initial urbanization grid
    boundary_list : List of tuples
        Initial list of (row, col) indices representing boundary points
    n_steps : int
        Number of growth steps to simulate
    
    Returns:
    --------
    grid : np.ndarray
        Final grid after n_steps
    boundary_list : List of tuples
        Final boundary list
    """
    for step in range(n_steps):
        if len(boundary_list) == 0:
            break
        grid, boundary_list = eden_growth_step(grid, boundary_list)
    
    return grid, boundary_list


def initialize_boundary(grid, initial_boundary_indices):
    """
    Helper function to convert boundary indices to Numba typed List.
    
    Parameters:
    -----------
    grid : np.ndarray
        The urbanization grid
    initial_boundary_indices : list of tuples
        List of (row, col) indices for initial boundary points
    
    Returns:
    --------
    numba.typed.List
        Typed list compatible with Numba
    """
    boundary_list = List()
    for idx in initial_boundary_indices:
        boundary_list.append(idx)
    return boundary_list

# ---------- PROFILE EXTRACTING --------------


@njit
def radial_profile_from_grid_single(grid, n_theta=10_000, ray_step=0.5):
    """
    Compute r(θ, t) by ray marching from the center.
    Single-threaded optimized version using Numba.
    
    For each angle θ_k = 2πk/n_theta, march along the ray from the center
    and record the furthest radius that intersects an occupied site.
    
    Parameters:
    -----------
    grid : np.ndarray (2D, dtype=np.int8 or bool)
        The urbanization grid where 1/True = urbanized, 0/False = non-urbanized
    n_theta : int
        Number of angular samples (default: 10,000)
    ray_step : float
        Step size for ray marching (default: 0.5)
    
    Returns:
    --------
    thetas : np.ndarray
        Array of angles in [0, 2π)
    r_vals : np.ndarray
        Array of maximum radii for each angle
    """
    h, w = grid.shape
    cx = (h - 1) / 2.0
    cy = (w - 1) / 2.0
    
    # Pre-compute angles and trig values (manually to avoid endpoint parameter)
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
    
    # Ray march for each angle
    for k in range(n_theta):
        ct = cos_t[k]
        st = sin_t[k]
        r = 0.0
        r_last = 0.0
        
        while r <= rmax:
            x = cx + r * ct
            y = cy + r * st
            
            # Round to nearest lattice cell
            ix = int(round(x))
            iy = int(round(y))
            
            # Check bounds
            if ix < 0 or ix >= h or iy < 0 or iy >= w:
                break
            
            # Check if occupied
            if grid[ix, iy]:
                r_last = r
            
            r += ray_step
        
        r_vals[k] = r_last
    
    return thetas, r_vals


@njit(parallel=True)
def radial_profile_from_grid_parallel(grid, n_theta=10_000, ray_step=0.5):
    """
    Compute r(θ, t) by ray marching from the center.
    Multi-threaded optimized version using Numba parallel execution.
    
    For each angle θ_k = 2πk/n_theta, march along the ray from the center
    and record the furthest radius that intersects an occupied site.
    
    Parameters:
    -----------
    grid : np.ndarray (2D, dtype=np.int8 or bool)
        The urbanization grid where 1/True = urbanized, 0/False = non-urbanized
    n_theta : int
        Number of angular samples (default: 10,000)
    ray_step : float
        Step size for ray marching (default: 0.5)
    
    Returns:
    --------
    thetas : np.ndarray
        Array of angles in [0, 2π)
    r_vals : np.ndarray
        Array of maximum radii for each angle
    """
    h, w = grid.shape
    cx = (h - 1) / 2.0
    cy = (w - 1) / 2.0
    
    # Pre-compute angles and trig values (manually to avoid endpoint parameter)
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
    
    # Parallel ray march for each angle
    for k in prange(n_theta):
        ct = cos_t[k]
        st = sin_t[k]
        r = 0.0
        r_last = 0.0
        
        while r <= rmax:
            x = cx + r * ct
            y = cy + r * st
            
            # Round to nearest lattice cell
            ix = int(round(x))
            iy = int(round(y))
            
            # Check bounds
            if ix < 0 or ix >= h or iy < 0 or iy >= w:
                break
            
            # Check if occupied
            if grid[ix, iy]:
                r_last = r
            
            r += ray_step
        
        r_vals[k] = r_last
    
    return thetas, r_vals


def radial_profile_from_grid(grid, n_theta=10_000, ray_step=0.5, parallel=True):
    """
    Compute r(θ, t) by ray marching from the center.
    
    Wrapper function that automatically selects single-threaded or parallel
    version based on the 'parallel' parameter.
    
    Parameters:
    -----------
    grid : np.ndarray (2D, dtype=np.int8 or bool)
        The urbanization grid where 1/True = urbanized, 0/False = non-urbanized
    n_theta : int
        Number of angular samples (default: 10,000)
    ray_step : float
        Step size for ray marching (default: 0.5)
    parallel : bool
        If True, use parallel version (default: True)
    
    Returns:
    --------
    thetas : np.ndarray
        Array of angles in [0, 2π)
    r_vals : np.ndarray
        Array of maximum radii for each angle
    """
    if parallel:
        return radial_profile_from_grid_parallel(grid, n_theta, ray_step)
    else:
        return radial_profile_from_grid_single(grid, n_theta, ray_step)

# ------ Compute of the statistics ----------- 

@njit
def _compute_sector_stats(r_sorted, N):
    """
    Helper function to compute sector statistics for a given N.
    Splits r_sorted into N sectors and computes means and variances.
    """
    n_samples = len(r_sorted)
    base_size = n_samples // N
    remainder = n_samples % N
    
    means = np.empty(N, dtype=np.float64)
    vars_ = np.empty(N, dtype=np.float64)
    
    idx = 0
    for i in range(N):
        # Calculate sector size (distribute remainder)
        sector_size = base_size + (1 if i < remainder else 0)
        
        if sector_size == 0:
            means[i] = np.nan
            vars_[i] = np.nan
            continue
        
        # Extract sector
        sector_end = idx + sector_size
        
        # Compute mean
        sector_sum = 0.0
        for j in range(idx, sector_end):
            sector_sum += r_sorted[j]
        mean_val = sector_sum / sector_size
        means[i] = mean_val
        
        # Compute variance
        var_sum = 0.0
        for j in range(idx, sector_end):
            diff = r_sorted[j] - mean_val
            var_sum += diff * diff
        vars_[i] = var_sum / sector_size
        
        idx = sector_end
    
    return means, vars_


@njit
def width_vs_arc_length_numba(theta, r, N_list):
    """
    Optimized Numba version of width_vs_arc_length.
    
    For each N in N_list (N sectors; Δθ = 2π/N):
    - Split the (θ, r) samples into N contiguous sectors of ~equal size.
    - For each sector i, compute ⟨r⟩_i and the sector variance ⟨(r-⟨r⟩_i)^2⟩.
    - Compute w^2 = (1/N) Σ_i var_i and w = sqrt(w^2).
    - Compute the *average* arc length ℓ̄ = Δθ * mean_i(⟨r⟩_i).
    
    Parameters:
    -----------
    theta : np.ndarray
        Array of angles
    r : np.ndarray
        Array of radii corresponding to theta
    N_list : np.ndarray
        Array of sector counts to test
    
    Returns:
    --------
    ell_list : np.ndarray
        Array of average arc lengths for each N
    w_list : np.ndarray
        Array of widths for each N
    """
    # Sort by theta
    order = np.argsort(theta)
    r_sorted = r[order]
    
    n_samples = len(r_sorted)
    n_N = len(N_list)
    
    ell_list = np.empty(n_N, dtype=np.float64)
    w_list = np.empty(n_N, dtype=np.float64)
    
    for idx in range(n_N):
        N = int(N_list[idx])
        # Clip N to valid range
        N = max(2, min(n_samples, N))
        
        # Compute sector statistics
        means, vars_ = _compute_sector_stats(r_sorted, N)
        
        # Filter out NaN values (empty sectors)
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


@njit(parallel=True)
def width_vs_arc_length_parallel(theta, r, N_list):
    """
    Parallel optimized Numba version of width_vs_arc_length.
    
    For each N in N_list (N sectors; Δθ = 2π/N):
    - Split the (θ, r) samples into N contiguous sectors of ~equal size.
    - For each sector i, compute ⟨r⟩_i and the sector variance ⟨(r-⟨r⟩_i)^2⟩.
    - Compute w^2 = (1/N) Σ_i var_i and w = sqrt(w^2).
    - Compute the *average* arc length ℓ̄ = Δθ * mean_i(⟨r⟩_i).
    
    Parameters:
    -----------
    theta : np.ndarray
        Array of angles
    r : np.ndarray
        Array of radii corresponding to theta
    N_list : np.ndarray
        Array of sector counts to test
    
    Returns:
    --------
    ell_list : np.ndarray
        Array of average arc lengths for each N
    w_list : np.ndarray
        Array of widths for each N
    """
    # Sort by theta
    order = np.argsort(theta)
    r_sorted = r[order]
    
    n_samples = len(r_sorted)
    n_N = len(N_list)
    
    ell_list = np.empty(n_N, dtype=np.float64)
    w_list = np.empty(n_N, dtype=np.float64)
    
    # Parallel loop over different N values
    for idx in prange(n_N):
        N = int(N_list[idx])
        # Clip N to valid range
        N = max(2, min(n_samples, N))
        
        # Compute sector statistics
        means, vars_ = _compute_sector_stats(r_sorted, N)
        
        # Filter out NaN values (empty sectors)
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


def width_vs_arc_length(theta, r, N_list, parallel=True):
    """
    Compute width vs arc length for multiple sector counts.
    
    For each N in N_list (N sectors; Δθ = 2π/N):
    - Split the (θ, r) samples into N contiguous sectors of ~equal size.
    - For each sector i, compute ⟨r⟩_i and the sector variance ⟨(r-⟨r⟩_i)^2⟩.
    - Compute w^2 = (1/N) Σ_i var_i and w = sqrt(w^2).
    - Compute the *average* arc length ℓ̄ = Δθ * mean_i(⟨r⟩_i).
    
    Parameters:
    -----------
    theta : np.ndarray or list
        Array of angles
    r : np.ndarray or list
        Array of radii corresponding to theta
    N_list : np.ndarray or list
        Array of sector counts to test
    parallel : bool
        If True, use parallel version (default: True)
    
    Returns:
    --------
    ell_list : np.ndarray
        Array of average arc lengths for each N
    w_list : np.ndarray
        Array of widths for each N
    """
    # Convert to numpy arrays if needed
    theta = np.asarray(theta, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    N_list = np.asarray(N_list, dtype=np.int64)
    
    if parallel:
        return width_vs_arc_length_parallel(theta, r, N_list)
    else:
        return width_vs_arc_length_numba(theta, r, N_list)











def simulate(grid_size,
             timesteps,
             output_file,
             metric_timestep,
             N_sampling=10000,
             num_N=20
             ):


    
    # ---------- INITIALIZATION ----------

    t=0

    if output_file:
        f = open(output_file, 'w', newline='')
        writer = csv.writer(f)
        writer.writerow(['timestep', 'l', 'w','urban_fraction'])
    else:
        writer = None
        w_results = []
        mean_results = []
        N_sectors_results = []
        urban_fraction_list = []

       # Random grid and city core
    grid = np.zeros((grid_size,grid_size),dtype=np.int8)
    center = int(grid_size//2.0)
    grid[center, center] = 1


    # Define initial boundary (neighbors of the seed)
    initial_boundary = [
        (center-1, center),  # up
        (center+1, center),  # down
        (center, center-1),  # left
        (center, center+1)   # right
        ]
    # Convert to Numba typed list
    boundary_list = initialize_boundary(grid, initial_boundary)

    w_list=[]
    ell_bar_list=[]
    while t < timesteps:
        # simulate for n=metric_timestep
        grid, boundary_list = eden_growth_simulate(
            grid, boundary_list, n_steps=metric_timestep)
       
        # udpdate the timestep
        t+=metric_timestep
        print(f'timestep {t}/{timesteps}')
        # ray_marching
        thetas,r=radial_profile_from_grid(grid=grid,n_theta=N_sampling,ray_step=0.5,parallel=True)

        # metric calculation
        # N_list estimation 
        total_points=len(r)
        min_N = 4
        max_N = total_points // 5
        if max_N >= min_N:
            num_N = min(num_N, max_N - min_N + 1)
            N_list = np.logspace(np.log10(min_N), np.log10(max_N), num_N, dtype=int)
            N_list = sorted(set(N_list.tolist()))
            
       
        ell,w = width_vs_arc_length_parallel(thetas, r, N_list)
        w_list.append(w)
        ell_bar_list.append(ell)
        urban_fraction=len(grid[grid==1])/grid_size**2

        # Write results
        if writer is not None:
            for ii in range(len(N_list)):
                writer.writerow([t,ell[ii],w[ii],urban_fraction])
        else :
            print("met un output chacal")
            break

    # Close CSV
    if writer is not None:
        f.close()
    print("Simulation done!")



            
                
                
             



    




