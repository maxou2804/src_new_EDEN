import numpy as np
from core import create_city_core
from simulation import simulate_with_competitive_distance
import matplotlib.pyplot as plt
import os
import random
import math

# Load parameters







# === CONFIGURABLE PARAMETERS ===

grid_sizes = [501]
num_simulations_per_size = 1  # 🟡 CHANGE THIS to control how many runs per grid size
core_shape = "circle"

r_lcc=50
r_periph=100
r_cluster=100
roughness=0.3
metric_rate=500
beta=0.5

# write 0 to have non gravity probability
gamma=1.5
merge_check_frequency=100
update_frequency=200


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
    center = (l / 2, l / 2)
    seed_configs.append({
        'position': center,
        'radius': r_lcc,
        'roughness': roughness
    })
    
    # Generate n seeds randomly placed on a circle around the LCC
    for i in range(n):
        # Random angle in radians
        angle = random.uniform(0, 2 * math.pi)
        
        # Calculate position on the circle of radius r around the center
        x = center[0] + r * math.cos(angle)
        y = center[1] + r * math.sin(angle)
        
        # Get radius from r_urb list
        radius = r_urb[i]
        
        seed_configs.append({
            'position': (x, y),
            'radius': radius,
            'roughness': roughness
        })
    
    return seed_configs


seed_configs = [
        {'position': (250, 250), 'radius': 50, 'roughness': 0.3},
        {'position': (350, 350), 'radius': 10, 'roughness': 0.3},
        {'position': (100, 100), 'radius': 10, 'roughness': 0.3},
        
        ]


# Function to compute timesteps based on grid size

def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2)*np.pi*0.3)


# Output base directory
base_output_dir = "C:\\Users\\trique\\Downloads\\EDEN_MAIN\\EDEN_output"
os.makedirs(base_output_dir, exist_ok=True)

# === RUN SIMULATIONS ===

for size in grid_sizes:
   
    seed_configs = generate_seed_configs(n=2,l=size,r=r_periph,r_lcc=r_lcc,r_urb=r_cluster,roughness=roughness)
    timesteps = compute_timesteps(size)
    metric_timestep=int(timesteps/metric_rate)
   

    print(f"\n📦 Grid size: {size}x{size} | Timesteps: {timesteps}")

    for run_id in range(1, num_simulations_per_size + 1):
        print(f" 🔁 Simulation run {run_id}/{num_simulations_per_size} for size {size}")

     

        # Name of the output file
        output_file = os.path.join(
            base_output_dir, f"simul_L_{size}_run_{run_id}_beta_{beta}.csv"
        )
     

        # Run simulation
        grid, largest_seed_stats, all_stats= simulate_with_competitive_distance (grid_size=size,
                                    seed_configs=seed_configs,
                                    timesteps=timesteps,
                                    output_file=output_file,
                                    beta=beta,
                                    gamma=gamma,
                                    metric_timestep=metric_rate,
                                    N_sampling=10000,
                                    num_N=30,
                                    update_frequency=update_frequency,
                                    merge_check_frequency=merge_check_frequency,
                                    create_animation=True,  # NEW: Enable/disable animation
                                    animation_interval=200,  # NEW: Capture frame every N steps
                                    animation_file=base_output_dir + f"\\animation_L_{size}_run_{run_id}_beta_{beta}.gif" ) # NEW: Animation output file)

          
        print(f"    📝 Results saved to {output_file}" )

      

print("\n✅ All simulations completed.")


