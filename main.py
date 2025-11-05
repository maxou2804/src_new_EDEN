import numpy as np
from simulation import simulate_with_competitive_distance, run_parallel_ensemble
import matplotlib.pyplot as plt
import os
import random
import math

# Load parameters







# === CONFIGURABLE PARAMETERS ===

size = 800
num_simulations_per_size = 10 # 🟡 CHANGE THIS to control how many runs per grid size
core_shape = "circle"


radius_collection=[5]
#[30,50,70,100,150,180]
r_lcc=50
r_cluster=[5]
roughness=0
metric_rate=1000
beta=0.5
spatial_variance = (10 ) ** 2

# write 0 to have non gravity probability
gamma=0
merge_check_frequency=100
update_frequency=100000


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
        # Random angle in radians
        angle = random.uniform(0, 2 * math.pi)
        
        # Calculate position on the circle of radius r around the center
        x =int( center[0] + r * math.cos(angle))
        y =int( center[1] + r * math.sin(angle))
        
        # Get radius from r_urb list
        radius = r_urb[i]
        
        seed_configs.append({
            'position': (x, y),
            'radius': radius,
            'roughness': roughness
        })
    
    return seed_configs



# Function to compute timesteps based on grid size

def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2-100**2)*np.pi*0.5)


# Output base directory


# === RUN SIMULATIONS ===

   

#     timesteps = compute_timesteps(size)

#     print(f"\n📦 Grid size: {size}x{size} | Timesteps: {timesteps}")

#     for run_id in range(1, num_simulations_per_size + 1):
#         print(f" 🔁 Simulation run {run_id}/{num_simulations_per_size} for size {size}")

#         seed_configs = generate_seed_configs(n=0,l=size,r=radius,r_lcc=r_lcc,r_urb=r_cluster,roughness=roughness)
   
#         metric_timestep=int(timesteps/metric_rate)

#         # Name of the output file
#         output_file = os.path.join(
#             base_output_dir, f"simul_r_periphery_{radius}_run_{run_id}.csv"
#         )
     


      

radius=200

base_output_dir = f"C:\\Users\\trique\\Downloads\\EDEN_MAIN\\EDEN_output\\r_study_new\\r_200"
os.makedirs(base_output_dir, exist_ok=True)


seed_configs = generate_seed_configs(n=1, l=size, r=radius, r_lcc=50, r_urb=[5], roughness=0.1)

timesteps = compute_timesteps(size)





        # Name of the output file
output_file = os.path.join(
            base_output_dir, f"simul.csv")


        # Run simulation
# grid, largest_seed_stats, all_stats= simulate_with_competitive_distance (grid_size=size,
#                                     seed_configs=seed_configs,
#                                     timesteps=timesteps,
#                                     output_file=output_file,
#                                     beta=beta,
#                                     gamma=gamma,
#                                     metric_timestep=metric_rate,
#                                     N_sampling=10000,
#                                     num_N=30,
#                                     update_frequency=update_frequency,
#                                     merge_check_frequency=merge_check_frequency,
#                                     create_animation=True,  # NEW: Enable/disable animation
#                                     animation_interval=5000,  # NEW: Capture frame every N steps
#                                     use_spatial_filter=False,
#                                     spatial_variance=spatial_variance,
#                                     animation_file=base_output_dir + f"\\animation_r_periphery_{radius}_run_test.gif",
#                                     use_competitive_distance=False ) # NEW: Animation output file)

          
# print(f"    📝 Results saved to {output_file}" )







if __name__ == '__main__':
    grid, largest_seed_stats=  run_parallel_ensemble (grid_size=size,
                                    n_realizations=50,
                                    seed_configs=seed_configs,
                                    timesteps=timesteps,
                                    output_dir=base_output_dir,
                                    beta=beta,
                                    gamma=gamma,
                                    metric_timestep=metric_rate,
                                    N_sampling=10000,
                                    num_N=30,
                                    update_frequency=update_frequency,
                                    merge_check_frequency=merge_check_frequency,
                                  # NEW: Capture frame every N steps
                                    use_spatial_filter=False,
                                    spatial_variance=spatial_variance,
                                    use_competitive_distance=False ) # NEW: Animation output file)

    print("\n✅ All simulations completed.")