import numpy as np
from simulation import simulate_with_competitive_distance, run_parallel_ensemble
import matplotlib.pyplot as plt
import os
import random
import math
import pandas as pd
# Load parameters







# === CONFIGURABLE PARAMETERS ===

size = 800
num_simulations_per_size = 10 # 🟡 CHANGE THIS to control how many runs per grid size
core_shape = "circle"


# radius_collection=[5]
# #[30,50,70,100,150,180]
# r_lcc=50
# r_cluster=[5]
# roughness=0
metric_rate=1000
beta=0.5
spatial_variance = (10 ) ** 2

# write 0 to have non gravity probability
gamma=0
merge_check_frequency=100
update_frequency=100000






# Function to compute timesteps based on grid size

def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2-100**2)*np.pi*0.4)


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



seed_configs={'n':7, 'l':size, 'r':250, 'r_lcc':100, 'r_urb':[20,20,20,20,20,20,20], 'roughness':0.1}
spatial_variance=50

def compute_timesteps(L,seed_configs):
    # 🟢 Replace with your actual expression
    return int((((L/2)**2-(seed_configs['r']+seed_configs['r_urb'][0]**2))*np.pi*0.4))

timesteps = compute_timesteps(size,seed_configs)
n_realizations=50

base_output_dir = f"C:\\Users\\trique\\Downloads\\MASTER_THESIS\\test\\runs_n={seed_configs['n']}_r={seed_configs['r']}"
os.makedirs(base_output_dir, exist_ok=True)



        # Name of the output file
output_file = os.path.join(
            base_output_dir, f"simul.csv")


        #Run simulation
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
                                    animation_interval=5000,  # NEW: Capture frame every N steps
                                    use_spatial_filter=False,
                                    spatial_variance=spatial_variance,
                                    animation_file=base_output_dir + f"\\animation.gif",
                                    use_competitive_distance=False ) # NEW: Animation output file)

          
print(f"    📝 Results saved to {output_file}" )







# if __name__ == '__main__':
#     grid, largest_seed_stats=  run_parallel_ensemble (grid_size=size,
#                                     n_realizations=n_realizations,
#                                     seed_configs=seed_configs,
#                                     timesteps=timesteps,
#                                     output_dir=base_output_dir,
#                                     beta=beta,
#                                     gamma=gamma,
#                                     metric_timestep=metric_rate,
#                                     N_sampling=10000,
#                                     num_N=30,
#                                     update_frequency=update_frequency,
#                                     merge_check_frequency=merge_check_frequency,
#                                   # NEW: Capture frame every N steps
#                                     use_spatial_filter=False,
#                                     spatial_variance=spatial_variance,
#                                     use_competitive_distance=False ) # NEW: Animation output file)

#     print("\n✅ All simulations completed.")