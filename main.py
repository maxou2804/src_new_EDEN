import numpy as np
from core import create_city_core
from simulation import simulate_with_competitive_distance
import matplotlib.pyplot as plt
import os

# Load parameters







# === CONFIGURABLE PARAMETERS ===

grid_sizes = [501]
num_simulations_per_size = 1  # 🟡 CHANGE THIS to control how many runs per grid size
core_shape = "circle"
radius=1
metric_rate=500
beta=0.5
gamma=1.0
merge_check_frequency=100
update_frequency=200




seed_configs = [
        {'position': (250, 250), 'radius': 50, 'roughness': 0.3},
        {'position': (350, 350), 'radius': 10, 'roughness': 0.3},
        {'position': (100, 100), 'radius': 10, 'roughness': 0.3},
        
        ]


# Function to compute timesteps based on grid size

def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2)*np.pi*0.1)


# Output base directory
base_output_dir = "C:\\Users\\trique\\Downloads\\EDEN_MAIN\\EDEN_output"
os.makedirs(base_output_dir, exist_ok=True)

# === RUN SIMULATIONS ===

for size in grid_sizes:
   
 
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


