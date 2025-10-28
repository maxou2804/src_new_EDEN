import numpy as np
from core import create_city_core
from eden_growth_optimized import simulate
import matplotlib.pyplot as plt
import os

# Load parameters







# === CONFIGURABLE PARAMETERS ===

grid_sizes = [5001]
num_simulations_per_size = 5  # 🟡 CHANGE THIS to control how many runs per grid size
core_shape = "circle"
radius=1
metric_rate=2000


def compute_timesteps(L):
    # 🟢 Replace with your actual expression
    return int(((L/2)**2-radius**2)*np.pi*0.5)


# Output base directory
base_output_dir = "C:\\Users\\trique\\Downloads\\MASTER_THESIS\\outputs\\new_run_grid"
os.makedirs(base_output_dir, exist_ok=True)

# === RUN SIMULATIONS ===

for size in grid_sizes:
   
 
    timesteps = compute_timesteps(size)
    metric_timestep=int(timesteps/metric_rate)
   

    print(f"\n📦 Grid size: {size}x{size} | Radius: {1} | Timesteps: {timesteps}")

    for run_id in range(1, num_simulations_per_size + 1):
        print(f" 🔁 Simulation run {run_id}/{num_simulations_per_size} for size {size}")

     

        # Name of the output file
        output_file = os.path.join(
            base_output_dir, f"simul_L_{size}_run_{run_id}.csv"
        )
     

        # Run simulation
        simulate(grid_size=size,
                 timesteps=timesteps,
                 output_file=output_file,
                 metric_timestep=metric_timestep,
                 N_sampling=10000,
                 num_N=40)

          
        print(f"    📝 Results saved to {output_file}" )

      

print("\n✅ All simulations completed.")


