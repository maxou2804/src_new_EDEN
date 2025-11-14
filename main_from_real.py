import numpy as np
from simulation import simulate_with_competitive_distance, run_parallel_ensemble,load_grid_auto
import matplotlib.pyplot as plt
import os
import random
import math
import pandas as pd
import csv



city='Bangkok'
metric_rate=100
timesteps =10000
update_frequency=10000
merge_check_frequency=200
animation_interval=200
size=1874
year=1985
n_clusters=10

mask_name=f'/Users/mika/Documents/DATA/masks/mask_{city}_n_clusters={n_clusters}_{year}.csv'

base_output_dir = f"/Users/mika/Documents/EDEN_inital_conditons/real_initial_conditions/real_initial_conditions_{city}_n_clusters={n_clusters}"
os.makedirs(base_output_dir, exist_ok=True)

grid =load_grid_auto(mask_name)
print(np.max(grid))

            # Name of the output file
output_file = os.path.join(
                base_output_dir, f"simul.csv")

grid, largest_seed_stats, all_stats= simulate_with_competitive_distance (grid_size=size,
                                        seed_configs=None,
                                        timesteps=timesteps,
                                        output_file=output_file,
                                        beta=2.0,
                                        gamma=1,
                                        metric_timestep=metric_rate,
                                        N_sampling=10000,
                                        num_N=30,
                                        update_frequency=update_frequency,
                                        merge_check_frequency=merge_check_frequency,
                                        create_animation=True,  # NEW: Enable/disable animation
                                        animation_interval=animation_interval,  # NEW: Capture frame every N steps
                                        use_spatial_filter=False,
                                        spatial_variance=10000,
                                        animation_file=base_output_dir + f"/{city}_animation.gif",
                                        use_competitive_distance=False,
                                        initial_grid_file=mask_name ) # NEW: Animation output file)

            
print(f"    📝 Results saved to {output_file}" )
