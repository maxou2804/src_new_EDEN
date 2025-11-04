
import pandas as pd
import matplotlib.pyplot as plt
directory ="C:\\Users\\trique\\Downloads\\EDEN_MAIN\\EDEN_output\\simul_L_501_run_1_beta_0.5.csv"


data=pd.read_csv(directory)
df=pd.DataFrame(data)

area=df["cells"]
timestep=df["timestep"]
a = list(set(area))
a.sort()
t = list(set(timestep))
t.sort()

plt.figure()
plt.plot(a,'o')
plt.xlabel('time')
plt.ylabel('Area')
plt.show()

