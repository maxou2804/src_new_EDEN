
import pandas as pd
import matplotlib.pyplot as plt
directory ="C:\\Users\\trique\\Downloads\\EDEN_MAIN\\EDEN_output\\simul_r_periphery_100_run_1.csv"


data=pd.read_csv(directory)
df=pd.DataFrame(data)

area=df["total_urbanized"]
timestep=df["timestep"]
cells=df["cells"]
a = list(set(area))
a.sort()
t = list(set(timestep))
t.sort()
c = list(set(cells))
c.sort()


# plt.figure()
# plt.plot(t,a,'o')
# plt.xlabel('time')
# plt.ylabel('Area')
# plt.show()

plt.figure()
plt.plot(t,a,'o')
plt.show()