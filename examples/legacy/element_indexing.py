from __future__ import annotations
import os
from pathlib import Path
if os.name == 'nt':
    import msvcrt
else:
    import select
from matplotlib import pyplot as plt
from openlifu.db import Database
import numpy as np
from scipy import ndimage
from matplotlib import colormaps

# medium parameters

# alpha_power         = 1.43     # Robertson et al., PMB 2017 usually between 1 and 3? from Treeby paper
alpha_power = 1.1
alpha_coeff_water   = 0.001        # [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017), see also 0.05 Fomenko et al., 2020?
alpha_coeff_min     = 4     
alpha_coeff_max     = 8.7      # [dB/(MHz cm)] Fry 1978 at 0.5MHz: 1 Np/cm (8.7 dB/cm) for both diploe and outer tables

# set focus
simulate = True
plot = False

xInput = 0
yInput = 0
zInput = 30

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system
freq = frequency_kHz*1e3

here = Path(__file__).parent.resolve()
db_path = here / ".." / "OpenLIFU_Database_DCVA"

db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1_002")

sensor_mask_pos = np.array([el.get_position(units='m') for el in arr.elements]).T*100
xs, ys, zs = (sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2])
n_ele = len(arr.elements)
cmap = colormaps['viridis'].resampled(n_ele)
color_i = []
for n in range(n_ele):
    color_i.append(cmap(n/n_ele))
    print((xs[n],ys[n],zs[n]))

if plot:
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) 
    ax.scatter(sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2],c=color_i)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
plt.show()

