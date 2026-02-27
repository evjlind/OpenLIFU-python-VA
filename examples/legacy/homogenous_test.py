from __future__ import annotations

import logging
import os
import sys
import threading
import time
from pathlib import Path

if os.name == 'nt':
    import msvcrt
else:
    import select

from matplotlib import pyplot as plt
from openlifu.bf import apod_methods, focal_patterns, delay_methods
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence
from openlifu.db import Database
from openlifu.geo import Point
from openlifu.plan import Protocol
from openlifu.sim import SimSetup
from openlifu.sim.kwave_if import get_karray, get_medium, get_source, get_sensor, get_kgrid
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
import numpy as np
import xarray as xa
from openlifu.util.units import getunitconversion
from scipy import ndimage
from scipy.io import savemat

# # medium parameters
# alpha_power         = 1.43     # Robertson et al., PMB 2017 usually between 1 and 3? from Treeby paper
alpha_power = 1.1
alpha_coeff_water   = 0.001       # [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017), see also 0.05 Fomenko et al., 2020?
alpha_coeff_min     = 4     
alpha_coeff_max     = 8.7      # [dB/(MHz cm)] Fry 1978 at 0.5MHz: 1 Np/cm (8.7 dB/cm) for both diploe and outer tables

# set focus
simulate = True
plot = True

xInput = 0
yInput = 0
zInput = 50

frequency_kHz = 400 # Frequency in kHz
duration_msec = 0.1 # Pulse Duration in milliseconds
interval_msec = 20 # Pulse Repetition Interval in milliseconds
num_modules = 2 # Number of modules in the system
freq = frequency_kHz*1e3

pulse = Pulse(frequency=frequency_kHz*1e3, duration=duration_msec*1e-3)
sequence = Sequence(
    pulse_interval=interval_msec*1e-3,
    pulse_count=int(60/(interval_msec*1e-3)),
    pulse_train_interval=0,
    pulse_train_count=1)

here = Path(__file__).parent.resolve()
db_path = here / ".." / "OpenLIFU_Database_DCVA"

db = Database(db_path)
arr = db.load_transducer(f"openlifu_{num_modules}x400_evt1_002")
arr.sort_by_pin()

simulation_options = SimulationOptions(
                        pml_auto=True,
                        pml_inside=False,
                        save_to_disk=True,
                        data_cast='single'
                    )

target = Point(position=(xInput,yInput,zInput), units="mm")

execution_options = SimulationExecutionOptions(is_gpu_simulation=True)
deg_accuracy = 2
spacing = 0.5/deg_accuracy
T0 = 1 / freq

set_dt = (T0 / 360) * deg_accuracy
sim_setup = SimSetup(spacing=spacing, dt=set_dt, t_end=100e-6)
focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
apod_method = apod_methods.MaxAngle()
delay_method = delay_methods.Preset()

protocol = Protocol(
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    sim_setup=sim_setup,
    delay_method=delay_method,
    apod_method=apod_method)

pts = protocol.focal_pattern.get_targets(target)
coords = protocol.sim_setup.get_coords()
params = protocol.seg_method.ref_params(coords)
kgrid = get_kgrid(coords)

delays, apod = protocol.beamform(arr=arr, target=pts[0], params=params)
amplitude = 1
cycles = 20
t = np.arange(0, cycles / freq, kgrid.dt)
t_pulse = np.arange(0,kgrid.dt,(0.5/freq))
src_pulse = np.sin(2*np.pi*freq*t_pulse)
input_signal = amplitude * np.sin(2 * np.pi * freq * t)
source_mat = arr.calc_output(input_signal, kgrid.dt, delays, apod)

units = [params[dim].attrs['units'] for dim in params.dims]
scl = getunitconversion(units[0], 'm')
array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]
bli_tolerance = 0.05
upsampling_rate = 1
karray = get_karray(arr,
                    translation=array_offset,
                    bli_tolerance=bli_tolerance,
                    upsampling_rate=upsampling_rate)

medium = get_medium(params)
sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
source = get_source(kgrid, karray, source_mat)

if simulate:
    output = kspaceFirstOrder3D(kgrid=kgrid,
                                    source=source,
                                    sensor=sensor,
                                    medium=medium,
                                    simulation_options=simulation_options,
                                    execution_options=execution_options)

    sz = list(params.coords.sizes.values())
    p_max = xa.DataArray(output['p_max'].reshape(sz, order='F'),
                            coords=params.coords,
                            name='p_max',
                            attrs={'units':'Pa', 'long_name':'PPP'})
    p_min = xa.DataArray(-1*output['p_min'].reshape(sz, order='F'),
                            coords=params.coords,
                            name='p_min',
                            attrs={'units':'Pa', 'long_name':'PNP'})
    Z = params['density'].data*params['sound_speed'].data
    intensity = xa.DataArray(1e-4*output['p_min'].reshape(sz, order='F')**2/(2*Z),
                            coords=params.coords,
                            name='I',
                            attrs={'units':'W/cm^2', 'long_name':'Intensity'})
    ds = xa.Dataset({'p_max':p_max, 'p_min':p_min, 'intensity':intensity})
    if plot:
        plt.figure()
        plt.imshow(p_max[round(kgrid.Nx/2),:,:])
        plt.title('Sim result (x)')
        plt.colorbar()
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.title('Sim result (y)')
        plt.colorbar()
        plt.figure()
        plt.imshow(p_max[:,:,round(kgrid.Nz/2)])
        plt.title('Sim result (z)')
        plt.colorbar()

if simulate:
    sim_result = {'p_max':p_max}
    savemat('sim_result.mat',sim_result)

if plot:
    plt.show()