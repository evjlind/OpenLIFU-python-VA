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
from kwave.utils.filters import extract_amp_phase
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
import numpy as np
import xarray as xa
from openlifu.util.units import getunitconversion
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from kwave.kmedium import kWaveMedium
from scipy import ndimage


xInput = 0
yInput = 0
zInput = 35

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
db_path = "../{here}/../OpenLIFU_Database_DCVA"

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
spacing = 0.25
# spacing = 0.125
sim_setup = SimSetup(spacing=spacing, dt=2e-8, t_end=100e-6)
focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
apod_method = apod_methods.Uniform()
delay_method = delay_methods.Direct()
protocol = Protocol(
    pulse=pulse,
    sequence=sequence,
    focal_pattern=focal_pattern,
    sim_setup=sim_setup)

pts = protocol.focal_pattern.get_targets(target)
coords = protocol.sim_setup.get_coords()
params = protocol.seg_method.ref_params(coords)
kgrid = get_kgrid(coords)

delays, apod = protocol.beamform(arr=arr, target=pts[0], params=params)

amplitude = 1
cycles = 20
t = np.arange(0, cycles / freq, kgrid.dt)
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

sensor_mask_pos = np.array([el.get_position(units='m') for el in arr.elements]).T*100

ele_bin = karray.get_array_binary_mask(kgrid)

rev_sig_t = np.arange(0,0.5/freq,kgrid.dt)
rev_puls = np.sin(2*np.pi*freq*rev_sig_t)
p = np.concatenate([rev_puls,np.zeros(kgrid.Nt - len(rev_puls))])
print(p)