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


# medium parameters
c_min               = 1540     # sound speed [m/s]
c_max               = 3100     # max. speed of sound in skull (F. A. Duck, 2013.) [m/s]
rho_min             = 1000     # density [kg/m^3]
rho_max             = 1900     # max. skull density [kg/m3]
# alpha_power         = 1.43     # Robertson et al., PMB 2017 usually between 1 and 3? from Treeby paper
alpha_power = 1.1
alpha_coeff_water   = 0        # [dB/(MHz^y cm)] close to 0 (Mueller et al., 2017), see also 0.05 Fomenko et al., 2020?
alpha_coeff_min     = 4     
alpha_coeff_max     = 8.7      # [dB/(MHz cm)] Fry 1978 at 0.5MHz: 1 Np/cm (8.7 dB/cm) for both diploe and outer tables

hu_min 	= 300
hu_max 	= 2000	

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def generate_ct_noise(kgrid):
    hu_max 	= 2000
    width = kgrid.Nx
    height = kgrid.Ny
    depth = kgrid.Nz
    x = np.arange(width)
    y = np.arange(height)
    z = np.arange(depth)

    X, Y, Z = np.meshgrid(x, y, z)
    p = np.random.rand(width,height,depth)
    wavelength_x = 100000 
    wavelength_y = 15000 
    amplitude = 1

    sine_image = amplitude * np.sin(2 * np.pi * (X / wavelength_x + Y / wavelength_y)) + p

    sine_image = np.abs(sine_image)
    max_val = sine_image.max()
    scale_factor = 1500/max_val
    sine_image = sine_image*scale_factor
    sine_image = ndimage.median_filter(sine_image,3)

    sine_image[sine_image<hu_min] = 0
    sine_image[sine_image>hu_max] = hu_max

    return sine_image

# set focus
simulate = True
plot = True
simulate2 = True
use_ct_noise = False

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

deg_accuracy = 2
spacing = 0.5/deg_accuracy
T0 = 1 / freq
set_dt = (T0 / 360) * deg_accuracy

sim_setup = SimSetup(spacing=spacing, dt=set_dt, t_end=100e-6)
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

ele_bin = np.zeros([375,375,315])

el_list = karray.get_element_positions()
least_x = min(el_list[0])
least_y = min(el_list[1])
least_z = min(el_list[2])

ele_ordering = np.zeros_like(ele_bin)
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
for ind in range(len(el_list[0])):
    ele_pos = [el_list[0][ind],el_list[1][ind],el_list[2][ind]]
    ix = round((ele_pos[0]+abs(least_x))/(kgrid.dx))
    iy = round((ele_pos[1]+abs(least_y))/(kgrid.dy))
    iz = round((ele_pos[2]+abs(least_z))/(kgrid.dz))
    ix = max(0,min(374,ix))
    iy = max(0,min(374,iy))
    iz = max(0,min(314,iz))
    ele_bin[ix][iy][iz] = 1
    ele_ordering[ix][iy][iz] = ind

    if plot:
        ax.scatter(ix,iy,iz)
        ax.text(ix,iy,iz,'{}'.format(ind+1))

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
    if plot == True:
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.title('forward sim')
        plt.colorbar()

sensor_mask_pos = np.array([el.get_position(units='m') for el in arr.elements]).T*100

if plot:
    xs, ys, zs = (sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2])
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    ax.set_box_aspect((np.ptp(xs), np.ptp(ys), np.ptp(zs))) 
    ax.scatter(sensor_mask_pos[0],sensor_mask_pos[1],sensor_mask_pos[2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')

if simulate2:
    p0 = p_max.to_numpy()
    db_cutoff = p0.max()/2
    p0[p0<db_cutoff] = 0
    if plot:
        plt.figure()
        plt.imshow(p0[:,round(kgrid.Ny/2),:])
        plt.title('initial pressure distribution (reverse)')
        plt.colorbar()
    sensor2 = kSensor(record=['p'])
    sensor2.mask = ele_bin
    source2 = kSource()
    source2.p0 = p0
    kgrid2 = get_kgrid(coords)
    if use_ct_noise:
        ct_noise = generate_ct_noise(kgrid2)
        if plot:
            plt.figure()
            plt.imshow(ct_noise[:,round(kgrid.Ny/2),:])
            plt.title('noisegen')
            plt.colorbar()
        model = np.zeros_like(ct_noise)
        model[:,:,zInput-5:zInput-3] = ct_noise[:,:,zInput-5:zInput-3]
        dmap = rho_min + (rho_max-rho_min)*(model-0)/(hu_max-0)
        cmap = c_min + (c_max-c_min)*(dmap-rho_min)/(rho_max-rho_min)
        amap = alpha_coeff_min+(alpha_coeff_max-alpha_coeff_min)*(1-(model-hu_min)/(hu_max-hu_min))**0.5
        medium2 = kWaveMedium(sound_speed=cmap,density=dmap,alpha_coeff=amap,alpha_power=alpha_power,alpha_mode='no_dispersion')
        if plot:
            plt.figure()
            plt.imshow(model[:,round(kgrid.Ny/2),:])
            plt.title('medium')
            plt.colorbar()

    else:
        medium2 = get_medium(params)


    sensor_data = kspaceFirstOrder3D(
        kgrid=kgrid2,
        source=source2,
        sensor=sensor2,
        medium=medium2,
        simulation_options=simulation_options,
        execution_options=execution_options
    )

    if plot:
        plt.figure(figsize=(10, 6))
        plt.imshow(sensor_data['p'].T, aspect='auto', extent=[
            0, kgrid.Nt * kgrid.dt * 1e6,  # Time in μs
            0, sensor_data['p'].shape[1]  # Sensor number
        ])
        plt.xlabel('Time (μs)')
        plt.ylabel('Sensor Number')
        plt.title('Recorded Pressure at Boundary Sensors')
        plt.colorbar(label='Pressure (Pa)')

delays_tr = np.zeros_like(delays)
first_val = np.zeros_like(delays)
order = []
for v in range(1,9):
    p = (v-1)*8
    q = 120-(v-1)*8
    order.append([p,q,p+1,q+1,p+2,q+2,p+3,q+3,p+4,q+4,p+5,q+5,p+6,q+6,p+7,q+7])
order = np.array(order).flatten()

if simulate2:
    for i in range(128):
        amp, phase, p_freq = extract_amp_phase(np.squeeze(sensor_data['p'].T[i]),1/kgrid.dt,freq,dim=0)
        delays_tr[order[i]]=phase/freq/(2*np.pi)
        first_val_p = abs(sensor_data['p'].T)[i] - min(abs(sensor_data['p'].T[i]))
        first_val[order[i]]=np.where(first_val_p>np.mean(first_val_p))[0][0]*kgrid.dt
    # first_val = first_val-min(first_val)
    # delays_tr = delays_tr-min(delays_tr)
    ## no zero delays even though the minimums are subtracted
    print('reverse sim delays')
    print(delays_tr)
    print('reverse sim first value')
    print(first_val)
    print('original delays')
    print(delays)

if simulate and simulate2:
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays_tr, apod)
    karray = get_karray(arr,
                        translation=array_offset,
                        bli_tolerance=bli_tolerance,
                        upsampling_rate=upsampling_rate)

    medium = get_medium(params)
    sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
    source = get_source(kgrid, karray, source_mat)
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
    if plot == True:
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.title('forward sim with TR delays')
        plt.colorbar()
        
if simulate and simulate2:
    source_mat = arr.calc_output(input_signal, kgrid.dt, first_val, apod)
    karray = get_karray(arr,
                        translation=array_offset,
                        bli_tolerance=bli_tolerance,
                        upsampling_rate=upsampling_rate)

    medium = get_medium(params)
    sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
    source = get_source(kgrid, karray, source_mat)
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
    if plot == True:
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.title('forward sim with first non 0 delays')
        plt.colorbar()

if simulate and simulate2 and use_ct_noise:
    source_mat = arr.calc_output(input_signal, kgrid.dt, delays_tr, apod)
    karray = get_karray(arr,
                        translation=array_offset,
                        bli_tolerance=bli_tolerance,
                        upsampling_rate=upsampling_rate)

    sensor = get_sensor(kgrid, record=['p_max', 'p_min'])
    source = get_source(kgrid, karray, source_mat)
    output = kspaceFirstOrder3D(kgrid=kgrid,
                                    source=source,
                                    sensor=sensor,
                                    medium=medium2,
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
    if plot == True:
        plt.figure()
        plt.imshow(p_max[:,round(kgrid.Ny/2),:])
        plt.title('forward sim with TR delays with ct noise')
        plt.colorbar()

if plot:
    plt.show()