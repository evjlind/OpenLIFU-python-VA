from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xa

from openlifu.bf.delay_methods import DelayMethod
from openlifu.geo import Point
from openlifu.util.annotations import OpenLIFUFieldData
from openlifu.xdc import Transducer
from openlifu.sim.time_reversal import TimeReversal

from openlifu.bf import apod_methods, focal_patterns, delay_methods
from openlifu.geo import Point
from openlifu.io.LIFUInterface import LIFUInterface
from openlifu.plan import Protocol
from openlifu.sim import SimSetup

from openlifu.util.units import getunitconversion
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D
from kwave.utils.filters import extract_amp_phase
from openlifu.sim.kwave_if import get_karray, get_medium, get_source, get_sensor, get_kgrid
from kwave.options.simulation_execution_options import SimulationExecutionOptions
from kwave.options.simulation_options import SimulationOptions
from kwave.ksensor import kSensor
from kwave.ksource import kSource
from openlifu.bf.pulse import Pulse
from openlifu.bf.sequence import Sequence

@dataclass
class TRDelay(DelayMethod):
    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Speed of sound in the medium (m/s)")] = 1480.0
    
    def __init__(self,kgrid,medium,sensor,pulse,sequence):
        self.kgrid = kgrid
        self.medium = medium
        self.sensor = sensor
        self.pulse = pulse
        self.sequence = sequence
        self.simulation_options = SimulationOptions(
                            pml_auto=True,
                            pml_inside=False,
                            save_to_disk=True,
                            data_cast='single'
                        )
        self.execution_options = SimulationExecutionOptions(is_gpu_simulation=True)


    def __post_init__(self):
        if not isinstance(self.c0, (int, float)):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)
    
    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset | None=None, transform:np.ndarray | None=None):
        if params is None:
            c = self.c0
        else:
            c = self.medium['sound_speed']
        
        simulation_options = SimulationOptions(
                        pml_auto=True,
                        pml_inside=False,
                        save_to_disk=True,
                        data_cast='single'
                    )
        
        execution_options = SimulationExecutionOptions(is_gpu_simulation=True)

        xIn,yIn,zIn = target.get_position()
        freq = 400e3
        spacing = 0.25
        T0 = 1/freq
        set_dt = (T0/360)*2
        sim_setup = SimSetup(spacing=spacing, dt=set_dt, t_end=100e-6)
        duration_msec = 0.1 # Pulse Duration in milliseconds
        interval_msec = 20 # Pulse Repetition Interval in milliseconds
        pulse = Pulse(frequency=freq, duration=duration_msec*1e-3)
        sequence = Sequence(
            pulse_interval=interval_msec*1e-3,
            pulse_count=int(60/(interval_msec*1e-3)),
            pulse_train_interval=0,
            pulse_train_count=1)
        focal_pattern = focal_patterns.SinglePoint(target_pressure=300e3)
        protocol = Protocol(
            pulse=pulse,
            sequence=sequence,
            focal_pattern=focal_pattern,
            sim_setup=sim_setup)
            
        
        coords = protocol.sim_setup.get_coords()
        kgrid = get_kgrid(coords)
        units = [params[dim].attrs['units'] for dim in params.dims]
        scl = getunitconversion(units[0], 'm')
        array_offset =[-float(coord.mean())*scl for coord in params.coords.values()]
        bli_tolerance = 0.05
        upsampling_rate = 5
        karray = get_karray(arr,
                            translation=array_offset,
                            bli_tolerance=bli_tolerance,
                            upsampling_rate=upsampling_rate)

        el_list = karray.get_element_positions()
        least_x = min(el_list[0])
        least_y = min(el_list[1])
        least_z = min(el_list[2])
        ele_ordering = np.zeros([kgrid.Nx,kgrid.Ny,kgrid.Nz])
        ele_bin = np.zeros_like(ele_ordering)

        for ind in range(len(el_list[0])):
            ele_pos = [el_list[0][ind],el_list[1][ind],el_list[2][ind]]
            ix = round((ele_pos[0]+abs(least_x))/(kgrid.dx))
            iy = round((ele_pos[1]+abs(least_y))/(kgrid.dy))
            iz = round((ele_pos[2]+abs(least_z))/(kgrid.dz))
            ix = max(0,min(kgrid.Nx-1,ix))
            iy = max(0,min(kgrid.Ny-1,iy))
            iz = max(0,min(kgrid.Nz-1,iz))
            ele_ordering[ix][iy][iz] = ind + 1 # numbering 1-128 

        sensor = kSensor(record=['p'])
        sensor.mask = ele_bin
        source = kSource()
        source.p_mask = np.zeros((kgrid.Nx,kgrid.Ny,kgrid.Nz))
        ix = round((xIn/1e3+abs(least_x))/(kgrid.dx))
        iy = round((yIn/1e3+abs(least_y))/(kgrid.dy))
        iz = round((zIn/1e3+abs(least_z)/kgrid.dz))
        source.p_mask[ix,iy,iz] = 1
        n_samples = max(1, int(np.round(0.5 / (freq * kgrid.dt))))
        t = np.arange(n_samples, dtype=np.float64) * kgrid.dt
        rev_puls = np.sin(2 * np.pi * freq * t)
        source.p = np.array([np.concatenate([rev_puls,np.zeros(kgrid.Nt - len(rev_puls))])])
        medium = get_medium(params)

        sensor_data = kspaceFirstOrder3D(
            kgrid=kgrid,
            source=source,
            sensor=sensor,
            medium=medium,
            simulation_options=simulation_options,
            execution_options=execution_options
        )

        delays = np.zeros(128)
        for i in range(128):
            amp, phase, p_freq = extract_amp_phase(np.squeeze(sensor_data['p'].T[i]),1/kgrid.dt,freq,dim=0)
            delays[ele_ordering[i]] = phase/freq/(2*np.pi)

        return delays

