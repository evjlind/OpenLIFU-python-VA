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
from scipy.io import loadmat

@dataclass
class Preset(DelayMethod):
    c0: Annotated[float, OpenLIFUFieldData("Speed of Sound (m/s)", "Speed of sound in the medium (m/s)")] = 1540.0
    """Speed of sound in the medium (m/s)"""

    def __post_init__(self):
        if not isinstance(self.c0, (int, float)):
            raise TypeError("Speed of sound must be a number")
        if self.c0 <= 0:
            raise ValueError("Speed of sound must be greater than 0")
        self.c0 = float(self.c0)

    def calc_delays(self, arr: Transducer, target: Point, params: xa.Dataset | None=None, transform:np.ndarray | None=None):
        try:
            delays = loadmat('delays.mat')
            delays = delays['delays'].reshape(-1)
        except:
            print('Delays not found')
            exit(0)
        return delays

    def to_table(self) -> pd.DataFrame:
        """
        Get a table of the delay method parameters

        :returns: Pandas DataFrame of the delay method parameters
        """
        records = [{"Name": "Type", "Value": "Preset", "Unit": ""},
       {"Name": "Default Sound Speed", "Value": self.c0, "Unit": "m/s"}]
        return pd.DataFrame.from_records(records)


