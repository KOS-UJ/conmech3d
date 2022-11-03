from dataclasses import dataclass
from typing import Union

import numpy as np


@dataclass
class Boundary:
    surfaces: np.ndarray
    node_indices: Union[slice, np.ndarray]  # slice or direct indices
    node_count: int