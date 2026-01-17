from dataclasses import dataclass
import numpy as np


@dataclass
class DistanceAmounts:
    main_distance: np.float64 = np.float64(0.0)
    anchor_distance: np.float64 = np.float64(0.0)
