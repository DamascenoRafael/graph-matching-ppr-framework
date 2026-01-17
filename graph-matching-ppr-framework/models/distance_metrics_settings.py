from dataclasses import dataclass


@dataclass
class DistanceMetricsSettings:
    amortization_sigma: float
    main_distance_importance_gamma: float
