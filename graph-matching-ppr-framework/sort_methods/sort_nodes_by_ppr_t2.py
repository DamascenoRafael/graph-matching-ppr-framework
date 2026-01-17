import numpy as np
import numpy.typing as npt
from scipy import sparse


def sort_nodes_by_ppr_t2(
    ppr_horizon_2: npt.NDArray[np.float64] | sparse.csr_matrix[np.float64]
) -> list[int]:
    number_of_nodes = ppr_horizon_2.shape[0]
    ppr_values: list[np.float64] = []

    for node_id in range(number_of_nodes):
        value = ppr_horizon_2[node_id, node_id]
        ppr_values.append(value)

    return sorted(range(number_of_nodes), key=lambda node_id: ppr_values[node_id], reverse=True)
