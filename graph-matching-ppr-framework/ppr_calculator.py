from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar
import networkx as nx
import numpy as np
import numpy.typing as npt
from scipy import sparse

from models import PageRankSettings


MatrixType = TypeVar("MatrixType", npt.NDArray[np.float64], sparse.csr_matrix[np.float64])

class PPRCalculator(Generic[MatrixType], ABC):
    def __init__(
        self,
        graph: nx.Graph[object],
        node_to_id: dict[object, int],
        id_to_node: dict[int, object],
        settings: PageRankSettings
    ):
        self.graph = graph
        self.node_to_id = node_to_id
        self.id_to_node = id_to_node
        self.settings = settings
        self.number_of_nodes = graph.number_of_nodes()

    def calculate(self) -> list[MatrixType]:
        ppr_values: list[MatrixType] = []
        transition_matrix = self._create_transition_matrix()
        values_matrix: MatrixType | Literal[1] = 1 # identity matrix
        for _ in range(self.settings.max_horizon):
            new_values_matrix = self._scalar_multiply(
                (1 - self.settings.return_alpha),
                self._dot_matrices(values_matrix, transition_matrix)
            )
            self._sum_to_diagonal(new_values_matrix, self.settings.return_alpha)
            ppr_values.append(new_values_matrix)
            values_matrix = self._copy_matrix(new_values_matrix)
        return ppr_values

    def _create_transition_matrix(self) -> MatrixType:
        row: list[int] = []
        col: list[int] = []
        data: list[np.float64] = []
        for node_id in range(self.number_of_nodes):
            node = self.id_to_node[node_id]
            neighbors = list(self.graph.neighbors(node))
            neighbor_ids = np.sort(
                [self.node_to_id[neighbor] for neighbor in neighbors]
            )  # ordered insertion is better optimized
            number_of_neighbors = len(neighbors)
            if number_of_neighbors == 0:
                continue
            transition_val = np.float64(1.0) / number_of_neighbors  # avoid recomputing value
            for neighbor_id in neighbor_ids:
                row.append(node_id)
                col.append(neighbor_id)
                data.append(transition_val)
        return self._create_transition_matrix_from_data(row, col, data)


    @abstractmethod
    def _create_transition_matrix_from_data(self, row: list[int], col: list[int], data: list[np.float64]) -> MatrixType:
        """Create an empty transition matrix."""

    @abstractmethod
    def _dot_matrices(self, matrix_a: MatrixType | Literal[1], matrix_b: MatrixType) -> MatrixType:
        """Multiply two matrices."""

    @abstractmethod
    def _scalar_multiply(self, scalar: float, matrix: MatrixType) -> MatrixType:
        """Multiply a matrix by a scalar."""

    @abstractmethod
    def _sum_to_diagonal(self, matrix: MatrixType, value: float) -> None:
        """Sum a value to the diagonal of a matrix."""

    @abstractmethod
    def _copy_matrix(self, matrix: MatrixType) -> MatrixType:
        """Copy a matrix."""


class SparsePPRCalculator(PPRCalculator[sparse.csr_matrix[np.float64]]):
    def _create_transition_matrix_from_data(
            self, row: list[int], col: list[int], data: list[np.float64]
    ) -> sparse.csr_matrix:
        return sparse.csr_matrix(
            (np.array(data), (np.array(row), np.array(col))), shape=(self.number_of_nodes, self.number_of_nodes) # type: ignore
        )

    def _dot_matrices(self, matrix_a: sparse.csr_matrix | Literal[1], matrix_b: sparse.csr_matrix) -> sparse.csr_matrix:
        if isinstance(matrix_a, int):
            return matrix_b
        return sparse.csr_matrix.dot(matrix_a, matrix_b)

    def _scalar_multiply(self, scalar: float, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        return scalar * matrix

    def _sum_to_diagonal(self, matrix: sparse.csr_matrix, value: float) -> None:
        current_diagonal = matrix.diagonal()
        new_diagonal = current_diagonal + value
        matrix.setdiag(new_diagonal)

    def _copy_matrix(self, matrix: sparse.csr_matrix) -> sparse.csr_matrix:
        return matrix.copy()


class DensePPRCalculator(PPRCalculator[npt.NDArray[np.float64]]):
    def _create_transition_matrix_from_data(
            self, row: list[int], col: list[int], data: list[np.float64]
    ) -> np.ndarray:
        matrix = np.zeros((self.number_of_nodes, self.number_of_nodes))
        for r, c, d in zip(row, col, data):
            matrix[r, c] = d
        return matrix

    def _dot_matrices(self, matrix_a: np.ndarray | Literal[1], matrix_b: np.ndarray) -> np.ndarray:
        if isinstance(matrix_a, int):
            return matrix_b
        return np.dot(matrix_a, matrix_b)

    def _scalar_multiply(self, scalar: float, matrix: np.ndarray) -> np.ndarray:
        return scalar * matrix

    def _sum_to_diagonal(self, matrix: np.ndarray, value: float) -> None:
        matrix_order = matrix.shape[0]
        for i in range(matrix_order):
            matrix[i, i] += value

    def _copy_matrix(self, matrix: np.ndarray) -> np.ndarray:
        return np.copy(matrix)
