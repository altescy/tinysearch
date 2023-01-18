from typing import Any, Callable, Mapping, Sequence, TypeVar

import numpy
from scipy import sparse

Analyzer = Callable[[str], Sequence[str]]
DenseMatrix = numpy.ndarray
SparseMatrix = sparse.csr_matrix

Document = TypeVar("Document", bound=Mapping[str, Any])
Matrix = TypeVar("Matrix", numpy.ndarray, sparse.csr_matrix)
