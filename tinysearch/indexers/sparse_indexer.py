from typing import Dict, List, Optional, Sequence, Tuple

from scipy import sparse

from tinysearch.indexers.indexer import Indexer
from tinysearch.util import csr_row_normalize


class SparseIndexer(Indexer[sparse.csr_matrix]):
    AVAILABLE_SPACES = {"dotprod", "cosine"}

    def __init__(self, space: str = "dotprod", threshold: float = 0.0) -> None:
        if space not in self.AVAILABLE_SPACES:
            raise ValueError(f"Unknown space {space}.")

        self._space = space
        self._threshold = threshold
        self._data = sparse.csr_matrix((0, 0))
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def _compute_similarity(self, source: sparse.csr_matrix, target: sparse.csr_matrix) -> sparse.csr_matrix:
        if self._space == "dotprod":
            return source @ target.T
        if self._space == "cosine":
            source = csr_row_normalize(source)
            target = csr_row_normalize(target)
            return source @ target.T
        raise ValueError(f"Unknown space {self._space}.")

    def insert(self, ids: Sequence[str], data: sparse.csr_matrix, update: bool = False) -> None:
        if len(ids) != data.shape[0]:
            raise ValueError("Number of ids must match number of rows in data.")
        for id_ in ids:
            if not update and id_ in self._id_to_index:
                raise ValueError(f"Duplicate id {id_}.")
            index = len(self._id_to_index)
            self._id_to_index[id_] = index
            self._index_to_id[index] = id_
        if self._data.shape[0] == 0:
            self._data = data
        else:
            self._data = sparse.vstack([self._data, data])

    def search(self, queries: sparse.csr_matrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        scores = self._compute_similarity(queries, self._data)
        scores.data[scores.data <= self._threshold] = 0.0
        scores.eliminate_zeros()
        results: List[List[Tuple[str, float]]] = []
        for row in scores:
            indices = sorted(row.indices.tolist(), key=lambda index: float(row[0, index]), reverse=True)
            if topk is not None:
                indices = indices[:topk]
            results.append([(self._index_to_id[index], float(row[0, index])) for index in indices])
        return results
