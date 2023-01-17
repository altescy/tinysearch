from typing import Dict, List, Optional, Sequence, Tuple

import numpy

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import DenseMatrix


class DenseIndexer(Indexer[DenseMatrix]):
    def __init__(self, dim: int, threshold: float = 0.0) -> None:
        self._threshold = threshold
        self._data = numpy.zeros((0, dim))
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def insert(self, ids: Sequence[str], data: DenseMatrix, update: bool = False) -> None:
        if len(ids) != data.shape[0]:
            raise ValueError("Number of ids must match number of rows in data.")
        for id_ in ids:
            if not update and id_ in self._id_to_index:
                raise ValueError(f"Duplicate id {id_}.")
            index = len(self._id_to_index)
            self._id_to_index[id_] = index
            self._index_to_id[index] = id_
        if len(self._data) == 0:
            self._data = data
        else:
            self._data = numpy.vstack([self._data, data])

    def search(self, queries: DenseMatrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        scores = queries @ self._data.T
        indices = numpy.argsort(scores, axis=1)[:, ::-1]
        results: List[List[Tuple[str, float]]] = []
        for row in range(len(queries)):
            results.append([])
            for col in range(min(topk or len(self._data), len(self._data))):
                if scores[row, col] <= self._threshold:
                    break
                id_ = self._index_to_id[indices[row, col]]
                score = float(scores[row, col])
                results[row].append((id_, score))
        return results
