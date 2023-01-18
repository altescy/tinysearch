from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import DenseMatrix


class DenseIndexer(Indexer[DenseMatrix]):
    AVAILABLE_SPACES = {"dotprod", "cosine", "l1", "l2", "linf"}

    def __init__(self, space: str) -> None:
        if space not in self.AVAILABLE_SPACES:
            raise ValueError(f"Unknown space {space}.")
        self._space = space
        self._data = numpy.zeros((0, 0))
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def _compute_similarity(self, source: DenseMatrix, target: DenseMatrix) -> DenseMatrix:
        if self._space == "dotprod":
            return cast(DenseMatrix, source @ target.T)
        if self._space == "cosine":
            source_norm = numpy.linalg.norm(source, axis=1, keepdims=True)
            target_norm = numpy.linalg.norm(target, axis=1, keepdims=True)
            return cast(DenseMatrix, source @ target.T / source_norm / target_norm.T)
        if self._space == "l1":
            return cast(DenseMatrix, -numpy.sum(numpy.abs(source[:, None, :] - target[None, :, :]), axis=2))
        if self._space == "l2":
            return cast(DenseMatrix, -numpy.linalg.norm(source[:, None, :] - target[None, :, :], axis=2))
        if self._space == "linf":
            return cast(DenseMatrix, -numpy.max(numpy.abs(source[:, None, :] - target[None, :, :]), axis=2))
        raise ValueError(f"Unknown space {self._space}.")

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
        if topk is None:
            raise ValueError("topk must be specified for dense indexers.")

        scores = self._compute_similarity(queries, self._data)
        indices = numpy.argsort(scores, axis=1)[:, ::-1]
        results = [
            [(self._index_to_id[index], float(scores[i, index])) for index in indices[i, :topk]]
            for i, row in enumerate(indices)
        ]

        return results
