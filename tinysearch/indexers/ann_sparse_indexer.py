import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import SparseMatrix
from tinysearch.util import distance_to_similarity


class AnnSparseIndexer(Indexer[SparseMatrix]):
    NMSLIB_SPACES = {
        "dotprod": "negdotprod_sparse",
        "cosine": "cosinesimil_sparse",
        "l1": "l1_sparse",
        "l2": "l2_sparse",
        "linf": "linf_sparse",
    }

    def __init__(self, space: str) -> None:
        import nmslib

        if space not in self.NMSLIB_SPACES:
            raise ValueError(f"Unknown space {space}.")

        self._index = nmslib.init(
            method="hnsw",
            space=self.NMSLIB_SPACES[space],
            data_type=nmslib.DataType.SPARSE_VECTOR,
        )
        self._distance_to_similarity = functools.partial(distance_to_similarity, space)
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def insert(self, ids: Sequence[str], data: SparseMatrix, update: bool = False) -> None:
        for id_ in ids:
            if not update and id_ in self._id_to_index:
                raise ValueError(f"Duplicate id {id_}.")
            index = len(self._id_to_index)
            self._id_to_index[id_] = index
            self._index_to_id[index] = id_
        indices = numpy.array([self._id_to_index[id_] for id_ in ids])
        self._index.addDataPointBatch(data, indices)

    def build(self, **kwargs: Any) -> None:
        self._index.createIndex(**kwargs)

    def search(self, queries: SparseMatrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        if topk is None:
            raise ValueError("topk must be specified for ann indexers.")

        knn_results = self._index.knnQueryBatch(queries, k=topk)
        results = [
            [(self._index_to_id[id_], self._distance_to_similarity(float(score))) for id_, score in zip(ids, scores)]
            for ids, scores in knn_results
        ]

        return results
