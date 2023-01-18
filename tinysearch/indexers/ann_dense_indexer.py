from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import DenseMatrix


class AnnDenseIndexer(Indexer[DenseMatrix]):
    def __init__(self) -> None:
        import nmslib

        self._index = nmslib.init(method="hnsw", space="cosinesimil")
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}

    def insert(self, ids: Sequence[str], data: DenseMatrix, update: bool = False) -> None:
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

    def search(self, queries: DenseMatrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        if topk is None:
            raise ValueError("topk must be specified for ann indexers.")

        knn_results = self._index.knnQueryBatch(queries, k=topk)
        results: List[List[Tuple[str, float]]] = []
        for ids, scores in knn_results:
            results.append([(self._index_to_id[id_], float(score)) for id_, score in zip(ids, scores)])

        return results
