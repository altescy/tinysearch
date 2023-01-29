from typing import Dict, List, Optional, Sequence, Tuple

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import SparseMatrix


class InvertedIndexer(Indexer[SparseMatrix]):
    def __init__(self) -> None:
        self._id_to_index: Dict[str, int] = {}
        self._index_to_id: Dict[int, str] = {}
        self._inverted_index: Dict[int, Dict[int, float]] = {}

    def insert(self, ids: Sequence[str], data: SparseMatrix, update: bool = False) -> None:
        rows, cols = data.nonzero()
        for row, col in zip(rows, cols):
            id_ = ids[row]
            if id_ not in self._id_to_index:
                if update:
                    continue
                index = len(self._id_to_index)
                self._id_to_index[id_] = index
                self._index_to_id[index] = id_
            else:
                index = self._id_to_index[id_]
            self._inverted_index.setdefault(col, {})[index] = float(data[row, col])

    def search(self, queries: SparseMatrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        rows, cols = queries.nonzero()
        document_scores: List[Dict[int, float]] = [{} for _ in range(queries.shape[0])]
        for row, col, value in zip(rows, cols, queries.data):
            for index, score in self._inverted_index.get(col, {}).items():
                document_scores[row][index] = document_scores[row].get(index, 0.0) + (score * value)
        return [
            [
                (self._index_to_id[index], score)
                for index, score in sorted(document_scores[row].items(), key=lambda x: x[1], reverse=True)[:topk]
            ]
            for row in range(queries.shape[0])
        ]
