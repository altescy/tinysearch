import functools
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy

from tinysearch.indexers.indexer import Indexer
from tinysearch.typing import DenseMatrix
from tinysearch.util import distance_to_similarity


class AnnDenseIndexer(Indexer[DenseMatrix]):
    NMSLIB_SPACES = {
        "dotprod": "negdotprod",
        "cosine": "cosinesimil",
        "l1": "l1",
        "l2": "l2",
        "linf": "linf",
    }

    def __init__(
        self,
        space: str = "dotprod",
        method: str = "hnsw",
        space_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        import nmslib

        if space not in self.NMSLIB_SPACES:
            raise ValueError(f"Unknown space {space}.")

        self._index_args = dict(
            method=method,
            space=self.NMSLIB_SPACES[space],
            data_type=nmslib.DataType.DENSE_VECTOR,
            space_params=space_params,
        )
        self._index = nmslib.init(**self._index_args)
        self._distance_to_similarity = functools.partial(distance_to_similarity, space)
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
        results = [
            [(self._index_to_id[id_], self._distance_to_similarity(float(score))) for id_, score in zip(ids, scores)]
            for ids, scores in knn_results
        ]

        return results

    def __getstate__(self) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory() as _workdir:
            workdir = Path(_workdir)
            index_filename = workdir / "index.bin"
            index_data_filename = workdir / "index.bin.dat"
            self._index.saveIndex(str(index_filename), save_data=True)
            assert index_data_filename.is_file()
            with index_filename.open("rb") as binfile:
                indexder_bytes = binfile.read()
            with index_data_filename.open("rb") as binfile:
                index_data_bytes = binfile.read()

        state = {
            "index": indexder_bytes,
            "index_data": index_data_bytes,
            "index_args": self._index_args,
            "distance_to_similarity": self._distance_to_similarity,
            "id_to_index": self._id_to_index,
            "index_to_id": self._index_to_id,
        }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        import nmslib

        self._index_args = state["index_args"]
        self._distance_to_similarity = state["distance_to_similarity"]
        self._id_to_index = state["id_to_index"]
        self._index_to_id = state["index_to_id"]

        self._index = nmslib.init(**self._index_args)

        with tempfile.TemporaryDirectory() as _workdir:
            workdir = Path(_workdir)
            index_filename = workdir / "index.bin"
            index_data_filename = workdir / "index.bin.dat"
            with index_filename.open("wb") as binfile:
                binfile.write(state["index"])
            with index_data_filename.open("wb") as binfile:
                binfile.write(state["index_data"])

            self._index.loadIndex(str(index_filename), load_data=True)
