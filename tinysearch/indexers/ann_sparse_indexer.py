import functools
import itertools
import tempfile
from pathlib import Path
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

    def __init__(
        self,
        space: str,
        method: str = "hnsw",
        threshold: float = 0.0,
        space_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        import nmslib

        if space not in self.NMSLIB_SPACES:
            raise ValueError(f"Unknown space {space}.")

        self._threshold = threshold
        self._index_args = dict(
            method=method,
            space=self.NMSLIB_SPACES[space],
            data_type=nmslib.DataType.SPARSE_VECTOR,
            space_params=space_params,
        )
        self._index = nmslib.init(**self._index_args)
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

    def build(self, *args: Any, **kwargs: Any) -> None:
        self._index.createIndex(*args, **kwargs)

    def search(self, queries: SparseMatrix, topk: Optional[int]) -> List[List[Tuple[str, float]]]:
        if topk is None:
            raise ValueError("topk must be specified for ann indexers.")

        knn_results = self._index.knnQueryBatch(queries, k=topk)
        results = [
            [
                (self._index_to_id[id_], score)
                for id_, score in itertools.takewhile(
                    lambda x: x[1] > self._threshold,
                    zip(ids, (self._distance_to_similarity(float(score)) for score in scores)),
                )
            ]
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
            "threshold": self._threshold,
        }
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        import nmslib

        self._index_args = state["index_args"]
        self._distance_to_similarity = state["distance_to_similarity"]
        self._id_to_index = state["id_to_index"]
        self._index_to_id = state["index_to_id"]
        self._threshold = state["threshold"]

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
