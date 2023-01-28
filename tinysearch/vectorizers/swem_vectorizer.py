from typing import Mapping, Sequence, cast

import numpy

from tinysearch.typing import DenseMatrix
from tinysearch.vectorizers.vectorizer import Vectorizer


class SwemVectorizer(Vectorizer[DenseMatrix]):
    def __init__(
        self,
        embeddings: Mapping[str, DenseMatrix],
        window_size: int = 3,
        smoothing: float = 1e-3,
        normalize: bool = False,
    ) -> None:
        self.window_size = window_size
        self.smoothing = smoothing
        self.normalize = normalize
        self.embeddings = embeddings
        self.embedding_dim = len(next(iter(embeddings.values())))

    def _get_swem_vector(self, document: Sequence[str]) -> DenseMatrix:
        vectors = numpy.array([self.embeddings[token] for token in document if token in self.embeddings])
        if len(vectors) == 0:
            return numpy.zeros(self.embedding_dim)
        if len(vectors) < self.window_size:
            padding_size = int(numpy.ceil((self.window_size - len(vectors)) / 2))
            vectors = numpy.pad(vectors, ((padding_size, padding_size), (0, 0)), "constant")
        output = vectors.min() * numpy.ones(self.embedding_dim)
        for offset in range(len(vectors) - self.window_size + 1):
            window = vectors[offset : offset + self.window_size]
            output = numpy.maximum(output, window.mean(0))
        return cast(DenseMatrix, output)

    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_swem_vector(query) for query in queies])

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> DenseMatrix:
        return numpy.vstack([self._get_swem_vector(document) for document in documents])
