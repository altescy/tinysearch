from typing import Sequence, cast

import numpy

from tinysearch.typing import DenseMatrix
from tinysearch.vectorizers.vectorizer import Vectorizer


class TransformerVectorizer(Vectorizer[DenseMatrix]):
    AVAILABLE_METHODS = {"first", "mean", "sum", "max", "min"}

    def __init__(self, model_name: str, method: str = "sum") -> None:
        if method not in self.AVAILABLE_METHODS:
            raise ValueError(f"method must be one of {self.AVAILABLE_METHODS}")

        from transformers import pipeline

        self._model = pipeline("feature-extraction", model=model_name)
        self._method = method

    def _get_vector(self, text: str) -> DenseMatrix:
        features = numpy.array(self._model(text, padding=False)[0])

        if self._method == "first":
            return cast(DenseMatrix, features[0])

        special_tokens_mask = self._model.tokenizer(text, return_special_tokens_mask=True, return_tensors="np")[
            "special_tokens_mask"
        ][0]
        if self._method == "sum":
            return cast(DenseMatrix, features[numpy.logical_not(special_tokens_mask)].sum(axis=0))
        if self._method == "mean":
            return cast(DenseMatrix, features[numpy.logical_not(special_tokens_mask)].mean(axis=0))
        if self._method == "max":
            return cast(DenseMatrix, features[numpy.logical_not(special_tokens_mask)].max(axis=0))
        if self._method == "min":
            return cast(DenseMatrix, features[numpy.logical_not(special_tokens_mask)].min(axis=0))

        raise ValueError(f"method must be one of {self.AVAILABLE_METHODS}")

    def _get_vectors(self, texts: Sequence[str]) -> DenseMatrix:
        return numpy.array([self._get_vector(text) for text in texts])

    def vectorize_queries(self, queies: Sequence[Sequence[str]]) -> DenseMatrix:
        if not queies:
            raise ValueError("queies must not be empty")
        if not isinstance(queies[0], str):
            raise ValueError("queies must be a sequence of strings")
        queries = cast(Sequence[str], queies)
        return self._get_vectors(queries)

    def vectorize_documents(self, documents: Sequence[Sequence[str]]) -> DenseMatrix:
        if not documents:
            raise ValueError("documents must not be empty")
        if not isinstance(documents[0], str):
            raise ValueError("documents must be a sequence of strings")
        documents = cast(Sequence[str], documents)
        return self._get_vectors(documents)
