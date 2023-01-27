from typing import Dict, List, Sequence, Tuple

import numpy


class Metrics:
    def __call__(self, gold: Sequence[Tuple[str, int]], pred: Sequence[Tuple[str, int]]) -> None:
        raise NotImplementedError

    def get_metrics(self) -> Dict[str, float]:
        raise NotImplementedError


class NDCG(Metrics):
    def __init__(self, topk: int) -> None:
        self._topk = topk
        self._total_score = 0.0
        self._total_count = 0

    def __call__(self, gold: Sequence[Tuple[str, int]], pred: Sequence[Tuple[str, int]]) -> None:
        gold_relevance = numpy.sort([r for _, r in gold])[::-1][: self._topk]
        pred_relevance = numpy.array([r for _, r in pred])[: self._topk]

        dcg = numpy.sum(pred_relevance / numpy.log2(numpy.arange(2, pred_relevance.size + 2)))
        idcg = numpy.sum(gold_relevance / numpy.log2(numpy.arange(2, gold_relevance.size + 2)))
        self._total_score += dcg / idcg
        self._total_count += 1

    def get_metrics(self) -> Dict[str, float]:
        return {"ndcg": self._total_score / self._total_count}


class FMeasure(Metrics):
    def __init__(self, topk: int, beta: float = 1.0) -> None:
        self._topk = topk
        self._beta = beta
        self._true_positives = 0
        self._false_positives = 0
        self._false_negatives = 0

    def __call__(self, gold: Sequence[Tuple[str, int]], pred: Sequence[Tuple[str, int]]) -> None:
        gold = sorted(gold, key=lambda x: x[1], reverse=True)[: self._topk]
        pred = pred[: self._topk]
        gold_ids = set([doc_id for doc_id, _ in gold])
        pred_ids = set([doc_id for doc_id, _ in pred])
        self._true_positives += len(gold_ids & pred_ids)
        self._false_positives += len(pred_ids - gold_ids)
        self._false_negatives += len(gold_ids - pred_ids)

    def get_metrics(self) -> Dict[str, float]:
        precision = (
            self._true_positives / (self._true_positives + self._false_positives)
            if (self._true_positives + self._false_positives) > 0
            else 0.0
        )
        recall = (
            self._true_positives / (self._true_positives + self._false_negatives)
            if (self._true_positives + self._false_negatives) > 0
            else 0.0
        )
        fbeta = (
            (1 + self._beta**2) * precision * recall / (self._beta**2 * precision + recall)
            if (self._beta**2 * precision + recall) > 0
            else 0.0
        )
        return {
            "precision": precision,
            "recall": recall,
            "fbeta": fbeta,
        }


class MultiMetrics(Metrics):
    def __init__(self, *metrics: Metrics) -> None:
        self._metrics = metrics

    def __call__(self, gold: Sequence[Tuple[str, int]], pred: Sequence[Tuple[str, int]]) -> None:
        for metric in self._metrics:
            metric(gold, pred)

    def get_metrics(self) -> Dict[str, float]:
        metrics = {}
        for metric in self._metrics:
            metrics.update(metric.get_metrics())
        return metrics
