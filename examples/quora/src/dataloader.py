from collections import defaultdict
from typing import Any, Dict, Iterator, List, Tuple

import ir_datasets


class DataLoader:
    def __init__(self, name: str) -> None:
        self.dataset = ir_datasets.load(name)

    def load_documents(self) -> Iterator[Dict[str, Any]]:
        for doc in self.dataset.docs_iter():
            yield {"id": doc.doc_id, "text": doc.text}

    def load_query(self) -> Iterator[Dict[str, Any]]:
        for query in self.dataset.queries_iter():
            yield {"id": query.query_id, "text": query.text}

    def load_relations(self) -> Dict[Tuple[str, str], int]:
        relations: Dict[Tuple[str, str], int] = defaultdict(lambda: 0)
        for qrel in self.dataset.qrels_iter():
            relations[(qrel.query_id, qrel.doc_id)] = qrel.relevance
        return relations

    def load_golds(self) -> Dict[str, List[Tuple[str, int]]]:
        gold: Dict[str, List[Tuple[str, int]]] = {}
        for qrel in self.dataset.qrels_iter():
            if qrel.query_id not in gold:
                gold[qrel.query_id] = []
            gold[qrel.query_id].append((qrel.doc_id, qrel.relevance))
        return gold
