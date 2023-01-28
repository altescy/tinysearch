# TinySearch

[![Actions Status](https://github.com/altescy/tinysearch/workflows/CI/badge.svg)](https://github.com/altescy/tinysearch/actions/workflows/ci.yml)
[![License](https://img.shields.io/github/license/altescy/tinysearch)](https://github.com/altescy/tinysearch/blob/main/LICENSE)

```python
import tinysearch

documents = [
    {"id": "0", "text": "hello there good man !"},
    {"id": "1", "text": "how is the weather today ?"},
    {"id": "2", "text": "it is quite windy in yokohama"},
]

searcher = tinysearch.bm25(documents)
# searcher = tinysearch.tfidf(documents)
# searcher = tinysearch.sif(documents, embeddings="path/to/embeddings.txt")
# searcher = tinysearch.transformer(documents, model_name="bert-base-uncased")

results = searcher.search("weather windy yokohama")
print(results)
# [{'id': '2', 'text': 'it is quite windy in yokohama'},
#  {'id': '1', 'text': 'how is the weather today ?'}]

searcher.save("tinysearch.bin")
searcher = tinyserach.load("tinysearch.bin")
```
