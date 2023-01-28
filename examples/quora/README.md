# Evaluate Search Performance with Quora Dataset

## Benchmark Results

### BM25 with NLTK analyzer

|      | nDCG@10 | Recall@10 |
|------|---------|-----------|
| dev  | 79.5    | 80.9      |
| test | 79.4    | 79.8      |

### ANN Dense Vector Search w/ GloVe [SIF Vectors](https://openreview.net/forum?id=SyK00v5xx)

|      | nDCG@10 | Recall@10 |
|------|---------|-----------|
| dev  | 71.8    | 73.6      |
| test | 72.0    | 72.4      |

## Usage

Setup python environment:

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build index:

```basdh
python build.py --indexer bm25
```

Run evaluation script:

```bash
python evaluate.py outputs/bm25.pkl
```

Search documents interactively:

```bash
python search.py outputs/bm25.pkl
```
