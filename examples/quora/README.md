# Evaluate Search Performance with Quora Dataset

## Benchmark Results

### BM25 with NLTK analyzer

|      | nDCG@10 | Recall@10 |
|------|---------|-----------|
| dev  | 79.5    | 80.9      |
| test | 79.4    | 79.8      |

## Usage

Setup python environment:

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Build index:

```basdh
python build.py python build.py outputs/index.pkl
```

Run evaluation script:

```bash
python evaluate.py outputs/index.pkl
```

Search documents interactively:

```bash
python search.py outputs/index.pkl
```
