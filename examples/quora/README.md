# Evaluate Search Performance with Quora Dataset

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
