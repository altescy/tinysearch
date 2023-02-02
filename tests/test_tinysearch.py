from pathlib import Path

import tinysearch


def test_version() -> None:
    assert tinysearch.__version__ == "0.2.0"


def test_bm25() -> None:
    documents = [
        {"id": "0", "text": "hello there good man !"},
        {"id": "1", "text": "how is the weather today ?"},
        {"id": "2", "text": "it is quite windy in yokohama"},
    ]

    searcher = tinysearch.bm25(documents)
    results = searcher.search("weather windy yokohama")
    assert len(results) == 2
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"


def test_tfidf() -> None:
    documents = [
        {"id": "0", "text": "hello there good man !"},
        {"id": "1", "text": "how is the weather today ?"},
        {"id": "2", "text": "it is quite windy in yokohama"},
    ]

    searcher = tinysearch.tfidf(documents)
    results = searcher.search("weather windy yokohama")
    assert len(results) == 2
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"


def test_sif() -> None:
    documents = [
        {"id": "0", "text": "you"},
        {"id": "1", "text": "make"},
        {"id": "2", "text": "mosquito"},
    ]
    embeddings = "tests/fixtures/embeddings.txt"

    searcher = tinysearch.sif(documents, embeddings)
    results = searcher.search("mosquito")
    assert len(results) == 3
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "0"
    assert results[2]["id"] == "1"


def test_swem() -> None:
    documents = [
        {"id": "0", "text": "you"},
        {"id": "1", "text": "make"},
        {"id": "2", "text": "mosquito"},
    ]
    embeddings = "tests/fixtures/embeddings.txt"

    searcher = tinysearch.swem(documents, embeddings)
    results = searcher.search("mosquito")
    assert len(results) == 3
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "0"
    assert results[2]["id"] == "1"


def test_transformer() -> None:
    documents = [
        {"id": "0", "text": "hello there good man !"},
        {"id": "1", "text": "how is the weather today ?"},
        {"id": "2", "text": "it is quite windy in yokohama"},
    ]

    searcher = tinysearch.transformer(documents, "prajjwal1/bert-tiny")
    results = searcher.search("weather windy yokohama")
    assert len(results) == 3
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"
    assert results[2]["id"] == "0"


def test_save_and_load(tmp_path: Path) -> None:
    documents = [
        {"id": "0", "text": "hello there good man !"},
        {"id": "1", "text": "how is the weather today ?"},
        {"id": "2", "text": "it is quite windy in yokohama"},
    ]

    filename = tmp_path / "searcher.pkl"
    tinysearch.bm25(documents, approximate_search=True).save(filename)
    searcher = tinysearch.load(filename)

    results = searcher.search("weather windy yokohama")
    assert len(results) == 2
    assert results[0]["id"] == "2"
    assert results[1]["id"] == "1"
