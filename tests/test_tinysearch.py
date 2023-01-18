import tinysearch


def test_version() -> None:
    assert tinysearch.__version__ == "0.1.0"


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
