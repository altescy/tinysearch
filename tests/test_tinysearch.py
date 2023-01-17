import tinysearch


def test_version() -> None:
    assert tinysearch.__version__ == "0.1.0"


def test_tinysearch() -> None:
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
