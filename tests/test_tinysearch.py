import tinysearch


def test_version() -> None:
    assert tinysearch.__version__ == "0.1.0"


def test_tinysearch() -> None:
    documents = [
        {"id": "0", "text": "hello there good man !"},
        {"id": "1", "text": "it is quite windy in yokohama"},
        {"id": "2", "text": "how is the weather today ?"},
    ]

    searcher = tinysearch.from_documents(documents)
    results = searcher.search("weather windy yokoham")
    assert len(results) == 2
    assert results[0]["id"] == "1"
    assert results[1]["id"] == "2"
