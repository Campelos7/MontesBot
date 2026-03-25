from langchain_core.documents import Document

from database.indexer import documents_within_max_distance


def test_documents_within_max_distance_filters():
    pairs = [
        (Document(page_content="a"), 0.3),
        (Document(page_content="b"), 1.5),
        (Document(page_content="c"), 0.9),
    ]
    out = documents_within_max_distance(pairs, max_distance=1.0)
    assert len(out) == 2
    assert out[0].page_content == "a"
    assert out[1].page_content == "c"
