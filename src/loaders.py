from langchain_core.documents import Document
from langchain_community.document_loaders import WikipediaLoader


def load_wikipedia(query: str) -> Document:
    """Takes in a query and returns a Document with the content of the first Wikipedia
    matching page."""
    loader = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=1000000)
    docs = loader.load()
    return docs[0]