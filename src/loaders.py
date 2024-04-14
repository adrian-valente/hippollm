import os
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, WikipediaLoader


def load_wikipedia(query: str) -> Document:
    """Takes in a query and returns a Document with the content of the first Wikipedia
    matching page."""
    loader = WikipediaLoader(query=query, load_max_docs=1, doc_content_chars_max=1000000)
    docs = loader.load()
    return docs[0]


def load_text(location: os.PathLike) -> Document:
    """Load a text document from a file."""
    loader = TextLoader(file_path=location)
    docs = loader.load()
    return docs[0]
    