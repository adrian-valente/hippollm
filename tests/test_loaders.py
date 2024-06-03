import os
from hippollm.loaders import *
from hippollm.storage import Source

assets_path = os.path.join(os.path.dirname(__file__), 'assets')

def test_load_wikipedia():
    doc = load_wikipedia("Paris")
    assert doc is not None
    assert len(doc.page_content) > 0
    source = Source.from_document(doc)
    

def test_load_text():
    doc = load_text(os.path.join(assets_path, 'rust.txt'))
    assert doc is not None
    assert len(doc.page_content) > 0
    source = Source.from_document(doc)