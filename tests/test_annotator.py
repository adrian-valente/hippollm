import os
from hippollm.annotator import Annotator
from hippollm import storage
from hippollm.loaders import load_text
storage.INTERACTIVE = False

assets_path = os.path.join(os.path.dirname(__file__), 'assets')

def test_annotator():
    if not 'HIPPODB_DO_LONG_TESTS' in os.environ:
        print('Skipping long test test_annotator. Set HIPPODB_DO_LONG_TESTS to run.')
        return
    annot = Annotator()
    annot.annotate(load_text(os.path.join(assets_path, 'short.txt')))
    assert annot.db._check_integrity()
    assert len(annot.db.entities) > 0
    assert len(annot.db.facts) > 0
    