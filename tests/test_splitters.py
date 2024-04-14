import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from splitters import *
from loaders import load_text

from typing import get_args

test_strategies = get_args(TSplitStrategyLiteral)
assets_path = os.path.join(os.path.dirname(__file__), 'assets')

if not 'HIPPODB_DO_LONG_TESTS' in os.environ:
    print('Skipping strategy "semantic" in test_splitters. Set HIPPODB_DO_LONG_TESTS to run.')
    test_strategies = [s for s in test_strategies if s != 'semantic']

def test_get_splitter():
    for val in test_strategies:
        print(f'test_splitters::test_get_splitter():{val}')
        splitter = get_splitter(val, chunk_size=1000)
        assert splitter is not None
        assert isinstance(splitter, TextSplitter)

def test_split_text():
    doc = load_text(os.path.join(assets_path, 'rust.txt'))
    for val in test_strategies:
        print(f'test_splitters::test_split_text():{val}')
        splitter = get_splitter(val, chunk_size=1000)
        chunks = splitter.split(doc.page_content)
        assert len(chunks) > 0
        assert isinstance(chunks[0], Chunk)
        assert len(chunks[0].text) > 0
        assert chunks[0].pos[0] == 0
        assert chunks[-1].pos[1] == len(doc.page_content)
