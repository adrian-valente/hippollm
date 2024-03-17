from langchain_community.embeddings import FakeEmbeddings

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from storage import EntityStore, Entity

def test_add_entities():
    emb_model = FakeEmbeddings(size=740)
    db = EntityStore(embedding_model=emb_model)
    db.add_entity(name="Paris", description="Capital of France")
    assert 'Paris' in db.entities
    assert len(db.entities) == 1
    ret = db.get_entity("Paris")
    assert type(ret) == Entity
    assert ret.name == "Paris"
    closest = db.get_closest_entities('City in France')
    assert len(closest) == 1
    ret = closest[0]
    assert ret.name == "Paris"
