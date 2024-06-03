from langchain_community.embeddings import FakeEmbeddings

from datetime import datetime as dt
import os
import sys
from tempfile import TemporaryDirectory
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from hippollm import storage
storage.INTERACTIVE = False

from hippollm.storage import EntityStore, Entity, Fact, Source


def add_some_fake_facts(db: EntityStore) -> None:
    db.add_entity(name="Paris", description="Capital of France")
    db.add_entity(name="London", description="Capital of the UK")
    db.add_entity(name="France", description="Country in Europe")
    db.add_entity(name="UK", description="Country in Europe")
    src = Source(name="Me", description="Just me", url="http://me.com", date=dt.now(), position=(0, 0))
    db.add_fact(text="Paris is the capital of France", entities=["Paris", "France"], source=src)
    db.add_fact(text="London is the capital of the UK", entities=["London", "UK"], source=None)

def test_add_retrieve_entities():
    emb_model = FakeEmbeddings(size=740)
    db = EntityStore(embedding_model=emb_model)
    db.add_entity(name="Paris", description="Capital of France")
    assert 'Paris' in db.entities
    assert len(db.entities) == 1
    assert db.chroma_entities._collection.count() == 1
    ret = db.get_entity("Paris")
    assert type(ret) == Entity
    assert ret.name == "Paris"
    closest = db.get_closest_entities('City in France')
    assert len(closest) == 1
    ret = closest[0]
    assert ret.name == "Paris"
    
    
def test_add_retrieve_facts():
    emb_model = FakeEmbeddings(size=740)
    db = EntityStore(embedding_model=emb_model)
    add_some_fake_facts(db)
    assert len(db.facts) == 2
    assert db.chroma_facts._collection.count() == 2
    assert len(db.entities["Paris"].facts) == 1
    assert len(db.entities["London"].facts) == 1
    closest = db.get_closest_facts('Capital of the UK', k=2)
    assert len(closest) == 2
    assert type(closest[0]) == Fact
    
    
    
def test_save_load():
    with TemporaryDirectory() as tmpdir:
        emb_model = FakeEmbeddings(size=740)
        db = EntityStore(embedding_model=emb_model, persist_dir=tmpdir)
        add_some_fake_facts(db)
        db.save()
        del db
        db2 = EntityStore(embedding_model=emb_model, persist_dir=tmpdir)
        assert len(db2.entities) == 4
        assert len(db2.facts) == 2
        assert db2.entities["Paris"].name == "Paris"
        assert db2.entities["London"].name == "London"
        assert db2.facts[0].text == "Paris is the capital of France"
        assert db2.facts[1].text == "London is the capital of the UK"
    
    
def test_restore_integrity():
    emb_model = FakeEmbeddings(size=740)
    db = EntityStore(embedding_model=emb_model)
    assert db._check_integrity() == True
    db.add_entity(name="Paris", description="Capital of France")
    db.add_entity(name="London", description="Capital of the UK")
    del db.entities["London"]
    assert db._check_integrity() == False
    db._restore_integrity()
    assert db.chroma_entities._collection.count() == 1
    assert db._check_integrity() == True
    

def test_hybrid_retrieval_facts():
    emb_model = FakeEmbeddings(size=740)
    db = EntityStore(embedding_model=emb_model)
    add_some_fake_facts(db)
    closest_paris = db.get_closest_facts_with_entities_union(
        "Capital", ["Paris"], k=2
    )
    assert len(closest_paris) == 1
    assert closest_paris[0].text == "Paris is the capital of France"
    closest_paris_or_london = db.get_closest_facts_with_entities_union(
        "Capital", ["Paris", "London"], k=2
    )
    assert len(closest_paris_or_london) == 2
    closest_paris_and_london = db.get_closest_facts_with_entities_intersection(
        "Capital", ["Paris", "London"], k=2
    )
    assert len(closest_paris_and_london) == 0
    closest_paris_and_france = db.get_closest_facts_with_entities_intersection(
        "Capital", ["Paris", "France"], k=2
    )
    assert len(closest_paris_and_france) == 1