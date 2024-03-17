from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
import os
import orjson
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma


@dataclass
class Entity:
    name: str
    description: str
    facts: List[int]  # List of indices of facts in the facts list


@dataclass
class Source:
    name: str
    description: str
    url: str
    date: datetime
    position: Tuple[int, int]
    
    def copy_with_new_position(self, position: Tuple[int, int]) -> 'Source':
        return Source(
            name=self.name,
            description=self.description,
            url=self.url,
            date=self.date,
            position=position
        )
        

@dataclass
class Fact:
    text: str
    entities: List[str]
    sources: List[Source]
    confidence: float
    id: int    # Index of the fact in the facts list


class EntityStore:
    entities: dict[str, Entity]
    facts: List[Fact]
    embedding_model: Embeddings
    chroma_entities: Chroma
    chroma_facts: Chroma
    
    def __init__(self, embedding_model: Embeddings, persist_dir: Optional[os.PathLike] = None):
        self.entities = {}
        self.facts = []
        self.embedding_model = embedding_model
        if persist_dir:
            self.persist_dir = persist_dir
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
            else:
                self.load()
            entities_persist_dir = os.path.join(persist_dir, "entities")
            facts_persist_dir = os.path.join(persist_dir, "facts")
        else:
            self.persist_dir = None
            entities_persist_dir = None
            facts_persist_dir = None
            
        self.chroma_entities = Chroma(
            embedding_function=embedding_model,
            persist_directory=entities_persist_dir,
        )
        self.chroma_facts = Chroma(
            embedding_function=embedding_model,
            persist_directory=facts_persist_dir,
        )

    def add_entity(self, 
                   name: str,
                   description: Optional[str] = '',
                   ) -> None:
        if name in self.entities:
            print(f"Entity {name} already exists in the store.")
            return
        desc = name  + " " + description
        entity = Entity(
            name=name, 
            description=description, 
            facts=[]
        )
        self.entities[name] = entity
        self.chroma_entities.add_texts([desc], ids=[name], metadatas=[{'name': name}])
        
    def get_entity(self, name: str) -> Entity:
        if name not in self.entities:
            print(f"Entity {name} does not exist in the store.")
            return None
        return self.entities[name]
    
    def add_fact(self,
                 text: str,
                 entities: List[str],
                 source: Source
                 ) -> None:
        fact = Fact(
            text=text,
            entities=entities,
            sources=[source],
            confidence=1.0,
            id=len(self.facts)
        )
        self.chroma_facts.add_texts([text], ids=[str(len(self.facts))])
        self.facts.append(fact)
        for entity in entities:
            self.entities[entity].facts.append(len(self.facts) - 1)
        
    def add_fact_source(self, fact_id: int, source: Source) -> None:
        self.facts[fact_id].sources.append(source)
    
    def get_closest_entities(self, query: str, k: int = 5) -> List[Entity]:
        emb = self.embedding_model.embed_query(query)
        try:
            closest = self.chroma_entities.similarity_search_by_vector(emb, k=k)
        except Exception as e:
            print(e)
            return []
        closest = [self.entities[c.metadata['name']] for c in closest]
        return closest
    
    def get_closest_facts(self, query: str, k: int = 5) -> List[Fact]:
        emb = self.embedding_model.embed_query(query)
        k = min(k, len(self.facts))
        try:
            closest = self.chroma_facts.similarity_search_by_vector(emb, k=k)
        except Exception as e:
            print(e)
            return []
        closest = [self.facts[int(c)] for c in closest]
        return [f[0] for f in closest]
    
    
    def load(self) -> None:
        entities_file = os.path.join(self.persist_dir, "entities.json")
        if os.path.exists(entities_file):
            with open(entities_file, "rb") as f:
                entities_bytes = f.read()
                entities = orjson.loads(entities_bytes)
                for k in entities:
                    self.entities[k] = Entity(**entities[k])
        facts_file = os.path.join(self.persist_dir, "facts.json")
        if os.path.exists(facts_file):
            with open(os.path.join(self.persist_dir, "facts.json"), "rb") as f:
                facts_bytes = f.read()
                facts = orjson.loads(facts_bytes)
                for f in facts:
                    f['sources'] = [Source(**s) for s in f['sources']]
                    self.facts.append(Fact(**f))
    
    def save(self) -> None:
        if self.persist_dir is None:
            print("Cannot save if no persist_dir is set at initialization.")
            return
        with open(os.path.join(self.persist_dir, "entities.json"), "wb") as f:
            entities_bytes = orjson.dumps(self.entities)
            f.write(entities_bytes)
        with open(os.path.join(self.persist_dir, "facts.json"), "wb") as f:
            facts_bytes = orjson.dumps(self.facts)
            f.write(facts_bytes)
        

        