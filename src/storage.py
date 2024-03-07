from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple
import numpy as np
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma


@dataclass
class Entity:
    name: str
    description: str
    embedding: np.array
    facts: List[int]  # List of indices of facts in the facts list


@dataclass
class Source:
    name: str
    description: str
    url: str
    date: datetime
    position: Tuple[int, int]
    
    def copy_with_new_position(self, position: Tuple[int, int]) -> Source:
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
    emb: np.array
    sources: List[Source]
    confidence: float
    id: int    # Index of the fact in the facts list


class EntityStore:
    entities: dict[str, Entity]
    facts: List[Fact]
    embedding_model: Embeddings
    chroma_entities: Chroma
    chroma_facts: Chroma
    
    def __init__(self, embedding_model: Embeddings):
        self.entities = {}
        self.facts = []
        self.embedding_model = embedding_model
        self.chroma_entities = Chroma(embedding_function=embedding_model)
        self.chroma_facts = Chroma(embedding_function=embedding_model)

    def add_entity(self, 
                   name: str,
                   description: Optional[str] = '',
                   ) -> None:
        if name in self.entities:
            print(f"Entity {name} already exists in the store.")
            return
        desc = name  + " " + description
        emb = np.array(self.embedding_model.embed_query(desc))
        entity = Entity(
            name=name, 
            description=description, 
            embedding=emb, 
            facts=[]
        )
        self.entities[name] = entity
        self.chroma_entities.add_texts([desc], ids=[name])
        
    def get_entity(self, name: str):
        if name not in self.entities:
            print(f"Entity {name} does not exist in the store.")
            return None
        return self.entities[name]
    
    def add_fact(self,
                 text: str,
                 entities: List[str],
                 source: Source
                 ) -> None:
        emb = np.array(self.embedding_model.embed_query(text))
        fact = Fact(
            text=text,
            entities=entities,
            emb=emb,
            sources=[source],
            confidence=1.0,
            id=len(self.facts)
        )
        self.chroma_facts.add_texts([text], ids=[str(len(self.facts))])
        self.facts.append(fact)
        for entity in entities:
            self.entities[entity].facts.append(len(self.facts) - 1)
        
    def add_fact_source(self, fact_id: int, source: Source):
        self.facts[fact_id].sources.append(source)
    
    def get_closest_entities(self, query: str, k: int = 5):
        emb = np.array(self.embedding_model.embed_query(query))
        closest = self.chroma_entities.similarity_search_by_vector(emb, k=k)
        closest = [self.entities[c] for c in closest]
        return closest
    
    def get_closest_facts(self, query: str, k: int = 5):
        emb = np.array(self.embedding_model.embed_query(query))
        k = min(k, len(self.facts))
        closest = self.chroma_facts.similarity_search_by_vector(emb, k=k)
        closest = [self.facts[int(c)] for c in closest]
        return [f[0] for f in closest]
        