from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Union
from omegaconf import OmegaConf
import os
import orjson

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from .helpers import is_yes

INTERACTIVE = True  # Set to False for automated tests

@dataclass
class Entity:
    name: str
    description: str
    facts: list[int]  # List of indices of facts in the facts list
    
    def __repr__(self) -> str:
        if self.description:
            return f"{self.name} ({self.description})"
        else:
            return self.name


@dataclass
class Source:
    name: str
    description: str
    url: str
    date: datetime
    position: tuple[int, int]
    
    def copy_with_new_position(self, position: tuple[int, int]) -> 'Source':
        return Source(
            name=self.name,
            description=self.description,
            url=self.url,
            date=self.date,
            position=position
        )
        
    @staticmethod
    def from_document(doc: Document, ctx: str = '') -> 'Source':
        title = doc.metadata.get('title', '')
        url = doc.metadata.get('source', '')
        
        return Source(
            name=title,
            description=ctx,
            url=url,
            date=None,
            position=(0, len(doc.page_content))
        )
        

@dataclass
class Fact:
    text: str
    entities: list[str]
    sources: list[Source]
    confidence: float          # Not used for the moment
    id: int                    # Index of the fact in the facts list
    
    def __repr__(self) -> str:
        return f"{self.text} [" + " ".join(self.entities) + "]"


class EntityStore:
    entities: dict[str, Entity]
    facts: list[Fact]
    embedding_model: Embeddings
    chroma_entities: Chroma
    chroma_facts: Chroma
    
    def __init__(
        self, 
        embedding_model: Optional[Embeddings] = None, 
        persist_dir: Optional[os.PathLike] = None,
        cfg: Optional[OmegaConf] = None,
        ) -> None:
        self.entities = {}
        self.facts = []
        self._modified = False
        
        if persist_dir:
            self.persist_dir = persist_dir
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)
            else:
                self._load()
            entities_persist_dir = os.path.join(persist_dir, "entities")
            facts_persist_dir = os.path.join(persist_dir, "facts")
        else:
            self.persist_dir = None
            entities_persist_dir = None
            facts_persist_dir = None
            
        # Source embedding model from config (if necessary in parameters.yaml)
        if not embedding_model and not cfg:
            if not persist_dir:
                raise ValueError("An embedding model is required if no persist_dir is set.")
            if not os.path.exists(os.path.join(persist_dir, "parameters.yaml")):
                raise ValueError("No parameters.yaml found in database, "
                                 "an embedding model needs to be set.")
            cfg = OmegaConf.load(os.path.join(persist_dir, "parameters.yaml"))
            
        if cfg:
            if 'annotator' in cfg:
                cfg = cfg.annotator
            if 'embedding_model' in cfg:
                embedding_model = SentenceTransformerEmbeddings(
                    model_name=cfg.embedding_model)
            else:
                raise ValueError("No embedding model found in the configuration.")
        
        self.embedding_model = embedding_model
            
        self.chroma_entities = Chroma(
            embedding_function=embedding_model,
            persist_directory=entities_persist_dir,
            collection_name='entities'
        )
        self.chroma_facts = Chroma(
            embedding_function=embedding_model,
            persist_directory=facts_persist_dir,
            collection_name='facts'
        )
        self._check_integrity()
   
    def _check_integrity(self) -> bool:
        """Check that the DB is not corrputed (graph and vector components are consistent)."""
        if len(self.entities) != self.chroma_entities._collection.count() or \
                len(self.facts) != self.chroma_facts._collection.count():
            print('The database appears corrupted: some entities or facts in the ChromaDB'
                  ' are missing in the graph DB. It was probably not saved when last used.') 
            if INTERACTIVE:
                inp = input('Do you want to restore integrity by removing these elements? (y/n)')
                if is_yes(inp):
                    self._restore_integrity()
                    return True
            else:
                return False
        return True
            
            
    def _restore_integrity(self) -> None:
        """Remove elements in the ChromaDB that are not in the entities or facts list."""
        entities_in_chroma = self.chroma_entities._collection.get(include=[])['ids']
        to_delete = []
        for k in entities_in_chroma:
            if k not in self.entities:
                to_delete.append(k)
        if len(to_delete) > 0:
            for batch in range(0, len(to_delete), 1000):
                self.chroma_entities.delete(to_delete[batch:batch+1000])
        
        # We assume that facts have not been deleted, but some facts may have been added
        # to the ChromaDB and not saved in the json. Hence, only facts at the end will be
        # deleted.
        chroma_len = self.chroma_facts._collection.count()
        if len(self.facts) < chroma_len:
            idxes = [str(i) for i in range(len(self.facts), chroma_len)]
            self.chroma_facts.delete(idxes)
            
    def _prune_lone_entities(self) -> None:
        """Remove entities without any fact."""
        to_delete = []
        for k in list(self.entities.keys()):
            if len(self.entities[k].facts) == 0:
                del self.entities[k]
                to_delete.append(k)
        if len(to_delete) > 0:
            for batch in range(0, len(to_delete), 1000):
                self.chroma_entities.delete(to_delete[batch:batch+1000])
            print(f"Removed {len(to_delete)} entities.")

    def add_entity(self, 
                   name: str,
                   description: Optional[str] = '',
                   ) -> None:
        """Add entity to the DB."""
        if name in self.entities:
            print(f"Entity {name} already exists in the store.")
            return
        desc = name  + " (" + description + ")"
        entity = Entity(
            name=name, 
            description=description, 
            facts=[]
        )
        self.entities[name] = entity
        self.chroma_entities.add_texts([desc], ids=[name], metadatas=[{'name': name}])
        self._modified = True
        
    def get_entity(self, name: str) -> Entity:
        """Get entity by name."""
        if name not in self.entities:
            return None
        return self.entities[name]
    
    def get_fact(self, id: int) -> Fact:
        """Get fact by id."""
        if id < 0 or id >= len(self.facts):
            return None
        return self.facts[id]
    
    def get_neighbours(
        self, name: str, return_facts: bool = False
    ) -> Union[list[str], list[tuple[str, list[int]]]]:
        ent = self.get_entity(name)
        neighbors = defaultdict(list)
        for f in ent.facts:
            new = [e for e in self.facts[f].entities if e != name]
            for n in new:
                neighbors[n].append(f)
        if return_facts:
            return list(neighbors.items())
        return list(neighbors.keys())
            
    
    def add_fact(self,
                 text: str,
                 entities: list[str],
                 source: Source
                 ) -> None:
        """Add fact to the DB."""
        fact = Fact(
            text=text,
            entities=entities,
            sources=[source],
            confidence=1.0,
            id=len(self.facts)
        )
        self.chroma_facts.add_texts(
            [text], 
            ids=[str(len(self.facts))], 
            metadatas=[{'id': str(len(self.facts))}]
        )
        self.facts.append(fact)
        for entity in entities:
            self.entities[entity].facts.append(len(self.facts) - 1)
        self._modified = True
        
    def add_fact_source(self, fact_id: int, source: Source) -> None:
        """Add a source to a fact."""
        self.facts[fact_id].sources.append(source)
        self._modified = True
    
    def get_closest_entities(self, query: str, k: int = 5) -> list[Entity]:
        """Get the k closest entities to a query."""
        emb = self.embedding_model.embed_query(query)
        try:
            closest = self.chroma_entities.similarity_search_by_vector(emb, k=k)
        except Exception as e:
            print(e)
            return []
        closest = [self.entities[c.metadata['name']] for c in closest]
        return closest
    
    def get_closest_facts(self, query: str, k: int = 5) -> list[Fact]:
        """Get the k closest facts to a query."""
        emb = self.embedding_model.embed_query(query)
        k = min(k, len(self.facts))
        try:
            closest = self.chroma_facts.similarity_search_by_vector(emb, k=k)
        except Exception as e:
            print(e)
            return []
        closest = [self.facts[int(c.metadata['id'])] for c in closest]
        return closest
    
    def _get_fact_ids_by_entities_union(self, entities: list[str]) -> list[Fact]:
        """Get ids of facts that involve any of the entities in the list."""
        facts = set()
        for e in entities:
            fs = self.entities[e].facts
            facts.update(fs)
        return facts
    
    def get_facts_by_entities_union(self, entities: list[str]) -> list[Fact]:
        """Get facts that involve any of the entities in the list."""
        facts = self._get_fact_ids_by_entities_union(entities)
        return [self.facts[f] for f in facts]
    
    def _get_fact_ids_by_entities_intersection(self, entities: list[str]) -> list[Fact]:
        """Get ids of facts that involve all of the entities in the list."""
        facts = set(self.entities[entities[0]].facts)
        for e in entities[1:]:
            fs = self.entities[e].facts
            facts = facts.intersection(fs)
        return facts
    
    def get_facts_by_entities_intersection(self, entities: list[str]) -> list[Fact]:
        """Get facts that involve all of the entities in the list."""
        facts = self._get_fact_ids_by_entities_intersection(entities)
        return [self.facts[f] for f in facts]
    
    def _get_closest_facts_with_ids(
        self, query: str, ids: list[int], k: int = 5
        ) -> list[Fact]:
        """Get the k closest facts to a query that are in the ids list."""
        if k > len(ids):
            return [self.facts[f] for f in ids]
        emb = self.embedding_model.embed_query(query)
        try:
            closest = self.chroma_facts.similarity_search_by_vector(
                emb, 
                k=k, 
                filter={
                    '$or': [{'id': str(f)} for f in ids]
                }
            )
        except Exception as e:
            print(e)
            return []
        closest = [self.facts[int(c.metadata['id'])] for c in closest]
        return closest
    
    def get_closest_facts_with_entities_union(
        self, query: str, entities: list[str], k: int = 5
        ) -> list[Fact]:
        """Get the k closest facts to a query that involve any of the entities."""
        facts = self._get_fact_ids_by_entities_union(entities)
        return self._get_closest_facts_with_ids(query, facts, k)
    
    def get_closest_facts_with_entities_intersection(
        self, query: str, entities: list[str], k: int = 5
        ) -> list[Fact]:
        """Get the k closest facts to a query that involve all of the entities."""
        facts = self._get_fact_ids_by_entities_intersection(entities)
        return self._get_closest_facts_with_ids(query, facts, k)
    
    def _load(self) -> None:
        """
        Load the entities and facts from the persist_dir.
        Implicitly called at initialization if persist_dir is set.
        """
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
                    if 'sources' in f and f['sources']:
                        f['sources'] = [Source(**s) for s in f['sources'] if s]
                    self.facts.append(Fact(**f))
        self._modified = False
    
    def save(self, cfg: Optional[OmegaConf] = None, prune: bool = False) -> None:
        """
        Save the entities and facts to the persist_dir.
        
        Args:
            prune: Remove entities without any fact.
        """
        if self.persist_dir is None:
            print("Cannot save if no persist_dir is set at initialization.")
            return
        if prune:
            self._prune_lone_entities()
        with open(os.path.join(self.persist_dir, "entities.json"), "wb") as f:
            entities_bytes = orjson.dumps(self.entities)
            f.write(entities_bytes)
        with open(os.path.join(self.persist_dir, "facts.json"), "wb") as f:
            facts_bytes = orjson.dumps(self.facts)
            f.write(facts_bytes)
        if cfg:
            OmegaConf.save(cfg, os.path.join(self.persist_dir, "parameters.yaml"))
        self._modified = False
            
    def __del__(self) -> None:
        if self.persist_dir and self._modified and INTERACTIVE:
            inp = input("The database has been modified. Do you want to save it? (y/n)")
            if is_yes(inp):
                self.save()
        elif not self.persist_dir:  # fixes an issue with ephemeral DBs in chroma
            self.chroma_entities.delete_collection()
            self.chroma_facts.delete_collection()
        

        