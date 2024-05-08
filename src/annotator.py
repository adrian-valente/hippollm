from typing import Optional

from langchain_core.documents import Document
from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from grammars import *
from helpers import *
from llm_backend import load_llm
from prompts import *
import storage
from nlp_additional import NLPModels
from splitters import get_splitter, TSplitStrategyLiteral


class Annotator:
    
    def __init__(self, 
                 db_location: Optional[str] = None, 
                 verbosity: int = 1,
                 llm_backend: str = 'llama-cpp',
                 llm_model: str = '/home/avalente/models/mistral-7b-instruct-v0.2.Q4_0.gguf',
                 llm_options: dict = {'n_gpu_layers': -1, 'n_ctx': 5000},
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 split_strategy: TSplitStrategyLiteral = 'recursive',
                 chunk_size: int = 1000,
                 ctx_size: int = 5000) -> None:
        self.verbosity = verbosity
        self.ctx_size = ctx_size
        
        # Load models
        self.nlp_models = NLPModels()
        self.llm = load_llm(model=llm_model, backend=llm_backend, **llm_options)
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
        
        # Text splitter
        self.splitter = get_splitter(split_strategy, chunk_size)
        
        # Load database
        print('Loading database', db_location)
        self.db = storage.EntityStore(
            embedding_model=self.embedding_model, 
            persist_dir=db_location
        )
        print(f'Loaded database with {len(self.db.entities)} entities and '
              f'{len(self.db.facts)} facts.')
        
    
    def _reformulate_fact(self, fact: str, ctx: str, chunk: str) -> str:
        reform_prompt = reformulation_prompt.format(fact=fact, context=ctx, text=chunk)
        fact = self.llm.invoke(reform_prompt).strip().split('\n')[0].strip()
        if self.verbosity:
            print("After reformulation: ", fact)
        return fact
    
    
    def _compare_fact(self, fact: str, ctx: str, source: storage.Source) -> bool:
        related_facts = self.db.get_closest_facts(fact)
        for related in related_facts:
            if self.nlp_models.detect_entailment(related.text, fact):
                if self.verbosity > 0:
                    print('Identified an overlapping fact in the database: ', related.text)
                prompt = confrontation_prompt.format(
                    fact=fact, 
                    context=ctx, 
                    other_fact=related.text
                )
                res = self.llm.invoke(prompt, optional_grammar=grammar_yn, max_tokens=3)
                if res.lower().strip().startswith('yes'):
                    self.db.add_fact_source(related.id, source)
                    if self.verbosity > 0:
                        print('Added source to the fact.')
                    return True
                    # TODO: add fact merging?
                else:
                    if self.verbosity > 0:
                        print('Considered non redundant')
        return False
    
    
    def _extract_entities(self, fact: str, ctx: str) -> list[str]:
        extraction_prompt = entity_extraction_prompt.format(fact=fact, context=ctx)
        ans = self.llm.invoke(extraction_prompt)
        entities = parse_bullet_points(ans, only_first_bullets=True)
        if self.verbosity > 0:
            print('Extracted entities:')
            print(entities)
        return entities
        
        
    def _find_equivalent_entity(self, entity: str, fact: str) -> Optional[str]:
        # Look for the closest entities in the database
        tmp_entities = self.db.get_closest_entities(entity, k=10)
        
        # Classify matches with NLI model
        top_matches_i, sc = self.nlp_models.entailment_classify(
            entity, 
            [e.name for e in tmp_entities]
        )
        top_matches = [tmp_entities[i] for i in top_matches_i]
        
        # Also look if there is an exact match in the database
        if (match := self.db.get_entity(entity)) is not None:
            top_matches.insert(0, match)
            log_action('db.entity_exact_match', entity, match)
       
        # TODO: replace with entity linking model?
        # Then look for equivalent entity with NLI model + prompt
        if tmp_entities:
            top_matches, sc = self.nlp_models.entailment_classify(entity, [e.name for e in tmp_entities])
            for match in top_matches:
                prompt = entity_equivalence_prompt.format(entity=entity, other=match.name)
                res = self.llm.invoke(prompt, optional_grammar=grammar_yn, max_tokens=3)
                if res.lower().strip().startswith('yes'):
                    return match
        return None
        
        
    def _fact_extractor(self, chunk: str, ctx: str, source: storage.Source) -> None:
        """Internal: extract and process facts from a chunk of text"""
        # Extract facts
        prompt = annotation_prompt.format(text=chunk, context=ctx)
        facts = parse_bullet_points(self.llm.invoke(prompt))
        
        for fact in facts:
            if self.verbosity > 0:
                print('Processing new fact:')
                print(fact)
            
            # Reformulate
            fact = self._reformulate_fact(fact, ctx, chunk)
            
            # Check if fact not already registered (if so add source and move on)
            if self._compare_fact(fact, ctx, source):
                continue
            
            # Extract entities related to the fact
            entities_extracted = self._extract_entities(fact, ctx)
            
            # Find identified entities in the database, or add them if they don't exist
            kept_entities = set()
            for entity in entities_extracted:
                top_match = self._find_equivalent_entity(entity, fact)
                # Keep track of equivalent entities, and add new ones to the database
                if top_match is not None:
                    kept_entities.add(top_match.name)
                    if self.verbosity > 0:
                        print('Entity', entity, 'considered equivalent to', top_match)
                else:
                    self.db.add_entity(name=entity)
                    kept_entities.add(entity)
                    if self.verbosity > 0:
                        print('Entity', entity, 'added to the database.') 
            
            if self.verbosity > 0:
                print('Final entities:')
                print(list(kept_entities))
            # Now add the fact to the database
            self.db.add_fact(text=fact, entities=list(kept_entities), source=source)
        
        
    def annotate(self, doc: Document) -> None:
        """Extract facts and entities from a document and save them to the database."""
        print("Processing document:", doc.metadata.get('title', 'Untitled'))
        print("Length:", len(doc.page_content))
        content = doc.page_content
        
        # Contextualization
        prompt = contextualization_prompt.format(text=content[:min(self.ctx_size, len(content))])
        ctx = first_sentence(self.llm.invoke(prompt, max_tokens=200, stop=['. ', '.\n']))
        
        # Create source object
        source = storage.Source.from_document(doc, ctx)
        
        # Loop through chunks of text
        chunks = self.splitter.split(content)
        for chunk in chunks:
            source = source.copy_with_new_position(chunk.pos)
            self._fact_extractor(chunk.text, ctx, source)
        
        # Save to disk
        self.db.save()
        
        
        