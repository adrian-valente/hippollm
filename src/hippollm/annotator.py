from typing import Optional

from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from .grammars import *
from .helpers import *
from .llm_backend import load_llm
from .log_helpers import log_action, log_message
from .prompts import *
from . import storage
from .nlp_additional import NLPModels
from .splitters import get_splitter, TSplitStrategyLiteral


class Annotator:
    
    def __init__(self, 
                 db_location: Optional[str] = None, 
                 llm_backend: str = 'llama-cpp',
                 llm_model: str = '/home/avalente/models/mistral-7b-instruct-v0.2.Q4_0.gguf',
                 llm_options: dict = {},
                 embedding_model: str = 'all-MiniLM-L6-v2',
                 split_strategy: TSplitStrategyLiteral = 'recursive',
                 chunk_size: int = 1000,
                 ctx_size: int = 5000) -> None:
        self.ctx_size = ctx_size
        # llm_options = {'n_gpu_layers': -1, 'n_ctx': 4096, 'chat_model': True}
        
        # Load models
        self.nlp_models = NLPModels()
        self.llm = load_llm(model=llm_model, backend=llm_backend, **llm_options)
        self.embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
        
        # Text splitter
        self.splitter = get_splitter(split_strategy, chunk_size)
        
        # Load database
        log_message(f'Loading database {db_location}')
        self.db = storage.EntityStore(
            embedding_model=self.embedding_model, 
            persist_dir=db_location
        )
        log_message(f'Loaded database with {len(self.db.entities)} entities and '
                    f'{len(self.db.facts)} facts.')
        
    
    def _reformulate_fact(self, fact: str, ctx: str, chunk: str) -> str:
        reform_prompt = reformulation_prompt.format(fact=fact, context=ctx, text=chunk)
        ans = self.llm.invoke(reform_prompt)
        fact = ans.strip().split('\n')[0].strip()
        log_action('llm.reformulation', reform_prompt, ans, fact=fact)
        return fact
    
    
    def _compare_fact(self, fact: str, ctx: str, source: storage.Source) -> bool:
        related_facts = self.db.get_closest_facts(fact)
        log_action('db.fact_lookup', fact, related_facts)
        for related in related_facts:
            entails = self.nlp_models.detect_entailment(related.text, fact)
            log_action('nlp.fact_entailment', [related.text, fact], entails)
            if entails:
                prompt = confrontation_prompt.format(
                    fact=fact, 
                    context=ctx, 
                    other_fact=related.text
                )
                res = self.llm.invoke(prompt, optional_grammar=grammar_yn, max_tokens=3)
                log_action('llm.fact_confrontation', prompt, res)
                if res.lower().strip().startswith('yes'):
                    self.db.add_fact_source(related.id, source)
                    log_action('db.added_fact', fact, '')
                    return True
                    # TODO: add fact merging?           
        return False
    
    
    def _extract_entities(self, fact: str, ctx: str) -> list[str]:
        extraction_prompt = entity_extraction_prompt.format(fact=fact, context=ctx)
        ans = self.llm.invoke(extraction_prompt)
        entities = parse_bullet_points(ans, only_first_bullets=True)
        log_action('llm.entity_extraction', extraction_prompt, ans, entities=entities)
        return entities
        
        
    def _find_equivalent_entity(self, entity: str, fact: str) -> Optional[str]:
        # Look for the closest entities in the database
        tmp_entities = self.db.get_closest_entities(entity, k=10)
        log_action('db.entity_lookup', entity, tmp_entities)
        
        # Classify matches with NLI model
        top_matches_i, sc = self.nlp_models.entailment_classify(
            entity, 
            [e.name for e in tmp_entities]
        )
        top_matches = [tmp_entities[i] for i in top_matches_i]
        log_action('nlp.entity_entailment_class', [entity, tmp_entities], top_matches, scores=sc)
        
        # Also look if there is an exact match in the database
        if (match := self.db.get_entity(entity)) is not None:
            top_matches.insert(0, match)
            log_action('db.entity_exact_match', entity, match)
       
        for match in top_matches:
            prompt = entity_equivalence_prompt.format(entity=entity, other=str(match), fact=fact)
            res = self.llm.invoke(prompt, optional_grammar=grammar_yn, max_tokens=3)
            log_action('llm.entity_equivalence', prompt, res)
            if res.lower().strip().startswith('yes'):
                return match
        return None
        
        
    def _fact_extractor(self, chunk: str, ctx: str, source: storage.Source) -> None:
        """Internal: extract and process facts from a chunk of text"""
        # Extract facts
        prompt = annotation_prompt.format(text=chunk, context=ctx)
        ans = self.llm.invoke(prompt)
        facts = parse_bullet_points(ans)
        log_action('llm.fact_extraction', prompt, ans, facts=facts)
        
        for fact in facts:
            
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
                else:
                    self.db.add_entity(name=entity)
                    kept_entities.add(entity)
                    log_action('db.add_entity', entity, '')
            # Now add the fact to the database
            self.db.add_fact(text=fact, entities=list(kept_entities), source=source)
            log_action('db.add_fact', str(fact), '', entities=list(kept_entities))
        
        
    def annotate(self, doc: Document) -> None:
        """Extract facts and entities from a document and save them to the database."""
        log_message(f"Processing document: {doc.metadata.get('title', 'Untitled')}")
        log_message(f'Length: {len(doc.page_content)}')
        content = doc.page_content
        
        # Contextualization
        prompt = contextualization_prompt.format(text=content[:min(self.ctx_size, len(content))])
        ans = self.llm.invoke(prompt, max_tokens=200, stop=['. ', '.\n'])
        ctx = first_sentence(ans)
        log_action('llm.contextualization', prompt, ans, context=ctx)
        
        # Create source object
        source = storage.Source.from_document(doc, ctx)
        
        # Loop through chunks of text
        chunks = self.splitter.split(content)
        for chunk in chunks:
            source = source.copy_with_new_position(chunk.pos)
            self._fact_extractor(chunk.text, ctx, source)
        
        # Save to disk
        self.db.save()
        
        
        