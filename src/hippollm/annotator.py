from copy import deepcopy
from omegaconf import OmegaConf
import os
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from .grammars import *
from .helpers import *
from .llm_backend import load_llm
from .log_helpers import log_action, log_message, log_setup
from .prompts import *
from . import storage
from .nlp_additional import NLPModels
from .splitters import get_splitter, TSplitStrategyLiteral


class Annotator:
    _params_to_save = [
        'llm_backend', 'llm_model', 'llm_options', 'embedding_model',
        'split_strategy', 'chunk_size', 'ctx_size'
    ]
    _configurable_params = [
        'llm_backend', 'llm_model', 'llm_options', 'embedding_model',
        'split_strategy', 'chunk_size', 'ctx_size'
    ]
    
    # Default instance variables
    llm_options: dict[str, Any] = {}
    embedding_model: str = 'all-MiniLM-L6-v2'
    split_strategy: str = 'recursive'
    chunk_size: int = 1000
    ctx_size: int = 5000
    
    def __init__(self, *,
                 db_location: Optional[str] = None, 
                 llm_backend: Optional[str] = None,
                 llm_model: Optional[str] = None,
                 llm_options: Optional[dict] = None,
                 embedding_model: Optional[str] = None,
                 split_strategy: Optional[TSplitStrategyLiteral] = None,
                 chunk_size: Optional[int] = None,
                 ctx_size: Optional[int] = None,
                 cfg: Optional[OmegaConf] = None,
                 try_load_db_config: bool = True,
                 log_in_db: bool = True) -> None:
        """
        Initialize the Annotator object.
        
        Args:
            db_location: The location of the database.
            llm_backend: The LLM backend to use.
            llm_model: The LLM model to use.
            llm_options: Additional options to pass to the LLM model.
            embedding_model: The SentenceTransformer model to use for embeddings.
            split_strategy: The strategy to use for splitting text into chunks.
            chunk_size: The size of the chunks to split the text into.
            ctx_size: The size of the context to use for contextualization.
            cfg: A configuration object (superseded by other arguments).
            try_load_db_config: if True and cfg is None, try to load the db config file.
            log_in_db: if True, the log location will be in the database directory.
        """
        # Load configuration (by default, try to load the db config file)
        if try_load_db_config and cfg is None and db_location is not None:
            if os.path.exists(
                cfg_path := os.path.join(db_location, 'parameters.yaml')):
                cfg = OmegaConf.load(cfg_path)
        self._load_config(locals(), cfg)
        
        # Load models
        self.nlp_models = NLPModels()
        self.llm = load_llm(model=self.llm_model, backend=self.llm_backend, **self.llm_options)
        self.embedding_model = SentenceTransformerEmbeddings(model_name=self.embedding_model)
        
        # Text splitter
        self.splitter = get_splitter(self.split_strategy, self.chunk_size)
        
        # Logging setup
        logging_path = None
        if log_in_db:
            if db_location is None:
                print("Warning: log_in_db is True but no db_location provided. Logging to default location.")
            else:
                logging_path = os.path.join(db_location, 'log')
        log_setup(logging_path)
        
        # Load database
        log_message(f'Loading database {db_location}')
        self.db = storage.EntityStore(
            embedding_model=self.embedding_model, 
            persist_dir=db_location
        )
        log_message(f'Loaded database with {len(self.db.entities)} entities and '
                    f'{len(self.db.facts)} facts.')
        
    def _load_config(self, local_vars: dict, cfg: Optional[OmegaConf]) -> None:
        if cfg is not None:
            if 'annotator' in cfg:
                cfg = cfg.annotator
        else:
            cfg = {}
        for attr in self._configurable_params:
            if attr in local_vars and local_vars[attr] is not None:
                setattr(self, attr, local_vars[attr])
            elif attr in cfg and cfg[attr] is not None:
                setattr(self, attr, cfg[attr])
            else:
                setattr(self, attr, getattr(self, attr, None))
        
        # Create config from arguments (for saving)
        self.cfg = OmegaConf.create(
            {k: getattr(self, k) for k in self._params_to_save}
        )
    
    def _reformulate_fact(self, fact: str, ctx: str, chunk: str) -> str:
        reform_prompt = reformulation_prompt.format(fact=fact, context=ctx, text=chunk)
        ans = self.llm.invoke(reform_prompt)
        if ans.strip().startswith("Here is"):
            ans = ans.split(':', 1)[1].strip()
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
                    log_action('db.added_fact_source', fact, '')
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
        if len(tmp_entities) == 0:
            return None
        
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
        prompt = contextualization_prompt.format(text=content[:min(self.cfg.ctx_size, len(content))])
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
        self.db.save(self.cfg)
