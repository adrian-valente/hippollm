from datetime import datetime
import sys

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from helpers import *
from prompts import *
import storage
from nlp_additional import NLPModels


CHUNK_SIZE = 1000
CTX_SIZE = 5000
MODEL = 'mistral'

if __name__ == "__main__":
    query = 'Paris' # sys.argv[1]
    db_location = 'wiki_test' # sys.argv[2]
    loader = WikipediaLoader(query=query, doc_content_chars_max=1000000)
    docs = loader.load()
    llm = Ollama(model=MODEL)
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    nlp_models = NLPModels()
    
    print('Loading database', db_location)
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir=db_location)
    print(f'Loaded database with {len(db.entities)} entities and {len(db.facts)} facts.')
    
    for doc in docs:
        print("Document:", doc.metadata['title'])
        print("Length:", len(doc.page_content))
        content = doc.page_content
        
        # Contextualization
        prompt = contextualization_prompt.format(text=content[:min(CTX_SIZE, len(content))])
        ctx = first_sentence(llm.invoke(prompt))
        source = storage.Source(
            name=doc.metadata['title'],
            description=ctx,
            url=doc.metadata['source'],
            date=None,
            position=(0, len(content))
        )
        
        for i in range(0, len(content), CHUNK_SIZE):
            chunk = content[i: min(i+CHUNK_SIZE, len(content))]
            source = source.copy_with_new_position((i, min(i+CHUNK_SIZE, len(content))))
            prompt = annotation_prompt.format(text=chunk, context=ctx)
            facts = parse_bullet_points(llm.invoke(prompt))
            
            for fact in facts:
                # Check if fact not already registered (if so add source)
                print('Processing new fact:')
                print(fact)
                
                # Reformulate
                reform_prompt = reformulation_prompt.format(fact=fact, context=ctx, text=chunk)
                fact = llm.invoke(reform_prompt).split('\n')[0].strip()
                
                related_facts = db.get_closest_facts(fact)
                for related in related_facts:
                    if nlp_models.detect_entailment(fact, related.text):
                        # TODO: add fact merging
                        # prompt = confrontation_prompt.format(
                        #     fact=fact, 
                        #     context=ctx, 
                        #     other_fact=related.text
                        # )
                        # res = llm.invoke(prompt)
                        # if res.lower().strip().startswith('yes'):
                        db.add_fact_source(related.id, source)
                        print('Identified an overlapping fact in the database: ', related.text)
                        print('Added source to the fact.')
                
                # Extract entities related to the fact
                kept_entities = set()
                # Prompt to extract entities in zero-shot fashion
                extraction_prompt = entity_extraction_prompt.format(fact=fact, context=ctx)
                ans = llm.invoke(extraction_prompt)
                entities_extracted = parse_bullet_points(ans, only_bullets=True)
                print('Extracted entities:')
                print(entities_extracted)
                
                # Find identified entities in the database, or add them if they don't exist
                for entity in entities_extracted:
                    # Prompt to find equivalent entities in the database
                    tmp_entities = db.get_closest_entities(entity, k=10)
                    print(tmp_entities)
                    if entity in [e.name for e in tmp_entities]:
                        top_match = [e for e in tmp_entities if e.name == entity][0]
                    elif tmp_entities:
                        top_match = nlp_models.top_entailment(entity, [e.name for e in tmp_entities])
                        if top_match is not None:
                            other = tmp_entities[top_match]
                            prompt = entity_equivalence_prompt.format(entity=entity, other=other.name)
                            res = llm.invoke(prompt)
                            if res.lower().strip().startswith('yes'):
                                top_match = other
                            else:
                                top_match = None
                    else:
                        top_match = None
                    # If an equivalent entity is found keep it, otherwise create new one
                    if top_match is not None:
                        kept_entities.add(top_match.name)
                        print('Entity', entity, 'considered equivalent to', top_match)
                    else:
                        db.add_entity(name=entity)
                        kept_entities.add(entity)
                        print('Entity', entity, 'added to the database.')
                    
                        
                # Find other entities that the model could have missed through sim search
                # tmp_entities = db.get_closest_entities(fact, k=10)
                # tmp_entities = [x for x in tmp_entities if x.name not in kept_entities]
                # print(tmp_entities)
                # # Finally, let the model decide which entities are really involved
                # for entity in tmp_entities:
                #     prompt = entity_selection_prompt.format(fact=fact, context=ctx, entity=entity)
                #     ans = llm.invoke(prompt).strip()
                #     if ans.lower().strip().startswith('yes'):
                #         kept_entities.add(entity.name)
                #         print('Also adding entity', entity)
                    
                print('Final entities:')
                print(list(kept_entities))
                # Now add the fact to the database
                db.add_fact(text=fact, entities=list(kept_entities), source=source)
            break
    db.save()
                
