from datetime import datetime
import sys

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from helpers import *
from prompts import *
import storage


CHUNK_SIZE = 1000
CTX_SIZE = 5000
MODEL = 'mistral'

if __name__ == "__main__":
    query = 'Paris' # sys.argv[1]
    db_location = 'wikidb' # sys.argv[2]
    loader = WikipediaLoader(query=query, doc_content_chars_max=1000000)
    docs = loader.load()
    llm = Ollama(model=MODEL)
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir=db_location)
    
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
                related_facts = db.get_closest_facts(fact)
                jump = False
                for related in related_facts:
                    prompt = confrontation_prompt.format(
                        fact=fact, 
                        context=ctx, 
                        other_fact=related.text
                    )
                    res = llm.invoke(confrontation_prompt)
                    if res.lower().strip().startswith('yes'):
                        db.add_fact_source(related.id, source)
                        print('Identified an equivalent fact in the database: ', related.text)
                        jump = True
                if jump:
                    continue
                
                # Extract entities related to the fact
                kept_entities = set()
                # Prompt to extract entities in zero-shot fashion
                extraction_prompt = entity_extraction_prompt.format(fact=fact, context=ctx)
                entities_extracted = parse_bullet_points(llm.invoke(extraction_prompt))
                print('Extracted entities:')
                print(entities_extracted)
                
                # Find identified entities in the database, or add them if they don't exist
                for entity in entities_extracted:
                    tmp_entities = db.get_closest_entities(entity, k=10)
                    print(tmp_entities)
                    prompt = entity_equivalence_prompt.format(
                        fact=fact,
                        context=ctx,
                        entity=entity,
                        choices=join_bullet_points(tmp_entities)
                    )
                    ans = choice_selection(llm.invoke(prompt).strip(), tmp_entities + ['none'])
                    if ans is not None:
                        kept_entities.add(ans.name)
                        print('Entity', entity, 'considered equivalent to', ans)
                    else:
                        db.add_entity(name=entity)
                        kept_entities.add(entity)
                        print('Entity', entity, 'added to the database.')
                    
                        
                # Find other entities that the model could have missed through sim search
                tmp_entities = db.get_closest_entities(fact, k=10)
                print(tmp_entities)
                # Finally, let the model decide which entities are really involved
                for entity in tmp_entities:
                    prompt = entity_selection_prompt.format(fact=fact, context=ctx, entity=entity)
                    ans = llm.invoke(prompt).strip()
                    if ans.lower().startswith('yes'):
                        kept_entities.add(entity.name)
                        print('Also adding entity', entity)
                    
                print('Final entities:')
                print(list(kept_entities))
                # Now add the fact to the database
                db.add_fact(text=fact, entities=list(kept_entities), source=source)
                
