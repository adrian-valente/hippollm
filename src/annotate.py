from datetime import datetime
import sys

from langchain_community.document_loaders import WikipediaLoader
from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from helpers import *
from prompts import *
import storage


CHUNK_SIZE = 1000
CTX_SIZE = 5000
MODEL = 'mistral'

if __name__ == "__main__":
    query = 'Paris' # sys.argv[1]
    loader = WikipediaLoader(query=query, doc_content_chars_max=1000000)
    docs = loader.load()
    llm = Ollama(model=MODEL)
    
    db = storage.EntityStore(OllamaEmbeddings(model=MODEL))
    
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
                related_facts = db.get_closest_facts(fact)
                for related in related_facts:
                    prompt = confrontation_prompt.format(
                        fact=fact, 
                        context=ctx, 
                        other_fact=related.text
                    )
                    res = llm.invoke(confrontation_prompt)
                    if res.lower().strip().startswith('yes'):
                        db.add_fact_source(related.id, source)
                        break 
                
                # Extract entities related to the fact
                entities = db.get_closest_entities(fact, n=10)
                kept_entities = []
                for entity in entities:
                    prompt = entity_selection_prompt.format(fact=fact, context=ctx, entity=entity)
                    ans = llm.invoke(prompt).strip()
                    if ans.lower().startswith('yes'):
                        kept_entities.append(entity)
                prompt = get_new_entities_prompt(fact=fact, context=ctx, entities=entities)
                new_entities = parse_bullet_points(llm.invoke(prompt))
                print(fact)
                print(new_entities)
                
                # Build the new entities and add them to the database
                for entity in new_entities:
                    db.add_entity(name=entity)
                    
                # Now add the fact to the database
                db.add_fact(text=fact, entities=kept_entities+new_entities, source=source)
                
