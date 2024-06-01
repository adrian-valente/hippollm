import sys

from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from hippollm.helpers import *
from hippollm.prompts import *
from hippollm import storage
from hippollm.nlp_additional import NLPModels

MODEL = 'mistral'
FACTS_K = 10

if __name__ == '__main__':
    db_location = sys.argv[1]
    llm = Ollama(model=MODEL)
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    nlp_models = NLPModels()
    
    print('Loading database', db_location)
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir=db_location)
    print(f'Loaded database with {len(db.entities)} entities and {len(db.facts)} facts.')
    
    while True:
        query = input('Enter your query: ')
        relevant_facts = db.get_closest_facts(query, k=FACTS_K)
        print('Most relevant facts:')
        print(itemize_list(relevant_facts))
        
        do_rag = input('Do you want to generate a response? (y/n): ').strip().lower().startswith('y')
        if do_rag:
            prompt = retrieval_prompt.format(query=query, facts=itemize_list(relevant_facts))
            print(llm.invoke(prompt))
            