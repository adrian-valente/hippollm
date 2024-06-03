import argparse
from omegaconf import OmegaConf

from langchain_community.llms import Ollama
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from hippollm.helpers import *
from hippollm.prompts import *
from hippollm import llm_backend, storage
from hippollm.nlp_additional import NLPModels

MODEL = 'mistral'
FACTS_K = 10

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Positional argument
    parser.add_argument(
        'db', 
        type=str,
        help='The database location'
    )
    # Additional arguments
    parser.add_argument(
        '--llm_model',
        type=str,
        help='The LLM model to use',
        default=None
    )
    parser.add_argument(
        '--llm_backend',
        type=str,
        help='The LLM backend to use',
        default=None
    )
    parser.add_argument(
        '--cfg',
        type=str,
        help='The configuration file to use',
        default=None,
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg) if args.cfg is not None else None
    if cfg is None and os.path.exists(os.path.join(args.db, 'parameters.yaml')):
        cfg = OmegaConf.load(os.path.join(args.db, 'parameters.yaml'))
        
    _llm_backend = args.llm_backend if args.llm_backend is not None \
                   else cfg.annotator.llm_backend if cfg is not None \
                   else None
    if _llm_backend is None:
        raise ValueError("No LLM backend specified, and no config file found.")
    llm_model = args.llm_model if args.llm_model is not None \
                else cfg.annotator.llm_model if cfg is not None \
                else None
    if llm_model is None:
        raise ValueError("No LLM model specified, and no config file found.")
    
    
    print('Loading database', args.db)
    db = storage.EntityStore(persist_dir=args.db, cfg=cfg)
    print(f'Loaded database with {len(db.entities)} entities and {len(db.facts)} facts.')
    
    llm = llm_backend.load_llm(model=llm_model, backend=_llm_backend)
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    nlp_models = NLPModels()
    
    
    while True:
        query = input('Enter your query: ')
        relevant_facts = db.get_closest_facts(query, k=FACTS_K)
        print('Most relevant facts:')
        print(itemize_list(relevant_facts))
        
        do_rag = input('Do you want to generate a response? (y/n): ').strip().lower().startswith('y')
        if do_rag:
            prompt = retrieval_prompt.format(query=query, facts=itemize_list(relevant_facts))
            print(llm.invoke(prompt))
            