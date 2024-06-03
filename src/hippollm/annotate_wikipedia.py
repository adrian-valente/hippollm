import argparse
from omegaconf import OmegaConf

from hippollm.annotator import Annotator
from hippollm.loaders import load_wikipedia

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Positional arguments
    parser.add_argument(
        'query', 
        type=str, 
        help='The query to search for in Wikipedia'
    )
    parser.add_argument(
        'db',
        type=str,
        help='The database location'
    )
    parser.add_argument(
        '--llm_model',
        type=str,
        help='The LLM model to use',
        default=None,
    )
    parser.add_argument(
        '--llm_backend',
        type=str,
        help='The LLM backend to use',
        default=None,
    )
    parser.add_argument(
        '--cfg',
        type=str,
        help='The configuration file to use',
        default=None,
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg) if args.cfg is not None else None
    
    doc = load_wikipedia(args.query)
    annotator = Annotator(
        db_location=args.db, 
        llm_model=args.llm_model, 
        llm_backend=args.llm_backend, 
        cfg=args.cfg
    )
    annotator.annotate(doc)