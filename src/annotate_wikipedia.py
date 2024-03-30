import argparse

from annotator import Annotator
from loaders import load_wikipedia

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    # Positional arguments
    argparser.add_argument(
        'query', 
        type=str, 
        help='The query to search for in Wikipedia', 
        required=True
    )
    argparse.add_argument(
        'db',
        type=str,
        help='The database location',
        required=True
    )
    argparse.add_argument('-v', 'verbose', action='store_true', help='Verbose mode')
    args = argparser.parse_args()
    
    doc = load_wikipedia(args.query)
    annotator = Annotator(db_location=args.db, verbose=int(args.verbose))
    annotator.annotate_document(doc)