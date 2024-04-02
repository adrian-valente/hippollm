import argparse

from annotator import Annotator
from loaders import load_wikipedia

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
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose mode')
    args = parser.parse_args()
    
    doc = load_wikipedia(args.query)
    annotator = Annotator(db_location=args.db, verbosity=int(args.verbose))
    annotator.annotate(doc)