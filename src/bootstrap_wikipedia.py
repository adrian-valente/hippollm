"""
This scripts populates a database with entities corresponding to Wikipedia's articles in 
english (titles + abstracts), recovered from the enwiki abstract dump file, keeping only
articles that have been seen at least min_views times in one day (from the pageviews dump file),
which decreases it to about 1.3M entities for min_views=5.

Note: the abstracts obtained through the dump file do not seem to be very informative, as
they are often empty or contained random arrangements of words and characters. I am still
looking for a better source of abstracts for higher quality embeddings.
See notably this reference: https://stackoverflow.com/questions/61449459/are-the-abstracts-in-in-enwiki-latest-abstract-xml-gz-corrupted

Usage:
python bootstrap_wikipedia.py --db_loc <db_location> \
    [--dump_loc <dump_location> --min_views <min_views>]
"""

import argparse
import os
import subprocess
from time import time
import xml.etree.ElementTree as ElementTree

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import storage

def parse_wiki_xml(file_name, limit=None):
    context = ElementTree.iterparse(file_name, events=("start", "end"))
    result = {}
    for event, elem in context:
        if event == "end" and elem.tag == 'doc':
            title = elem.find('title')
            abstract = elem.find('abstract')
            if title is not None:
                if title.text.startswith('Wikipedia: '):
                    title = title.text[11:]
                else:
                    title = title.text
                result[title] = abstract.text if abstract is not None else ''
                title = None
                abstract = None
                if limit is not None and len(result) > limit:
                    return result
            elem.clear()
    return result


def parse_pageviews(location, min_views):
    counts = dict()
    for file in os.listdir(location):
        with open(os.path.join(location, file), 'r') as f:
            print('Parsing', file)
            lines = f.readlines()
            for l in lines:
                if l.startswith('en '):
                    parts = l.split(' ')
                    title = parts[1].replace('_', ' ')
                    count = int(parts[2])
                    if title in counts:
                        counts[title] += count
                    else:
                        counts[title] = count
    return {k for k, v in counts.items() if v > min_views}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_loc', type=str, help='The database location', required=True)
    parser.add_argument('--dump_loc', type=str, help='Location for dumps', default='.')
    parser.add_argument('--min_views', type=int, help='Minimum number of views to keep an article',
                        default=5)
    args = parser.parse_args()
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir=args.db_loc)
    
    # Load files if necessary
    loc_titles_dump = os.path.join(args.dump_loc, 'enwiki-latest-abstract.xml')
    if not os.path.exists(loc_titles_dump):
        print('Downloading abstract dump file...')
        pr = subprocess.Popen(['wget', '-O', f'{loc_titles_dump}.gz',
                               'https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-abstract.xml.gz'], stdout=subprocess.PIPE)
        while pr.poll() is None:
            print(pr.stdout.readline().strip())
        print('Uncompressing...')
        subprocess.Popen(['gzip', '-d', f'{loc_titles_dump}.gz'])
        
    loc_pageviews = os.path.join(args.dump_loc, 'pageviews')
    if not os.path.exists(loc_pageviews):
        os.makedirs(loc_pageviews)
    for i in range(23):
        file_id = f'pageviews-20240301-{i:02d}0000'
        if not os.path.exists(os.path.join(loc_pageviews, file_id)):
            print(f'Downloading pageviews dump file {i}...')
            pr = subprocess.Popen([
                'wget', '-O', 
                os.path.join(loc_pageviews, file_id+'.gz'),
                f'https://dumps.wikimedia.org/other/pageviews/2024/2024-03/{file_id}.gz'], 
                stdout=subprocess.PIPE
            )
            while pr.poll() is None:
                print(pr.stdout.readline().strip())
            pr = subprocess.Popen(['gzip', '-d', os.path.join(loc_pageviews, file_id+'.gz')])
    
    # Retrieve all titles and abstracts (in english)
    t0 = time()
    titles_abstracts = parse_wiki_xml(loc_titles_dump)
    print(f'Parsed {len(titles_abstracts)} titles and abstracts in {time() - t0:.2f} seconds.')
    
    # Retrieve articles visited in the past month (from the pageviews dump)
    viewed_titles = parse_pageviews(loc_pageviews, args.min_views)
    titles_abstracts = {k: v for k, v in titles_abstracts.items() if k in viewed_titles}
    print('Remaining titles:', len(titles_abstracts))
    print('Est. time to add:', len(titles_abstracts) * 160 / (10000 * 60), 'minutes.')
    
    t0 = time()
    for i, kv in enumerate(titles_abstracts.items()):
        name = kv[0]
        desc = kv[1] if kv[1] is not None else ''
        db.add_entity(name=name, description=desc)
    print(f'Added {i} entities in {time() - t0:.2f} seconds.')
    db.save()
    