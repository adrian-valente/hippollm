"""
This scripts populates a database with entities corresponding to Wikipedia's articles in english 
(titles + short descriptions), keeping only articles that have been seen at least min_views times 
in the past month (from a set of pageviews dump file), which decreases it to about 568k entities 
for min_views=100.


Usage:
python bootstrap_wikipedia.py --db_loc <db_location> \
    [--dump_loc <dump_location> --min_views <min_views>]
"""

import aiohttp
import argparse
import asyncio
from calendar import monthrange
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import itertools
import os
import orjson
import requests
import subprocess
from time import time

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

from hippollm import storage


def parse_pageviews(file):
    counts = dict()
    with open(file, 'r') as f:
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
    return counts


async def get_descriptions_async(titles, batch_size=50):
    api_url = 'https://en.wikipedia.org/w/api.php'
    
    async def get_description_batch(session, titles, semaphore):
        async with semaphore:
            await asyncio.sleep(5)  # Avoid rate-limiting 429s
            titles_fmt = '|'.join(titles)
            params = {
                'action': 'query',
                'format': 'json',
                'titles': titles_fmt,
                'prop': 'description',
            }
            async with session.get(api_url, params=params) as response:
                try:
                    if response.status != 200:
                        print(f'Error with status code {response.status}')
                        return {}
                    response = await response.json()
                except Exception as e:
                    print(e)
                    return {}
                elts = response['query']['pages']
                titles_descs = {
                    v['title']: v['description']
                    for k, v in elts.items() 
                    if 'missing' not in v and 'description' in v
                }
                print('Retrieved', len(titles_descs), 'descriptions.')
                return titles_descs
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(100)  # Avoid rate-limiting 429s
        batches = [titles[i:i+batch_size] for i in range(0, len(titles), batch_size)]
        tasks = [get_description_batch(session, batch, semaphore) for batch in batches]
        results = await asyncio.gather(*tasks)
        return {k: v for d in results for k, v in d.items()}


def load_and_parse_pageviews(datetime):
    date, time = datetime
    file_id = f'pageviews-{date.strftime("%Y%m%d")}-{time:02d}0000'
    print(f'Downloading pageviews dump file {file_id}...')
    try:
        subprocess.run(
            [
                'wget', '-t', '3', '-O',
                os.path.join(loc_pageviews, file_id+'.gz'),
                f'https://dumps.wikimedia.org/other/pageviews/'
                    f'{date.strftime("%Y")}/{date.strftime("%Y-%m")}/{file_id}.gz'
            ],
            check=True
        )
        subprocess.run(['gzip', '-f', '-d', os.path.join(loc_pageviews, file_id+'.gz')], check=True)
    except:
        print(f'Failed for {file_id}')
        return dict()
    
    counts = parse_pageviews(os.path.join(loc_pageviews, file_id))
    try:
        subprocess.run(['rm', os.path.join(loc_pageviews, file_id)], check=True)
        subprocess.run(['rm', os.path.join(loc_pageviews, file_id+'.gz')], check=True)
        print(f'Finished with file {file_id}')
    except:
        print(f'Error when cleaning up {file_id}')
    finally:
        return counts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--db_loc', type=str, help='The database location', required=True)
    parser.add_argument('--dump_loc', type=str, help='Location for dumps', default='.')
    parser.add_argument('--min_views', type=int, help='Minimum number of views to keep an article',
                        default=5)
    args = parser.parse_args()
    
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir=args.db_loc)
    
    # viewed_titles = defaultdict(int)
    if not os.path.exists(os.path.join(args.dump_loc, 'aggregated_views.json')):
        t0 = time()
        loc_pageviews = os.path.join(args.dump_loc, 'pageviews')
        if not os.path.exists(loc_pageviews):
            os.makedirs(loc_pageviews)
        # Get past month
        past_month = (dt.datetime.now().date().replace(day=1) - dt.timedelta(days=1)).\
            replace(day=1)
        dates = []
        for day in range(1, monthrange(past_month.year, past_month.month)[1]+1):
            dates.append(past_month.replace(day=day))
        times = list(range(24))
        datesxtimes = itertools.product(dates, times)
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = executor.map(load_and_parse_pageviews, datesxtimes)
        viewed_titles = defaultdict(int)
        for d in results:
            for k in d:
                viewed_titles[k] += d[k]
        print('Total time to load and parse pageviews:', time() - t0, 'seconds')           
        # Save dictionary
        with open(os.path.join(args.dump_loc, 'aggregated_views.json'), 'wb') as f:
            b = orjson.dumps(viewed_titles)
            f.write(b)
         
    else:
        # Load dict
        with open(os.path.join(args.dump_loc, 'aggregated_views.json'), 'rb') as f:
            b = f.read()
            viewed_titles = orjson.loads(b)
    print(f'Retrieved views information about {len(viewed_titles)} titles.')
    
    # Retrieve all descriptions
    if not os.path.exists(os.path.join(args.dump_loc, 'titles_abstracts.json')):
        t0 = time()
        titles_abstracts = asyncio.run(get_descriptions_async(list(viewed_titles.keys())))
        print('Retrieved', len(titles_abstracts), 'descriptions in', time() - t0, 'seconds.')
        # Save them
        with open(os.path.join(args.dump_loc, 'titles_abstracts.json'), 'wb') as f:
            b = orjson.dumps(titles_abstracts)
            f.write(b)
    else:
        with open(os.path.join(args.dump_loc, 'titles_abstracts.json'), 'rb') as f:
            b = f.read()
            titles_abstracts = orjson.loads(b)
        print(f'Retrieved {len(titles_abstracts)} descriptions.')
    
    # Keep most seen titles in the past month
    viewed_titles = {k for k, v in viewed_titles.items() if v >= args.min_views}
    print(f'Titles seen at least {args.min_views} times in the past month:', len(viewed_titles))
    titles_abstracts = {k: v for k, v in titles_abstracts.items() if k in viewed_titles}
    
    # Adding to the database
    print('Adding to the database...')
    print('Est. time to add:', len(titles_abstracts) * 160 / (10000 * 60), 'minutes.')
    t0 = time()
    for i, kv in enumerate(titles_abstracts.items()):
        name = kv[0]
        desc = kv[1] if kv[1] is not None else ''
        db.add_entity(name=name, description=desc)
    print(f'Added {i} entities in {time() - t0:.2f} seconds.')
    db.save()
    
