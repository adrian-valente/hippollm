import requests
import xml.etree.ElementTree as ElementTree
from time import time

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings

import storage

# def get_wikipedia_titles(continuation=None):
#     url = 'https://en.wikipedia.org/w/api.php'
#     params = {
#         'action': 'query',
#         'format': 'json',
#         'list': 'allpages',
#         'aplimit': 500,  # Maximum number of titles to retrieve per request
#         'apfrom': continuation,
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#     titles = [page['title'] for page in data['query']['allpages']]
#     if 'continue' in data:
#         continuation = data['continue']['apcontinue']
#         titles.extend(get_wikipedia_titles(continuation))
#     return titles


# def get_top_pages(limit=10000):
#     url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/user'
#     params = {
#         'project': 'en.wikipedia',
#         'platform': 'all-access',
#         'agent': 'user',
#         'granularity': 'monthly',
#         'start': '2024-01-01',
#         'end': '2024-02-01',
#         'limit': limit,
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#     top_pages = [item['article'] for item in data['items']]
#     return top_pages


# def get_page_info(page_titles):
#     url = 'https://en.wikipedia.org/w/api.php'
#     params = {
#         'action': 'query',
#         'format': 'json',
#         'prop': 'extracts',
#         'titles': '|'.join(page_titles),
#         'exintro': 1,
#         'explaintext': 1,
#         'redirects': 1,
#     }
#     response = requests.get(url, params=params)
#     data = response.json()
#     page_info = []
#     for page_id, page_data in data['query']['pages'].items():
#         title = page_data['title']
#         extract = page_data['extract']
#         first_sentence = extract.split('.')[0] + '.'
#         page_info.append((title, first_sentence))
#     return page_info

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


def parse_pageviews(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return [line.split()[1].replace('_', ' ') for line in lines]


if __name__ == '__main__':
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir='wikidb')
    
    # Retrieve all titles and abstracts (in english)
    t0 = time()
    titles_abstracts = parse_wiki_xml('../../enwiki-20240301-abstract.xml')
    print(f'Parsed {len(titles_abstracts)} titles and abstracts in {time() - t0:.2f} seconds.')
    
    # Retrieve articles visited in the past month (from the pageviews dump)
    viewed_titles = set(parse_pageviews('../../pageviews-20240301-000000'))
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
    