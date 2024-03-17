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
    result = []
    for event, elem in context:
        if event == "end" and elem.tag == 'doc':
            title = elem.find('title')
            abstract = elem.find('abstract')
            if title is not None:
                if title.text.startswith('Wikipedia: '):
                    title = title.text[11:]
                else:
                    title = title.text
                result.append((title, abstract.text if abstract is not None else ''))
                title = None
                abstract = None
                if limit is not None and len(result) > limit:
                    return result
            elem.clear()
    return result

if __name__ == '__main__':
    embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = storage.EntityStore(embedding_model=embedding_model, persist_dir='wikidb')
    t0 = time()
    titles_abstracts = parse_wiki_xml('../../enwiki-20240301-abstract.xml', limit=10000)
    print(f'Parsed {len(titles_abstracts)} titles and abstracts in {time() - t0:.2f} seconds.')
    t0 = time()
    for i, ta in enumerate(titles_abstracts):
        db.add_entity(name=ta[0], description=ta[1])
        if i > 10000:
            break
    print(f'Added {i} entities in {time() - t0:.2f} seconds.')
    db.save()
    