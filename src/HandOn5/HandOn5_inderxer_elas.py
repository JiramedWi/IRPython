import json
import os
import pickle
from pathlib import Path

from elasticsearch import Elasticsearch


class Indexer:
    def __init__(self):
        self.crawled_folder = Path(__file__).parent / '../crawled/'
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)
        self.es_client = Elasticsearch("localhost:9200", http_auth=[
                                       "elastic", "changeme"],)

    def run_indexer(self):
        self.es_client.indices.create(index='simple', ignore=400)
        self.es_client.indices.delete(index='simple', ignore=[400, 404])

        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(os.path.join(self.crawled_folder, file)))
                j['id'] = j['url']
                print(j)
                self.es_client.index(index='simple', body=j)


if __name__ == '__main__':
    s = Indexer()
    s.run_indexer()
    query = {'query': {'bool': {'must': [{'match': {'text': 'camt'}}]}}}
    results = s.es_client.search(index='simple', body=query)
    print("Got %d Hits:" % results['hits']['total']['value'])
    for hit in results['hits']['hits']:
        print("The title is '{0} ({1})'.".format(
            hit["_source"]['title'], hit["_source"]['url']))
