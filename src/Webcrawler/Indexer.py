import json
import os
import pickle
from pathlib import Path

from elasticsearch import Elasticsearch

from src.Webcrawler.pr import Pr


class Indexer:

    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')) / 'crawled'
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)
        self.es_client = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "wfElebF0u9Z*h+xdgv6+"),
                   ca_certs="/Users/Jumma/http_ca.crt")

    def run_indexer(self):
        self.pr = Pr(alpha=0.85)
        self.pr.pr_calc()
        self.es_client.indices.create(index='simple', ignore=400)
        self.es_client.indices.delete(index='simple', ignore=[400, 404])

        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(self.crawled_folder / file))
                j['id'] = j['url']
                j['pagerank'] = self.pr.pr_result.loc[j['id']].score
                print(j)
                self.es_client.index(index='simple', body=j)


if __name__ == '__main__':
    indexer = Indexer()
    indexer.run_indexer()