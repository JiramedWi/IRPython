import json
import os
import pickle
from pathlib import Path

from elasticsearch import Elasticsearch

from src.Webcrawler.pr import Pr


class Indexer:

    def __init__(self):
        self.crawled_folder = Path(os.path.abspath('')) / 'crawled_new'
        with open(self.crawled_folder / 'url_list.pickle', 'rb') as f:
            self.file_mapper = pickle.load(f)
        self.es_client = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "2mUm9sEz9uZL4-S-KCX0"),
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

    def run_indexer_custom(self):
        self.es_client.options(ignore_status=[400, 404]).indices.delete(index='custom')
        index_body = {
            "settings": {
                "similarity": {
                    "custom_similarity": {
                        "type": "scripted",
                        "script": {
                            "source": """
        10 double idf = Math.log((field.docCount + 1.0) / (term.docFreq + 1.0)) + 1.0;
        11 return idf;
        12 """
                        }
                    }
                }
            },

            "mappings": {

                "properties": {

                    "text": {

                        "type": "text",

                        "similarity": "custom_similarity"
                    }
                }
            }
        }
        self.es_client.options(ignore_status=400).indices.create(index='custom', body=index_body)
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

