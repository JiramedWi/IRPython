from elasticsearch import Elasticsearch
import json

es = Elasticsearch("https://localhost:9200", basic_auth=("elastic", "2mUm9sEz9uZL4-S-KCX0"), ca_certs="/Users/Jumma/http_ca.crt")
a = json.dumps(es.info().body)
print(a)