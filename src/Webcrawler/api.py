import time

import pandas as pd
from elasticsearch import Elasticsearch
from flask import Flask, request

app = Flask(__name__)
app.es_client = Elasticsearch("https://localhost:9200")


@app.route('/search', methods=['GET'])
def search():
    start = time.time()
    response_object = {'status': 'success'}
    argList = request.args.to_dict(flat=False)
    query_term = argList['query'][0]
    results = app.es_client.search(index='simple', source_excludes=['url_lists'], size=100,
                                   query={
                                       "script_score":
                                           {"query":
                                                {"match":
                                                     {"text": query_term}},
                                            "script": {"source": "_score * doc['pagerank'].value"}}})
    end = time.time()
    total_hit = results['hits']['total']['value']
    result_df = pd.DataFrame([[hit["_source"]['title'], hit["_source"]['url'], hit["_source"]
                                                                               ['text'][:100], hit["_score"]] for hit in
                              results['hits']['hits']], columns=['title', 'url', 'text',
                                                                 'score'])
    response_object['total_hit'] = total_hit
    response_object['elaspe'] = end - start
    response_object['result'] = result_df.to_dict(orient='records')
    return response_object


if __name__ == '__main__':
    app.run(debug=True, port=5001)
