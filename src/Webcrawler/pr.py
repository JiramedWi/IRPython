import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch


class Pr:
    def __init__(self, alpha):
        self.pr_result = None
        self.crawled_folder = Path(os.path.abspath('')) / 'crawled_new'
        self.alpha = alpha

    def url_extractor(self):
        url_maps = {}
        all_urls = set([])
        for file in os.listdir(self.crawled_folder):
            if file.endswith(".txt"):
                j = json.load(open(self.crawled_folder / file))
                all_urls.add(j['url'])
                for s in j['url_lists']:
                    all_urls.add(s)
                url_maps[j['url']] = list(set(j['url_lists']))
        all_urls = list(all_urls)
        return url_maps, all_urls

    def pr_calc(self):
        url_maps, all_urls = self.url_extractor()
        url_matrix = pd.DataFrame(0, columns=all_urls, index=all_urls, dtype=float)

        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                url_matrix.loc[url, url_maps[url]] += (self.alpha * (1 / len(url_maps[url])))

        # Handle dangling nodes (nodes with no outgoing links)
        url_matrix.loc[url_matrix.sum(axis=1) == 0, :] = 1 / len(all_urls)

        # Convert to numpy matrix
        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)

        # Power iteration
        prev_Px = x0
        Px = x0 @ P  # Ensure correct order
        i = 0

        while (any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8)):
            i += 1
            prev_Px = Px
            Px = Px @ P  # Correct order

        # Normalize PageRank scores to sum to 1
        Px = Px / Px.sum()
        # Debugging print statement
        print("Final PageRank sum:", Px.sum())

        print('Converged in {0} iterations: {1}'.format(i, np.around(np.asarray(Px).flatten().astype(float), 5)))

        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T

    import numpy as np
    import pandas as pd

    def pr_calc_old(self):
        url_maps, all_urls = self.url_extractor()
        url_matrix = pd.DataFrame(columns=all_urls, index=all_urls)

        for url in url_maps:
            if len(url_maps[url]) > 0 and len(all_urls) > 0:
                url_matrix.loc[url] = (1 - self.alpha) * (1 / len(all_urls))
                url_matrix.loc[url, url_maps[url]] = url_matrix.loc[url, url_maps[url]] + (
                            self.alpha * (1 / len(url_maps[url])))

        url_matrix.loc[url_matrix.isnull().all(axis=1), :] = (1 / len(all_urls))

        x0 = np.matrix([1 / len(all_urls)] * len(all_urls))
        P = np.asmatrix(url_matrix.values)
        prev_Px = x0
        Px = x0 * P  # Matrix multiplication (potential issue)

        i = 0
        while (any(abs(np.asarray(prev_Px).flatten() - np.asarray(Px).flatten()) > 1e-8)):
            i += 1
            prev_Px = Px
            Px = Px * P  # Potential issue with matrix multiplication

        print('Converged in {0} iterations: {1}'.format(i, np.around(np.asarray(Px).flatten().astype(float), 5)))

        self.pr_result = pd.DataFrame(Px, columns=url_matrix.index, index=['score']).T

        # Debugging: Print sum of PageRank scores
        print("Final PageRank sum:", Px.sum())


if __name__ == '__main__':
    pr = Pr(alpha=0.85)
    pr_2 = Pr(alpha=0.85)
    pr.pr_calc()
    pr_2.pr_calc_old()
