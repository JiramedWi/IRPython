import json
import multiprocessing
import os.path
import pickle
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Queue
from pathlib import Path
from queue import Empty
from urllib.parse import urlparse, urljoin

import joblib
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


class MultiThreadCrawler:
    def __init__(self, base_url, depth):
        self.base_url = base_url
        extracted_url = urlparse(base_url)
        parent = extracted_url.path[:extracted_url.path.rfind('/') + 1]
        self.root = '{}://{}{}'.format(extracted_url.scheme, extracted_url.netloc, parent)
        self.pool = ThreadPoolExecutor(max_workers=multiprocessing.cpu_count() - 1)
        self.to_crawl = Queue()
        self.to_crawl.put((self.base_url, depth))
        self.stored_folder = Path(os.path.abspath('')) / 'crawled'

        if not Path(self.stored_folder).exists():
            os.makedirs(self.stored_folder)

        if Path(self.stored_folder / 'url_list.pickle').exists():
            with open(self.stored_folder / 'url_list.pickle', 'rb') as f:
                self.crawled_pages = pickle.load(f)
            print(self.crawled_pages)
        else:
            self.crawled_pages = set([])

    def parse_links(self, html, depth):
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        url_list = []
        for link in links:
            url = link['href']
            url = urljoin(self.root, url)
            if depth >= 0 and '..' not in url and url not in self.crawled_pages:
                print("Adding {}".format(url))
                self.to_crawl.put({url, depth})
            url_list.append(url)
        return url_list

    def parse_contents(self, url, html, url_list):
        def tag_visible(element):
            if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
                return False
            if isinstance(element, Comment):
                return False
            return True

        try:
            soup = BeautifulSoup(html, 'html.parser')
            texts = soup.findAll(string=True)
            visible_texts = filter(tag_visible, texts)
            title = soup.find('title').string.strip()
            text = u" ".join(t.strip() for t in visible_texts).strip()
            data = pd.DataFrame({'url': [url], 'title': [title], 'text': [text], 'url_lists': [url_list]})
            data.to_csv(self.stored_folder / (str(hash(url)) + '.csv'), index=False, encoding='utf-8')
            # with open(self.stored_folder / (str(hash(url)) + '.txt'), 'w', encoding='utf-8') as f:
            #     json.dump({'url': url, 'title': title, 'text': text, 'url_lists': url_list}, f, ensure_ascii=False)
        except:
            pass

    def extract_page(self, obj):
        if obj.result():
            result, url, depth = obj.result()
            if result and result.status_code == 200:
                url_lists = self.parse_links(result.text, depth)
                self.parse_contents(url, result.text, url_lists)

    def get_page(self, url, depth):
        try:
            res = requests.get(url, timeout=(3, 30))
            return res, url, depth
        except requests.RequestException:
            return None, url, depth

    def run_scraper(self):
        while True:
            try:
                target = self.to_crawl.get(timeout=10)
                url, depth = target
                if url not in self.crawled_pages:
                    self.crawled_pages.add(url)
                    job = self.pool.submit(self.get_page, url, depth - 1)
                    job.add_done_callback(self.extract_page)
            except Empty:
                with open(self.stored_folder / 'url_list.pickle', 'wb') as f:
                    pickle.dump(self.crawled_pages, f, pickle.HIGHEST_PROTOCOL)
                with open(self.stored_folder / 'url_list.pickle', 'rb') as f:
                    print(pickle.load(f))
                break
            except Exception as e:
                print(e)
                continue


if __name__ == '__main__':
    crawler = MultiThreadCrawler("https://camt.cmu.ac.th/index.php/en/", 5)
    crawler.run_scraper()