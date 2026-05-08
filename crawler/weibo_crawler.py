import os
import re
import time
import json
import requests
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from .base_crawler import BaseCrawler


class WeiboCrawler(BaseCrawler):
    def __init__(
        self,
        output_dir: str,
        cookie: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        super().__init__(output_dir)
        self.cookie = cookie or os.getenv("WEIBO_COOKIE", "")
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
        self.headers = {
            "User-Agent": self.user_agent,
            "Cookie": self.cookie,
            "Referer": "https://weibo.com",
        }
        self.session = requests.Session()

    def crawl(self, keywords: List[str], max_count: int = 1000) -> List[Dict]:
        corpus = []
        for keyword in keywords:
            print(f"Crawling keyword: {keyword}")
            keyword_corpus = self._search_by_keyword(keyword, max_count // len(keywords))
            corpus.extend(keyword_corpus)
            time.sleep(2)
        return corpus

    def _search_by_keyword(self, keyword: str, max_count: int) -> List[Dict]:
        url = "https://weibo.com/ajax/statuses/search"
        params = {
            "q": keyword,
            "count": 20,
            "page": 1,
        }
        corpus = []
        page = 1

        while len(corpus) < max_count:
            params["page"] = page
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=self.headers,
                    timeout=10
                )
                if response.status_code != 200:
                    break
                data = response.json()
                posts = data.get("data", {}).get("list", [])
                if not posts:
                    break
                for post in posts:
                    item = self._parse_post(post)
                    if item:
                        corpus.append(item)
                if len(corpus) >= max_count:
                    break
                page += 1
                time.sleep(1)
            except Exception as e:
                print(f"Error crawling page {page}: {e}")
                break

        return corpus[:max_count]

    def _parse_post(self, post: Dict) -> Optional[Dict]:
        try:
            text = post.get("text_raw", post.get("text", ""))
            text = self._clean_html(text)
            if not text or len(text) < 10:
                return None
            return {
                "id": str(post.get("id", "")),
                "text": text,
                "user_id": str(post.get("user_id", "")),
                "created_at": post.get("created_at", ""),
                "keyword": post.get("keyword", ""),
                "attitudes_count": post.get("attitudes_count", 0),
                "comments_count": post.get("comments_count", 0),
                "reposts_count": post.get("reposts_count", 0),
            }
        except Exception as e:
            print(f"Error parsing post: {e}")
            return None

    def _clean_html(self, text: str) -> str:
        soup = BeautifulSoup(text, "html.parser")
        clean_text = soup.get_text()
        clean_text = re.sub(r"#\w+#", "", clean_text)
        clean_text = re.sub(r"@\w+", "", clean_text)
        clean_text = re.sub(r"http\S+", "", clean_text)
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        return clean_text

    def crawl_by_topic(self, topic_ids: List[str], max_count: int = 500) -> List[Dict]:
        url = "https://weibo.com/ajax/statuses/topic"
        corpus = []

        for topic_id in topic_ids:
            params = {"id": topic_id, "count": 20, "page": 1}
            page = 1
            topic_corpus = []

            while len(topic_corpus) < max_count // len(topic_ids):
                params["page"] = page
                try:
                    response = self.session.get(
                        url,
                        params=params,
                        headers=self.headers,
                        timeout=10
                    )
                    if response.status_code != 200:
                        break
                    data = response.json()
                    posts = data.get("data", {}).get("list", [])
                    if not posts:
                        break
                    for post in posts:
                        item = self._parse_post(post)
                        if item:
                            item["topic_id"] = topic_id
                            topic_corpus.append(item)
                    page += 1
                    time.sleep(1)
                except Exception as e:
                    print(f"Error crawling topic {topic_id} page {page}: {e}")
                    break

            corpus.extend(topic_corpus)
            time.sleep(2)

        return corpus
