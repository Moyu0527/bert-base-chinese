import abc
import json
import os
from typing import List, Dict, Optional
from datetime import datetime


class BaseCrawler(abc.ABC):
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    @abc.abstractmethod
    def crawl(self, keywords: List[str], max_count: int = 1000) -> List[Dict]:
        pass

    def save_corpus(self, corpus: List[Dict], filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"corpus_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(corpus, f, ensure_ascii=False, indent=2)
        return filepath

    def load_corpus(self, filepath: str) -> List[Dict]:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
