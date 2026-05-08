import os
import json
import config
from crawler.weibo_crawler import WeiboCrawler
from lda.lda_model import LDABiasGraphBuilder


def main():
    crawled_dir = config.ensure_dir(config.DATA_CONFIG["crawled_corpus_dir"])
    processed_dir = config.ensure_dir(config.DATA_CONFIG["processed_corpus_dir"])
    bias_graph_dir = config.ensure_dir(config.DATA_CONFIG["chinese_bias_graph_dir"])

    print("=" * 60)
    print("Phase 1-2: Social Media Crawling and LDA Bias Graph Construction")
    print("=" * 60)

    keywords = [
        "职场性别歧视",
        "地域歧视",
        "职业偏见",
        "性别刻板印象",
        "贫富差距",
        "学历歧视",
        "年龄歧视",
        "农村城市",
    ]

    topic_ids = [
        "1022:职场女性",
        "1022:性别平等",
        "1022:就业歧视",
    ]

    crawler = WeiboCrawler(output_dir=crawled_dir)

    print("\n[Step 1] Crawling social media corpus...")
    try:
        keyword_corpus = crawler.crawl(keywords=keywords, max_count=500)
        print(f"Crawled {len(keyword_corpus)} posts from keyword search")
    except Exception as e:
        print(f"Keyword search failed: {e}")
        keyword_corpus = []

    try:
        topic_corpus = crawler.crawl_by_topic(topic_ids=topic_ids, max_count=300)
        print(f"Crawled {len(topic_corpus)} posts from topics")
    except Exception as e:
        print(f"Topic crawling failed: {e}")
        topic_corpus = []

    all_corpus = keyword_corpus + topic_corpus
    if all_corpus:
        corpus_file = crawler.save_corpus(all_corpus, "weibo_bias_corpus.json")
        print(f"Saved corpus to {corpus_file}")
    else:
        print("No corpus collected, using simulated data for demonstration")
        sample_corpus = [
            {"id": "1", "text": "男人应该赚钱养家，女人应该相夫教子", "created_at": "2024-01-01"},
            {"id": "2", "text": "女性在职场中遭遇玻璃天花板", "created_at": "2024-01-02"},
            {"id": "3", "text": "南方人精明能干，北方人豪爽大方", "created_at": "2024-01-03"},
            {"id": "4", "text": "程序员都是宅男，不善交际", "created_at": "2024-01-04"},
            {"id": "5", "text": "护士都是女性，医生都是男性", "created_at": "2024-01-05"},
            {"id": "6", "text": "农村出来的人勤奋朴实", "created_at": "2024-01-06"},
            {"id": "7", "text": "985毕业生能力更强", "created_at": "2024-01-07"},
            {"id": "8", "text": "年轻人不靠谱，老年人保守", "created_at": "2024-01-08"},
            {"id": "9", "text": "北方人喜欢吃面食，南方人喜欢吃米饭", "created_at": "2024-01-09"},
            {"id": "10", "text": "女性不适合学理工科", "created_at": "2024-01-10"},
        ]
        corpus_file = crawler.save_corpus(sample_corpus, "weibo_bias_corpus.json")
        print(f"Saved sample corpus to {corpus_file}")

    print("\n[Step 2] Building LDA Bias Graph...")
    corpus = crawler.load_corpus(corpus_file)

    bias_seed_words = {
        dimension: pairs for dimension, pairs in config.ATTRIBUTE_PAIRS.items()
    }

    lda_builder = LDABiasGraphBuilder(
        num_topics=config.HYPERPARAM_CONFIG["lda_num_topics"],
        passes=config.HYPERPARAM_CONFIG["lda_passes"]
    )

    try:
        bias_graph = lda_builder.build_from_corpus(corpus, bias_seed_words)
        bias_graph_file = os.path.join(bias_graph_dir, "chinese_bias_graph.json")
        lda_builder.save_bias_graph(bias_graph_file)
        print(f"Bias graph saved to {bias_graph_file}")

        print("\n[Summary] Bias Graph Structure:")
        for dimension, data in bias_graph["attribute_pairs"].items():
            print(f"  {dimension}: {len(data.get('positive', []))} positive, {len(data.get('negative', []))} negative attributes")

    except Exception as e:
        print(f"LDA model building failed: {e}")
        print("Using config-based bias graph as fallback")
        fallback_graph = {
            "dimension": {k: v for k, v in config.ATTRIBUTE_PAIRS.items()},
            "attribute_pairs": config.ATTRIBUTE_PAIRS,
            "cooccurrence_matrix": {}
        }
        bias_graph_file = os.path.join(bias_graph_dir, "chinese_bias_graph.json")
        with open(bias_graph_file, "w", encoding="utf-8") as f:
            json.dump(fallback_graph, f, ensure_ascii=False, indent=2)
        print(f"Fallback bias graph saved to {bias_graph_file}")

    print("\n" + "=" * 60)
    print("Phase 1-2 completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
