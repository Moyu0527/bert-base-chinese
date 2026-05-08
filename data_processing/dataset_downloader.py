import os
import json
from typing import Dict, List, Optional, Tuple


DATASET_INFO = {
    "weat": {
        "name": "WEATHub",
        "huggingface_url": "https://huggingface.co/datasets/iamshnoo/WEATHub",
        "paper": "Global Voices, Local Biases: Socio-Cultural Prejudices across Languages (EMNLP 2023)",
        "github_url": None,
        "languages": ["ar", "bn", "ckb", "da", "de", "el", "es", "fa", "fr", "hi", "it", "ja", "ko", "ku", "mr", "pa", "ru", "te", "th", "tl", "tr", "ur", "vi", "zh"],
        "chinese_code": "zh",
        "description": "24语言WEAT词表，包含target words和attribute words用于测量词嵌入偏见关联",
    },
    "cdial_bias": {
        "name": "CDial-Bias",
        "huggingface_url": "https://huggingface.co/datasets/para-zhou/CDial-Bias",
        "paper": "Towards Identifying Social Bias in Dialog Systems: Framework, Dataset, and Benchmark (EMNLP 2022 Findings)",
        "github_url": "https://github.com/para-zhou/CDial-Bias",
        "nlpcc_task_url": "https://constellate.xlore.cn/nlpcc2022/sharedtask7.html",
        "description": "首个中文对话偏见数据集，包含17,371条上下文敏感和独立的对话样本，涵盖Race、Gender、Region、Occupation四个偏见维度",
    }
}


def download_weat_from_huggingface(output_dir: str, language: str = "zh") -> str:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先安装datasets库: pip install datasets")

    print(f"[Download] 从HuggingFace下载WEATHub ({language})...")
    dataset = load_dataset("iamshnoo/WEATHub", language, trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"weat_official_{language}.json")

    all_data = []
    for split_name, split_data in dataset.items():
        for item in split_data:
            item_dict = {
                "language": item.get("language", language),
                "weat": item.get("weat", ""),
                "attr1_category": item.get("attr1", {}).get("category", ""),
                "attr1_examples": item.get("attr1", {}).get("examples", []),
                "attr2_category": item.get("attr2", {}).get("category", ""),
                "attr2_examples": item.get("attr2", {}).get("examples", []),
                "targ1_category": item.get("targ1", {}).get("category", ""),
                "targ1_examples": item.get("targ1", {}).get("examples", []),
                "targ2_category": item.get("targ2", {}).get("category", ""),
                "targ2_examples": item.get("targ2", {}).get("examples", []),
                "split": split_name,
            }
            all_data.append(item_dict)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    print(f"[Download] WEAT官方数据已保存至: {output_path}")
    print(f"[Info] 共 {len(all_data)} 条记录")
    return output_path


def download_cdial_bias_from_huggingface(output_dir: str) -> Tuple[str, str]:
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请先安装datasets库: pip install datasets")

    print("[Download] 从HuggingFace下载CDial-Bias...")
    dataset = load_dataset("para-zhou/CDial-Bias", trust_remote_code=True)

    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "cdial_bias_train.json")
    test_path = os.path.join(output_dir, "cdial_bias_test.json")

    if "train" in dataset:
        train_data = [item for item in dataset["train"]]
        with open(train_path, "w", encoding="utf-8") as f:
            json.dump(train_data, f, ensure_ascii=False, indent=2)
        print(f"[Download] 训练集已保存至: {train_path} ({len(train_data)} 条)")

    if "test" in dataset:
        test_data = [item for item in dataset["test"]]
        with open(test_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f, ensure_ascii=False, indent=2)
        print(f"[Download] 测试集已保存至: {test_path} ({len(test_data)} 条)")

    return train_path, test_path


def download_all_datasets(output_base_dir: str):
    weat_dir = os.path.join(output_base_dir, "weat")
    cdial_dir = os.path.join(output_base_dir, "cdial_bias")

    print("=" * 60)
    print("下载官方数据集")
    print("=" * 60)

    print("\n[1/2] WEAT (WEATHub)")
    print(f"    来源: {DATASET_INFO['weat']['huggingface_url']}")
    print(f"    论文: {DATASET_INFO['weat']['paper']}")
    print(f"    语言: {', '.join(DATASET_INFO['weat']['languages'])}")
    try:
        weat_path = download_weat_from_huggingface(weat_dir, language="zh")
        print(f"    状态: 成功 ✓")
    except Exception as e:
        print(f"    状态: 失败 ✗ - {e}")
        weat_path = None

    print("\n[2/2] CDial-Bias")
    print(f"    来源: {DATASET_INFO['cdial_bias']['huggingface_url']}")
    print(f"    论文: {DATASET_INFO['cdial_bias']['paper']}")
    print(f"    官网: {DATASET_INFO['cdial_bias'].get('nlpcc_task_url', 'N/A')}")
    try:
        train_path, test_path = download_cdial_bias_from_huggingface(cdial_dir)
        print(f"    状态: 成功 ✓")
    except Exception as e:
        print(f"    状态: 失败 ✗ - {e}")
        train_path = test_path = None

    print("\n" + "=" * 60)
    print("下载完成!")
    print("=" * 60)
    return weat_path, train_path, test_path


if __name__ == "__main__":
    print("""
    ============================================================
    官方数据集下载工具
    ============================================================
    
    本工具提供以下官方数据集的下载接口:
    
    1. WEAT (WEATHub)
       - 来源: HuggingFace datasets (iamshnoo/WEATHub)
       - 论文: Global Voices, Local Biases (EMNLP 2023)
       - 语言: 24种语言，包括中文(zh)
       - 内容: 词嵌入偏见关联测试词表
       
    2. CDial-Bias  
       - 来源: HuggingFace datasets (para-zhou/CDial-Bias)
       - 论文: Towards Identifying Social Bias in Dialog Systems (EMNLP 2022)
       - 内容: 中文对话偏见数据集，17,371条样本
       - 维度: Gender, Region, Occupation, Race
    
    ============================================================
    """)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    data_dir = os.path.join(project_root, "data")
    
    download_all_datasets(data_dir)
