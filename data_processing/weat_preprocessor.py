import os
import json
import torch
from typing import Dict, List, Tuple, Optional, Set
from transformers import BertTokenizer, BertModel
import config


class WEATPreprocessor:
    WEAT_EN_TERMS = {
        "career_family": {
            "target_A_male": ["John", "James", "Robert", "Michael", "William", "David", "Richard", "Joseph", "Thomas", "Charles"],
            "target_B_female": ["Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica", "Sarah", "Karen"],
            "attribute_X_career": ["executive", "manager", "doctor", "lawyer", " CEO ", "firm", "business", "corporation", "company", "professional"],
            "attribute_Y_family": ["home", "house", "children", "family", "marriage", "wedding", "parents", "kids", "home", "domestic"],
        },
        "science_arts": {
            "target_A_science": ["math", "physics", "chemistry", "biology", "science", "NASA", "mathematics", "engineer", "technology", "computer"],
            "target_B_arts": ["poetry", "art", "dance", "music", "Shakespeare", "literature", "novel", "symphony", "museum", "painting"],
            "attribute_X_science_pos": ["genius", "brilliant", "smart", "clever", "intelligent", " cerebral ", "logically", "analytic"],
            "attribute_Y_arts_pos": [" beautiful ", "artistic", "creative", "sensitive", "mysterious", "imaginative", "deeply", "emotional"],
        },
    }

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.MODEL_CONFIG["model_name"]
        self.tokenizer = None
        self.model = None
        self.vocab_set = None
        self.weat_chinese = {}
        self.official_data = {}

    def load_bert_vocab(self):
        print(f"Loading BERT vocabulary from {self.model_name}...")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.vocab_set = set(self.tokenizer.vocab.keys())
        print(f"BERT vocab size: {len(self.vocab_set)}")

    def translate_to_chinese(self, en_terms: Dict, translation_dict: Dict) -> Dict:
        chinese_terms = {}
        for category, term_lists in en_terms.items():
            chinese_terms[category] = {}
            for subcategory, terms in term_lists.items():
                if subcategory in translation_dict:
                    translated = translation_dict[subcategory]
                    if isinstance(translated, list):
                        chinese_terms[category][subcategory] = translated
                    else:
                        chinese_terms[category][subcategory] = [translated]
                else:
                    chinese_terms[category][subcategory] = []
        return chinese_terms

    def validate_chinese_terms(self, terms: List[str]) -> Tuple[List[str], List[str]]:
        if self.vocab_set is None:
            self.load_bert_vocab()

        valid_terms = []
        invalid_terms = []

        for term in terms:
            term_clean = term.strip()
            if term_clean in self.vocab_set:
                valid_terms.append(term_clean)
            else:
                for variant in [term_clean, term_clean.replace(" ", ""), term_clean.lower()]:
                    if variant in self.vocab_set:
                        valid_terms.append(variant)
                        break
                else:
                    invalid_terms.append(term_clean)

        return valid_terms, invalid_terms

    def load_official_weat(self, input_path: Optional[str] = None) -> Dict:
        if input_path is None:
            input_path = os.path.join(config.DATA_CONFIG["weat_data_dir"], "weat_official_zh.json")
        
        if not os.path.exists(input_path):
            print(f"[Warning] Official WEAT file not found at {input_path}")
            return {}
        
        with open(input_path, "r", encoding="utf-8") as f:
            self.official_data = json.load(f)
        
        print(f"[WEAT] Loaded official WEATHub data from {input_path}")
        print(f"[WEAT] Number of WEAT tests: {len(self.official_data)}")
        
        return self.official_data

    def get_weat_by_dimension(self, weat_data: Dict, dimension: str) -> List[Dict]:
        """按维度获取WEAT数据"""
        dimension_map = {
            "gender": ["WEAT6", "WEAT7", "WEAT8"],
            "occupation": ["WEAT6", "WEAT7", "WEAT8"],
            "science_arts": ["WEAT7", "WEAT8"],
            "emotion": ["WEAT9"],
            "default": ["WEAT1", "WEAT2"],
        }
        
        target_weats = dimension_map.get(dimension, dimension_map["default"])
        
        results = []
        for item in weat_data:
            if item.get("weat") in target_weats:
                results.append(item)
        
        return results

    def build_chinese_weat(self) -> Dict:
        career_family_translation = {
            "target_A_male": ["他", "男人", "男孩", "丈夫", "父亲", "男士"],
            "target_B_female": ["她", "女人", "女孩", "妻子", "母亲", "女士"],
            "attribute_X_career": ["成功", "事业", "领导", "管理", "专业", "工作"],
            "attribute_Y_family": ["家庭", "孩子", "婚姻", "家务", "照顾", "生活"],
        }

        science_arts_translation = {
            "target_A_science": ["数学", "物理", "化学", "生物", "科学", "工程", "技术", "电脑"],
            "target_B_arts": ["诗歌", "艺术", "舞蹈", "音乐", "文学", "小说", "绘画"],
            "attribute_X_science_pos": ["聪明", "智慧", "理性", "逻辑", "分析"],
            "attribute_Y_arts_pos": ["美丽", "创意", "敏感", "想象", "情感", "艺术"],
        }

        gender_translation = {
            "target_A_male": ["男性", "男孩", "丈夫", "父亲", "男士", "他"],
            "target_B_female": ["女性", "女孩", "妻子", "母亲", "女士", "她"],
            "attribute_X_career": ["强壮", "勇敢", "独立", "果断", "竞争", "成就"],
            "attribute_Y_family": ["温柔", "体贴", "善良", "关怀", "细腻", "情感"],
        }

        region_translation = {
            "target_A_north": ["北方人", "北方", "北方的", "东北人", "北方人"],
            "target_B_south": ["南方人", "南方", "南方的", "广东人", "江浙人"],
            "attribute_X_positive": ["勤劳", "豪爽", "直率", "大方", "朴实"],
            "attribute_Y_negative": ["精明", "小气", "计较", "柔弱", "娇气"],
        }

        occupation_translation = {
            "target_A_stem": ["工程师", "医生", "科学家", "程序员", "数学家", "物理学家"],
            "target_B_humanities": ["教师", "护士", "社工", "图书管理员", "艺术家"],
            "attribute_X_positive": ["收入高", "体面", "专业", "精英", "杰出", "领导"],
            "attribute_Y_negative": ["收入低", "卑微", "普通", "平凡", "一般"],
        }

        self.weat_chinese = {
            "career_family": self.translate_to_chinese(self.WEAT_EN_TERMS["career_family"], career_family_translation),
            "science_arts": self.translate_to_chinese(self.WEAT_EN_TERMS["science_arts"], science_arts_translation),
            "gender": gender_translation,
            "region": region_translation,
            "occupation": occupation_translation,
        }

        self._validate_all_terms()

        return self.weat_chinese

    def _validate_all_terms(self):
        print("\n[Validation] Checking Chinese terms in BERT vocab...")
        for category, term_lists in self.weat_chinese.items():
            print(f"\n  Category: {category}")
            for subcategory, terms in term_lists.items():
                valid, invalid = self.validate_chinese_terms(terms)
                if invalid:
                    print(f"    {subcategory}: {len(valid)} valid, {len(invalid)} invalid (removed)")
                else:
                    print(f"    {subcategory}: {len(valid)} valid")

    def get_validated_weat(self) -> Dict:
        if not self.weat_chinese:
            self.build_chinese_weat()

        validated = {}
        for category, term_lists in self.weat_chinese.items():
            validated[category] = {}
            for subcategory, terms in term_lists.items():
                valid, _ = self.validate_chinese_terms(terms)
                validated[category][subcategory] = valid

        return validated

    def save_weat_data(self, output_dir: str) -> str:
        validated = self.get_validated_weat()
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "weat_chinese.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(validated, f, ensure_ascii=False, indent=2)
        print(f"WEAT Chinese data saved to {output_path}")
        return output_path

    def load_weat_data(self, input_path: str) -> Dict:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def get_weat_pairs(self, category: str) -> List[Dict]:
        validated = self.get_validated_weat()
        if category not in validated:
            raise ValueError(f"Category {category} not found. Available: {list(validated.keys())}")

        data = validated[category]
        pairs = []

        target_keys = [k for k in data.keys() if k.startswith("target_")]
        attribute_keys = [k for k in data.keys() if k.startswith("attribute_")]

        if len(target_keys) >= 2 and len(attribute_keys) >= 2:
            target_a = data.get("target_A_male") or data.get("target_A_science") or data.get("target_A_north") or data.get("target_A_stem") or []
            target_b = data.get("target_B_female") or data.get("target_B_arts") or data.get("target_B_south") or data.get("target_B_humanities") or []

            attr_x = data.get("attribute_X_career") or data.get("attribute_X_science_pos") or data.get("attribute_X_positive") or []
            attr_y = data.get("attribute_Y_family") or data.get("attribute_Y_arts_pos") or data.get("attribute_Y_negative") or []

            if target_a and target_b and attr_x and attr_y:
                pairs.append({
                    "category": category,
                    "target_A": target_a,
                    "target_B": target_b,
                    "attribute_X": attr_x,
                    "attribute_Y": attr_y,
                })

        return pairs
