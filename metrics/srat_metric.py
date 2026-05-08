"""
文件用途：SRAT (句表征关联测试) 核心算法实现模块
主要功能：
1. 使用预训练 BERT 模型对整个句子进行编码，提取并返回语境化向量 (Contextualized Vector)。
2. 计算带有不同属性词/引导词（刻板印象 vs 反刻板印象）的句子对在向量空间中的重心偏移量 (Vector Shift Magnitude)。
3. 根据计算得到的多个模板偏移量取其期望值，消除单一模板的偶发性误差，实现偏见分数的量化。
"""
import torch
import numpy as np
from typing import List, Dict, Tuple
from transformers import BertTokenizer, BertModel
import config


class SRATMetric:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or config.MODEL_CONFIG["model_name"]
        self.device = config.MODEL_CONFIG["device"]
        self.dtype = config.MODEL_CONFIG["dtype"]
        self.batch_size = config.MODEL_CONFIG["batch_size"]
        
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading SRAT model: {self.model_name}")
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = BertModel.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cuda":
            self.model = self.model.to(self.device)
            self.model = self.model.half()
        self.model.eval()
        print("SRAT model loaded")

    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        对句子进行编码，获取语境化向量
        
        Args:
            sentences: 句子列表
        
        Returns:
            embeddings: 句子嵌入矩阵，shape: [n_sentences, hidden_dim]
        """
        embeddings = []
        
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MODEL_CONFIG["max_seq_length"],
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded)
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.extend(cls_embeddings.detach().cpu().numpy())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        return np.array(embeddings)

    def calculate_vector_shift(
        self,
        stereotype_sentences: List[str],
        anti_stereotype_sentences: List[str]
    ) -> Tuple[np.ndarray, float]:
        """
        计算语境化向量重心偏移量
        
        Args:
            stereotype_sentences: 刻板印象句子列表
            anti_stereotype_sentences: 反刻板印象句子列表
        
        Returns:
            shift_vector: 偏移向量
            shift_magnitude: 偏移量大小
        """
        stereotype_embeddings = self.encode_sentences(stereotype_sentences)
        anti_stereotype_embeddings = self.encode_sentences(anti_stereotype_sentences)
        
        stereotype_center = np.mean(stereotype_embeddings, axis=0)
        anti_stereotype_center = np.mean(anti_stereotype_embeddings, axis=0)
        
        shift_vector = stereotype_center - anti_stereotype_center
        shift_magnitude = np.linalg.norm(shift_vector)
        
        return shift_vector, shift_magnitude

    def calculate_expected_effect_size(
        self,
        stereotype_templates: List[str],
        anti_stereotype_templates: List[str],
        target_words: List[str]
    ) -> float:
        """
        计算期望效应量（Expected Value of Effect Size）
        
        Args:
            stereotype_templates: 刻板印象模板列表
            anti_stereotype_templates: 反刻板印象模板列表
            target_words: 目标词列表
        
        Returns:
            expected_effect_size: 期望效应量
        """
        effect_sizes = []
        
        for target in target_words:
            stereotype_sentences = [t.replace("{target}", target) for t in stereotype_templates]
            anti_stereotype_sentences = [t.replace("{target}", target) for t in anti_stereotype_templates]
            
            shift_vector, shift_magnitude = self.calculate_vector_shift(
                stereotype_sentences,
                anti_stereotype_sentences
            )
            
            effect_sizes.append(shift_magnitude)
        
        if len(effect_sizes) == 0:
            return 0.0
        
        expected_effect_size = np.mean(effect_sizes)
        
        return expected_effect_size

    def run_srat_test(
        self,
        stereotype_templates: List[str],
        anti_stereotype_templates: List[str],
        target_group_a: List[str],
        target_group_b: List[str],
        n_permutations: int = 1000
    ) -> Dict[str, float]:
        """
        运行完整的SRAT（Sentence-level Representation Association Test）测试
        
        Args:
            stereotype_templates: 刻板印象模板
            anti_stereotype_templates: 反刻板印象模板
            target_group_a: 目标组A（如男性代词）
            target_group_b: 目标组B（如女性代词）
            n_permutations: 置换次数
        
        Returns:
            result: 包含SRAT得分、p值和效应量的字典
        """
        effect_a = self.calculate_expected_effect_size(
            stereotype_templates,
            anti_stereotype_templates,
            target_group_a
        )
        
        effect_b = self.calculate_expected_effect_size(
            stereotype_templates,
            anti_stereotype_templates,
            target_group_b
        )
        
        srat_score = effect_a - effect_b
        
        all_targets = target_group_a + target_group_b
        n_a = len(target_group_a)
        
        permutation_scores = []
        for _ in range(n_permutations):
            shuffled = np.random.permutation(all_targets)
            perm_a = list(shuffled[:n_a])
            perm_b = list(shuffled[n_a:])
            
            perm_effect_a = self.calculate_expected_effect_size(
                stereotype_templates,
                anti_stereotype_templates,
                perm_a
            )
            
            perm_effect_b = self.calculate_expected_effect_size(
                stereotype_templates,
                anti_stereotype_templates,
                perm_b
            )
            
            permutation_scores.append(perm_effect_a - perm_effect_b)
        
        permutation_scores = np.array(permutation_scores)
        
        p_value = (np.sum(np.abs(permutation_scores) >= np.abs(srat_score)) + 1) / (n_permutations + 1)
        
        mean_null = np.mean(permutation_scores)
        std_null = np.std(permutation_scores)
        
        if std_null == 0:
            effect_size = 0.0
        else:
            effect_size = (srat_score - mean_null) / std_null
        
        return {
            "srat_score": srat_score,
            "effect_a": effect_a,
            "effect_b": effect_b,
            "p_value": p_value,
            "effect_size": effect_size,
            "n_permutations": n_permutations,
            "n_target_a": len(target_group_a),
            "n_target_b": len(target_group_b),
            "n_templates": len(stereotype_templates)
        }

    def calculate_layerwise_shift(
        self,
        sentences: List[str],
        layers: List[int] = None
    ) -> Dict[int, np.ndarray]:
        """
        计算不同隐藏层的向量偏移
        
        Args:
            sentences: 句子列表
            layers: 要分析的层索引列表
        
        Returns:
            layer_shifts: 各层的偏移向量
        """
        if layers is None:
            layers = list(range(self.model.config.num_hidden_layers))
        
        layer_embeddings = {layer: [] for layer in layers}
        
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i:i+self.batch_size]
            
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MODEL_CONFIG["max_seq_length"],
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
            
            with torch.no_grad():
                outputs = self.model(**encoded, output_hidden_states=True)
                
                for layer in layers:
                    layer_output = outputs.hidden_states[layer]
                    cls_embeddings = layer_output[:, 0, :]
                    layer_embeddings[layer].extend(cls_embeddings.detach().cpu().numpy())
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        layer_centers = {}
        for layer in layers:
            embeddings = np.array(layer_embeddings[layer])
            layer_centers[layer] = np.mean(embeddings, axis=0)
        
        return layer_centers

    def release_memory(self):
        """释放模型占用的显存"""
        if self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()
            print("SRAT model memory released")
