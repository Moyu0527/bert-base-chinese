"""
文件用途：静态词向量提取模块
主要功能：
1. 加载并初始化指定的预训练语言模型（如 BERT）。
2. 从模型的底层嵌入层（Embedding Layer）中批量提取目标词或属性词的静态词嵌入（Static Embeddings），以供后续的 WRAT 计算使用。
"""
import os
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from transformers import BertTokenizer, BertModel
import config


class WordEmbeddingExtractor:
    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or config.MODEL_CONFIG["model_name"]
        self.device = config.MODEL_CONFIG["device"]
        self.dtype = config.MODEL_CONFIG["dtype"]
        self.batch_size = config.MODEL_CONFIG["batch_size"]
        self.max_seq_length = config.MODEL_CONFIG["max_seq_length"]
        
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        print(f"Loading model: {self.model_name}")
        print(f"Device: {self.device}, dtype: {self.dtype}")
        
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
        print(f"Model loaded successfully")

    def extract_embeddings(self, words: List[str], layer: int = -1) -> Tuple[np.ndarray, List[str]]:
        """
        提取词的静态词嵌入
        
        Args:
            words: 待提取嵌入的词列表
            layer: 提取哪个层的嵌入，-1表示最后一层
        
        Returns:
            embeddings: 词向量矩阵，shape: [num_valid_words, hidden_dim]
            valid_words: 有效的词列表（过滤掉BERT词表中不存在的词）
        """
        embeddings = []
        valid_words = []
        
        for i in range(0, len(words), self.batch_size):
            batch_words = words[i:i+self.batch_size]
            
            batch_embeddings = self._extract_batch_embeddings(batch_words, layer)
            embeddings.extend(batch_embeddings)
            valid_words.extend(batch_words)
            
            if self.device == "cuda":
                torch.cuda.empty_cache()
        
        if embeddings:
            embeddings = np.vstack(embeddings)
        else:
            embeddings = np.array([])
        
        return embeddings, valid_words

    def _extract_batch_embeddings(self, words: List[str], layer: int) -> List[np.ndarray]:
        """提取一批词的嵌入"""
        embeddings = []
        
        for word in words:
            embedding = self._extract_single_word_embedding(word, layer)
            if embedding is not None:
                embeddings.append(embedding)
        
        return embeddings

    def _extract_single_word_embedding(self, word: str, layer: int) -> Optional[np.ndarray]:
        """提取单个词的嵌入（处理subword情况）"""
        tokens = self.tokenizer.tokenize(word)
        
        if not tokens:
            return None
        
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        input_ids = torch.tensor([input_ids], device=self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states
            layer_output = hidden_states[layer]
            
            seq_len = layer_output.size(1)
            
            if seq_len <= 2:
                embedding = layer_output[0, 0, :].detach().cpu().numpy()
            elif len(tokens) == 1:
                embedding = layer_output[0, 0, :].detach().cpu().numpy()
            else:
                subword_embeddings = layer_output[0, 1:-1, :]
                embedding = torch.mean(subword_embeddings, dim=0).detach().cpu().numpy()
        
        return embedding

    def extract_target_attribute_embeddings(
        self,
        target_words: List[str],
        attribute_words: List[str],
        layer: int = -1
    ) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
        """
        提取目标词和属性词的嵌入
        
        Returns:
            target_embeddings: 目标词向量矩阵
            attribute_embeddings: 属性词向量矩阵
            valid_targets: 有效的目标词
            valid_attributes: 有效的属性词
        """
        print(f"Extracting embeddings for {len(target_words)} target words...")
        target_embeddings, valid_targets = self.extract_embeddings(target_words, layer)
        
        print(f"Extracting embeddings for {len(attribute_words)} attribute words...")
        attribute_embeddings, valid_attributes = self.extract_embeddings(attribute_words, layer)
        
        return target_embeddings, attribute_embeddings, valid_targets, valid_attributes

    def get_vocab_size(self) -> int:
        """获取词表大小"""
        return len(self.tokenizer.vocab)

    def is_in_vocab(self, word: str) -> bool:
        """检查词是否在词表中"""
        tokens = self.tokenizer.tokenize(word)
        return len(tokens) > 0

    def get_embedding_dim(self) -> int:
        """获取嵌入维度"""
        return self.model.config.hidden_size

    def release_memory(self):
        """释放模型占用的显存"""
        if self.device == "cuda":
            del self.model
            torch.cuda.empty_cache()
            print("Model memory released")


class EmbeddingCache:
    """嵌入缓存管理器"""
    def __init__(self):
        self.cache = {}
    
    def get(self, key: str) -> Optional[np.ndarray]:
        return self.cache.get(key)
    
    def set(self, key: str, embedding: np.ndarray):
        self.cache[key] = embedding
    
    def contains(self, key: str) -> bool:
        return key in self.cache
    
    def clear(self):
        self.cache = {}
    
    def size(self) -> int:
        return len(self.cache)
