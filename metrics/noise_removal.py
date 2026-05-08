"""
文件用途：向量噪声消除模块
主要功能：
1. 采用全集归一化（Global Centering）技术。
2. 将所有提取出的静态或语境化词向量减去整个向量空间的质心，拉平模型中词频导致的频率分布偏差。
3. 使后续的余弦距离计算更专注于语义方向，而非词频带来的长度与偏移。
"""
import numpy as np
from typing import List, Optional
from scipy import stats


class NoiseRemoval:
    @staticmethod
    def global_centering(embeddings: np.ndarray) -> np.ndarray:
        """
        全集归一化（Global Centering）
        将所有词向量减去空间质心，拉平频率分布
        
        Args:
            embeddings: 词向量矩阵，shape: [n_samples, n_features]
        
        Returns:
            centered_embeddings: 中心化后的词向量矩阵
        """
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array")
        
        mean_vec = np.mean(embeddings, axis=0, keepdims=True)
        centered = embeddings - mean_vec
        
        return centered

    @staticmethod
    def z_score_normalization(embeddings: np.ndarray) -> np.ndarray:
        """
        Z-score归一化
        
        Args:
            embeddings: 词向量矩阵
        
        Returns:
            normalized_embeddings: Z-score归一化后的矩阵
        """
        mean = np.mean(embeddings, axis=0)
        std = np.std(embeddings, axis=0)
        std[std == 0] = 1e-6
        
        normalized = (embeddings - mean) / std
        return normalized

    @staticmethod
    def unit_length_normalization(embeddings: np.ndarray) -> np.ndarray:
        """
        单位长度归一化（L2归一化）
        
        Args:
            embeddings: 词向量矩阵
        
        Returns:
            normalized_embeddings: L2归一化后的矩阵
        """
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1e-6
        
        normalized = embeddings / norms
        return normalized

    @staticmethod
    def frequency_bias_removal(
        embeddings: np.ndarray,
        word_frequencies: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        频率偏差消除
        
        Args:
            embeddings: 词向量矩阵
            word_frequencies: 词频数组（可选）
        
        Returns:
            debiased_embeddings: 去噪后的词向量
        """
        centered = NoiseRemoval.global_centering(embeddings)
        
        if word_frequencies is not None:
            freq_weights = 1.0 / (word_frequencies + 1e-6)
            freq_weights = freq_weights / np.max(freq_weights)
            centered = centered * freq_weights[:, np.newaxis]
        
        normalized = NoiseRemoval.unit_length_normalization(centered)
        
        return normalized

    @staticmethod
    def remove_pc_direction(embeddings: np.ndarray, n_components: int = 1) -> np.ndarray:
        """
        移除主成分方向（用于去偏）
        
        Args:
            embeddings: 词向量矩阵
            n_components: 要移除的主成分数量
        
        Returns:
            debiased_embeddings: 去偏后的词向量
        """
        centered = NoiseRemoval.global_centering(embeddings)
        
        cov_matrix = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, sorted_indices]
        
        for i in range(n_components):
            pc = eigenvectors[:, i:i+1]
            projection = centered @ pc @ pc.T
            centered = centered - projection
        
        return centered

    @staticmethod
    def apply_full_debiasing(
        embeddings: np.ndarray,
        word_frequencies: Optional[np.ndarray] = None,
        remove_pc: bool = True
    ) -> np.ndarray:
        """
        应用完整的去噪流程
        
        Args:
            embeddings: 词向量矩阵
            word_frequencies: 词频数组
            remove_pc: 是否移除主成分
        
        Returns:
            debiased_embeddings: 去噪后的词向量
        """
        embeddings = NoiseRemoval.global_centering(embeddings)
        
        if word_frequencies is not None:
            embeddings = NoiseRemoval.frequency_bias_removal(embeddings, word_frequencies)
        
        if remove_pc:
            embeddings = NoiseRemoval.remove_pc_direction(embeddings, n_components=1)
        
        embeddings = NoiseRemoval.unit_length_normalization(embeddings)
        
        return embeddings
