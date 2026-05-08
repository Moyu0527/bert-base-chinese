"""
文件用途：WRAT (词表征关联测试) 核心算法实现模块
主要功能：
1. 提供两个词向量余弦相似度计算。
2. 依据 WRAT 核心公式，实现单个词的偏见得分计算、集合测试统计量计算、效应量计算 (Effect Size)。
3. 提供置换检验 (Permutation Test) 来计算统计显著性 P-value。
"""
import numpy as np
from typing import List, Tuple, Dict
from scipy.spatial.distance import cosine


class WRATMetric:
    @staticmethod
    def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0
        return 1 - cosine(vec1, vec2)

    @staticmethod
    def calculate_single_word_score(
        word_embedding: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray,
    ) -> float:
        """
        Formula 3.1: 单个词语的偏见得分
        s(ω, A, B) = mean_{a∈A} cos(ω, a) - mean_{b∈B} cos(ω, b)

        Args:
            word_embedding: 单个词向量
            attribute_a_embeddings: 属性词组A嵌入矩阵
            attribute_b_embeddings: 属性词组B嵌入矩阵

        Returns:
            s: 单个词的偏见得分
        """
        if len(attribute_a_embeddings) == 0 or len(attribute_b_embeddings) == 0:
            return 0.0

        mean_a = np.mean([WRATMetric.cosine_similarity(word_embedding, a)
                          for a in attribute_a_embeddings])
        mean_b = np.mean([WRATMetric.cosine_similarity(word_embedding, b)
                          for b in attribute_b_embeddings])

        return mean_a - mean_b

    @staticmethod
    def calculate_set_statistic(
        target_x_embeddings: np.ndarray,
        target_y_embeddings: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray
    ) -> float:
        """
        Formula 3.2: 集合测试统计量
        s(X,Y,A,B) = Σ_{x∈X} s(x,A,B) - Σ_{y∈Y} s(y,A,B)

        Args:
            target_x_embeddings: 目标词组X嵌入矩阵
            target_y_embeddings: 目标词组Y嵌入矩阵
            attribute_a_embeddings: 属性词组A嵌入矩阵
            attribute_b_embeddings: 属性词组B嵌入矩阵

        Returns:
            s: 集合测试统计量
        """
        sum_x = np.sum([WRATMetric.calculate_single_word_score(x, attribute_a_embeddings, attribute_b_embeddings)
                        for x in target_x_embeddings])
        sum_y = np.sum([WRATMetric.calculate_single_word_score(y, attribute_a_embeddings, attribute_b_embeddings)
                        for y in target_y_embeddings])

        return sum_x - sum_y

    @staticmethod
    def calculate_effect_size(
        target_x_embeddings: np.ndarray,
        target_y_embeddings: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray
    ) -> float:
        """
        Formula 3.3: 效应量计算 (类似Cohen's d)
        d = (mean_{x∈X} s(x,A,B) - mean_{y∈Y} s(y,A,B)) / std_{ω∈X∪Y} s(ω,A,B)

        Args:
            target_x_embeddings: 目标词组X嵌入矩阵
            target_y_embeddings: 目标词组Y嵌入矩阵
            attribute_a_embeddings: 属性词组A嵌入矩阵
            attribute_b_embeddings: 属性词组B嵌入矩阵

        Returns:
            d: 效应量 (d > 0.8表示存在显著的强社会偏见)
        """
        scores_x = [WRATMetric.calculate_single_word_score(x, attribute_a_embeddings, attribute_b_embeddings)
                    for x in target_x_embeddings]
        scores_y = [WRATMetric.calculate_single_word_score(y, attribute_a_embeddings, attribute_b_embeddings)
                    for y in target_y_embeddings]

        all_scores = np.array(scores_x + scores_y)

        if len(all_scores) < 2:
            return 0.0

        mean_x = np.mean(scores_x)
        mean_y = np.mean(scores_y)
        std_all = np.std(all_scores)

        if std_all == 0:
            return 0.0

        return (mean_x - mean_y) / std_all

    @staticmethod
    def calculate_permutation_test(
        target_x_embeddings: np.ndarray,
        target_y_embeddings: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray,
        n_permutations: int = 10000
    ) -> Tuple[float, float]:
        """
        执行置换检验计算p-value

        Args:
            target_x_embeddings: 目标词组X嵌入矩阵
            target_y_embeddings: 目标词组Y嵌入矩阵
            attribute_a_embeddings: 属性词组A嵌入矩阵
            attribute_b_embeddings: 属性词组B嵌入矩阵
            n_permutations: 置换次数

        Returns:
            p_value: p值
            effect_size: 效应量
        """
        observed_statistic = WRATMetric.calculate_set_statistic(
            target_x_embeddings,
            target_y_embeddings,
            attribute_a_embeddings,
            attribute_b_embeddings
        )

        all_targets = np.vstack([target_x_embeddings, target_y_embeddings])
        n_x = len(target_x_embeddings)
        n_total = len(all_targets)

        extreme_count = 0
        for _ in range(n_permutations):
            indices = np.random.permutation(n_total)
            permuted_x = all_targets[indices[:n_x]]
            permuted_y = all_targets[indices[n_x:]]

            permuted_statistic = WRATMetric.calculate_set_statistic(
                permuted_x,
                permuted_y,
                attribute_a_embeddings,
                attribute_b_embeddings
            )

            if np.abs(permuted_statistic) >= np.abs(observed_statistic):
                extreme_count += 1

        p_value = (extreme_count + 1) / (n_permutations + 1)

        effect_size = WRATMetric.calculate_effect_size(
            target_x_embeddings,
            target_y_embeddings,
            attribute_a_embeddings,
            attribute_b_embeddings
        )

        return p_value, effect_size

    @staticmethod
    def run_weat_test(
        target_x_embeddings: np.ndarray,
        target_y_embeddings: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray,
        n_permutations: int = 10000
    ) -> Dict[str, float]:
        """
        运行完整的WEAT测试

        Args:
            target_x_embeddings: 目标词组X嵌入矩阵 (如: 程序员, 医生)
            target_y_embeddings: 目标词组Y嵌入矩阵 (如: 护士, 老师)
            attribute_a_embeddings: 属性词组A嵌入矩阵 (如: 男, 他)
            attribute_b_embeddings: 属性词组B嵌入矩阵 (如: 女, 她)
            n_permutations: 置换次数

        Returns:
            result: 包含WEAT统计量、p值和效应量的字典
        """
        weat_statistic = WRATMetric.calculate_set_statistic(
            target_x_embeddings,
            target_y_embeddings,
            attribute_a_embeddings,
            attribute_b_embeddings
        )

        p_value, effect_size = WRATMetric.calculate_permutation_test(
            target_x_embeddings,
            target_y_embeddings,
            attribute_a_embeddings,
            attribute_b_embeddings,
            n_permutations
        )

        return {
            "weat_statistic": weat_statistic,
            "p_value": p_value,
            "effect_size": effect_size,
            "n_permutations": n_permutations,
            "n_target_x": len(target_x_embeddings),
            "n_target_y": len(target_y_embeddings),
            "n_attr_a": len(attribute_a_embeddings),
            "n_attr_b": len(attribute_b_embeddings),
            "bias_significant": effect_size > 0.8
        }

    @staticmethod
    def run_wrat_test(
        target_embeddings: np.ndarray,
        attribute_a_embeddings: np.ndarray,
        attribute_b_embeddings: np.ndarray,
        n_permutations: int = 10000
    ) -> Dict[str, float]:
        """
        运行WRAT测试（单个目标词组）

        Args:
            target_embeddings: 目标词嵌入矩阵
            attribute_a_embeddings: 属性词组A嵌入矩阵
            attribute_b_embeddings: 属性词组B嵌入矩阵
            n_permutations: 置换次数

        Returns:
            result: 包含WRAT得分、p值和效应量的字典
        """
        wrat_score = np.mean([WRATMetric.calculate_single_word_score(
            target, attribute_a_embeddings, attribute_b_embeddings
        ) for target in target_embeddings])

        all_attributes = np.vstack([attribute_a_embeddings, attribute_b_embeddings])
        n_a = len(attribute_a_embeddings)
        n_total = len(all_attributes)

        scores = []
        for _ in range(n_permutations):
            indices = np.random.permutation(n_total)
            permuted_a = all_attributes[indices[:n_a]]
            permuted_b = all_attributes[indices[n_a:]]

            permuted_score = np.mean([WRATMetric.calculate_single_word_score(
                target, permuted_a, permuted_b
            ) for target in target_embeddings])
            scores.append(permuted_score)

        scores = np.array(scores)
        p_value = (np.sum(np.abs(scores) >= np.abs(wrat_score)) + 1) / (n_permutations + 1)

        effect_size = wrat_score / (np.std(scores) if np.std(scores) > 0 else 1)

        return {
            "wrat_score": wrat_score,
            "p_value": p_value,
            "effect_size": effect_size,
            "n_permutations": n_permutations,
            "n_targets": len(target_embeddings),
            "n_attr_a": len(attribute_a_embeddings),
            "n_attr_b": len(attribute_b_embeddings)
        }
