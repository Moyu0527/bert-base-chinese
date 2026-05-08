import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from typing import Dict, List, Optional


class BiasVisualizer:
    def __init__(self, results_path: Optional[str] = None):
        self.results = {}
        if results_path:
            self.load_results(results_path)
    
    def load_results(self, results_path: str):
        """加载评估结果"""
        with open(results_path, "r", encoding="utf-8") as f:
            self.results = json.load(f)
    
    def plot_effect_size_comparison(self, output_path: str):
        """
        绘制WRAT和SRAT的效应量对比图表
        
        Args:
            output_path: 输出图片路径
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        dimensions = []
        wrat_effects = []
        srat_effects = []
        
        for dimension, wrat_data in self.results.get("wrat_results", {}).items():
            dimensions.append(dimension)
            
            wrat_effect = 0.0
            count = 0
            for result in wrat_data.get("results", []):
                wrat_effect += result.get("effect_size", 0.0)
                count += 1
            if count > 0:
                wrat_effect /= count
            wrat_effects.append(wrat_effect)
        
        for dimension in dimensions:
            srat_data = self.results.get("srat_results", {}).get(dimension, {})
            srat_result = srat_data.get("result", {})
            srat_effects.append(srat_result.get("effect_size", 0.0))
        
        x = np.arange(len(dimensions))
        width = 0.35
        
        rects1 = ax.bar(x - width/2, wrat_effects, width, label='WRAT')
        rects2 = ax.bar(x + width/2, srat_effects, width, label='SRAT')
        
        ax.set_xlabel('Bias Dimension')
        ax.set_ylabel('Effect Size')
        ax.set_title('WRAT vs SRAT Effect Size Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(dimensions)
        ax.legend()
        
        ax.bar_label(rects1, padding=3, fmt='%.3f')
        ax.bar_label(rects2, padding=3, fmt='%.3f')
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Effect size comparison plot saved to {output_path}")
    
    def plot_p_value_distribution(self, output_path: str):
        """
        绘制p值分布图
        
        Args:
            output_path: 输出图片路径
        """
        p_values = []
        labels = []
        
        for dimension, wrat_data in self.results.get("wrat_results", {}).items():
            for result in wrat_data.get("results", []):
                p_values.append(result.get("p_value", 1.0))
                labels.append(f"{dimension}-{result.get('weat_name', '')}")
        
        for dimension, srat_data in self.results.get("srat_results", {}).items():
            srat_result = srat_data.get("result", {})
            p_values.append(srat_result.get("p_value", 1.0))
            labels.append(f"{dimension}-SRAT")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        y_pos = np.arange(len(p_values))
        
        ax.barh(y_pos, p_values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('p-value')
        ax.set_title('Statistical Significance (p-values)')
        ax.axvline(x=0.05, color='red', linestyle='--', label='p=0.05')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"P-value distribution plot saved to {output_path}")
    
    def plot_embedding_pca(self, embeddings: np.ndarray, labels: List[str], 
                          output_path: str, title: str = "PCA Projection"):
        """
        使用PCA绘制词嵌入的二维投影图
        
        Args:
            embeddings: 词嵌入矩阵
            labels: 词标签列表
            output_path: 输出图片路径
            title: 图表标题
        """
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(labels):
            ax.scatter(reduced[i, 0], reduced[i, 1], alpha=0.6)
            ax.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)
        
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_title(title)
        ax.grid(True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"PCA projection plot saved to {output_path}")
    
    def plot_embedding_tsne(self, embeddings: np.ndarray, labels: List[str],
                           output_path: str, title: str = "t-SNE Projection",
                           perplexity: int = 30):
        """
        使用t-SNE绘制词嵌入的二维投影图
        
        Args:
            embeddings: 词嵌入矩阵
            labels: 词标签列表
            output_path: 输出图片路径
            title: 图表标题
            perplexity: t-SNE perplexity参数
        """
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        reduced = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        for i, label in enumerate(labels):
            ax.scatter(reduced[i, 0], reduced[i, 1], alpha=0.6)
            ax.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=8)
        
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_title(title)
        ax.grid(True)
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"t-SNE projection plot saved to {output_path}")
    
    def plot_bias_dimension_projection(self, embeddings: np.ndarray, labels: List[str],
                                       bias_dimension: str, output_path: str):
        """
        绘制词汇在特定偏见轴上的投影图
        
        Args:
            embeddings: 词嵌入矩阵
            labels: 词标签列表
            bias_dimension: 偏见维度名称
            output_path: 输出图片路径
        """
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        scatter = ax.scatter(reduced[:, 0], reduced[:, 1], c=reduced[:, 0], 
                            cmap='coolwarm', alpha=0.7)
        
        for i, label in enumerate(labels):
            ax.annotate(label, (reduced[i, 0], reduced[i, 1]), fontsize=9)
        
        ax.set_xlabel(f'{bias_dimension} Bias Axis (PC1)')
        ax.set_ylabel('Secondary Axis (PC2)')
        ax.set_title(f'Word Projection on {bias_dimension} Bias Axis')
        
        plt.colorbar(scatter, label='Bias Score')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bias dimension projection plot saved to {output_path}")
    
    def plot_score_heatmap(self, output_path: str):
        """
        绘制偏见得分热力图
        
        Args:
            output_path: 输出图片路径
        """
        dimensions = list(self.results.get("wrat_results", {}).keys())
        metrics = ["WRAT", "SRAT"]
        
        scores = []
        
        for dimension in dimensions:
            row = []
            
            wrat_data = self.results.get("wrat_results", {}).get(dimension, {})
            wrat_score = 0.0
            count = 0
            for result in wrat_data.get("results", []):
                wrat_score += result.get("weat_score", 0.0)
                count += 1
            if count > 0:
                wrat_score /= count
            row.append(wrat_score)
            
            srat_data = self.results.get("srat_results", {}).get(dimension, {})
            srat_result = srat_data.get("result", {})
            row.append(srat_result.get("srat_score", 0.0))
            
            scores.append(row)
        
        scores = np.array(scores)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(scores, cmap='RdBu', vmin=-1, vmax=1)
        
        ax.set_xticks(np.arange(len(metrics)))
        ax.set_yticks(np.arange(len(dimensions)))
        ax.set_xticklabels(metrics)
        ax.set_yticklabels(dimensions)
        
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        for i in range(len(dimensions)):
            for j in range(len(metrics)):
                text = ax.text(j, i, f"{scores[i, j]:.3f}",
                              ha="center", va="center", color="w")
        
        ax.set_title('Bias Score Heatmap')
        plt.colorbar(im)
        
        fig.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Heatmap plot saved to {output_path}")
    
    def generate_all_plots(self, output_dir: str):
        """
        生成所有可视化图表
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        if self.results:
            self.plot_effect_size_comparison(os.path.join(output_dir, "effect_size_comparison.png"))
            self.plot_p_value_distribution(os.path.join(output_dir, "p_value_distribution.png"))
            self.plot_score_heatmap(os.path.join(output_dir, "bias_score_heatmap.png"))
        
        print(f"\nAll plots saved to {output_dir}")


def main():
    import sys
    sys.path.insert(0, '.')
    import config
    
    results_path = os.path.join(config.DATA_CONFIG["output_dir"], "bias_evaluation_results.json")
    
    if os.path.exists(results_path):
        visualizer = BiasVisualizer(results_path)
        visualizer.generate_all_plots(config.DATA_CONFIG["output_dir"])
    else:
        print(f"Results file not found: {results_path}")
        print("Please run main.py first to generate evaluation results.")


if __name__ == "__main__":
    main()
