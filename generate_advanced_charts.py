import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import torch
from transformers import BertTokenizer, BertModel

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_word_embedding(model, tokenizer, word):
    input_ids = tokenizer.encode(word, add_special_tokens=False)
    with torch.no_grad():
        embs = model.embeddings.word_embeddings(torch.tensor(input_ids))
        return embs.mean(dim=0).numpy()

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def plot_heatmap(ax, data, row_labels, col_labels, title):
    sns.heatmap(data, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                xticklabels=col_labels, yticklabels=row_labels, ax=ax)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis='x', rotation=45)
    ax.tick_params(axis='y', rotation=0)

def generate_heatmap_report(pdf_path, orig_model_path, ft_model_path):
    print("正在提取词汇余弦相似度用于绘制热力图...")
    orig_tokenizer = BertTokenizer.from_pretrained(orig_model_path)
    orig_model = BertModel.from_pretrained(orig_model_path)
    
    ft_tokenizer = BertTokenizer.from_pretrained(ft_model_path)
    ft_model = BertModel.from_pretrained(ft_model_path)
    
    # 选取代表性的目标词和属性词
    targets = ["男性", "女性", "北京", "河南", "老板", "保安"]
    attributes = ["科学", "艺术", "富裕", "贫穷", "成功", "失败"]
    
    def calc_sim_matrix(model, tokenizer):
        matrix = np.zeros((len(targets), len(attributes)))
        for i, t in enumerate(targets):
            t_emb = get_word_embedding(model, tokenizer, t)
            for j, a in enumerate(attributes):
                a_emb = get_word_embedding(model, tokenizer, a)
                matrix[i, j] = cos_sim(t_emb, a_emb)
        return matrix

    orig_matrix = calc_sim_matrix(orig_model, orig_tokenizer)
    ft_matrix = calc_sim_matrix(ft_model, ft_tokenizer)
    
    # 计算相对变化矩阵
    diff_matrix = ft_matrix - orig_matrix

    with PdfPages(pdf_path) as pdf:
        # 第一页：相似度热力图对比
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plot_heatmap(ax1, orig_matrix, targets, attributes, "原始模型：目标词与属性词余弦相似度")
        plot_heatmap(ax2, ft_matrix, targets, attributes, "去偏模型：目标词与属性词余弦相似度")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # 第二页：相似度差值热力图
        fig, ax = plt.subplots(figsize=(10, 8))
        plot_heatmap(ax, diff_matrix, targets, attributes, "微调前后相似度变化 (正值代表关联增强，负值代表关联减弱)")
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
    print(f"热力图报告已生成: {pdf_path}")

def plot_radar_chart(pdf_path, json_path):
    print("正在绘制雷达图...")
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    orig = data['original']
    ft = data['finetuned']
    
    dimensions = list(orig.keys())
    num_vars = len(dimensions)
    
    # 准备雷达图的角度
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # 闭合多边形
    
    def get_radar_data(model_data, metric):
        vals = [abs(model_data[dim][metric]) for dim in dimensions]
        vals += vals[:1]
        return vals

    orig_srat = get_radar_data(orig, 'srat_shift')
    ft_srat = get_radar_data(ft, 'srat_shift')
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    ax.plot(angles, orig_srat, color='red', linewidth=2, label='原始模型 (Original)')
    ax.fill(angles, orig_srat, color='red', alpha=0.25)
    
    ax.plot(angles, ft_srat, color='blue', linewidth=2, label='去偏模型 (Fine-tuned)')
    ax.fill(angles, ft_srat, color='blue', alpha=0.25)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimensions, fontsize=12)
    ax.set_title('多维度 SRAT 偏移量雷达图对比 (面积越小偏见越低)', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
    plt.close()
    print(f"雷达图报告已生成: {pdf_path}")

def main():
    os.makedirs('results', exist_ok=True)
    orig_model = "bert-base-chinese"
    ft_model = "models/bert-base-chinese-weat-cdial"
    json_path = 'results/multidim_evaluation.json'
    
    if not os.path.exists(ft_model) or not os.path.exists(json_path):
        print("缺少前置模型或数据文件，请先运行前面的步骤。")
        return
        
    generate_heatmap_report('results/Heatmap_Bias_Analysis.pdf', orig_model, ft_model)
    plot_radar_chart('results/Radar_Bias_Analysis.pdf', json_path)
    print("\n✅ 所有进阶图表生成完毕！")

if __name__ == '__main__':
    main()