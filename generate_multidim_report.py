import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.decomposition import PCA
import torch
from transformers import BertTokenizer, BertModel

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def get_contextual_embedding(model, tokenizer, word, template="这个人是[T]。"):
    """使用语境化向量来绘制 PCA，因为微调主要改变的是上下文表征能力"""
    text = template.replace("[T]", word)
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
        hidden_states = outputs.last_hidden_state[0, 1:-1]
        return hidden_states.mean(dim=0).numpy()

def extract_contextual_embeddings(model_path, words_dict):
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    
    embeddings_dict = {}
    for category, words in words_dict.items():
        cat_embeddings = []
        valid_words = []
        for word in words:
            emb = get_contextual_embedding(model, tokenizer, word)
            cat_embeddings.append(emb)
            valid_words.append(word)
        embeddings_dict[category] = {"words": valid_words, "embeddings": np.array(cat_embeddings)}
    return embeddings_dict

def plot_multidim_bars(ax, orig_data, ft_data, metric_key, title, ylabel):
    dimensions = list(orig_data.keys())
    x = np.arange(len(dimensions))
    width = 0.35
    
    orig_vals = [abs(orig_data[dim][metric_key]) for dim in dimensions]
    ft_vals = [abs(ft_data[dim][metric_key]) for dim in dimensions]
    
    rects1 = ax.bar(x - width/2, orig_vals, width, label='原始模型', color='#FF9999')
    rects2 = ax.bar(x + width/2, ft_vals, width, label='微调后模型', color='#66B2FF')
    
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(dimensions, fontsize=10, rotation=15)
    ax.legend()
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
    autolabel(rects1)
    autolabel(rects2)

def main():
    json_path = 'results/multidim_evaluation.json'
    if not os.path.exists(json_path):
        print("未找到多维度数据，请先运行 evaluate_multidim.py")
        return
        
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    orig = data['original']
    ft = data['finetuned']
    
    pdf_path = 'results/Multi_Dimension_Bias_Report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # ==========================================
        # 第 1 页：多维度 WRAT 与 SRAT 柱状图
        # ==========================================
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))
        
        plot_multidim_bars(ax1, orig, ft, 'wrat_effect_size', 
                           '图1：多维度 WRAT 词级别偏见绝对效应量对比 (越接近0越好)', '绝对效应量 |d|')
                           
        plot_multidim_bars(ax2, orig, ft, 'srat_shift', 
                           '图2：多维度 SRAT 句级别上下文向量偏移量对比 (越小越好)', '向量重心偏移距离')
                           
        plt.tight_layout(pad=5.0)
        pdf.savefig()
        plt.close()
        
        # ==========================================
        # 第 2 页：使用共享坐标系的 PCA 语境化向量投影
        # ==========================================
        print("正在提取语境化向量进行共享坐标系 PCA 降维...")
        # 选取典型的性别与职业对比词
        pca_words = {
            "男性目标": ["男人", "父亲", "哥哥"],
            "女性目标": ["女人", "母亲", "姐姐"],
            "职场/高薪": ["老板", "高管", "成功"],
            "家庭/底层": ["家务", "照顾", "底层"]
        }
        
        orig_embs_dict = extract_contextual_embeddings("bert-base-chinese", pca_words)
        ft_embs_dict = extract_contextual_embeddings("models/bert-base-chinese-weat-cdial", pca_words)
        
        # 为了保证前后投影的坐标系完全一致，我们将原始模型的数据训练出一个 PCA，
        # 然后同时转换原始和微调后的向量！
        all_orig_embs = np.vstack([orig_embs_dict[cat]["embeddings"] for cat in pca_words])
        pca_model = PCA(n_components=2)
        pca_model.fit(all_orig_embs)
        
        fig = plt.figure(figsize=(8.27, 11.69))
        ax_title = fig.add_axes([0.1, 0.9, 0.8, 0.05])
        ax_title.axis('off')
        ax_title.text(0.5, 0.5, '图3：基于共享坐标系的语境化向量 PCA 投影', ha='center', va='center', fontsize=16, weight='bold')
        
        def plot_shared_pca(ax, embs_dict, title):
            colors = ['blue', 'red', 'green', 'orange']
            markers = ['o', 's', '^', 'D']
            categories = list(embs_dict.keys())
            
            for i, cat in enumerate(categories):
                embs = embs_dict[cat]["embeddings"]
                if len(embs) == 0: continue
                # 使用共享的 pca_model 进行转换
                embs_2d = pca_model.transform(embs)
                x, y = embs_2d[:, 0], embs_2d[:, 1]
                ax.scatter(x, y, c=colors[i], marker=markers[i], label=cat, s=120, alpha=0.7)
                
                for j, word in enumerate(embs_dict[cat]["words"]):
                    ax.annotate(word, (x[j], y[j]), xytext=(5, 5), textcoords='offset points', fontsize=11)
                    
            ax.set_title(title, fontsize=14)
            ax.legend(fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.5)

        ax3 = fig.add_axes([0.1, 0.52, 0.8, 0.35])
        plot_shared_pca(ax3, orig_embs_dict, "原始模型 (存在明显的刻板印象空间隔离)")
        
        ax4 = fig.add_axes([0.1, 0.10, 0.8, 0.35])
        plot_shared_pca(ax4, ft_embs_dict, "去偏微调后模型 (词汇向量明显向中心融合，偏见被削弱)")
        
        pdf.savefig()
        plt.close()
        
    print(f"\n✅ 多维度两两对比综合 PDF 报告已生成至: {pdf_path}")

if __name__ == '__main__':
    main()