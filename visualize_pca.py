import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os
from matplotlib.backends.backend_pdf import PdfPages

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def extract_embeddings(model_path, word_lists):
    """提取给定词表的静态词嵌入 (来自底层的 word_embeddings)"""
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model = BertModel.from_pretrained(model_path)
    
    embeddings_dict = {}
    word_embeddings_layer = model.embeddings.word_embeddings
    
    for category, words in word_lists.items():
        cat_embeddings = []
        valid_words = []
        for word in words:
            # 获取词的 token id (如果是单字或存在于词表)
            input_ids = tokenizer.encode(word, add_special_tokens=False)
            if len(input_ids) == 1:  # 只处理单字或完整词汇以保证准确性
                token_id = input_ids[0]
                with torch.no_grad():
                    emb = word_embeddings_layer(torch.tensor([token_id])).squeeze().numpy()
                cat_embeddings.append(emb)
                valid_words.append(word)
        
        embeddings_dict[category] = {
            "words": valid_words,
            "embeddings": np.array(cat_embeddings)
        }
        
    return embeddings_dict

def plot_pca_projection(orig_embs, ft_embs, pdf_path):
    """使用 PCA 进行降维并绘制散点图对比"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    categories = list(orig_embs.keys())
    colors = ['blue', 'red', 'green', 'orange']
    markers = ['o', 's', '^', 'D']
    
    def plot_single_axis(ax, embs_dict, title):
        # 将所有向量合并以训练统一的 PCA 空间
        all_embs = []
        all_labels = []
        all_words = []
        
        for i, cat in enumerate(categories):
            if len(embs_dict[cat]["embeddings"]) > 0:
                all_embs.append(embs_dict[cat]["embeddings"])
                all_labels.extend([i] * len(embs_dict[cat]["words"]))
                all_words.extend(embs_dict[cat]["words"])
                
        all_embs = np.vstack(all_embs)
        
        # 降维至 2D
        pca = PCA(n_components=2)
        embs_2d = pca.fit_transform(all_embs)
        
        # 绘图
        current_idx = 0
        for i, cat in enumerate(categories):
            count = len(embs_dict[cat]["words"])
            if count == 0: continue
            
            x = embs_2d[current_idx:current_idx+count, 0]
            y = embs_2d[current_idx:current_idx+count, 1]
            
            ax.scatter(x, y, c=colors[i], marker=markers[i], label=cat, s=100, alpha=0.7)
            
            # 添加词汇标注
            for j, word in enumerate(embs_dict[cat]["words"]):
                ax.annotate(word, (x[j], y[j]), xytext=(5, 5), textcoords='offset points', fontsize=10)
                
            current_idx += count
            
        ax.set_title(title, fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.5)

    plot_single_axis(ax1, orig_embs, "原始模型: 静态词向量 PCA 投影")
    plot_single_axis(ax2, ft_embs, "去偏微调后: 静态词向量 PCA 投影")
    
    plt.tight_layout()
    
    with PdfPages(pdf_path) as pdf:
        pdf.savefig(fig)
        
    plt.close()
    print(f"PCA 降维投影对比图已生成: {pdf_path}")

def main():
    # 我们测试特定偏见维度的词：男/女 (目标) vs 职业/家庭 (属性)
    # 因为 BERT-base-chinese 词表主要是单字，所以选取具有代表性的单字
    word_lists = {
        "男性目标 (Male)": ["男", "爸", "哥", "爷", "弟", "公"],
        "女性目标 (Female)": ["女", "妈", "姐", "奶", "妹", "婆"],
        "职业/权力属性 (Career)": ["商", "官", "将", "王", "相", "总"],
        "家庭/柔弱属性 (Family)": ["家", "娇", "弱", "幼", "温", "软"]
    }
    
    print("正在提取原始模型词向量...")
    orig_embs = extract_embeddings("bert-base-chinese", word_lists)
    
    print("正在提取去偏模型词向量...")
    # 假设微调后的模型在这个路径
    ft_model_path = "models/bert-base-chinese-weat-cdial"
    if not os.path.exists(ft_model_path):
        print(f"找不到去偏模型 {ft_model_path}，请确保预训练已经完成。")
        return
        
    ft_embs = extract_embeddings(ft_model_path, word_lists)
    
    os.makedirs('results', exist_ok=True)
    pdf_path = "results/PCA_Word_Embeddings_Projection.pdf"
    plot_pca_projection(orig_embs, ft_embs, pdf_path)

if __name__ == '__main__':
    main()