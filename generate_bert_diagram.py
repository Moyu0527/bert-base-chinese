import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体，确保在不同系统下都能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 10))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 2-1 BERT 模型整体架构与双向注意力机制示意图", fontsize=24, fontweight='bold', y=0.95)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=12, ls='-', zorder=1):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        linestyle=ls, lw=2.5, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#111111', zorder=zorder+1, fontweight='bold')
    return (x, y - h/2), (x, y + h/2), (x - w/2, y), (x + w/2, y)

def draw_arrow(start, end, rad=0.0, lw=2.5, ls='-', color="#555555"):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}"))

# ==========================================
# 左侧面板：BERT 整体架构 (Macro View)
# ==========================================
ax.text(25, 88, "BERT 整体输入与层级架构", fontsize=16, ha='center', va='center', color="#333333", fontweight='bold')

# 输入层 Embeddings
draw_box(10, 15, 12, 6, "Token\nEmbeddings", "#E8F0FE", "#4A76A8", fontsize=11)
draw_box(25, 15, 12, 6, "Segment\nEmbeddings", "#E8F0FE", "#4A76A8", fontsize=11)
draw_box(40, 15, 12, 6, "Position\nEmbeddings", "#E8F0FE", "#4A76A8", fontsize=11)

# 融合节点
ax.text(25, 25, "⊕", fontsize=24, ha='center', va='center', color="#C25953")
draw_arrow((10, 18), (23, 25), rad=0.15)
draw_arrow((25, 18), (25, 23))
draw_arrow((40, 18), (27, 25), rad=-0.15)

# BERT 堆叠层
_, top_in, _, _ = draw_box(25, 33, 22, 6, "Input Representation", "#F5F5F5", "#888888")
draw_arrow((25, 27), (25, 30))

_, top_t1, _, _ = draw_box(25, 45, 26, 8, "Transformer Encoder 1", "#E6F4EA", "#57A66C", fontsize=14)
draw_arrow((25, 36), (25, 41))

ax.text(25, 55, "......", fontsize=24, ha='center', va='center', color="#888888", rotation=90)
draw_arrow((25, 49), (25, 52))

_, top_tn, _, _ = draw_box(25, 65, 26, 8, "Transformer Encoder 12", "#E6F4EA", "#57A66C", fontsize=14)
draw_arrow((25, 58), (25, 61))

draw_box(25, 78, 22, 6, "Contextual Embeddings\n(语境化输出)", "#FCE8E6", "#C25953", fontsize=12)
draw_arrow((25, 69), (25, 75))

# ==========================================
# 右侧面板：Transformer 编码器内部细节 (Micro View)
# ==========================================
# 放大连线
ax.plot([38, 55], [45, 15], color="#57A66C", ls='--', lw=1.5, alpha=0.6)
ax.plot([38, 55], [45, 88], color="#57A66C", ls='--', lw=1.5, alpha=0.6)

# 背景框
rect_bg = patches.FancyBboxPatch(
    (55, 15), 40, 73, boxstyle="round,pad=1",
    edgecolor="#57A66C", facecolor="#F9FCFA", linestyle='--', lw=2.5, zorder=0
)
ax.add_patch(rect_bg)
ax.text(75, 91, "Transformer Encoder 内部结构 (双向注意力)", fontsize=16, ha='center', va='center', color="#57A66C", fontweight='bold')

# 1. Input
bot_enc_in, top_enc_in, left_enc_in, _ = draw_box(75, 20, 14, 5, "Input", "#FFFFFF", "#888888")

# 2. Q, K, V 投影
_, top_q, _, _ = draw_box(63, 33, 8, 5, "Q\n(Query)", "#FFF2CC", "#F1C232")
_, top_k, _, _ = draw_box(75, 33, 8, 5, "K\n(Key)", "#FFF2CC", "#F1C232")
_, top_v, _, _ = draw_box(87, 33, 8, 5, "V\n(Value)", "#FFF2CC", "#F1C232")

# Input -> Q, K, V
draw_arrow((75, 22.5), (63, 30.5), rad=0.1)
draw_arrow((75, 22.5), (75, 30.5))
draw_arrow((75, 22.5), (87, 30.5), rad=-0.1)

# 3. Multi-Head Attention
bot_mha, top_mha, _, _ = draw_box(75, 48, 30, 8, "Multi-Head Attention\n(多头自注意力机制)", "#E8F0FE", "#4A76A8", fontsize=13)
# Q, K, V -> MHA
draw_arrow((63, 35.5), (68, 44), rad=-0.1)
draw_arrow((75, 35.5), (75, 44))
draw_arrow((87, 35.5), (82, 44), rad=0.1)

# 4. Add & Norm 1
bot_an1, top_an1, left_an1, _ = draw_box(75, 60, 18, 5, "Add & Norm", "#FCE8E6", "#C25953")
draw_arrow((75, 52), (75, 57.5))

# Residual Connection 1 (Input -> Add & Norm 1)
ax.plot([left_enc_in[0], 58], [20, 20], color="#C25953", lw=2.5, ls=':')
ax.plot([58, 58], [20, 60], color="#C25953", lw=2.5, ls=':')
draw_arrow((58, 60), (left_an1[0], 60), color="#C25953", ls=':')
ax.text(60, 40, "Residual\nConnection", fontsize=10, color="#C25953", rotation=90, va='center')

# 5. Feed Forward
bot_ffn, top_ffn, _, _ = draw_box(75, 72, 22, 6, "Feed Forward\n(前馈神经网络)", "#E8F0FE", "#4A76A8", fontsize=13)
draw_arrow((75, 62.5), (75, 69))

# 6. Add & Norm 2
bot_an2, top_an2, left_an2, _ = draw_box(75, 84, 18, 5, "Add & Norm", "#FCE8E6", "#C25953")
draw_arrow((75, 75), (75, 81.5))

# Residual Connection 2 (Add & Norm 1 -> Add & Norm 2)
ax.plot([left_an1[0], 60], [60, 60], color="#C25953", lw=2.5, ls=':')
ax.plot([60, 60], [60, 84], color="#C25953", lw=2.5, ls=':')
draw_arrow((60, 84), (left_an2[0], 84), color="#C25953", ls=':')

os.makedirs('results', exist_ok=True)
plt.savefig('results/BERT_Architecture_Attention.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/BERT_Architecture_Attention.png', dpi=300, bbox_inches='tight')
print("BERT 架构图生成完毕！保存在 results/ 目录下。")
