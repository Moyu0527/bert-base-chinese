import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体，确保在不同系统下都能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 18))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

def draw_layer_bg(y_start, y_end, title, edgecolor, facecolor):
    """绘制每层的背景大框和标题"""
    rect = patches.FancyBboxPatch(
        (2, y_start), 96, y_end - y_start,
        boxstyle="round,pad=2",
        edgecolor=edgecolor, facecolor=facecolor, alpha=0.08,
        linestyle='--', lw=2.5
    )
    ax.add_patch(rect)
    ax.text(4, y_end - 1.5, title, fontsize=20, fontweight='bold', color=edgecolor, va='top', ha='left')

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=14):
    """绘制具体的模块节点"""
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.8",
        edgecolor=edgecolor, facecolor=facecolor, lw=2.5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', linespacing=1.8, color='#222222')
    return (x, y - h/2), (x, y + h/2)

def draw_arrow_ortho(start, end):
    """绘制直角折线箭头"""
    x1, y1 = start
    x2, y2 = end
    y_mid = (y1 + y2) / 2
    ax.plot([x1, x1], [y1, y_mid], color='#666666', lw=2.5)
    ax.plot([x1, x2], [y_mid, y_mid], color='#666666', lw=2.5)
    ax.annotate("", xy=(x2, y2), xytext=(x2, y_mid),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=2.5, shrinkA=0, shrinkB=3))

# --- 1. 绘制背景层级 ---
draw_layer_bg(76, 98, "一、输入层：本土化中文多维度偏见数据集构建", "#3C5488", "#4C72B0")
draw_layer_bg(51, 73, "二、处理层：模型表征偏见度量核心算法", "#448C55", "#55A868")
draw_layer_bg(26, 48, "三、干预层：预训练模型去偏微调", "#A84144", "#C44E52")
draw_layer_bg(1, 23, "四、输出与可视化评估层：多维量化评估", "#68589B", "#8172B2")

# --- 2. 绘制各个模块节点 ---
# 第一层：输入层
b1_bot, _ = draw_box(20, 85, 24, 7, "WEAT 中文词表\n(多维度属性映射)", "#F0F4FA", "#3C5488")
b2_bot, _ = draw_box(50, 85, 28, 7, "CDial-Bias & Safety-Prompts\n(真实对话与安全场景)", "#F0F4FA", "#3C5488")
b3_bot, _ = draw_box(80, 85, 24, 7, "CrowS-Pairs 语料\n(刻板印象最小对比对)", "#F0F4FA", "#3C5488")

# 第二层：处理层
l2_t1 = "WRAT 模块 (静态词表征关联)\n" + "-"*26 + "\n• 提取预训练底层词向量特征\n• 引入全集归一化 (频率去噪)\n• 目标与属性词余弦效应量计算"
b4_bot, b4_top = draw_box(30, 60, 36, 10, l2_t1, "#EEF7E8", "#448C55")

l2_t2 = "SRAT 模块 (语境化句表征关联)\n" + "-"*26 + "\n• 融合多场景动态掩码模板生成\n• 提取模型深层语境隐藏状态\n• 隐空间重心偏移量与期望值计算"
b5_bot, b5_top = draw_box(70, 60, 36, 10, l2_t2, "#EEF7E8", "#448C55")

# 第三层：干预层
l3_t = "MLM 微调去偏模块 (Masked Language Modeling)\n" + "-"*40 + "\n• 基于动态掩码与对比语料持续预训练\n• 引入 AMP 混合精度与 GradScaler 梯度缩放优化\n• 解耦并削弱模型固有表征中的隐性偏见"
b6_bot, b6_top = draw_box(50, 35, 54, 10, l3_t, "#FCEBEB", "#A84144")

# 第四层：输出层
b7_bot, b7_top = draw_box(16, 10, 18, 7, "雷达图\n(多维偏见全景对比)", "#EBE6F2", "#68589B")
b8_bot, b8_top = draw_box(38, 10, 18, 7, "PCA 降维投影\n(空间位移可视化)", "#EBE6F2", "#68589B")
b9_bot, b9_top = draw_box(62, 10, 18, 7, "热力图\n(语义关联距离聚类)", "#EBE6F2", "#68589B")
b10_bot, b10_top = draw_box(84, 10, 18, 7, "定量数据表\n(显著性与绝对效应量)", "#EBE6F2", "#68589B")

# --- 3. 绘制连接箭头 ---
# Layer 1 -> Layer 2
draw_arrow_ortho(b1_bot, (30, b4_top[1]))
draw_arrow_ortho((50, b2_bot[1]), (30, b4_top[1]))
draw_arrow_ortho((50, b2_bot[1]), (70, b5_top[1]))
draw_arrow_ortho(b3_bot, (70, b5_top[1]))

# Layer 2 -> Layer 3
draw_arrow_ortho(b4_bot, (50, b6_top[1]))
draw_arrow_ortho(b5_bot, (50, b6_top[1]))

# Layer 3 -> Layer 4 (绘制主干后分支)
trunk_start = b6_bot
trunk_mid_y = 20
ax.plot([trunk_start[0], trunk_start[0]], [trunk_start[1], trunk_mid_y], color='#666666', lw=2.5)
ax.plot([b7_top[0], b10_top[0]], [trunk_mid_y, trunk_mid_y], color='#666666', lw=2.5)
for target in [b7_top, b8_top, b9_top, b10_top]:
    ax.annotate("", xy=target, xytext=(target[0], trunk_mid_y),
                arrowprops=dict(arrowstyle="->", color="#666666", lw=2.5, shrinkA=0, shrinkB=3))

os.makedirs('results', exist_ok=True)
plt.savefig('results/Project_Architecture_Flowchart.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Project_Architecture_Flowchart.png', dpi=300, bbox_inches='tight')
print("流程图生成完毕！保存在 results/ 目录下。")
