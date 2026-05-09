import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(14, 16))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 4-2 偏见度量系统总体架构设计框图", fontsize=24, fontweight='bold', y=0.92)

def draw_layer_bg(y_center, h, title, color):
    """绘制每层的大虚线背景框"""
    w = 80
    x_center = 50
    rect = patches.FancyBboxPatch(
        (x_center - w/2, y_center - h/2), w, h,
        boxstyle="round,pad=1",
        edgecolor=color, facecolor=color, alpha=0.06,
        linestyle='--', lw=2, zorder=0
    )
    ax.add_patch(rect)
    # 左上角标明层级名称
    ax.text(x_center - w/2 + 2, y_center + h/2 - 3, title, fontsize=16, fontweight='bold', color=color, ha='left', va='center', zorder=1)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, zorder=2):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        lw=2.5, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)

def draw_arrow(start, end, text="", color="#444444", lw=2.5, x_offset=0):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if text:
        mid_x = (start[0] + end[0]) / 2 + x_offset
        mid_y = (start[1] + end[1]) / 2
        # 给文字加白底，防止和线条重合
        ax.text(mid_x, mid_y, text, fontsize=13, color=color, ha='left', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2))

# ==========================================
# 层级划分参数
# ==========================================
L1_Y = 80
L2_Y = 58
L3_Y = 36
L4_Y = 14
LAYER_H = 14

# ==========================================
# 第一层：数据接入与处理层
# ==========================================
draw_layer_bg(L1_Y, LAYER_H, "数据接入与处理层 (Data Layer)", "#3C5488")
draw_box(25, L1_Y-1, 20, 8, "多源语料采集库\n(WEAT / CDial)", "#F0F4FA", "#3C5488")
draw_box(50, L1_Y-1, 20, 8, "数据清洗与语义增强\n(同义替换 / 模板化)", "#F0F4FA", "#3C5488")
draw_box(75, L1_Y-1, 20, 8, "CrowS-Pairs 构建器\n(最小对立句对生成)", "#F0F4FA", "#3C5488")

# ==========================================
# 第二层：模型干预与优化层
# ==========================================
draw_layer_bg(L2_Y, LAYER_H, "模型干预与优化层 (Model Layer)", "#D9822B")
draw_box(25, L2_Y-1, 20, 8, "预训练基础模型\n(BERT-base-chinese)", "#FFF4E6", "#D9822B")
draw_box(50, L2_Y-1, 20, 8, "硬件资源适配引擎\n(AMP / 显存回收)", "#FFF4E6", "#D9822B")
draw_box(75, L2_Y-1, 20, 8, "MLM 去偏微调模块\n(动态掩码 / 梯度裁剪)", "#FFF4E6", "#D9822B")

# ==========================================
# 第三层：核心偏见度量层
# ==========================================
draw_layer_bg(L3_Y, LAYER_H, "核心偏见度量层 (Measurement Layer)", "#448C55")
draw_box(25, L3_Y-1, 20, 8, "WRAT 静态词表征测试\n(全集归一化去噪)", "#EEF7E8", "#448C55")
draw_box(50, L3_Y-1, 20, 8, "SRAT 动态句表征测试\n(隐空间重心偏移)", "#EEF7E8", "#448C55")
draw_box(75, L3_Y-1, 20, 8, "统计显著性检验\n(Permutation Test)", "#EEF7E8", "#448C55")

# ==========================================
# 第四层：可视化与评估层
# ==========================================
draw_layer_bg(L4_Y, LAYER_H, "可视化与评估层 (Visualization Layer)", "#68589B")
draw_box(25, L4_Y-1, 20, 8, "降维空间投影图\n(PCA Shared Space)", "#EBE6F2", "#68589B")
draw_box(50, L4_Y-1, 20, 8, "多维偏见全景雷达图\n(Radar Chart)", "#EBE6F2", "#68589B")
draw_box(75, L4_Y-1, 20, 8, "综合量化评估报告\n(Metrics PDF Report)", "#EBE6F2", "#68589B")

# ==========================================
# 绘制层级间的连接箭头 (数据流向)
# ==========================================
# L1 -> L2
draw_arrow((50, L1_Y - LAYER_H/2), (50, L2_Y + LAYER_H/2), text="训练与测试数据流", x_offset=2)

# L2 -> L3
draw_arrow((50, L2_Y - LAYER_H/2), (50, L3_Y + LAYER_H/2), text="模型权重与高维特征流", x_offset=2)

# L3 -> L4
draw_arrow((50, L3_Y - LAYER_H/2), (50, L4_Y + LAYER_H/2), text="量化指标与显著性数据流", x_offset=2)

# 补充一个侧边的反馈流 L3 -> L2 (诊断反馈)
draw_arrow((88, L3_Y), (88, L2_Y), text="偏见诊断反馈", color="#A84144", x_offset=2)
# 绘制横向折线连接反馈流
ax.plot([85, 88], [L3_Y, L3_Y], color="#A84144", lw=2.5)
ax.plot([88, 85], [L2_Y, L2_Y], color="#A84144", lw=2.5)

os.makedirs('results', exist_ok=True)
plt.savefig('results/System_Architecture_Diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/System_Architecture_Diagram.png', dpi=300, bbox_inches='tight')
print("系统总体架构设计框图生成完毕！保存在 results/ 目录下。")