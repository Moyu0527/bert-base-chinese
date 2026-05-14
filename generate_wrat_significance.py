import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(12, 14))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
# plt.suptitle("图 4-6 WRAT 模块效应量计算与显著性检验算法逻辑图", fontsize=22, fontweight='bold', y=0.96)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, shape="round", zorder=2):
    if shape == "round":
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.5",
            edgecolor=edgecolor, facecolor=facecolor,
            lw=2.5, zorder=zorder
        )
    elif shape == "diamond":
        rect = patches.Polygon([
            (x, y + h/2), (x + w/2, y),
            (x, y - h/2), (x - w/2, y)
        ], edgecolor=edgecolor, facecolor=facecolor, lw=2.5, zorder=zorder)
    
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)
    
    if shape == "diamond":
        return (x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)
    return (x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)

def draw_arrow(start, end, text="", color="#555555", lw=2.5, rad=0.0, text_offset=(0,0), zorder_arrow=1, zorder_text=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"), zorder=zorder_arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=12, color=color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=zorder_text)

# ==========================================
# 核心节点坐标设计 (自上而下，带有右侧大回环)
# ==========================================
CX = 40
W, H = 26, 10

# 1. 输入层
IN_Y = 85
# 2. 计算原始效应量
CALC_Y = 65
# 3. 随机置换打乱
SHUFFLE_Y = 45
# 4. 判断循环次数 (菱形)
COND_Y = 25
# 5. 统计 P-Value 并输出
OUT_Y = 5

# ==========================================
# 绘制节点
# ==========================================
_, _, in_bot, _ = draw_box(CX, IN_Y, W, H, "输入处理后的张量\n(Target & Attribute Tensors)", "#F0F4FA", "#3C5488")

calc_top, calc_right, calc_bot, _ = draw_box(CX, CALC_Y, W, H, "计算原始效应量 d\n(Cosine Similarity Diff)", "#EEF7E8", "#448C55")

shuf_top, shuf_right, shuf_bot, _ = draw_box(CX, SHUFFLE_Y, W, H, "执行随机置换 (Permutation)\n打乱目标词与属性词映射", "#FFF4E6", "#D9822B")

cond_top, cond_right, cond_bot, cond_left = draw_box(CX, COND_Y, 26, 12, "置换次数\n是否达到 10,000 次？", "#FFF4E6", "#D9822B", shape="diamond")

out_top, _, _, _ = draw_box(CX, OUT_Y, W, H, "统计 P-Value 分布\n输出量化评估报告", "#EBE6F2", "#68589B")

# ==========================================
# 绘制连接箭头
# ==========================================
# 1 -> 2
draw_arrow(in_bot, calc_top)

# 2 -> 3
draw_arrow(calc_bot, shuf_top, text="进入显著性检验阶段", text_offset=(0, 0))

# 3 -> 4
draw_arrow(shuf_bot, cond_top, text="记录当前 d'", text_offset=(0, 0))

# 4 -> 3 (循环回环，代表 10000 次循环)
draw_arrow(cond_right, shuf_right, text="否 (No)\n继续下一次随机置换", color="#D9822B", rad=0.8, text_offset=(18, -4), zorder_arrow=1, zorder_text=3)

# 4 -> 5 (结束循环，输出结果)
draw_arrow(cond_bot, out_top, text="是 (Yes)\n计算完成", color="#448C55", text_offset=(0, 0))

os.makedirs('results', exist_ok=True)
plt.savefig('results/WRAT_Statistical_Significance_Flow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/WRAT_Statistical_Significance_Flow.png', dpi=300, bbox_inches='tight')
plt.savefig('results/WRAT_Statistical_Significance_Flow.svg', format='svg', bbox_inches='tight')
print("WRAT 模块效应量计算与显著性检验算法逻辑图生成完毕！保存在 results/ 目录下。")