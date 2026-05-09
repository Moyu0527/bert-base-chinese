import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 13))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

plt.suptitle("图 4-3 系统核心功能模块交互逻辑图", fontsize=24, fontweight='bold', y=0.96)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        lw=2.5, zorder=5
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=6, fontweight='bold', linespacing=1.5)
    return (x, y, w, h)

def draw_center_circle(x, y, r, text, facecolor, edgecolor):
    circle = patches.Circle((x, y), r, edgecolor=edgecolor, facecolor=facecolor, lw=3, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, fontsize=16, ha='center', va='center', color='#D9822B', zorder=4, fontweight='bold', linespacing=1.5)
    return (x, y, r)

def draw_arrow(start, end, text="", color="#555555", lw=2, rad=0.0, text_offset=(0,0), style="->", text_color=None):
    if text_color is None: text_color = color
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=12, color=text_color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=2)

# Coordinates (Pentagon layout)
CX, CY = 50, 50
CR = 14

D_X, D_Y = 50, 86   # Data (Top)
M_X, M_Y = 86, 55   # Measure (Right)
V_X, V_Y = 74, 18   # Visual (Bottom Right)
B_X, B_Y = 26, 18   # Debias (Bottom Left)
L_X, L_Y = 14, 55   # Model (Left)

W, H = 22, 10

# Center Engine
draw_center_circle(CX, CY, CR, "核心计算引擎\n(PyTorch / GPU\nAutograd & AMP)", "#FFF4E6", "#D9822B")

# Peripheral Modules
draw_box(D_X, D_Y, W, H, "数据处理模块\n(WEAT/CDial 语料)", "#F0F4FA", "#3C5488")
draw_box(M_X, M_Y, W, H, "偏见度量模块\n(WRAT/SRAT 评估)", "#EEF7E8", "#448C55")
draw_box(V_X, V_Y, W, H, "可视化模块\n(雷达图/多维投影)", "#EBE6F2", "#68589B")
draw_box(B_X, B_Y, W, H, "微调去偏模块\n(MLM 动态掩码)", "#FCEBEB", "#A84144")
draw_box(L_X, L_Y, W, H, "模型管理模块\n(HuggingFace IO)", "#E6F3F7", "#2B7B8C")

# Bidirectional arrows to center (Hardware/Tensor exchange)
draw_arrow((D_X, D_Y-H/2), (CX, CY+CR), style="<->", color="#CCCCCC", lw=2)
draw_arrow((M_X-W/2, M_Y), (CX+CR, CY+2), style="<->", color="#CCCCCC", lw=2)
draw_arrow((V_X-W/2+2, V_Y+H/2), (CX+CR*0.7, CY-CR*0.7), style="<->", color="#CCCCCC", lw=2)
draw_arrow((B_X+W/2-2, B_Y+H/2), (CX-CR*0.7, CY-CR*0.7), style="<->", color="#CCCCCC", lw=2)
draw_arrow((L_X+W/2, L_Y), (CX-CR, CY+2), style="<->", color="#CCCCCC", lw=2)

ax.text(CX, CY-CR-3, "【底层张量与显存调度总线】", fontsize=12, color="#999999", ha='center', zorder=2)

# ==========================================
# Logical Data Flows
# ==========================================

# 1. Data -> Measure
draw_arrow((D_X+W/2-2, D_Y-H/2), (M_X-W/4, M_Y+H/2), "提供 CrowS-Pairs\n基准测试集", rad=-0.2, text_offset=(6, 4))

# 2. Data -> Debias
draw_arrow((D_X-W/2, D_Y), (B_X-W/4, B_Y+H/2+2), "下发微调训练语料", rad=0.2, text_offset=(-10, 4))

# 3. Model -> Measure
draw_arrow((L_X+W/2, L_Y+H/2), (M_X-W/2, M_Y+H/2), "加载对比基线权重", rad=-0.2, text_offset=(0, 2))

# 4. Model -> Debias
draw_arrow((L_X, L_Y-H/2), (B_X-W/4, B_Y+H/2), "初始化基础权重", rad=-0.2, text_offset=(-6, -2))

# 5. Measure -> Visual
draw_arrow((M_X, M_Y-H/2), (V_X+W/4, V_Y+H/2), "输出量化评估 JSON", rad=-0.1, text_offset=(8, 0))

# 6. Debias -> Measure (CRITICAL FEEDBACK)
draw_arrow((B_X+W/2, B_Y+H/2), (M_X-W/2, M_Y-H/2), "反馈去偏新权重 (触发二次评估)", color="#A84144", lw=2.5, rad=0.2, text_offset=(0, -8))


os.makedirs('results', exist_ok=True)
plt.savefig('results/Module_Interaction_Logic.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Module_Interaction_Logic.png', dpi=300, bbox_inches='tight')
print("核心功能模块交互逻辑图生成完毕！保存在 results/ 目录下。")