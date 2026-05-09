import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 9))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 4-8 MLM 去偏微调模块执行流程图", fontsize=22, fontweight='bold', y=0.96)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, zorder=2, ls='-'):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        lw=2.5, linestyle=ls, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)
    return (x - w/2, y), (x + w/2, y), (x, y + h/2), (x, y - h/2)

def draw_arrow(start, end, text="", color="#555555", lw=2.5, rad=0.0, text_offset=(0,0), zorder_arrow=1, zorder_text=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"), zorder=zorder_arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=12, color=color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=zorder_text)

# ==========================================
# 核心节点坐标设计 (水平布局，带反向传播回环)
# ==========================================
IN_X, IN_Y = 15, 50
BERT_X, BERT_Y = 50, 50
OUT_X, OUT_Y = 85, 50

W, H = 22, 14

# ==========================================
# 绘制输入层
# ==========================================
in_top, in_right, in_bot, in_left = draw_box(IN_X, IN_Y, W, 20, "【输入端】\n\n加载本土化平衡语料\n(CrowS-Pairs)\n\n例:\n'她是 [MASK]'\n'他是 [MASK]'", "#F0F4FA", "#3C5488")

# ==========================================
# 绘制核心层 (BERT Encoder)
# ==========================================
# 背景虚线框
rect_mid = patches.FancyBboxPatch(
    (BERT_X - 16, BERT_Y - 20), 32, 40,
    boxstyle="round,pad=1",
    edgecolor="#D9822B", facecolor="#FFF4E6", alpha=0.3,
    linestyle='--', lw=2.5, zorder=0
)
ax.add_patch(rect_mid)
ax.text(BERT_X, BERT_Y + 22, "【核心层】BERT 模型微调引擎", fontsize=15, fontweight='bold', color="#D9822B", ha='center', va='bottom')

# 内部 Encoder 堆叠 (本项目中没有冻结下层权重，而是全量微调并使用 AMP)
b1_top, b1_right, b1_bot, b1_left = draw_box(BERT_X, BERT_Y - 10, 24, 8, "AMP 混合精度与梯度缩放\n(GradScaler & Autocast)", "#FFF4E6", "#D9822B")
b2_top, b2_right, b2_bot, b2_left = draw_box(BERT_X, BERT_Y + 10, 24, 8, "全层 Encoder (梯度更新)\n(Layer 1-12)", "#FCEBEB", "#A84144")

# ==========================================
# 绘制前向传播连接线 (Forward Pass)
# ==========================================
# 输入 -> AMP 层 (直线)
draw_arrow((in_right[0], BERT_Y - 10), b1_left, text="前向传播\n(Forward)", text_offset=(0, 2))

# AMP 层 -> BERT 编码器 (直线，垂直向上)
draw_arrow(b1_top, b2_bot, text="FP16\n计算流", text_offset=(3, 0))

# BERT 编码器 -> 输出端 (直线)
draw_arrow(b2_right, (out_left[0], BERT_Y + 10), text="输出表征", text_offset=(0, 2))

# ==========================================
# 绘制反向传播回环 (Backward Pass)
# ==========================================
# 输出端 -> AMP层 (计算梯度，反向大回环，指向 AMP 层进行 Unscale)
draw_arrow((out_top[0], out_top[1] + 2), (b1_top[0] + 6, b1_top[1] + 2), text="Loss 梯度反向传播\n(Unscale & Clip Gradients)", color="#A84144", lw=3, rad=-0.4, text_offset=(10, 15), zorder_arrow=5, zorder_text=6)

# BERT 内部梯度更新循环标识 (指向红色全层Encoder自身)
draw_arrow((b2_left[0] - 2, b2_left[1] + 2), (b2_left[0], b2_left[1] - 2), text="全量\n微调", color="#A84144", lw=2, rad=1.5, text_offset=(-6, 0), zorder_arrow=5, zorder_text=6)

os.makedirs('results', exist_ok=True)
plt.savefig('results/MLM_FineTuning_Workflow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/MLM_FineTuning_Workflow.png', dpi=300, bbox_inches='tight')
print("MLM 去偏微调模块执行流程图生成完毕！保存在 results/ 目录下。")