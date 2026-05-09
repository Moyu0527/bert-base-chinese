import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 4-4 系统业务处理与实验验证闭环流程图", fontsize=24, fontweight='bold', y=0.95)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, shape="round", zorder=2):
    if shape == "round":
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.5",
            edgecolor=edgecolor, facecolor=facecolor,
            lw=2.5, zorder=zorder
        )
    elif shape == "diamond": # 菱形用于决策
        rect = patches.Polygon([
            (x, y + h/2 + 2), (x + w/2 + 2, y),
            (x, y - h/2 - 2), (x - w/2 - 2, y)
        ], edgecolor=edgecolor, facecolor=facecolor, lw=2.5, zorder=zorder)
    
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)
    
    if shape == "diamond":
        return (x, y + h/2 + 2), (x + w/2 + 2, y), (x, y - h/2 - 2), (x - w/2 - 2, y)
    return (x, y + h/2), (x + w/2, y), (x, y - h/2), (x - w/2, y)

def draw_arrow(start, end, text="", color="#444444", lw=2.5, rad=0.0, text_offset=(0,0)):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw, connectionstyle=f"arc3,rad={rad}"), zorder=1)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=13, color=color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=2)

# ==========================================
# 核心节点坐标设计 (环形拓扑)
# ==========================================
# 1. 输入阶段 (左上)
IN_X, IN_Y = 20, 75
# 2. 摸底探测 (右上)
P1_X, P1_Y = 80, 75
# 3. 干预决策 (右中，菱形)
DEC_X, DEC_Y = 80, 45
# 4. 微调实施 (左下)
FT_X, FT_Y = 20, 20
# 5. 效能评估 (右下)
EVAL_X, EVAL_Y = 80, 20

W, H = 24, 12

# ==========================================
# 绘制节点
# ==========================================
# [1] 输入阶段
in_top, in_right, in_bot, in_left = draw_box(IN_X, IN_Y, W, H, "【输入阶段】\n加载原始模型权重\n(bert-base-chinese)\n与本土化测试集", "#F0F4FA", "#3C5488")

# [2] 摸底探测
p1_top, p1_right, p1_bot, p1_left = draw_box(P1_X, P1_Y, W, H, "【摸底探测】\n执行 WRAT/SRAT 算法\n产出 Baseline 偏见报告", "#EEF7E8", "#448C55")

# [3] 干预决策 (菱形)
dec_top, dec_right, dec_bot, dec_left = draw_box(DEC_X, DEC_Y, 20, 10, "【干预决策】\n判定效应量 d\n是否 > 0.8 ?", "#FFF4E6", "#D9822B", shape="diamond")

# [4] 微调实施
ft_top, ft_right, ft_bot, ft_left = draw_box(FT_X, FT_Y, W, H, "【微调实施】\n在去偏语料库上\n执行 MLM 梯度反向传播", "#FCEBEB", "#A84144")

# [5] 效能评估
eval_top, eval_right, eval_bot, eval_left = draw_box(EVAL_X, EVAL_Y, W, H, "【效能评估】\n加载微调后模型\n对比雷达图面积收缩率", "#EBE6F2", "#68589B")

# ==========================================
# 绘制连接与闭环数据流
# ==========================================
# 1 -> 2 (输入送去探测)
draw_arrow(in_right, p1_left, text="初始化量化度量", text_offset=(0, 3))

# 2 -> 3 (探测结果送去决策)
draw_arrow(p1_bot, dec_top)

# 3 -> 4 (Yes: 需要微调，走左边)
draw_arrow(dec_left, ft_top, text="是 (Yes)\n偏见超标，进入微调", rad=-0.2, color="#A84144", text_offset=(-6, 4))

# 3 -> 5 (No: 偏见达标，直接走下面评估)
draw_arrow(dec_bot, eval_top, text="否 (No)\n偏见在安全域内", color="#448C55", text_offset=(-14, 0))

# 4 -> 5 (微调完送去评估)
draw_arrow(ft_right, eval_left, text="输出去偏新权重", text_offset=(0, 3))

# 5 -> 3 (闭环：评估完再送回决策进行判断)
# 画一条大弧线从底部的评估节点回到中间的决策节点
draw_arrow(eval_right, dec_right, text="二次度量反馈\n(闭环迭代)", rad=-0.6, color="#68589B", text_offset=(-18, 0))

# ==========================================
# 补充：输出出口 (达到去偏最终状态)
# ==========================================
OUT_X, OUT_Y = 50, 45
out_top, out_right, out_bot, out_left = draw_box(OUT_X, OUT_Y, 18, 8, "去偏后状态\n(输出最终报告)", "#E6F3F7", "#2B7B8C")

# 当决策为 No，或者闭环迭代达到阈值后，最终导向输出
draw_arrow(dec_left, out_right, text="完成迭代", color="#2B7B8C", text_offset=(6, 3))

os.makedirs('results', exist_ok=True)
plt.savefig('results/Experiment_Closed_Loop_Workflow.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Experiment_Closed_Loop_Workflow.png', dpi=300, bbox_inches='tight')
print("系统业务处理与实验验证闭环流程图生成完毕！保存在 results/ 目录下。")