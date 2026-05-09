import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 4-1 系统硬件适配与显存优化逻辑示意图", fontsize=24, fontweight='bold', y=0.96)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, ls='-', zorder=2):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        linestyle=ls, lw=2.5, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)
    return (x - w/2, y), (x + w/2, y), (x, y + h/2), (x, y - h/2)

def draw_arrow(start, end, text="", color="#666666", lw=2.5, y_offset=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + y_offset, text, fontsize=12, color=color, ha='center', va='bottom', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=1))

# ==========================================
# 左侧：输入特征层
# ==========================================
left_in, right_in, top_in, bot_in = draw_box(15, 50, 22, 40, "输入特征层\n\n• BERT 预训练模型\n  (768维隐状态)\n\n• 偏见评测语料\n  (CrowS-Pairs)\n\n• 批次数据输入\n  (Batch Tensors)", "#F0F4FA", "#3C5488", fontsize=14)

# ==========================================
# 中间：优化引擎层
# ==========================================
# 绘制中间的虚线大框
rect_opt = patches.FancyBboxPatch(
    (32, 15), 36, 70,
    boxstyle="round,pad=1",
    edgecolor="#D9822B", facecolor="#FFF4E6", alpha=0.3,
    linestyle='--', lw=2.5, zorder=0
)
ax.add_patch(rect_opt)
ax.text(50, 80, "受限算力优化引擎层 (适配 8GB VRAM)", fontsize=16, fontweight='bold', color="#D9822B", ha='center', va='center')

left_opt1, right_opt1, _, _ = draw_box(50, 65, 30, 10, "AMP 自动混合精度\n(FP32 权重 + FP16 计算)", "#FFFFFF", "#D9822B", fontsize=13)
left_opt2, right_opt2, _, _ = draw_box(50, 45, 30, 10, "动态显存回收机制\n(empty_cache & GC 释放碎片)", "#FFFFFF", "#D9822B", fontsize=13)
left_opt3, right_opt3, _, _ = draw_box(50, 25, 30, 10, "梯度范数裁剪\n(防止微调梯度爆炸)", "#FFFFFF", "#D9822B", fontsize=13)

# ==========================================
# 右侧：硬件资源层
# ==========================================
left_out, right_out, top_out, bot_out = draw_box(85, 50, 22, 40, "硬件资源层\n\n[GPU 状态评估]\n\n[+] 显存节省 >40%\n\n[+] 计算吞吐量提升 30%\n\n[+] 稳定执行高维投影", "#EEF7E8", "#448C55", fontsize=14)

# ==========================================
# 绘制连接箭头
# ==========================================
draw_arrow(right_in, left_opt1)
draw_arrow(right_in, left_opt2)
draw_arrow(right_in, left_opt3)

# 对右侧指向资源层的箭头分别设置不同的 Y 轴偏移量，防止文字遮挡箭头或过于偏离
draw_arrow(right_opt1, left_out, text="降低内存分配", y_offset=4)
draw_arrow(right_opt2, left_out, text="提供连续显存", y_offset=2)
draw_arrow(right_opt3, left_out, text="稳定显存曲线", y_offset=-6)

os.makedirs('results', exist_ok=True)
plt.savefig('results/Hardware_Optimization_Logic.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Hardware_Optimization_Logic.png', dpi=300, bbox_inches='tight')
print("系统硬件适配与显存优化逻辑示意图生成完毕！保存在 results/ 目录下。")