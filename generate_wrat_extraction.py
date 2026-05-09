import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(15, 6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
plt.suptitle("图 4-5 WRAT 模块词特征抽取与索引映射流程图", fontsize=20, fontweight='bold', y=0.95)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=12, shape="round", zorder=2):
    if shape == "round":
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=0.5",
            edgecolor=edgecolor, facecolor=facecolor,
            lw=2.5, zorder=zorder
        )
    elif shape == "cylinder":
        # 用圆角矩形模拟数据库/文件输入
        rect = patches.FancyBboxPatch(
            (x - w/2, y - h/2), w, h,
            boxstyle="round,pad=1",
            edgecolor=edgecolor, facecolor=facecolor,
            lw=2.5, zorder=zorder
        )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, fontweight='bold', linespacing=1.6)
    return (x - w/2, y), (x + w/2, y), (x, y + h/2), (x, y - h/2)

def draw_arrow(start, end, text="", color="#555555", lw=2.5, text_offset=(0,0), text2="", text2_offset=(0,0), zorder_arrow=1, zorder_text=2):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw), zorder=zorder_arrow)
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, fontsize=12, color=color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=zorder_text)
    if text2:
        mid_x2 = (start[0] + end[0]) / 2 + text2_offset[0]
        mid_y2 = (start[1] + end[1]) / 2 + text2_offset[1]
        ax.text(mid_x2, mid_y2, text2, fontsize=12, color=color, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', pad=2, alpha=0.9), zorder=zorder_text)

# ==========================================
# 左侧：输入层 (文件)
# ==========================================
IN_X = 15
_, in_right, _, _ = draw_box(IN_X, 50, 18, 30, "【输入层】\n\n词表文本文件\n(JSON / CSV)\n\n• 目标词 (如: 男/女)\n• 属性词 (如: 职业)", "#F0F4FA", "#3C5488", shape="cylinder")

# ==========================================
# 中间：处理层 (映射与抽取)
# ==========================================
MID_X1 = 45
MID_X2 = 70

# 虚线大框包裹中间处理层
rect_mid = patches.FancyBboxPatch(
    (32, 25), 48, 50,
    boxstyle="round,pad=1",
    edgecolor="#D9822B", facecolor="#FFF4E6", alpha=0.2,
    linestyle='--', lw=2, zorder=0
)
ax.add_patch(rect_mid)
ax.text(56, 70, "词嵌入特征提取引擎 (PyTorch + HuggingFace)", fontsize=14, fontweight='bold', color="#D9822B", ha='center', va='center')

left_t, right_t, _, _ = draw_box(MID_X1, 50, 16, 16, "Tokenizer\n映射层\n(Word -> ID)", "#FFF4E6", "#D9822B")
left_e, right_e, _, bot_e = draw_box(MID_X2, 50, 16, 16, "Embedding\n权重查找\n(ID -> Vector)", "#FFF4E6", "#D9822B")

# 全集归一化去噪模块 (可选挂载在下面)
draw_box(MID_X2, 30, 16, 8, "全集归一化\n(频率去噪)", "#EEF7E8", "#448C55")
# 连接查找 -> 去噪
draw_arrow((MID_X2, 42), (MID_X2, 34), color="#448C55", lw=2)

# ==========================================
# 右侧：输出层 (张量)
# ==========================================
OUT_X = 92
out_left, _, _, _ = draw_box(OUT_X, 50, 14, 30, "【输出层】\n\n可计算张量\n\n• Target Tensor\n• Attribute Tensor\n(FP16 / FP32)", "#EBE6F2", "#68589B")

# ==========================================
# 连接箭头
# ==========================================
# 输入 -> Tokenizer
draw_arrow(in_right, left_t, text="加载词汇", text_offset=(0, 0))

# Tokenizer -> Embedding
# “传入”在箭头上，“Token IDs”在箭头下
draw_arrow(right_t, left_e, text="传入", text_offset=(0, 3), text2="Token IDs", text2_offset=(0, -3))

# Embedding -> 输出 (直接水平连接，提高 zorder 确保穿透虚线框)
draw_arrow(right_e, (out_left[0], 50), text="抽取\n特征", text_offset=(0, 0), zorder_arrow=5, zorder_text=6)

# 全集归一化 (去噪后) -> 输出 (提高 zorder 确保穿透虚线框)
draw_arrow((MID_X2 + 8, 30), (out_left[0], 40), text="高维投影", color="#448C55", text_offset=(0, 0), zorder_arrow=5, zorder_text=6)

os.makedirs('results', exist_ok=True)
plt.savefig('results/WRAT_Feature_Extraction_Logic.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/WRAT_Feature_Extraction_Logic.png', dpi=300, bbox_inches='tight')
print("WRAT 模块词特征抽取与索引映射流程图生成完毕！保存在 results/ 目录下。")