import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

# 配置中文字体，确保在不同系统下都能正常显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(16, 7))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# 全局标题
# plt.suptitle("图 3-1 中文偏见数据集构建与清洗流程图", fontsize=22, fontweight='bold', y=0.92)

def draw_box(x, y, w, h, text, facecolor, edgecolor, fontsize=13, zorder=2):
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.5",
        edgecolor=edgecolor, facecolor=facecolor,
        lw=2.5, zorder=zorder
    )
    ax.add_patch(rect)
    ax.text(x, y, text, fontsize=fontsize, ha='center', va='center', color='#222222', zorder=zorder+1, linespacing=1.6)
    return (x - w/2, y), (x + w/2, y), (x, y + h/2), (x, y - h/2)

def draw_arrow(start, end, color="#666666", lw=2.5):
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))

def draw_step_bg(x, y, w, h, title, color):
    """绘制每个大阶段的虚线背景框"""
    rect = patches.FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=1",
        edgecolor=color, facecolor=color, alpha=0.08,
        linestyle='--', lw=2, zorder=0
    )
    ax.add_patch(rect)
    # 将标题向上移动
    ax.text(x, y + h/2 + 5, title, fontsize=15, fontweight='bold', color=color, ha='center', va='bottom', zorder=1)

# ==========================================
# 第一阶段：多源数据输入
# ==========================================
# 稍微降低背景框的高度以给上方文字留出空间，并下移一点位置
draw_step_bg(15, 42, 24, 60, "第一阶段\n多源原始数据输入", "#3C5488")
_, right_d1, _, _ = draw_box(15, 60, 22, 10, "WEAT 英文词表\n(基础属性与目标词)", "#F0F4FA", "#3C5488")
_, right_d2, _, _ = draw_box(15, 40, 22, 10, "CDial-Bias 语料\n(中文社交真实对话)", "#F0F4FA", "#3C5488")
_, right_d3, _, _ = draw_box(15, 20, 22, 10, "Safety-Prompts\n(安全伦理测试场景)", "#F0F4FA", "#3C5488")

# ==========================================
# 第二阶段：数据处理核心流 (清洗 -> 增强 -> 校验)
# ==========================================
draw_step_bg(50, 42, 40, 60, "第二阶段\n数据处理与语义工程", "#448C55")

# 数据清洗
left_c1, right_c1, _, _ = draw_box(38, 50, 16, 32, "数据清洗\n\n- 中文映射\n- 正则表达式过滤\n- 异常字符去除\n- 长度标准化", "#EEF7E8", "#448C55", fontsize=12)

# 语义增强
left_e1, right_e1, _, _ = draw_box(62, 50, 16, 32, "语义增强\n\n- 同义词替换\n- 动态模板生成\n- 语境多样性扩充\n- 中性化处理", "#EEF7E8", "#448C55", fontsize=12)

# 人工校对 (位于下方)
left_m1, right_m1, top_m1, bot_m1 = draw_box(50, 20, 38, 10, "专家人工校对 / 标注\n(交叉验证逻辑一致性)", "#FFF4E6", "#D9822B", fontsize=13)

# 连接 第一阶段 -> 数据清洗
draw_arrow(right_d1, (left_c1[0], 60))
draw_arrow(right_d2, (left_c1[0], 40))
draw_arrow(right_d3, (left_c1[0], 20))

# 连接 清洗 -> 增强
draw_arrow(right_c1, left_e1)

# 连接 增强 <-> 校对 (双向迭代循环)
draw_arrow((62, 34), (62, 25)) # 增强下发到校对
draw_arrow((38, 25), (38, 34)) # 校对打回给清洗
ax.text(64, 29.5, "审核", fontsize=11, color="#666666")
ax.text(34, 29.5, "重构", fontsize=11, color="#666666")

# ==========================================
# 第三阶段：输出层
# ==========================================
draw_step_bg(85, 42, 24, 60, "第三阶段\n基准数据集输出", "#68589B")

# 调整了 y 坐标以居中，并加大了高度 (28 -> 32)
left_out, _, _, bot_out = draw_box(85, 42, 22, 38, "CrowS-Pairs 格式\n中文偏见数据集\n\n四大维度：\n1. 性别偏见\n2. 职业偏见\n3. 地域偏见\n4. 民族/种族偏见", "#EBE6F2", "#68589B", fontsize=12)

# 增强 -> 输出
draw_arrow(right_e1, left_out)

os.makedirs('results', exist_ok=True)
plt.savefig('results/Data_Processing_Pipeline.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Data_Processing_Pipeline.png', dpi=300, bbox_inches='tight')
plt.savefig('results/Data_Processing_Pipeline.svg', format='svg', bbox_inches='tight')
print("数据清洗流程图生成完毕！保存在 results/ 目录下。")
