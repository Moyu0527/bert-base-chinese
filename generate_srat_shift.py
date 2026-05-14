import matplotlib.pyplot as plt
import numpy as np
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 模拟数据生成 (模拟高维张量降维到 2D 的结果)
# ==========================================
np.random.seed(42)

# 模拟属性 A 的句子向量簇 (例如："他在...")
# 中心点设定在左下方
center_A = np.array([2.5, 3.0])
# 生成 30 个正态分布的散点
points_A = center_A + np.random.randn(30, 2) * 0.6

# 模拟属性 B 的句子向量簇 (例如："她在...")
# 中心点设定在右上方，形成明显的位移
center_B = np.array([7.5, 6.5])
# 生成 30 个正态分布的散点
points_B = center_B + np.random.randn(30, 2) * 0.6

# ==========================================
# 2. 绘制图形
# ==========================================
fig, ax = plt.subplots(figsize=(10, 8))

# 绘制点簇 A (蓝色，代表属性 A 引导的隐藏状态)
ax.scatter(points_A[:, 0], points_A[:, 1], c='#6fb4f6', s=60, alpha=0.7, edgecolors='white', label='属性 A 语境向量簇 (如: 他在...)')

# 绘制点簇 B (粉色，代表属性 B 引导的隐藏状态)
ax.scatter(points_B[:, 0], points_B[:, 1], c='#f69a9e', s=60, alpha=0.7, edgecolors='white', label='属性 B 语境向量簇 (如: 她在...)')

# 绘制中心点 C_A 和 C_B
ax.scatter(center_A[0], center_A[1], c='#1E4B8B', s=200, marker='X', edgecolors='white', zorder=5)
ax.scatter(center_B[0], center_B[1], c='#9C2B30', s=200, marker='X', edgecolors='white', zorder=5)

# 标注中心点文本
ax.text(center_A[0] - 0.3, center_A[1] + 0.3, r'$C_A$', fontsize=16, fontweight='bold', color='#1E4B8B')
ax.text(center_B[0] + 0.3, center_B[1] - 0.3, r'$C_B$', fontsize=16, fontweight='bold', color='#9C2B30')

# ==========================================
# 3. 绘制偏移量连线与标注
# ==========================================
# 绘制带有箭头的虚线，表示位移方向
ax.annotate("", xy=center_B, xytext=center_A,
            arrowprops=dict(arrowstyle="->", color="#333333", lw=2.5, linestyle="--", shrinkA=10, shrinkB=10))

# 计算连线的中点，用于放置标注文字
mid_x = (center_A[0] + center_B[0]) / 2
mid_y = (center_A[1] + center_B[1]) / 2

# 在连线中心偏左上的位置添加带白底的公式标注
ax.text(mid_x - 0.5, mid_y + 0.5, r'偏移量 $Score_{SRAT}$ = $||C_A - C_B||$', 
        fontsize=14, fontweight='bold', color='#222222',
        bbox=dict(facecolor='white', edgecolor='#cccccc', boxstyle='round,pad=0.5', alpha=0.9), zorder=6)

# ==========================================
# 4. 图形样式优化
# ==========================================
# 隐藏上边框和右边框，使其看起来像标准的数学坐标系
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 加粗左侧和下侧坐标轴
ax.spines['left'].set_linewidth(2)
ax.spines['bottom'].set_linewidth(2)

# 设置轴标签
ax.set_xlabel('Principal Component 1 (主成分 1)', fontsize=12, fontweight='bold')
ax.set_ylabel('Principal Component 2 (主成分 2)', fontsize=12, fontweight='bold')

# 设置图例
ax.legend(loc='lower right', fontsize=12, framealpha=0.9, edgecolor='#cccccc')

# 标题
# plt.title("图 4-7 SRAT 模块语境向量重心偏移计算示意图", fontsize=18, fontweight='bold', pad=20)

# 网格线
ax.grid(True, linestyle=':', alpha=0.5)

os.makedirs('results', exist_ok=True)
plt.savefig('results/SRAT_Vector_Shift_Diagram.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/SRAT_Vector_Shift_Diagram.png', dpi=300, bbox_inches='tight')
plt.savefig('results/SRAT_Vector_Shift_Diagram.svg', format='svg', bbox_inches='tight')
print("SRAT 模块语境向量重心偏移计算示意图生成完毕！保存在 results/ 目录下。")