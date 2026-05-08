import matplotlib.pyplot as plt
import os

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

import json

# 读取最新的 JSON 语料统计数据
with open('data/crows_pairs_chinese_annotated.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

counts = {"gender": 0, "region": 0, "profession": 0, "race": 0}
for item in data:
    counts[item["bias_type"]] += 1

total = len(data)

# 准备数据
labels = [
    f'性别偏见\n(Gender)\n{counts["gender"]}条', 
    f'地域偏见\n(Region)\n{counts["region"]}条', 
    f'职业偏见\n(Profession)\n{counts["profession"]}条', 
    f'民族/种族偏见\n(Race)\n{counts["race"]}条'
]
sizes = [counts["gender"], counts["region"], counts["profession"], counts["race"]]
# 使用高对比度且符合学术审美的莫兰迪色系
colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B2']
explode = (0.05, 0, 0, 0)  # 将占比最高的性别偏见稍微突出显示

# 创建图表
fig, ax = plt.subplots(figsize=(10, 8))

# 过滤掉数量为0的数据以避免饼图绘制警告
valid_indices = [i for i, s in enumerate(sizes) if s > 0]
valid_labels = [labels[i] for i in valid_indices]
valid_sizes = [sizes[i] for i in valid_indices]
valid_colors = [colors[i] for i in valid_indices]
valid_explode = [explode[i] for i in valid_indices]

# 绘制环形图 (Donut Chart)
wedges, texts, autotexts = ax.pie(
    valid_sizes, 
    explode=valid_explode, 
    labels=valid_labels, 
    colors=valid_colors, 
    autopct='%1.1f%%',
    shadow=False, 
    startangle=90,
    pctdistance=0.75,
    textprops={'fontsize': 14, 'fontweight': 'bold'},
    wedgeprops={'width': 0.4, 'edgecolor': 'w', 'linewidth': 2}
)

# 调整标签文字属性
for text in texts:
    text.set_fontsize(13)
    text.set_color('#333333')

for autotext in autotexts:
    autotext.set_color('white')

# 添加中心文字
ax.text(0, 0, f'核心测试集\n总计: {total}条', ha='center', va='center', fontsize=16, fontweight='bold', color='#333333')

# 设置标题
plt.title("图 3-3 核心测试集偏见维度分布比例图", fontsize=18, fontweight='bold', pad=20)

# 确保输出目录存在
os.makedirs('results', exist_ok=True)

# 保存图片
plt.savefig('results/Bias_Dimension_Distribution.pdf', dpi=300, bbox_inches='tight')
plt.savefig('results/Bias_Dimension_Distribution.png', dpi=300, bbox_inches='tight')

print("偏见维度分布比例图生成完毕！保存在 results/ 目录下。")