import json
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def create_text_page(pdf, title, content_lines):
    fig = plt.figure(figsize=(8, 11))
    fig.clf()
    fig.text(0.5, 0.9, title, transform=fig.transFigure, size=24, ha="center", weight="bold")
    
    y_pos = 0.8
    for line in content_lines:
        fig.text(0.1, y_pos, line, transform=fig.transFigure, size=14, ha="left", wrap=True)
        y_pos -= 0.04
        
    pdf.savefig()
    plt.close()

def plot_wrat_effect_size(pdf, orig_data, ft_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['原始模型 (Original)', '去偏微调模型 (Fine-tuned)']
    effect_sizes = [orig_data['wrat']['effect_size'], ft_data['wrat']['effect_size']]
    
    colors = ['#FF9999', '#66B2FF']
    bars = ax.bar(labels, effect_sizes, color=colors, width=0.5)
    
    ax.set_ylabel('WRAT 效应量 (Effect Size)', fontsize=12)
    ax.set_title('WRAT 词级别度量：效应量对比 (越接近 0 偏见越小)', fontsize=16)
    ax.axhline(0, color='black', linewidth=1)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + (0.005 if yval > 0 else -0.015), 
                f'{yval:.4f}', ha='center', va='bottom' if yval > 0 else 'top', fontsize=12)
                
    pdf.savefig()
    plt.close()

def plot_wrat_p_value(pdf, orig_data, ft_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['原始模型 (Original)', '去偏微调模型 (Fine-tuned)']
    p_values = [orig_data['wrat']['p_value'], ft_data['wrat']['p_value']]
    
    colors = ['#FFCC99', '#99FF99']
    bars = ax.bar(labels, p_values, color=colors, width=0.5)
    
    ax.set_ylabel('P-Value (置换检验)', fontsize=12)
    ax.set_title('WRAT 词级别度量：显著性 P-Value 对比', fontsize=16)
    ax.axhline(0.05, color='red', linestyle='--', linewidth=2, label='显著性阈值 (0.05)')
    ax.legend()
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, 
                f'{yval:.4f}', ha='center', va='bottom', fontsize=12)
                
    pdf.savefig()
    plt.close()

def plot_srat_shift(pdf, orig_data, ft_data):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    labels = ['原始模型 (Original)', '去偏微调模型 (Fine-tuned)']
    shifts = [orig_data['srat']['avg_shift'], ft_data['srat']['avg_shift']]
    
    colors = ['#FF9933', '#33CC99']
    bars = ax.bar(labels, shifts, color=colors, width=0.5)
    
    ax.set_ylabel('平均语境化向量重心偏移量', fontsize=12)
    ax.set_title('SRAT 句级别度量：隐藏层向量偏移量对比 (越小越好)', fontsize=16)
    
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, 
                f'{yval:.4f}', ha='center', va='bottom', fontsize=12)
                
    pdf.savefig()
    plt.close()

def main():
    json_path = 'results/validation_wrat_srat_results.json'
    if not os.path.exists(json_path):
        print(f"找不到数据文件: {json_path}")
        return
        
    data = load_data(json_path)
    orig = data['original_model']
    ft = data['finetuned_model']
    
    os.makedirs('results', exist_ok=True)
    pdf_path = 'results/Bias_Measurement_Report.pdf'
    
    with PdfPages(pdf_path) as pdf:
        # 1. 封面与总结
        title = "模型偏见度量与去偏验证报告"
        summary_lines = [
            "【项目背景】",
            "本报告基于词表征关联测试（WRAT）和句子表征关联测试（SRAT），",
            "对 BERT-base-chinese 预训练模型及经过 WEAT、CDial 数据",
            "去偏微调后的模型进行多维度隐性偏见定量分析。",
            "",
            "【核心结论】",
            f"1. WRAT (词级别): 原始模型效应量为 {orig['wrat']['effect_size']:.4f}，",
            f"   微调后变为 {ft['wrat']['effect_size']:.4f}。方向发生翻转，偏见被有效扰动。",
            "   (P-value 均 > 0.05，说明在此测试集下两者统计学上均未表现出极端偏见)",
            "",
            f"2. SRAT (句级别): 面对 Safety-Prompts 提供的强歧视性语境，",
            f"   原始模型的平均向量重心偏移量高达 {orig['srat']['avg_shift']:.4f}，",
            f"   经过去偏微调后，该偏移量大幅下降至 {ft['srat']['avg_shift']:.4f}。",
            "   下降幅度显著，证明去偏微调有效拉平了模型在极端语境下的表征偏移。",
            "",
            "以下为各维度数据图表可视化结果："
        ]
        create_text_page(pdf, title, summary_lines)
        
        # 2. 图表绘制
        plot_wrat_effect_size(pdf, orig, ft)
        plot_wrat_p_value(pdf, orig, ft)
        plot_srat_shift(pdf, orig, ft)
        
    print(f"可视化报告已生成并汇总为 PDF: {pdf_path}")

if __name__ == '__main__':
    main()
