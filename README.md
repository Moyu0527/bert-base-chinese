# 基于词嵌入与句编码的模型表征偏见度量方法实现

## 项目概述

本系统实现了完整的偏见度量方法，包括：
- **WRAT (Word Representation Association Test)**: 词级别的偏见度量
- **SRAT (Sentence Representation Association Test)**: 句级别的偏见度量
- **多维度偏见评估**: 性别、职业、地域、交叉偏见
- **PCA可视化**: 词向量在偏见轴上的投影

## 数据来源

### 1. WEATHub 官方数据集
- 位置: `data/weat/weat_official_zh.json`
- 来源: HuggingFace Hub (iamshnoo/WEATHub)
- 内容: 6种WEAT测试的中文词表
  - WEAT1: 职业vs评价
  - WEAT2: 职业vs刻板印象
  - WEAT6: 性别vs职业
  - WEAT7: 性别vs学科
  - WEAT8: 性别vs领域

### 2. CDial-Bias 对话数据集
- 位置: `data/cdial_bias/cdial_bias_official.json`
- 来源: HuggingFace Hub (para-zhou/CDial-Bias)
- 内容: 真实的中文对话，包含社会偏见
  - 性别偏见
  - 职业偏见
  - 地域偏见

### 3. 中文句子模板 (100+)
- 位置: `training_pipeline.py` (ChineseTemplateGenerator)
- 类别:
  - 性别职业 (30+)
  - 性别能力 (12+)
  - 地域刻板印象 (12+)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行训练/评估系统

```bash
python training_pipeline.py
```

这个系统会：
1. 加载 WEATHub + CDial-Bias 数据集
2. 准备数据用于 WRAT/SRAT 评估
3. 加载 100+ 中文句子模板
4. 运行偏见度量评估

### 运行完整的增强验证

```bash
python full_validation_enhanced.py
```

## 项目结构

```
bert-base-chinese/
├── config.py                      # 全局配置
├── requirements.txt               # 依赖包
├── main.py                        # 完整评估入口
├── full_validation.py             # 完整验证 (WRAT + SRAT)
├── full_validation_enhanced.py    # 增强版验证 (带 PCA/报告)
├── training_pipeline.py           # 训练/评估系统 (集成所有数据)
├── simple_visualizer.py           # 简单可视化
├── metrics/                       # 核心度量模块
│   ├── __init__.py
│   ├── wrat_metric.py             # WRAT 词级别评估
│   ├── srat_metric.py             # SRAT 句级别评估
│   ├── word_embeddings.py         # 词向量提取 (BERT)
│   └── noise_removal.py           # 去噪 (全局归一化)
├── templates/                     # 模板生成
│   ├── __init__.py
│   └── template_generator.py      # 中性化增强模板
├── data_processing/               # 数据处理
│   ├── __init__.py
│   ├── weat_preprocessor.py       # WEAT 中文映射
│   ├── cdial_parser.py            # CDial-Bias 解析
│   ├── crows_pairs_generator.py   # 对比语料生成
│   └── dataset_downloader.py      # 数据集下载
├── crawler/                       # 爬虫模块
├── lda/                           # LDA 主题建模
├── visualization/                 # 可视化模块
├── data/                          # 数据目录
│   ├── weat/                      # WEAT 数据
│   └── cdial_bias/                # CDial-Bias 数据
├── results/                       # 输出结果
│   ├── bias_assessment_report.txt # 量化评估报告
│   └── full_enhanced_results.json # 完整结果
└── test/                          # 测试文件
```

## 度量方法

### WRAT - 词表征关联测试

**核心公式:**

1. **单个词偏见得分 (Formula 3.1):**
   ```
   s(ω, A, B) = mean_{a∈A} cos(ω, a) - mean_{b∈B} cos(ω, b)
   ```

2. **集合测试统计量 (Formula 3.2):**
   ```
   s(X, Y, A, B) = Σ_{x∈X} s(x, A, B) - Σ_{y∈Y} s(y, A, B)
   ```

3. **效应量计算 (Formula 3.3):**
   ```
   d = [mean_X s(x) - mean_Y s(y)] / std_{X∪Y} s(ω)
   ```

**偏见程度判定标准:**
- d > 0.8: 强社会偏见 ⚠️
- 0.5 < d ≤ 0.8: 中等偏见
- d ≤ 0.5: 弱/无偏见

### SRAT - 句表征关联测试

**核心特性:**
- 使用 BERT 句嵌入
- 3种 Pooling 策略: CLS, Mean, Max
- 100+ 中文句子模板
- 中性化增强 (同义词替换、句法变换)

## 评估流程

1. **加载数据**: WEATHub + CDial-Bias + 中文模板
2. **提取词向量**: BERT-base-chinese (fp16)
3. **去噪处理**: 全局归一化 (Global Centering)
4. **WRAT评估**: 词级别偏见测试
5. **SRAT评估**: 句级别偏见测试
6. **PCA可视化**: 偏见轴投影 + 皮尔逊相关
7. **报告生成**: 量化评估报告

## 可视化

运行可视化:
```bash
python simple_visualizer.py
```

## 硬件优化

为了在 8GB 显存设备上稳定运行，系统采用:
- **fp16 半精度**加载模型
- **batch_size = 1-2**
- 推理后自动 **torch.cuda.empty_cache()**

## 输出示例

### 量化评估报告

```
================================================================================
                    偏见度量量化评估报告
================================================================================

一、WRAT 词表征关联测试结果
--------------------------------------------------------------------------------
测试数量: 1
效应量 d: 0.8542
p-value: 0.1525
PCA解释方差比: 45.00%
皮尔逊相关系数: 0.8200 (p=0.0010)

二、SRAT 句表征关联测试结果
--------------------------------------------------------------------------------
测试数量: 1
句子模板数量: 54+
支持Pooling策略: cls, mean, max

Pooling策略效果对比:
  - CLS Pooling: d=0.62, p=0.18
  - Mean Pooling: d=0.58, p=0.21
  - Max Pooling: d=0.55, p=0.25
```

## 参考文献

1. Caliskan et al. (2017). Semantics Derived Automatically from Language Corpora Contain Human-Like Biases. Science.
2. Nissim et al. (2020). Fair is Better Than Sensational: Man is to Doctor as Woman is to Doctor. ACL.
3. WEATHub (2023). Global Voices, Local Biases. EMNLP.
4. CDial-Bias (2022). Towards Identifying Social Bias in Dialog Systems. ACL Findings.

## 许可证

本项目采用 MIT 许可证。
