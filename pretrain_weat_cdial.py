#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文件用途：预训练阶段执行脚本
主要功能：
1. 从 WEAT 和 CDial-Bias 数据集中提取文本数据，并进行清洗与去重，生成预训练语料。
2. 使用基于 BERT 的 Masked Language Modeling (MLM) 任务对模型进行微调（去偏）。
3. 严格遵循硬件约束：使用 fp16 半精度加载，设置 batch_size=2，每个 epoch 结束后强制调用 torch.cuda.empty_cache() 清理显存，确保在 8GB 显存设备上稳定运行。
"""
import os
import sys

import json

import torch
import pandas as pd
from transformers import BertTokenizer, BertForMaskedLM
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

# 配置参数
MODEL_NAME = "bert-base-chinese"
OUTPUT_DIR = "models/bert-base-chinese-weat-cdial"
BATCH_SIZE = 2
EPOCHS = 3  # 将 Epochs 提高到 3，确保模型能够充分学习去偏语料
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_pretrain_data():
    """提取WEAT和CDial内容并保存为预训练数据"""
    print("开始提取WEAT和CDial数据...")
    corpus = []
    
    # 1. 提取WEAT数据
    weat_path = "data/weat/weat_official_zh.json"
    if os.path.exists(weat_path):
        with open(weat_path, 'r', encoding='utf-8') as f:
            weat_data = json.load(f)
        for item in weat_data:
            # 组合WEAT词汇成句子，或者直接将词汇作为文本加入
            for word_list in [item.get('targ1_examples', []), item.get('targ2_examples', []),
                              item.get('attr1_examples', []), item.get('attr2_examples', [])]:
                for word in word_list:
                    corpus.append(f"这是一个词汇：{word}。")
    
    # 2. 提取CDial-Bias数据
    cdial_path = "CDial-Bias/train.csv"
    if os.path.exists(cdial_path):
        df = pd.read_csv(cdial_path)
        for _, row in df.iterrows():
            q = str(row.get('q', ''))
            a = str(row.get('a', ''))
            context = str(row.get('context', ''))
            
            if q and q != 'nan': corpus.append(q)
            if a and a != 'nan': corpus.append(a)
            if context and context != 'nan': corpus.append(context)
            
    # 清洗和去重
    corpus = list(set([text.strip() for text in corpus if len(text.strip()) > 3]))
    
    # 为了演示和在8GB显存上快速验证，这里如果数据量太大可以进行截断
    # 但保留全部数据进行保存
    os.makedirs("data", exist_ok=True)
    corpus_path = "data/pretrain_corpus_weat_cdial.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in corpus:
            f.write(line + "\n")
            
    print(f"预训练数据生成完毕，共 {len(corpus)} 条，保存在 {corpus_path}")
    
    # 限制预训练数据量以缩短时间（取前2000条即可演示去偏训练效果）
    if len(corpus) > 2000:
        corpus = corpus[:2000]
        print(f"为了加快预训练演示，截取前 {len(corpus)} 条数据进行训练")
        
    return corpus

class MLMDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.texts = texts
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(
            text, 
            max_length=self.max_length, 
            padding="max_length", 
            truncation=True, 
            return_tensors="pt"
        )
        
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)
        
        # MLM掩码逻辑：优化向量化操作，避免 for 循环带来的 CPU 瓶颈
        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        # 忽略[CLS], [SEP], [PAD]
        mask_arr = (rand < 0.15) & (input_ids != self.tokenizer.cls_token_id) & \
                   (input_ids != self.tokenizer.sep_token_id) & (input_ids != self.tokenizer.pad_token_id)
        
        # 80% 的概率替换为 [MASK]
        mask_80_idx = mask_arr & (torch.rand(input_ids.shape) < 0.8)
        input_ids[mask_80_idx] = self.tokenizer.mask_token_id
        
        # 10% 的概率替换为随机词
        mask_10_idx = mask_arr & ~mask_80_idx & (torch.rand(input_ids.shape) < 0.5)
        random_words = torch.randint(1, self.tokenizer.vocab_size, input_ids.shape)
        input_ids[mask_10_idx] = random_words[mask_10_idx]
        
        # 其余 10% 保持不变
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def pretrain():
    corpus = generate_pretrain_data()
    
    print(f"当前使用的训练设备: {DEVICE}")
    print(f"加载模型和分词器 {MODEL_NAME} (AMP 混合精度)...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    # 训练阶段必须以默认(FP32)加载主权重，由 autocast 自动将前向传播转为 FP16。
    # 如果强行用 FP16 加载主权重，会导致 GradScaler 无法放大梯度（报错 Attempting to unscale FP16 gradients）
    model = BertForMaskedLM.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    
    dataset = MLMDataset(corpus, tokenizer)
    # CPU 训练时，必须设置 num_workers=0
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    
    # 引入 GradScaler 防止 FP16 训练时的梯度下溢/溢出导致 NaN
    scaler = torch.amp.GradScaler('cuda' if DEVICE == 'cuda' else 'cpu', enabled=(DEVICE == 'cuda'))
    
    model.train()
    print("开始预训练...")
    for epoch in range(EPOCHS):
        total_loss = 0
        
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # 解决 CPU 上 FP16 的兼容性问题：如果在 CPU 上运行，模型必须转为 float32
            if DEVICE == "cpu":
                model.to(torch.float32)
                
            # 使用混合精度 (AMP) 进行前向传播，保证交叉熵损失计算在 FP32 空间，防止 nan
            with torch.amp.autocast('cuda' if DEVICE == 'cuda' else 'cpu', enabled=(DEVICE == 'cuda')):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
            
            # 缩放 loss 并反向传播
            if DEVICE == 'cuda':
                scaler.scale(loss).backward()
                # 梯度裁剪，防止梯度爆炸导致 nan
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:  # 恢复为每 10 步打印一次
                print(f"Epoch {epoch+1}/{EPOCHS}, Step {step}, Loss: {loss.item():.4f}", flush=True)
            
            # 及时释放不再使用的 tensor
            del input_ids, attention_mask, labels, outputs, loss
            
        # 核心约束：循环后强制执行显存清理
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
                
        print(f"Epoch {epoch+1} 平均损失: {total_loss / len(dataloader):.4f}")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"预训练模型已保存至 {OUTPUT_DIR}")

if __name__ == "__main__":
    pretrain()
