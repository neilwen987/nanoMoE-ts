# WikiText-2 Dataset

WikiText-2 是用于语言建模评估的经典数据集，主要用于测量困惑度 (perplexity)。

## 数据集描述

- **来源**: 维基百科文章的高质量子集
- **分割**: 训练集、验证集、测试集
- **评估指标**: 困惑度 (Perplexity)
- **特点**: 真实的、多样化的英语文本

## 数据统计

处理后：
- 训练集: ~2M tokens
- 验证集: ~200K tokens  
- 测试集: ~240K tokens

## 使用方法

运行准备脚本:
```bash
python prepare.py
```

这会:
1. 下载WikiText-2数据集
2. 清理和预处理文本
3. 使用GPT-2分词器tokenize
4. 保存为二进制格式用于高效加载

## 评估

使用 `evaluate_wikitext2.py` 脚本来评估模型困惑度。 