# LAMBADA Dataset

LAMBADA (LAnguage Modeling Broadened to Account for Discourse Aspects) 是一个专门设计用来测试语言模型理解上下文能力的数据集。

## 数据集描述

- **任务**: 预测段落的最后一个词，需要理解整个上下文
- **测试集**: 5,153个样本
- **评估指标**: 准确率 (Accuracy)
- **特点**: 需要理解长距离依赖关系和语义上下文

## 使用方法

运行准备脚本:
```bash
python prepare.py
```

这会:
1. 下载LAMBADA测试集
2. 使用GPT-2分词器处理数据
3. 保存为numpy格式用于评估

## 评估

使用 `evaluate_lambada.py` 脚本来评估模型性能。 