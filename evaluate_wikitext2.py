"""
Evaluate models on the WikiText dataset.
WikiText is used for language modeling evaluation, measuring perplexity.
Supports multiple dataset variants: wikitext-2-v1, wikitext-2-raw-v1, wikitext-103-v1, wikitext-103-raw-v1
"""

import os
import math
import pickle
import numpy as np
import torch
from contextlib import nullcontext
from tqdm import tqdm
import argparse

from model import GPTConfig, GPT

# Available WikiText dataset variants
AVAILABLE_DATASETS = {
    'wikitext-2-v1': 'wikitext-2-v1',
    'wikitext-2-raw-v1': 'wikitext-2-raw-v1', 
    'wikitext-103-v1': 'wikitext-103-v1',
    'wikitext-103-raw-v1': 'wikitext-103-raw-v1'
}

def load_wikitext_data_from_huggingface(dataset_name='wikitext-2-v1', split='test'):
    """Load WikiText data from Hugging Face datasets"""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("请安装 datasets 库: pip install datasets")
    
    # Load the specific dataset variant
    print(f"从 Hugging Face 加载 {dataset_name} 数据集的 {split} 分割...")
    dataset = load_dataset("wikitext", dataset_name, split=split, cache_dir="data/wikitext2/.cache")
    
    # Extract text and concatenate
    texts = []
    for example in dataset:
        if example['text'].strip():  # Skip empty lines
            texts.append(example['text'].strip())
    
    # Join all text with newlines
    full_text = '\n'.join(texts)
    
    return full_text

def tokenize_and_prepare_data(text, tokenizer_name="gpt2"):
    """Tokenize text and prepare for evaluation"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(tokenizer_name)
    except ImportError:
        raise ImportError("请安装 tiktoken 库: pip install tiktoken")
    
    # Tokenize the text
    tokens = enc.encode(text)
    
    # Convert to numpy array
    data = np.array(tokens, dtype=np.uint16)
    
    # Create meta info
    meta = {
        'vocab_size': enc.n_vocab,
        'tokenizer': tokenizer_name
    }
    
    return data, meta

def load_wikitext_data(data_dir='data/wikitext2', dataset_name='wikitext-2-v1', split='test'):
    """Load WikiText data - supports both local binary files and HuggingFace datasets"""
    
    # First try to load from local binary files (legacy format)
    data_path = os.path.join(data_dir, f'{split}.bin')
    meta_path = os.path.join(data_dir, 'meta.pkl')
    
    if os.path.exists(data_path) and os.path.exists(meta_path):
        print(f"从本地二进制文件加载 {split} 数据...")
        # Load binary data
        data = np.memmap(data_path, dtype=np.uint16, mode='r')
        
        # Load meta info
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        
        return data, meta
    
    # If binary files don't exist, load from HuggingFace and tokenize on the fly
    else:
        print(f"本地文件不存在，从 HuggingFace 加载 {dataset_name} 数据集...")
        
        # Validate dataset name
        if dataset_name not in AVAILABLE_DATASETS:
            raise ValueError(f"不支持的数据集: {dataset_name}. 可用数据集: {list(AVAILABLE_DATASETS.keys())}")
        
        # Load text data
        text = load_wikitext_data_from_huggingface(dataset_name, split)
        
        # Tokenize and prepare
        data, meta = tokenize_and_prepare_data(text)
        
        print(f"加载完成: {len(data)} 个 tokens")
        return data, meta

def get_batch_wikitext(data, block_size, batch_size, device):
    """Get a batch of data from WikiText"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if 'cuda' in str(device):
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    
    return x, y

def evaluate_wikitext(model, device, data, meta, eval_iters=100, batch_size=1):
    """
    Evaluate model on WikiText dataset.
    Returns perplexity calculated from cross-entropy loss.
    """
    model.eval()
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    losses = []
    
    with torch.no_grad():
        for _ in tqdm(range(eval_iters), desc="评估中"):
            X, Y = get_batch_wikitext(data, model.config.block_size, batch_size, device)
            
            with ctx:
                logits, loss = model(X, Y)
                losses.append(loss.item())
    
    # Calculate average loss and perplexity
    avg_loss = np.mean(losses)
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

def evaluate_full_sequence(model, device, data, max_seq_len=1024):
    """
    Evaluate model on the full sequence without batching.
    This gives a more accurate perplexity measurement.
    """
    model.eval()
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    total_loss = 0
    total_tokens = 0
    
    # Process data in chunks
    data_len = len(data)
    stride = max_seq_len // 2  # Overlapping windows
    
    with torch.no_grad():
        for i in tqdm(range(0, data_len - max_seq_len, stride), desc="全序列评估"):
            # Get sequence
            seq = data[i:i+max_seq_len]
            x = torch.from_numpy(seq[:-1].astype(np.int64)).unsqueeze(0).to(device)
            y = torch.from_numpy(seq[1:].astype(np.int64)).unsqueeze(0).to(device)
            
            with ctx:
                logits, loss = model(x, y)
                
                # Accumulate loss
                total_loss += loss.item() * (max_seq_len - 1)
                total_tokens += (max_seq_len - 1)
    
    # Calculate average loss and perplexity
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    perplexity = math.exp(avg_loss)
    
    return perplexity, avg_loss

def evaluate_all_datasets(model, device, data_dir='data/wikitext2', splits=['test'], eval_iters=100, batch_size=1, full_sequence=False, max_seq_len=1024):
    """评估所有WikiText数据集变体"""
    results = {}
    
    for dataset_name in AVAILABLE_DATASETS.keys():
        print(f"\n{'='*60}")
        print(f"评估数据集: {dataset_name}")
        print(f"{'='*60}")
        
        dataset_results = {}
        
        for split in splits:
            print(f"\n评估 {split} 分割...")
            try:
                # Load data
                data, meta = load_wikitext_data(data_dir, dataset_name, split)
                print(f"加载了 {len(data)} 个 tokens")
                
                # Evaluate
                if full_sequence:
                    perplexity, loss = evaluate_full_sequence(model, device, data, max_seq_len)
                else:
                    perplexity, loss = evaluate_wikitext(model, device, data, meta, eval_iters, batch_size)
                
                dataset_results[split] = {
                    'perplexity': perplexity,
                    'loss': loss,
                    'num_tokens': len(data)
                }
                
                print(f"{dataset_name} {split} - 损失: {loss:.4f}, 困惑度: {perplexity:.2f}")
                
            except Exception as e:
                print(f"评估 {dataset_name} {split} 时出错: {e}")
                dataset_results[split] = {'error': str(e)}
        
        results[dataset_name] = dataset_results
    
    return results

def main():
    parser = argparse.ArgumentParser(description='在WikiText数据集上评估模型')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--data_dir', type=str, default='data/wikitext2/wikitext2_data',
                       help='WikiText数据目录路径')
    parser.add_argument('--dataset', type=str, default='wikitext-2-v1',
                       choices=list(AVAILABLE_DATASETS.keys()),
                       help='要评估的数据集变体')
    parser.add_argument('--split', type=str, default='test', 
                       choices=['train', 'validation', 'test'],
                       help='要评估的数据分割')
    parser.add_argument('--eval_iters', type=int, default=100,
                       help='评估迭代次数')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='评估批次大小')
    parser.add_argument('--device', type=str, default='cuda',
                       help='评估使用的设备')
    parser.add_argument('--full_sequence', action='store_true',
                       help='使用全序列评估以获得更准确的困惑度')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                       help='全序列评估的最大序列长度')
    parser.add_argument('--eval_all', action='store_true',
                       help='评估所有数据集变体')
    parser.add_argument('--splits', nargs='+', default=['test'],
                       choices=['train', 'validation', 'test'],
                       help='要评估的数据分割列表')
    
    args = parser.parse_args()
    
    # Load model
    print(f"从 {args.model_path} 加载模型")
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    if args.model_path.endswith('.pt'):
        # Load from checkpoint
        checkpoint = torch.load(args.model_path, map_location=device)
        model_args = checkpoint['model_args']
        
        # Create model
        gptconf = GPTConfig(**model_args)
        model = GPT(gptconf)
        
        # Load state dict
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        
    else:
        # Load pretrained model (e.g., 'gpt2', 'gpt2-medium', etc.)
        model = GPT.from_pretrained(args.model_path, dict(dropout=0.0))
    
    model.to(device)
    model.eval()
    
    print(f"模型已加载。参数数量: {model.get_num_params()/1e6:.2f}M")
    
    # Evaluate
    if args.eval_all:
        print("开始评估所有WikiText数据集变体...")
        results = evaluate_all_datasets(
            model, device, args.data_dir, args.splits, 
            args.eval_iters, args.batch_size, args.full_sequence, args.max_seq_len
        )
        
        # Print summary
        print(f"\n{'='*80}")
        print("评估结果汇总")
        print(f"{'='*80}")
        
        for dataset_name, dataset_results in results.items():
            print(f"\n{dataset_name}:")
            for split, split_results in dataset_results.items():
                if 'error' in split_results:
                    print(f"  {split}: 错误 - {split_results['error']}")
                else:
                    print(f"  {split}: 损失={split_results['loss']:.4f}, 困惑度={split_results['perplexity']:.2f}, 词元数={split_results['num_tokens']}")
        
    else:
        # Single dataset evaluation
        print(f"评估单个数据集: {args.dataset}")
        
        # Load data
        data, meta = load_wikitext_data(args.data_dir, args.dataset, args.split)
        print(f"加载了 {len(data)} 个 tokens")
        
        # Evaluate
        if args.full_sequence:
            print("开始全序列WikiText评估...")
            perplexity, loss = evaluate_full_sequence(
                model, device, data, args.max_seq_len
            )
        else:
            print("开始批次WikiText评估...")
            perplexity, loss = evaluate_wikitext(
                model, device, data, meta, args.eval_iters, args.batch_size
            )
        
        # Print results
        print(f"\n{args.dataset} 在 {args.split} 集上的结果:")
        print(f"损失: {loss:.4f}")
        print(f"困惑度: {perplexity:.2f}")

if __name__ == "__main__":
    main() 