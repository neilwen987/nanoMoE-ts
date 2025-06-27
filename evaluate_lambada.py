"""
Evaluate models on the LAMBADA dataset.
LAMBADA tests the model's ability to predict the final word of passages 
that require understanding of broader linguistic context.
"""

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import tiktoken
from contextlib import nullcontext
from tqdm import tqdm
import argparse

from model import GPTConfig, GPT

def load_lambada_data(data_dir='data/lambada'):
    """Load processed LAMBADA data"""
    data_path = os.path.join(data_dir, 'lambada_processed.npy')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"LAMBADA data not found at {data_path}. Run data/lambada/prepare.py first.")
    
    data = np.load(data_path, allow_pickle=True).item()
    return data['contexts'], data['targets'], data['examples']

def evaluate_lambada(model, device, contexts, targets, examples, batch_size=1):
    """
    Evaluate model on LAMBADA dataset.
    Returns accuracy and perplexity.
    """
    model.eval()
    correct = 0
    total = len(contexts)
    total_loss = 0.0
    valid_examples = 0
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
    
    with torch.no_grad():
        for i in tqdm(range(total), desc="Evaluating LAMBADA"):
            context_tokens = contexts[i]
            target_tokens = targets[i]
            
            # Skip if target is empty
            if len(target_tokens) == 0:
                continue
                
            # Prepare input
            context_tensor = torch.tensor(context_tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            # Truncate context if too long
            if context_tensor.size(1) > model.config.block_size - len(target_tokens):
                context_tensor = context_tensor[:, -(model.config.block_size - len(target_tokens)):]
            
            with ctx:
                # Get model predictions
                logits, _ = model(context_tensor)
                
                # Get the logits for next token prediction
                next_token_logits = logits[0, -1, :]  # Last position
                
                # Calculate perplexity for the target token(s)
                # For LAMBADA, we typically care about the first target token
                target_token = target_tokens[0]
                
                # Calculate cross-entropy loss for perplexity
                loss = F.cross_entropy(next_token_logits.unsqueeze(0), 
                                     torch.tensor([target_token], device=device))
                total_loss += loss.item()
                valid_examples += 1
                
                # Get the most likely next token for accuracy
                predicted_token = torch.argmax(next_token_logits).item()
                
                # Check if prediction matches target
                if predicted_token == target_token:
                    correct += 1
    
    # Calculate metrics
    accuracy = correct / total * 100 if total > 0 else 0
    perplexity = torch.exp(torch.tensor(total_loss / valid_examples)).item() if valid_examples > 0 else float('inf')
    
    print(f"LAMBADA Accuracy: {accuracy:.2f}% ({correct}/{total})")
    print(f"LAMBADA Perplexity: {perplexity:.2f}")
    
    return accuracy, perplexity, correct, total

def main():
    parser = argparse.ArgumentParser(description='Evaluate model on LAMBADA dataset')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/lambada',
                       help='Path to LAMBADA data directory')
    parser.add_argument('--batch_size', type=int, default=1,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use for evaluation')
    
    args = parser.parse_args()
    
    # Load data
    print("Loading LAMBADA data...")
    contexts, targets, examples = load_lambada_data(args.data_dir)
    print(f"Loaded {len(contexts)} LAMBADA examples")
    
    # Load model
    print(f"Loading model from {args.model_path}")
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
    
    print(f"Model loaded. Parameters: {model.get_num_params()/1e6:.2f}M")
    
    # Evaluate
    print("Starting LAMBADA evaluation...")
    accuracy, perplexity, correct, total = evaluate_lambada(
        model, device, contexts, targets, examples, args.batch_size
    )
    
    # Print results
    print(f"\nLAMBADA Results:")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"Correct: {correct}")
    print(f"Total: {total}")

if __name__ == "__main__":
    main() 