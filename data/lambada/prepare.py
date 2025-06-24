"""
Downloads and prepares the LAMBADA dataset for evaluation.
The LAMBADA dataset tests language models' ability to predict the final word
of passages that require understanding of broader linguistic context.
"""

import os
import json
import requests
import tiktoken
import numpy as np
from tqdm import tqdm

# Download LAMBADA dataset
def download_lambada():
    """Download LAMBADA test dataset"""
    url = "https://raw.githubusercontent.com/cybertronai/bflm/master/lambada_test.jsonl"
    
    if not os.path.exists('lambada_test.jsonl'):
        print("Downloading LAMBADA test set...")
        response = requests.get(url)
        with open('lambada_test.jsonl', 'w') as f:
            f.write(response.text)
        print("Download complete!")
    else:
        print("LAMBADA test set already exists.")

def prepare_lambada():
    """Prepare LAMBADA dataset for evaluation"""
    
    # Download data if needed
    download_lambada()
    
    # Load the dataset
    examples = []
    with open('lambada_test.jsonl', 'r') as f:
        for line in f:
            examples.append(json.loads(line))
    
    print(f"Loaded {len(examples)} LAMBADA examples")
    
    # Initialize tokenizer (GPT-2 tokenizer)
    enc = tiktoken.get_encoding("gpt2")
    
    # Prepare contexts and targets
    contexts = []
    targets = []
    
    for example in tqdm(examples, desc="Processing examples"):
        text = example['text']
        
        # The last word is the target
        words = text.strip().split()
        context = ' '.join(words[:-1])
        target = words[-1]
        
        # Tokenize
        context_tokens = enc.encode(context)
        target_tokens = enc.encode(' ' + target)  # Add space for proper tokenization
        
        contexts.append(context_tokens)
        targets.append(target_tokens)
    
    # Save processed data
    data = {
        'contexts': contexts,
        'targets': targets,
        'examples': examples
    }
    
    np.save('lambada_processed.npy', data)
    print(f"Saved processed LAMBADA data with {len(contexts)} examples")
    
    # Create meta file
    meta = {
        'vocab_size': enc.n_vocab,
        'num_examples': len(contexts)
    }
    
    with open('meta.json', 'w') as f:
        json.dump(meta, f)
    
    print("LAMBADA dataset preparation complete!")

if __name__ == "__main__":
    prepare_lambada() 