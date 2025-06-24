"""
Downloads and prepares the WikiText-2 dataset for evaluation.
WikiText-2 is used for language modeling evaluation, focusing on perplexity measurement.
"""

import os
import requests
import zipfile
import tiktoken
import numpy as np
from tqdm import tqdm

def download_wikitext2():
    """Download WikiText-2 dataset"""
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-2-v1.zip"
    
    if not os.path.exists('wikitext-2-v1.zip'):
        print("Downloading WikiText-2 dataset...")
        response = requests.get(url)
        with open('wikitext-2-v1.zip', 'wb') as f:
            f.write(response.content)
        print("Download complete!")
    
    # Extract if needed
    if not os.path.exists('wikitext-2'):
        print("Extracting WikiText-2...")
        with zipfile.ZipFile('wikitext-2-v1.zip', 'r') as zip_ref:
            zip_ref.extractall('.')
        print("Extraction complete!")
    else:
        print("WikiText-2 already extracted.")

def prepare_wikitext2():
    """Prepare WikiText-2 dataset for evaluation"""
    
    # Download data if needed
    download_wikitext2()
    
    # Initialize tokenizer (GPT-2 tokenizer)
    enc = tiktoken.get_encoding("gpt2")
    
    # Process each split
    splits = ['train', 'valid', 'test']
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Read the text file
        filename = f'wikitext-2/wiki.{split}.tokens'
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Clean the text - remove empty lines and special tokens
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip empty lines and section headers
            if line and not line.startswith('='):
                cleaned_lines.append(line)
        
        # Join lines with space
        cleaned_text = ' '.join(cleaned_lines)
        
        # Tokenize
        print(f"Tokenizing {split} split...")
        tokens = enc.encode(cleaned_text)
        
        print(f"{split}: {len(tokens)} tokens")
        
        # Save as binary file
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(f'{split}.bin')
    
    # Create meta file
    meta = {
        'vocab_size': enc.n_vocab,
    }
    
    import pickle
    with open('meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    
    print("WikiText-2 dataset preparation complete!")
    print(f"Vocabulary size: {enc.n_vocab}")

if __name__ == "__main__":
    prepare_wikitext2() 