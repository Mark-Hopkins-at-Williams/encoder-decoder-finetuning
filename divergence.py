from math import log2
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

def extract_smoothed_unigram_lm(filename, alpha=1.0):
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    vocab_size = len(tokenizer)
    counts = torch.zeros(vocab_size, dtype=torch.float64)
    total_tokens = 0
    with open(filename) as reader:
        for line in tqdm(reader):
            line = line.strip()
            for token_id in tokenizer(line)['input_ids']:            
                counts[token_id] += 1
    smoothed_counts = counts + alpha  # apply Laplace smoothing    
    smoothed_total = total_tokens + alpha * vocab_size
    probs = smoothed_counts / smoothed_total 
    return probs

def compute_kl_divergence(p, q):
    total = 0.0
    for i in range(len(p)):
        total += p[i] * (log2(p[i]) - log2(q[i]))
    return total

if __name__ == "__main__":
    langs = ['cs', 'da', 'de', 'es', 'fi', 
             'et', 'fr', 'it', 'nl', 'pl', 
             'pt', 'ro', 'sk', 'sl', 'sv']
    distributions = dict()
    for lang in langs:
        distributions[lang] = extract_smoothed_unigram_lm(f'data/dev.{lang}')
    
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    for i in range(len(langs)):
        for j in range(len(langs)):
            if i != j:
                divergence = compute_kl_divergence(distributions[langs[i]], distributions[langs[j]])
                print(f'{langs[i]},{langs[j]},{divergence}')