import glob
import os

configs = glob.glob('experiments/01_muon_vs_adamw_baseline/configs/*.yaml')
for c in configs:
    with open(c, 'r') as f:
        content = f.read()
    
    content = content.replace('train_tokens: 500000', 'train_tokens: 1700000000')
    content = content.replace('detailed_log_every: 5', 'detailed_log_every: 250')
    content = content.replace('log_every: 5', 'log_every: 50')
    
    with open(c, 'w') as f:
        f.write(content)
print("Updated all configs to 1.7B tokens.")
