import glob
for f in glob.glob("experiments/01_muon_vs_adamw_baseline/configs/*.yaml"):
    with open(f, "r") as file:
        content = file.read()
    content = content.replace("train_tokens: 2000000", "train_tokens: 500000")
    content = content.replace("detailed_log_every: 20", "detailed_log_every: 5")
    content = content.replace("log_every: 10", "log_every: 5")
    with open(f, "w") as file:
        file.write(content)
