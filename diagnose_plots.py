import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from research_muon.track_manifold import plot_results

print("Loading data...")
with open('plots/metrics_200000000_20260222_081654.json', 'r') as f:
    data = json.load(f)

print("Data loaded. History keys:", data['history'].keys())
if 'manifold_history' in data['history']:
    print("Manifold history found. Keys:", data['history']['manifold_history'].keys())

print("Plotting results...")
plot_results(data['history'], 'results/research_plots')
print("Plots saved to results/research_plots")
