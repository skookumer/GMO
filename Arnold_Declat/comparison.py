from PAMI.frequentPattern.basic import ECLAT
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.basic import Apriori
from PAMI.frequentPattern.closed import CHARM
from PAMI.extras.convert import denseDF2DB as db

import pandas as pd
import numpy as np
import time
import tracemalloc
import matplotlib.pyplot as plt

from original_testbed import xclat, xxclat, xclat_broken, gen_tidset, gen_diffset, read_csv, oh_encode

data = read_csv([i for i in range(13)], "Member_number")
diffset = gen_diffset(data)
tidset = gen_tidset(data)
N = len(data)

keys = list(tidset.keys())
encoded = oh_encode(data)
obj = db.denseDF2DB(pd.DataFrame(encoded))
obj.convert2TransactionalDatabase('binneddata.csv', '>=', 1)

inputFile = 'binneddata.csv'


all_results = {
    'pami_Apriori': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
    'pami_ECLAT': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
    'pami_CHARM': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
    'pami_FPGrowth': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
    'xclat_ec': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
    'xclat_de': {'support': [], 'runtime': [], 'memory': [], 'patterns': []},
}

# Test different support thresholds
for n in range(200, 30, -10):
    support = n / N
    print(f"\n{'='*50}")
    print(f"Testing with minSup = {n} ({support:.4f})")
    print(f"{'='*50}")
    
    algorithms = {
        'pami_Apriori': ('pami', Apriori.Apriori),
        'pami_ECLAT': ('pami', ECLAT.ECLAT),
        'pami_CHARM': ('pami', CHARM.CHARM),
        'pami_FPGrowth': ('pami', FPGrowth.FPGrowth),
        'xclat_ec': ('custom', lambda: xclat(diffset.copy(), tidset.copy(), keys.copy(), keys.copy(), n, mode="ec")),
        'xclat_de': ('custom', lambda: xclat(diffset.copy(), tidset.copy(), keys.copy(), keys.copy(), n, mode="de")),
    }
    
    for name, (algo_type, algo) in algorithms.items():
        print(f"Running {name}...", end=' ')
        tracemalloc.start()
        start = time.time()
        
        try:
            if algo_type == 'pami':
                obj = algo(iFile=inputFile, minSup=n, sep='\t')
                obj.startMine()
                patterns = obj.getPatterns()
                # patterns = [p for p in obj.getPatterns() if len(p) > 1]
            else:  # custom
                patterns = algo()
                # patterns = [p for p in obj.getPatterns() if len(p) > 1]
            
            runtime = time.time() - start
            current, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Store results
            all_results[name]['support'].append(n)
            all_results[name]['runtime'].append(runtime)
            all_results[name]['memory'].append(peak_memory / 1024 / 1024)
            all_results[name]['patterns'].append(len(patterns))
            
            print(f"✓ Time: {runtime:.3f}s, Memory: {peak_memory/1024/1024:.2f}MB, Patterns: {len(patterns)}")
        
        except Exception as e:
            print(f"✗ Error: {e}")
            tracemalloc.stop()

# Colors for each algorithm
colors = {
    'pami_Apriori': '#1f77b4',
    'pami_ECLAT': '#ff7f0e',
    'pami_CHARM': '#e377c2',
    'pami_FPGrowth': '#2ca02c',
    'xclat_ec': '#9467bd',
    'xclat_de': '#8c564b',
}

# Plot 1: Runtime vs Support
plt.figure(figsize=(10, 6))
for name, data in all_results.items():
    if len(data['support']) > 0:
        plt.plot(data['support'], data['runtime'], marker='o', 
                label=name, color=colors[name], linewidth=2, markersize=6)
plt.xlabel('Minimum Support Count', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Runtime vs Support Threshold', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()  # Higher support on left
plt.tight_layout()
plt.savefig('runtime_vs_support.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 2: Memory vs Support
plt.figure(figsize=(10, 6))
for name, data in all_results.items():
    if len(data['support']) > 0:
        plt.plot(data['support'], data['memory'], marker='s', 
                label=name, color=colors[name], linewidth=2, markersize=6)
plt.xlabel('Minimum Support Count', fontsize=12)
plt.ylabel('Peak Memory (MB)', fontsize=12)
plt.title('Memory Usage vs Support Threshold', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.tight_layout()
plt.savefig('memory_vs_support.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 3: Number of Patterns vs Support
plt.figure(figsize=(10, 6))
for name, data in all_results.items():
    if len(data['support']) > 0:
        plt.plot(data['support'], data['patterns'], marker='^', 
                label=name, color=colors[name], linewidth=2, markersize=6)
plt.xlabel('Minimum Support Count', fontsize=12)
plt.ylabel('Number of Patterns', fontsize=12)
plt.title('Patterns Found vs Support Threshold', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.gca().invert_xaxis()
plt.yscale('log')  # Log scale for patterns
plt.tight_layout()
plt.savefig('patterns_vs_support.png', dpi=300, bbox_inches='tight')
plt.show()

# Plot 4: Runtime vs Memory (scatter)
plt.figure(figsize=(10, 6))
for name, data in all_results.items():
    if len(data['support']) > 0:
        plt.scatter(data['memory'], data['runtime'], 
                   label=name, color=colors[name], s=100, alpha=0.6)
plt.xlabel('Peak Memory (MB)', fontsize=12)
plt.ylabel('Runtime (seconds)', fontsize=12)
plt.title('Runtime vs Memory Usage', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('runtime_vs_memory.png', dpi=300, bbox_inches='tight')
plt.show()

# Print summary statistics
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)
for name, data in all_results.items():
    if len(data['runtime']) > 0:
        print(f"\n{name}:")
        print(f"  Avg Runtime: {np.mean(data['runtime']):.3f}s")
        print(f"  Avg Memory:  {np.mean(data['memory']):.2f}MB")
        print(f"  Total Patterns Range: {min(data['patterns'])} - {max(data['patterns'])}")