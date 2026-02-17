#!/usr/bin/env python3
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="Plot Pasqal vs Classical F_bond results")
    parser.add_argument("json_file", help="Path to JSON results file")
    parser.add_argument("-o", "--output", default="fbond_pasqal_comparison.pdf", help="Output PDF filename")
    args = parser.parse_args()

    try:
        with open(args.json_file, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        sys.exit(1)

    # Filter out redundant systems or reorder if needed
    # Desired order: Al4_aro, Al4_anti, B12, B6N6, Cs3Al8
    order = ["Al4_aromatic", "Al4_antiaromatic", "B12_planar", "B6N6_planar", "Cs3Al8"]
    
    # Map to display names
    labels = {
        "Al4_aromatic": "Al$_4^{2-}$ (Aro)",
        "Al4_antiaromatic": "Al$_4^{4-}$ (Anti)",
        "B12_planar": "B$_{12}$ (Sheet)",
        "B6N6_planar": "B$_6$N$_6$",
        "Cs3Al8": "Cs$_3$Al$_8^{-}$"
    }

    # Extract data in order
    systems = []
    q_se = []
    c_se = []
    
    # Store data in a dict for easy lookup
    data_map = {item["system"]: item for item in data}

    for key in order:
        if key in data_map:
            item = data_map[key]
            systems.append(labels.get(key, key))
            q_se.append(item["quantum_SE_max"])
            c_se.append(item["classical_SE_max"])

    x = np.arange(len(systems))
    width = 0.35

    # Create plot with dual y-axes
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Classical bars (left axis, blue)
    rects1 = ax1.bar(x - width/2, c_se, width, label='Classical CCSD $S_{E,max}$', color='#1f77b4', alpha=0.9)
    ax1.set_ylabel('Classical $S_{E,max}$ (nats)', color='#1f77b4', fontsize=12, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='#1f77b4', labelsize=10)
    ax1.set_ylim(0, max(c_se) * 1.2)

    # Quantum bars (right axis, red)
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, q_se, width, label='Quantum Pasqal $S_{E,max}$', color='#d62728', alpha=0.9)
    ax2.set_ylabel('Quantum (Rydberg) $S_{E,max}$ (nats)', color='#d62728', fontsize=12, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#d62728', labelsize=10)
    ax2.set_ylim(0, max(q_se) * 1.2)

    # X-axis formatting
    ax1.set_xticks(x)
    ax1.set_xticklabels(systems, fontsize=11, rotation=0)
    ax1.set_xlabel("Molecular System", fontsize=12, fontweight='bold')

    # Title
    plt.title("F-bond Entanglement: Classical CCSD vs. Pasqal Quantum Simulation", fontsize=14, pad=20)

    # Legend
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, frameon=True)

    # Add text labels on bars
    def autolabel(rects, ax, vals, color):
        for rect, val in zip(rects, vals):
            height = rect.get_height()
            ax.annotate(f'{val:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', color=color, fontsize=9, fontweight='bold')

    autolabel(rects1, ax1, c_se, '#1f77b4')
    autolabel(rects2, ax2, q_se, '#d62728')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Generated plot: {args.output}")

if __name__ == "__main__":
    main()
