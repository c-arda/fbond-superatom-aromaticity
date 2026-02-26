#!/usr/bin/env python3
"""
Regenerate all manuscript figures with updated N_D nomenclature.
Produces fig1-fig5 PDFs for the v5 manuscript.

Usage:
    python regenerate_figures.py

Output:
    ../figures/fig1_nd_bar.pdf           - N_D bar chart (all 11 systems)
    ../figures/fig2_fe_bar.pdf          - f_e bar chart (all 11 systems)
    ../figures/fig3_nd_vs_ne.pdf         - N_D vs N_e scatter
    ../figures/fig4_truncation_comparison.pdf - Truncated vs full N_D
    ../figures/fig5_fractional_orbitals.pdf   - Fractional orbital percentages
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# Output directory
FIGDIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGDIR, exist_ok=True)

# Complete dataset (11 systems, sorted by N_D)
SYSTEMS = [
    # (label, N_e, N_corr, M, M_frac, E_corr, N_D, f_e)
    ("C$_6$H$_6$\n(Benzene)",  42,  30, 114,  75, 0.821, 2.49, 0.083),
    ("Al$_4^{2-}$\n(Aro.)",    54,  46,  72,  49, 0.300, 3.84, 0.083),
    ("Al$_4^{4-}$\n(Sing.)",   56,  48,  72,  52, 0.352, 4.03, 0.084),
    ("B$_{12}$\n(Planar)",     60,  36, 168, 102, 1.037, 4.42, 0.123),
    ("B$_{12}$\n(Ico.)",       60,  36, 168, 106, 1.173, 4.99, 0.139),
    ("B$_6$N$_6$\n(Planar)",   72,  48, 168, 115, 1.529, 5.11, 0.106),
    ("Cs$_3$Al$_8^-$",        132, 116, 216, 208, 0.836, 5.58, 0.048),
    ("Al$_4^{4-}$\n(Trip.)",   56,  48,  72,  52, 0.342, 4.17, 0.087),
    ("Au$_{13}^-$\n(Ico.)",   248, 222, 300, 121, 1.417, 6.76, 0.030),
    ("Cs$_3$Al$_{12}^-$",     184, 160, 288, 276, 1.184, 7.10, 0.044),
    ("B$_{12}$N$_{12}$\n(Cage)", 144,  96, 360, 104, 2.888, 7.18, 0.075),
]

labels  = [s[0] for s in SYSTEMS]
N_e     = [s[1] for s in SYSTEMS]
N_corr  = [s[2] for s in SYSTEMS]
M       = [s[3] for s in SYSTEMS]
M_frac  = [s[4] for s in SYSTEMS]
E_corr  = [s[5] for s in SYSTEMS]
N_D     = [s[6] for s in SYSTEMS]
f_e     = [s[7] for s in SYSTEMS]

# Color scheme: small clusters = blue family, superatoms/cages = orange family, outlier = green
colors = []
for nd, fe in zip(N_D, f_e):
    if fe > 0.15:   # triplet outlier
        colors.append('#2ca02c')
    elif fe > 0.06: # small clusters
        colors.append('#1f77b4')
    else:           # superatoms/cages
        colors.append('#ff7f0e')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 0.8,
    'figure.dpi': 300,
})


# ========== Fig 1: N_D bar chart ==========
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(labels))
bars = ax.bar(x, N_D, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
ax.set_ylabel('$N_D$ (Takatsuka--Head-Gordon index)', fontsize=13, fontweight='bold')
ax.set_xlabel('Molecular System', fontsize=13, fontweight='bold')
ax.set_title('Aggregate Electron Correlation $N_D$ Across 11 Systems (CCSD/def2-SVP)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, ha='center')
for bar, val in zip(bars, N_D):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
            f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, max(N_D) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig1_nd_bar.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig1_nd_bar.pdf")


# ========== Fig 2: f_e bar chart ==========
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, f_e, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
ax.set_ylabel('$f_e = N_D / N_{\\mathrm{corr}}$ (per-electron correlation)',
              fontsize=13, fontweight='bold')
ax.set_xlabel('Molecular System', fontsize=13, fontweight='bold')
ax.set_title('Per-Electron Correlation Density $f_e$ Across 11 Systems',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, ha='center')
for bar, val in zip(bars, f_e):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
# Regime separation lines
ax.axhspan(0.06, 0.20, alpha=0.08, color='blue', label='Small cluster regime')
ax.axhspan(0.02, 0.06, alpha=0.08, color='orange', label='Superatom/cage regime')
ax.legend(fontsize=10, loc='upper right')
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
ax.set_ylim(0, max(f_e) * 1.2)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig2_fe_bar.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig2_fe_bar.pdf")


# ========== Fig 3: N_D vs N_e scatter ==========
fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(N_e, N_D, c=f_e, cmap='viridis', s=120, edgecolors='black',
                     linewidth=0.8, zorder=5)
cbar = plt.colorbar(scatter, ax=ax, label='$f_e = N_D / N_{\\mathrm{corr}}$')
ax.set_xlabel('Total electron count $N_e$', fontsize=13, fontweight='bold')
ax.set_ylabel('$N_D$', fontsize=13, fontweight='bold')
ax.set_title('$N_D$ vs. Total Electron Count', fontsize=14, fontweight='bold')
# Annotate each point
short_labels = ['C$_6$H$_6$', 'Al$_4^{2-}$', 'Al$_4^{4-}$', 'B$_{12}$(p)',
                'B$_{12}$(i)', 'B$_6$N$_6$', 'Cs$_3$Al$_8^-$', 'Al$_4^{4-}$(t)',
                'Au$_{13}^-$', 'Cs$_3$Al$_{12}^-$', 'B$_{12}$N$_{12}$']
for i, txt in enumerate(short_labels):
    ax.annotate(txt, (N_e[i], N_D[i]), textcoords="offset points",
                xytext=(8, 5), fontsize=7.5)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig3_nd_vs_ne.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig3_nd_vs_ne.pdf")


# ========== Fig 4: Truncation comparison ==========
truncation_data = [
    # (label, M_trunc, N_D_trunc, N_D_full, ratio)
    ("Al$_4^{2-}$",        6, 0.0006, 3.84, 6200),
    ("Al$_4^{4-}$",        6, 0.0007, 4.03, 5700),
    ("B$_{12}$ (planar)", 18, 0.43,   4.42,   10),
    ("B$_{12}$ (ico.)",   18, 0.42,   4.99,   12),
    ("B$_6$N$_6$",        27, 0.72,   5.11,    7),
    ("Cs$_3$Al$_8^-$",   20, 0.013,  5.58,  440),
    ("Cs$_3$Al$_{12}^-$",20, 0.013,  7.10,  550),
]
trunc_labels = [d[0] for d in truncation_data]
trunc_nd = [d[2] for d in truncation_data]
full_nd = [d[3] for d in truncation_data]
ratios = [d[4] for d in truncation_data]

fig, ax = plt.subplots(figsize=(12, 6))
x_t = np.arange(len(trunc_labels))
width = 0.35
bars1 = ax.bar(x_t - width/2, trunc_nd, width, label='$N_D^{\\mathrm{trunc}}$ (active space)',
               color='#d62728', edgecolor='black', linewidth=0.5, alpha=0.9)
bars2 = ax.bar(x_t + width/2, full_nd, width, label='$N_D^{\\mathrm{full}}$ (all orbitals)',
               color='#1f77b4', edgecolor='black', linewidth=0.5, alpha=0.9)
# Add ratio labels
for i, r in enumerate(ratios):
    ax.annotate(f'{r:,}x', xy=(x_t[i], max(trunc_nd[i], full_nd[i])),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontsize=10, fontweight='bold', color='#d62728')
ax.set_yscale('log')
ax.set_ylabel('$N_D$', fontsize=13, fontweight='bold')
ax.set_xlabel('System', fontsize=13, fontweight='bold')
ax.set_title('Effect of Orbital Space Truncation on $N_D$', fontsize=14, fontweight='bold')
ax.set_xticks(x_t)
ax.set_xticklabels(trunc_labels, fontsize=11)
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig4_truncation_comparison.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig4_truncation_comparison.pdf")


# ========== Fig 5: Fractional orbital percentages ==========
frac_pct = [100 * mf / m for mf, m in zip(M_frac, M)]
fig, ax = plt.subplots(figsize=(14, 6))
bars = ax.bar(x, frac_pct, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
ax.set_ylabel('Fractionally Occupied Orbitals (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('Molecular System', fontsize=13, fontweight='bold')
ax.set_title('Percentage of Orbitals with Fractional Occupation ($|n_i - 2| > 10^{-6}$ or $n_i > 10^{-6}$)',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=9, ha='center')
for bar, val in zip(bars, frac_pct):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylim(0, 105)
ax.yaxis.grid(True, alpha=0.3, linestyle='--')
ax.set_axisbelow(True)
plt.tight_layout()
plt.savefig(os.path.join(FIGDIR, 'fig5_fractional_orbitals.pdf'), dpi=300, bbox_inches='tight')
plt.close()
print("Generated fig5_fractional_orbitals.pdf")

print("\nAll 5 figures regenerated with N_D nomenclature.")
