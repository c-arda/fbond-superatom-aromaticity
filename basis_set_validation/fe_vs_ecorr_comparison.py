#!/usr/bin/env python3
"""
f_e vs |E_corr|/N_corr scatter plot — demonstrates non-redundancy
==================================================================
If f_e were simply a proxy for correlation energy per electron,
they'd correlate perfectly. This plot shows they don't.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

# All 11 systems from Table 1 (CCSD/def2-SVP)
data = {
    'C₆H₆':          {'N_D': 2.49, 'f_e': 0.083, 'E_corr': 0.821, 'N_corr': 30,  'regime': 'small'},
    'Al₄²⁻':         {'N_D': 3.84, 'f_e': 0.083, 'E_corr': 0.300, 'N_corr': 46,  'regime': 'small'},
    'Al₄⁴⁻ (S)':     {'N_D': 4.03, 'f_e': 0.084, 'E_corr': 0.352, 'N_corr': 48,  'regime': 'small'},
    'Al₄⁴⁻ (T)':     {'N_D': 4.17, 'f_e': 0.087, 'E_corr': 0.342, 'N_corr': 48,  'regime': 'small'},
    'B₁₂ (2D)':      {'N_D': 4.42, 'f_e': 0.123, 'E_corr': 1.037, 'N_corr': 36,  'regime': 'small'},
    'B₁₂ (3D)':      {'N_D': 4.99, 'f_e': 0.139, 'E_corr': 1.173, 'N_corr': 36,  'regime': 'small'},
    'B₆N₆':          {'N_D': 5.11, 'f_e': 0.106, 'E_corr': 1.529, 'N_corr': 48,  'regime': 'small'},
    'Cs₃Al₈⁻':       {'N_D': 5.58, 'f_e': 0.048, 'E_corr': 0.836, 'N_corr': 116, 'regime': 'super'},
    'Au₁₃⁻':         {'N_D': 6.76, 'f_e': 0.030, 'E_corr': 1.417, 'N_corr': 222, 'regime': 'super'},
    'Cs₃Al₁₂⁻':      {'N_D': 7.10, 'f_e': 0.044, 'E_corr': 1.184, 'N_corr': 160, 'regime': 'super'},
    'B₁₂N₁₂':        {'N_D': 7.18, 'f_e': 0.075, 'E_corr': 2.888, 'N_corr': 96,  'regime': 'inter'},
}

# Compute |E_corr|/N_corr for each system
names = list(data.keys())
f_e_vals = np.array([data[s]['f_e'] for s in names])
ecorr_per_e = np.array([data[s]['E_corr'] / data[s]['N_corr'] for s in names])
regimes = [data[s]['regime'] for s in names]

# Colors by regime
colors = {'small': '#00d2ff', 'super': '#f43f5e', 'inter': '#f59e0b'}
point_colors = [colors[r] for r in regimes]

# Correlations
rho_spearman, p_spearman = stats.spearmanr(f_e_vals, ecorr_per_e)
r_pearson, p_pearson = stats.pearsonr(f_e_vals, ecorr_per_e)

# Plot
fig, ax = plt.subplots(figsize=(7, 5.5))
fig.patch.set_facecolor('#080b18')
ax.set_facecolor('#0c1024')

for i, name in enumerate(names):
    ax.scatter(ecorr_per_e[i], f_e_vals[i], c=point_colors[i],
               s=100, zorder=5, edgecolors='white', linewidths=0.5, alpha=0.9)
    # Label offset
    dx, dy = 0.0003, 0.002
    if name in ('Al₄²⁻', 'Al₄⁴⁻ (S)'):
        dy = -0.004
    if name == 'B₁₂N₁₂':
        dx = -0.003
        dy = -0.004
    ax.annotate(name, (ecorr_per_e[i] + dx, f_e_vals[i] + dy),
                fontsize=7.5, color=(1,1,1,0.85),
                fontfamily='monospace')

# Perfect correlation line (if f_e = k * |E_corr|/N_corr)
slope, intercept = np.polyfit(ecorr_per_e, f_e_vals, 1)
x_fit = np.linspace(ecorr_per_e.min() * 0.9, ecorr_per_e.max() * 1.1, 100)
ax.plot(x_fit, slope * x_fit + intercept, '--', color=(1,1,1,0.25),
        linewidth=1, zorder=2, label=f'Linear fit (R² = {r_pearson**2:.2f})')

# Regime separation zone
ax.axhspan(0.055, 0.065, color=(1,1,1,0.05), zorder=1)
ax.axhline(0.06, color=(1,1,1,0.15), linestyle=':', linewidth=0.8, zorder=1)

ax.set_xlabel('|E$_{corr}$| / N$_{corr}$ (Ha/electron)', fontsize=11,
              color=(1,1,1,0.8), fontfamily='monospace')
ax.set_ylabel('f$_e$ = N$_D$ / N$_{corr}$', fontsize=11,
              color=(1,1,1,0.8), fontfamily='monospace')
ax.set_title('Non-redundancy of f$_e$ and |E$_{corr}$|/N$_{corr}$',
             fontsize=13, color=(1,1,1,0.95), fontfamily='monospace',
             pad=12)

# Stats box
stats_text = (f'Spearman ρ = {rho_spearman:.3f} (p = {p_spearman:.3f})\n'
              f'Pearson r = {r_pearson:.3f} (R² = {r_pearson**2:.2f})')
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
        fontsize=8, color=(1,1,1,0.7), fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=(0.047,0.063,0.141,0.8),
                  edgecolor=(0,0.824,1,0.15)))

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#00d2ff',
           markersize=8, label='Small cluster', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f43f5e',
           markersize=8, label='Metallic superatom', linestyle='None'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#f59e0b',
           markersize=8, label='Intermediate (B₁₂N₁₂)', linestyle='None'),
]
legend = ax.legend(handles=legend_elements, loc='lower right', fontsize=8,
                   facecolor='#0c1024', edgecolor=(0,0.824,1,0.15),
                   labelcolor=(1,1,1,0.8))

ax.tick_params(colors=(1,1,1,0.6), labelsize=9)
for spine in ax.spines.values():
    spine.set_color((0,0.824,1,0.08))

plt.tight_layout()
plt.savefig('fe_vs_ecorr_per_electron.png', dpi=200, facecolor='#080b18',
            bbox_inches='tight')
plt.savefig('fe_vs_ecorr_per_electron.pdf', facecolor='#080b18',
            bbox_inches='tight')

# Also save a white-background version for SI
fig2, ax2 = plt.subplots(figsize=(7, 5.5))

for i, name in enumerate(names):
    marker = {'small': 'o', 'super': 's', 'inter': 'D'}[regimes[i]]
    color = {'small': '#1f77b4', 'super': '#d62728', 'inter': '#ff7f0e'}[regimes[i]]
    ax2.scatter(ecorr_per_e[i], f_e_vals[i], c=color, marker=marker,
                s=80, zorder=5, edgecolors='black', linewidths=0.5)
    dx, dy = 0.0003, 0.002
    if name in ('Al₄²⁻', 'Al₄⁴⁻ (S)'):
        dy = -0.005
    if name == 'B₁₂N₁₂':
        dx = -0.003
        dy = -0.005
    if name == 'Al₄⁴⁻ (T)':
        dy = -0.005
    ax2.annotate(name, (ecorr_per_e[i] + dx, f_e_vals[i] + dy),
                 fontsize=7, fontfamily='serif')

ax2.plot(x_fit, slope * x_fit + intercept, '--', color='gray',
         linewidth=1, zorder=2)
ax2.axhline(0.06, color='gray', linestyle=':', linewidth=0.8, zorder=1)

ax2.set_xlabel('|E$_{corr}$| / N$_{corr}$ (Ha/electron)', fontsize=11)
ax2.set_ylabel('f$_e$ = N$_D$ / N$_{corr}$', fontsize=11)

stats_text2 = (f'Spearman ρ = {rho_spearman:.3f} (p = {p_spearman:.3f})\n'
               f'Pearson r = {r_pearson:.3f} (R² = {r_pearson**2:.2f})')
ax2.text(0.03, 0.97, stats_text2, transform=ax2.transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                   edgecolor='gray', alpha=0.8))

from matplotlib.lines import Line2D
legend_elements2 = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4',
           markeredgecolor='black', markersize=8, label='Small cluster', linestyle='None'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='#d62728',
           markeredgecolor='black', markersize=8, label='Metallic superatom', linestyle='None'),
    Line2D([0], [0], marker='D', color='w', markerfacecolor='#ff7f0e',
           markeredgecolor='black', markersize=8, label='Intermediate (B₁₂N₁₂)', linestyle='None'),
]
ax2.legend(handles=legend_elements2, loc='lower right', fontsize=8)

plt.tight_layout()
plt.savefig('fig_SI_fe_vs_ecorr.pdf', bbox_inches='tight')
plt.savefig('fig_SI_fe_vs_ecorr.png', dpi=200, bbox_inches='tight')

print(f"\n{'='*60}")
print(f"  CORRELATION ANALYSIS: f_e vs |E_corr|/N_corr")
print(f"{'='*60}")
print(f"  Spearman ρ = {rho_spearman:.4f}  (p = {p_spearman:.4f})")
print(f"  Pearson  r = {r_pearson:.4f}  (R² = {r_pearson**2:.3f})")
print(f"{'='*60}")
if r_pearson**2 < 0.70:
    print(f"  ✓ R² = {r_pearson**2:.2f} < 0.70 → f_e is NOT a proxy for |E_corr|/N_corr")
    print(f"    They capture different aspects of electron correlation.")
else:
    print(f"  ⚠ R² = {r_pearson**2:.2f} ≥ 0.70 → some redundancy detected")
print(f"{'='*60}")

# Key deviations from perfect correlation
print(f"\n  Key deviations:")
for i, name in enumerate(names):
    predicted_fe = slope * ecorr_per_e[i] + intercept
    residual = f_e_vals[i] - predicted_fe
    if abs(residual) > 0.015:
        print(f"    {name:15s}: f_e = {f_e_vals[i]:.3f}, predicted = {predicted_fe:.3f}, "
              f"residual = {residual:+.3f}")

print(f"\n  Saved: fe_vs_ecorr_per_electron.png/pdf (dark)")
print(f"  Saved: fig_SI_fe_vs_ecorr.png/pdf (SI-ready, white bg)")
