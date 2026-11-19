#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 13:43:14 2026

@author: feipeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITD-ILD index = (VE_ITD - VE_ILD) / (VE_ITD + VE_ILD) vs channel number.
One figure, one column per stimulus rate.
x = index (-1 = ILD only, +1 = ITD only), y = channel number
(channel 1 = most superficial, at the top).
Per channel: box plot across units, with individual units overlaid (colour = animal).
An arrow on the left indicates recording depth (superficial at ch 1, deep at the last channel).
"""

import os
from sys import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import warnings
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

# ---------------------------------------------------------------------------
# 1. Settings and data loading
# ---------------------------------------------------------------------------
if platform == "darwin":
    results_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/'
    Figure_path = '/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/'
elif platform == "linux":
    results_path = '/disks/CIdata/2023_02_23Results/'
    Figure_path = '/disks/CIdata/2_ITD_ILD_CueTrading/Figures/'
else:
    results_path = './Data/AMUA_analysis_v2/'
    Figure_path = './Figures/'
os.makedirs(Figure_path, exist_ok=True)

npz_name = 'all_files_VE_data_3to30ms.npz'
stiRate = [1, 100, 900]
ITD, ILD = 0, 1

INVERT_Y = True         # True: channel 1 (most superficial) at the top
DOT_JITTER = 0.25       # vertical jitter of dots within each box (channels)
BOX_WIDTH = 0.7         # box height in channel units
XLIM = (-1.05, 1.05)
ARROW_X = -0.20         # arrow position in axes fraction (more negative = further left)

with np.load(os.path.join(results_path, npz_name), allow_pickle=True) as d:
    ve_value = d['VE_value']       # (nfiles, nrates, nchans, 2)
    unit_index = d['unit_index']   # (nfiles, nchans)
    fnames = d['fnames'].tolist()

nfiles, nrates, nchans = ve_value.shape[:3]

rows = []
for ff in range(nfiles):
    animal = fnames[ff].split('_')[0]
    for cc in np.where(unit_index[ff] == 1)[0]:
        for rr in range(nrates):
            rows.append(dict(
                animal=animal, file=ff, chan=int(cc) + 1,   # 1-based channel number
                rate=stiRate[rr],
                ve_itd=float(ve_value[ff, rr, cc, ITD]),
                ve_ild=float(ve_value[ff, rr, cc, ILD]),
            ))
df = pd.DataFrame(rows)

# ITD-ILD index; NaN where both VEs are zero (denominator = 0)
den = df.ve_itd + df.ve_ild
df['index'] = np.where(den > 0, (df.ve_itd - df.ve_ild) / den, np.nan)
n_nan = int(df['index'].isna().sum())
if n_nan:
    print(f'Warning: {n_nan} unit-rate entries have VE_ITD + VE_ILD = 0 and are excluded')
df = df.dropna(subset=['index']).copy()

animals = sorted(df.animal.unique())
animal_colors = {a: plt.colormaps['tab20'](i % 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}
print(f'{len(animals)} animals, {df.groupby(["file", "chan"]).ngroups} responsive units')

rng = np.random.default_rng(0)
unit_key = df.file.astype(str) + '_' + df.chan.astype(str)
jit_map = {k: rng.uniform(-DOT_JITTER, DOT_JITTER) for k in unit_key.unique()}
df['y'] = df.chan + unit_key.map(jit_map)   # same jitter for a unit across rates


# ---------------------------------------------------------------------------
# 1b. Linear mixed-effects model: index ~ channel + (1 + channel | animal)
# ---------------------------------------------------------------------------
def fit_lmm(sub):
    """Random intercept + slope per animal; fall back to random intercept only."""
    d = sub[['index', 'chan', 'animal']].rename(columns={'index': 'idx'}).copy()
    d['chan_c'] = d.chan - d.chan.mean()          # centred channel (slope unchanged)
    for re_formula, label in (('~chan_c', 'random intercept + slope'),
                              ('~1', 'random intercept')):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                res = smf.mixedlm('idx ~ chan_c', d, groups=d['animal'],
                                  re_formula=re_formula).fit(reml=True, method='powell')
            if res.converged:
                return res, label
        except Exception as e:
            print(f'  LMM failed ({label}): {e}')
    return None, 'failed'

stat_rows = []
for rate in stiRate:
    sub = df[df.rate == rate]
    res, label = fit_lmm(sub)
    
    print(res.llf)      # compare between optimisers; keep the higher value
    print(res.cov_re)   # random-effect covariance: look for ~0 variance or correlation = ±1
    print(res.converged)
    if res is None:
        stat_rows.append(dict(rate=rate, n_units=len(sub),
                              n_animals=sub.animal.nunique(), model='failed'))
        continue
    ci = res.conf_int().loc['chan_c']
    stat_rows.append(dict(
        rate=rate, n_units=len(sub), n_animals=sub.animal.nunique(),
        model=label, beta_per_channel=res.params['chan_c'],
        ci_low=ci[0], ci_high=ci[1], z=res.tvalues['chan_c'],
        p=res.pvalues['chan_c']))

stats_df = pd.DataFrame(stat_rows)
ok = stats_df['p'].notna()
stats_df.loc[ok, 'p_bonferroni'] = multipletests(stats_df.loc[ok, 'p'], method='bonferroni')[1]
stats_df['beta_per_10ch'] = stats_df['beta_per_channel'] * 10
pd.set_option('display.width', 200)
print('\nLMM: index ~ channel, animal as random effect (bonferroni-corrected across rates)')
print(stats_df.round(4).to_string(index=False))
stats_df.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_vs_channel_LMM.csv'), index=False)

def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'

# stat_label = {}
# for _, r in stats_df.iterrows():
#     if pd.notna(r.get('p_holm', np.nan)):
#         stat_label[r.rate] = f'\u03b2 = {r.beta_per_channel:.3f}/ch, {fmt_p(r.p_holm)} (Holm)'
#     else:
#         stat_label[r.rate] = 'LMM failed'

stat_label = {}
for _, r in stats_df.iterrows():
    if pd.notna(r.get('p_bonferroni', np.nan)):
        stat_label[r.rate] = (f'\u03b2 = {r.beta_per_channel:.3f}/ch '
                              f'[{r.ci_low:.3f}, {r.ci_high:.3f}]\n'
                              f'{fmt_p(r.p_bonferroni)} (bonferroni)')
    else:
        stat_label[r.rate] = 'LMM failed'

# ---------------------------------------------------------------------------
# 2. Plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, nrates, figsize=(15, 9), sharey=True, sharex=True,
                         gridspec_kw={'wspace': 0.08})
all_chans = np.arange(1, nchans + 1)

for ax, rate in zip(axes, stiRate):
    sub = df[df.rate == rate]

    # one box per channel (only channels with data)
    chans = [c for c in all_chans if (sub.chan == c).any()]
    data = [sub.loc[sub.chan == c, 'index'].values for c in chans]
    ax.boxplot(data, positions=chans, vert=False, widths=BOX_WIDTH,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='whitesmoke', edgecolor='k', linewidth=0.8),
               medianprops=dict(color='k', linewidth=1.5),
               whiskerprops=dict(linewidth=0.8), capprops=dict(linewidth=0.8),
               zorder=1)

    # individual units on top
    for animal in animals:
        a = sub[sub.animal == animal]
        if a.empty:
            continue
        ax.scatter(a['index'], a.y, marker='o', s=18,
                   color=animal_colors[animal], edgecolors='k',
                   linewidth=0.3, alpha=0.9, zorder=3)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, zorder=2)
    # ax.set_title(f'{rate} pps')
    ax.set_title(f'{rate} pps\n{stat_label[rate]}', fontsize=11)
    ax.set_xlabel('(VE_ITD - VE_ILD) / (VE_ITD + VE_ILD)')
    ax.set_xlim(*XLIM)
    ax.set_yticks(all_chans)
    if INVERT_Y:
        ax.set_ylim(nchans + 0.5, 0.5)
    else:
        ax.set_ylim(0.5, nchans + 0.5)
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    if rate == stiRate[0]:
        ax.set_ylabel('Channel number')
        ax.yaxis.set_label_coords(-0.08, 0.5)

# depth arrow on the far left of the first panel (axes coordinates)
# top = channel 1 (superficial), bottom = deepest channel; arrow points to deep
ax0 = axes[0]
ARROW_TOP, ARROW_BOT = 0.62, 0.38
ax0.annotate('', xy=(ARROW_X, ARROW_BOT), xytext=(ARROW_X, ARROW_TOP),
             xycoords='axes fraction', textcoords='axes fraction',
             arrowprops=dict(arrowstyle='-|>', lw=2, color='k',
                             mutation_scale=18, shrinkA=0, shrinkB=0),
             annotation_clip=False)
ax0.text(ARROW_X - 0.04, ARROW_TOP, 'Superficial', transform=ax0.transAxes,
         rotation=90, ha='center', va='top', fontsize=11, clip_on=False)
ax0.text(ARROW_X - 0.04, ARROW_BOT, 'Deep', transform=ax0.transAxes,
         rotation=90, ha='center', va='bottom', fontsize=11, clip_on=False)

# animal legend
animal_handles = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=animal_colors[a], markeredgecolor='k',
                         markersize=8, label=animal_labels[a]) for a in animals]
fig.legend(handles=animal_handles, title='Animal', loc='upper left',
           bbox_to_anchor=(0.90, 0.92), fontsize=9, title_fontsize=10)

fig.tight_layout(rect=[0, 0, 0.90, 1])
out = os.path.join(Figure_path, 'ITD_ILD_index_boxplot_by_channel.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('Saved', out)

# ---------------------------------------------------------------------------
# 3. Sanity check: units per animal
# ---------------------------------------------------------------------------
for rate in stiRate:
    sub = df[df.rate == rate]
    print(f'\n{rate} pps: units per animal')
    print(sub.groupby('animal').size())

plt.show()

#%%
# ---------------------------------------------------------------------------
# 1b. Per-animal trend + Wilcoxon signed-rank test across animals
# ---------------------------------------------------------------------------
from scipy import stats
from statsmodels.stats.multitest import multipletests

METRIC = 'spearman'     # 'spearman' (rho) or 'slope' (linear slope per channel)
MIN_UNITS = 5           # minimum units per animal and rate
MIN_CHANNELS = 3        # minimum number of distinct channels per animal

per_animal = []
for rate in stiRate:
    for animal in animals:
        a = df[(df.rate == rate) & (df.animal == animal)]
        if len(a) < MIN_UNITS or a.chan.nunique() < MIN_CHANNELS:
            continue
        rho, _ = stats.spearmanr(a.chan, a['index'])
        slope = stats.linregress(a.chan, a['index']).slope
        per_animal.append(dict(rate=rate, animal=animal, n_units=len(a),
                               n_channels=a.chan.nunique(),
                               rho=rho, slope=slope))
pa = pd.DataFrame(per_animal)
value_col = 'rho' if METRIC == 'spearman' else 'slope'

wil_rows = []
for rate in stiRate:
    v = pa.loc[pa.rate == rate, value_col].dropna().values
    if len(v) < 5:
        wil_rows.append(dict(rate=rate, n_animals=len(v), median=np.nan,
                             W=np.nan, p=np.nan, n_positive=np.nan))
        continue
    res = stats.wilcoxon(v, alternative='two-sided')     # H0: median = 0
    wil_rows.append(dict(rate=rate, n_animals=len(v), median=np.median(v),
                         W=res.statistic, p=res.pvalue,
                         n_positive=int((v > 0).sum())))
wil = pd.DataFrame(wil_rows)
ok = wil.p.notna()
wil.loc[ok, 'p_bonferroni'] = multipletests(wil.loc[ok, 'p'], method='bonferroni')[1]

pd.set_option('display.width', 200)
print(f'\nPer-animal {value_col} of index vs channel:')
print(pa.round(3).to_string(index=False))
print(f'\nWilcoxon signed-rank test across animals ({value_col} vs 0):')
print(wil.round(4).to_string(index=False))
pa.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_per_animal_trend.csv'), index=False)
wil.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_wilcoxon.csv'), index=False)

# optional: one dot per animal, per rate
fig2, ax2 = plt.subplots(1, 1, figsize=(5, 5))
for i, rate in enumerate(stiRate):
    s = pa[pa.rate == rate]
    xj = i + rng.uniform(-0.1, 0.1, len(s))
    ax2.scatter(xj, s[value_col], s=40,
                c=[animal_colors[a] for a in s.animal], edgecolors='k', zorder=3)
    if len(s):
        ax2.hlines(s[value_col].median(), i - 0.25, i + 0.25, color='k', lw=2)
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_xticks(range(len(stiRate)))
ax2.set_xticklabels([f'{r} pps' for r in stiRate])
ax2.set_ylabel("Spearman \u03c1 (index vs channel)" if METRIC == 'spearman'
               else 'Slope (index per channel)')
fig2.tight_layout()
fig2.savefig(os.path.join(Figure_path, 'ITD_ILD_index_per_animal_trend.png'),
             dpi=300, bbox_inches='tight')