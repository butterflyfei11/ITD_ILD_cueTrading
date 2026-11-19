#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 16:38:45 2026

@author: feipeng
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ITD-ILD index = (VE_ITD - VE_ILD) / (VE_ITD + VE_ILD) vs channel number.

Outputs
  1) Fig: box plot per channel, one column per stimulus rate (channel 1 = most superficial, at top);
     panel titles show the LMM slope, 95% CI and Bonferroni-corrected p.
  2) LMM (per rate): index ~ channel + (1 + channel | animal), best of several optimisers.
  3) Wilcoxon signed-rank test across animals on per-animal Spearman rho (index vs channel).
  4) Fig: per-animal Spearman rho.
  5) CSV files with the statistics.
"""

import os
import warnings
from sys import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from scipy import stats

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

INVERT_Y = True          # True: channel 1 (most superficial) at the top
DOT_JITTER = 0.25
BOX_WIDTH = 0.7
XLIM = (-1.05, 1.05)
ARROW_X = -0.20

CORRECTION = 'bonferroni'          # or 'holm'
OPTIMIZERS = ['lbfgs', 'powell', 'bfgs', 'nm']
VERBOSE = True
MIN_UNITS = 5                      # per animal and rate (Wilcoxon analysis)
MIN_CHANNELS = 3                   # distinct channels per animal (Wilcoxon analysis)

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

den = df.ve_itd + df.ve_ild
df['index'] = np.where(den > 0, (df.ve_itd - df.ve_ild) / den, np.nan)
n_nan = int(df['index'].isna().sum())
if n_nan:
    print(f'Warning: {n_nan} unit-rate entries have VE_ITD + VE_ILD <= 0 and are excluded')
df = df.dropna(subset=['index']).copy()

animals = sorted(df.animal.unique())
animal_colors = {a: plt.colormaps['tab20'](i % 20) for i, a in enumerate(animals)}
animal_labels = {a: f'Rat {i + 1}' for i, a in enumerate(animals)}
print(f'{len(animals)} animals, {df.groupby(["file", "chan"]).ngroups} responsive units')

rng = np.random.default_rng(0)
unit_key = df.file.astype(str) + '_' + df.chan.astype(str)
jit_map = {k: rng.uniform(-DOT_JITTER, DOT_JITTER) for k in unit_key.unique()}
df['y'] = df.chan + unit_key.map(jit_map)

# ---------------------------------------------------------------------------
# 2. LMM: index ~ channel + (1 + channel | animal)
# ---------------------------------------------------------------------------
def fit_lmm(sub, verbose=True):
    """Random intercept + slope per animal (fallback: random intercept only).
    Tries several optimisers and keeps the converged fit with the highest REML log-likelihood."""
    d = sub[['index', 'chan', 'animal']].rename(columns={'index': 'idx'}).copy()
    d['chan_c'] = d.chan - d.chan.mean()

    for re_formula, label in (('~chan_c', 'random intercept + slope'),
                              ('~1', 'random intercept')):
        fits = []
        for m in OPTIMIZERS:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    r = smf.mixedlm('idx ~ chan_c', d, groups=d['animal'],
                                    re_formula=re_formula).fit(reml=True, method=m)
                if r.converged and np.isfinite(r.llf):
                    fits.append((r.llf, m, r))
                    if verbose:
                        print(f'    {label:26s} {m:7s} llf = {r.llf:9.3f}  '
                              f'beta = {r.params["chan_c"]:.5f}  p = {r.pvalues["chan_c"]:.4f}')
            except Exception as e:
                if verbose:
                    print(f'    {label:26s} {m:7s} failed: {e}')
        if not fits:
            continue

        llf, method, res = max(fits, key=lambda t: t[0])
        info = dict(model=label, optimizer=method, llf=llf,
                    slope_sd=np.nan, re_corr=np.nan, singular=False)
        if re_formula == '~chan_c':
            cov = np.asarray(res.cov_re)
            v_int, v_slope = cov[0, 0], cov[1, 1]
            info['slope_sd'] = np.sqrt(max(v_slope, 0))
            if v_int > 0 and v_slope > 0:
                info['re_corr'] = cov[0, 1] / np.sqrt(v_int * v_slope)
            info['singular'] = bool(v_slope < 1e-6 or v_int < 1e-6 or
                                    (np.isfinite(info['re_corr']) and abs(info['re_corr']) > 0.99))
        return res, label, info
    return None, 'failed', dict(model='failed')


stat_rows = []
for rate in stiRate:
    sub = df[df.rate == rate]
    print(f'\n{rate} pps (LMM fits):')
    res, label, info = fit_lmm(sub, verbose=VERBOSE)
    if res is None:
        stat_rows.append(dict(rate=rate, n_units=len(sub),
                              n_animals=sub.animal.nunique(), model='failed'))
        continue
    ci = res.conf_int().loc['chan_c']
    stat_rows.append(dict(
        rate=rate, n_units=len(sub), n_animals=sub.animal.nunique(),
        beta_per_channel=res.params['chan_c'], ci_low=ci[0], ci_high=ci[1],
        z=res.tvalues['chan_c'], p=res.pvalues['chan_c'], **info))

lmm = pd.DataFrame(stat_rows)
for c in ('p', 'beta_per_channel', 'ci_low', 'ci_high'):
    if c not in lmm.columns:
        lmm[c] = np.nan
ok = lmm.p.notna()
lmm['p_corrected'] = np.nan
if ok.any():
    lmm.loc[ok, 'p_corrected'] = multipletests(lmm.loc[ok, 'p'], method=CORRECTION)[1]
lmm['beta_per_10ch'] = lmm.beta_per_channel * 10

pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 30)
print(f'\nLMM: index ~ channel, animal as random effect ({CORRECTION}-corrected across rates)')
print(lmm.round(4).to_string(index=False))
lmm.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_vs_channel_LMM.csv'), index=False)


def fmt_p(p):
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


stat_label = {}
for _, r in lmm.iterrows():
    if pd.notna(r['p_corrected']):
        stat_label[r['rate']] = (f"\u03b2 = {r['beta_per_channel']:.3f}/ch "
                                 f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]\n"
                                 f"{fmt_p(r['p_corrected'])} ({CORRECTION.capitalize()})")
    else:
        stat_label[r['rate']] = 'LMM failed'

# ---------------------------------------------------------------------------
# 3. Per-animal Spearman rho + Wilcoxon signed-rank test across animals
# ---------------------------------------------------------------------------
per_animal = []
for rate in stiRate:
    for animal in animals:
        a = df[(df.rate == rate) & (df.animal == animal)]
        if len(a) < MIN_UNITS or a.chan.nunique() < MIN_CHANNELS:
            continue
        rho, _ = stats.spearmanr(a.chan, a['index'])
        per_animal.append(dict(rate=rate, animal=animal, n_units=len(a),
                               n_channels=a.chan.nunique(), rho=rho))
pa = pd.DataFrame(per_animal)

wil_rows = []
for rate in stiRate:
    v = pa.loc[pa.rate == rate, 'rho'].dropna().values
    if len(v) < 5:
        wil_rows.append(dict(rate=rate, n_animals=len(v)))
        continue
    res_w = stats.wilcoxon(v, alternative='two-sided')
    wil_rows.append(dict(rate=rate, n_animals=len(v), median=np.median(v),
                         W=res_w.statistic, p=res_w.pvalue,
                         n_positive=int((v > 0).sum())))
wil = pd.DataFrame(wil_rows)
if 'p' in wil.columns and wil.p.notna().any():
    okw = wil.p.notna()
    wil.loc[okw, 'p_corrected'] = multipletests(wil.loc[okw, 'p'], method=CORRECTION)[1]

print(f'\nPer-animal Spearman rho (index vs channel):')
print(pa.round(3).to_string(index=False))
print(f'\nWilcoxon signed-rank test across animals (rho vs 0; {CORRECTION}-corrected):')
print(wil.round(4).to_string(index=False))
pa.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_per_animal_rho.csv'), index=False)
wil.to_csv(os.path.join(Figure_path, 'ITD_ILD_index_wilcoxon.csv'), index=False)

# ---------------------------------------------------------------------------
# 4. Main figure: box plot per channel, one column per rate
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, nrates, figsize=(15, 9.5), sharey=True, sharex=True,
                         gridspec_kw={'wspace': 0.08})
all_chans = np.arange(1, nchans + 1)

for ax, rate in zip(axes, stiRate):
    sub = df[df.rate == rate]
    chans = [c for c in all_chans if (sub.chan == c).any()]
    data = [sub.loc[sub.chan == c, 'index'].values for c in chans]
    ax.boxplot(data, positions=chans, vert=False, widths=BOX_WIDTH,
               showfliers=False, patch_artist=True,
               boxprops=dict(facecolor='whitesmoke', edgecolor='k', linewidth=0.8),
               medianprops=dict(color='k', linewidth=4),
               whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2),
               zorder=1)

    for animal in animals:
        a = sub[sub.animal == animal]
        if a.empty:
            continue
        ax.scatter(a['index'], a.y, marker='o', s=18, color=animal_colors[animal],
                   edgecolors='k', linewidth=0.3, alpha=0.8, zorder=3)

    ax.axvline(0, color='k', linestyle='--', linewidth=1, zorder=2)
    if rate == 1:
        ax.set_title('single pulse', fontsize=10)
    else: 
        # ax.set_title(f'{rate} pps\n{stat_label[rate]}', fontsize=10)
        ax.set_title(f'{rate} pps', fontsize=10)
    ax.set_xlabel('(VE_ITD - VE_ILD) / (VE_ITD + VE_ILD)')
    ax.set_xlim(*XLIM)
    ax.set_yticks(all_chans)
    ax.set_ylim((nchans + 0.5, 0.5) if INVERT_Y else (0.5, nchans + 0.5))
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    if rate == stiRate[0]:
        ax.set_ylabel('Channel number')
        ax.yaxis.set_label_coords(-0.08, 0.5)

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

animal_handles = [Line2D([0], [0], marker='o', color='w',
                         markerfacecolor=animal_colors[a], markeredgecolor='k',
                         markersize=8, label=animal_labels[a]) for a in animals]
fig.legend(handles=animal_handles, title='Animal', loc='upper left',
           bbox_to_anchor=(0.90, 0.92), fontsize=9, title_fontsize=10)

fig.tight_layout(rect=[0, 0, 0.90, 0.97])
out = os.path.join(Figure_path, 'ITD_ILD_index_boxplot_by_channel.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print('Saved', out)

#%%
# ---------------------------------------------------------------------------
# 5. Per-animal rho figure
# ---------------------------------------------------------------------------
fig2, ax2 = plt.subplots(figsize=(5, 5))
for i, rate in enumerate(stiRate):
    s = pa[pa.rate == rate]
    xj = i + rng.uniform(-0.1, 0.1, len(s))
    ax2.scatter(xj, s.rho, s=40, c=[animal_colors[a] for a in s.animal],
                edgecolors='k', zorder=3)
    if len(s):
        ax2.hlines(s.rho.median(), i - 0.25, i + 0.25, color='k', lw=2)
ax2.axhline(0, color='k', ls='--', lw=1)
ax2.set_xticks(range(len(stiRate)))
# ax2.set_xticklabels([f'{r} pps' for r in stiRate])
ax2.set_xticks(range(len(stiRate)))
ax2.set_xticklabels(['single pulse' if r == 1 else f'{r} pps' for r in stiRate])
ax2.set_ylabel('Spearman \u03c1 (ratio vs channel)')
fig2.tight_layout()
fig2.savefig(os.path.join(Figure_path, 'ITD_ILD_index_per_animal_rho.png'),
             dpi=300, bbox_inches='tight')

plt.show()