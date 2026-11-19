#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plot the signed ITD-ILD index by animal and estimate three-level variance components (animal, penetration-within-animal, and unit/residual) at each rate.

Index:
    signed_index = (VE_ITD - VE_ILD) / (VE_ITD + VE_ILD)

Input:
    all_files_VE_data_3to30ms.npz
Expected arrays:
    VE_value:  (nfiles, nrates, nchans, 2), last dimension [ITD, ILD]
    unit_index: (nfiles, nchans), 1 = responsive unit
    fnames:     one recording filename per file

Animal IDs follow the convention already used in this repository:
    animal = fnames[file_index].split('_')[0]

Variance model:
    signed_index ~ 1 + (1 | animal) + (1 | animal:penetration)

The model partitions total variance into animal, penetration-within-animal,
and unit/residual components. It is a descriptive variance decomposition,
not a test-retest reliability ICC.
"""

import os
import warnings
from sys import platform

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from scipy.stats import chi2


# ---------------------------------------------------------------------
# 1. Settings
# ---------------------------------------------------------------------
if platform == "darwin":
    RESULTS_PATH = "/Volumes/Disk/4_ITD_ILD_cueTrading/Data/AMUA_analysis_v2/"
    FIGURE_PATH = "/Volumes/Disk/4_ITD_ILD_cueTrading/Figures_paper/"
elif platform == "linux":
    RESULTS_PATH = "/disks/CIdata/2023_02_23Results/"
    FIGURE_PATH = "/disks/CIdata/2_ITD_ILD_CueTrading/Figures/"
else:
    RESULTS_PATH = "./Data/AMUA_analysis_v2/"
    FIGURE_PATH = "./Figures/"

NPZ_NAME = "all_files_VE_data_3to30ms.npz"

# The repository's established rate order.
STIMULUS_RATES = [1, 100, 900]
ITD_INDEX = 0
ILD_INDEX = 1

# True: include only responsive units (unit_index == 1), matching the
# existing VE ratio scripts. False: include every channel.
RESPONSIVE_UNITS_ONLY = True

os.makedirs(FIGURE_PATH, exist_ok=True)


# ---------------------------------------------------------------------
# 2. Load data and build one row per recording-unit-rate
# ---------------------------------------------------------------------
npz_path = os.path.join(RESULTS_PATH, NPZ_NAME)
if not os.path.isfile(npz_path):
    raise FileNotFoundError(
        f"Cannot find {npz_path}. Check RESULTS_PATH and NPZ_NAME."
    )

with np.load(npz_path, allow_pickle=True) as data:
    ve_value = data["VE_value"]
    unit_index = data["unit_index"]
    fnames = data["fnames"].tolist()

nfiles, nrates, nchans, _ = ve_value.shape

if nrates != len(STIMULUS_RATES):
    raise ValueError(
        f"The file contains {nrates} rates, but STIMULUS_RATES has "
        f"{len(STIMULUS_RATES)} entries. Update STIMULUS_RATES."
    )

rows = []
for file_i in range(nfiles):
    # Same animal-ID convention as the existing repository scripts.
    animal = str(fnames[file_i]).split("_")[0]

    if RESPONSIVE_UNITS_ONLY:
        channels = np.flatnonzero(unit_index[file_i] == 1)
    else:
        channels = np.arange(nchans)

    for channel_i in channels:
        for rate_i, rate in enumerate(STIMULUS_RATES):
            ve_itd = float(ve_value[file_i, rate_i, channel_i, ITD_INDEX])
            ve_ild = float(ve_value[file_i, rate_i, channel_i, ILD_INDEX])
            denominator = ve_itd + ve_ild

            if not np.isfinite(denominator) or denominator <= 0:
                signed = np.nan
            else:
                signed = (ve_itd - ve_ild) / denominator

            rows.append({
                "animal": animal,
                # Each recording/file corresponds to one penetration in this dataset.
                "penetration": str(fnames[file_i]),
                "file": file_i,
                "filename": str(fnames[file_i]),
                "channel": int(channel_i) + 1,
                "rate": rate,
                "VE_ITD": ve_itd,
                "VE_ILD": ve_ild,
                "signed_index": signed,
                "absolute_index": abs(signed) if np.isfinite(signed) else np.nan,
            })

df = pd.DataFrame(rows)
n_invalid = int(df["signed_index"].isna().sum())
if n_invalid:
    print(f"Excluded {n_invalid} rows with invalid/non-positive VE denominator.")

df = df.dropna(subset=["signed_index", "absolute_index"]).copy()

if df.empty:
    raise ValueError("No valid unit-rate observations remain after filtering.")

print("\nData summary")
print(f"Files/recordings: {nfiles}")
print(f"Animals: {df['animal'].nunique()}")
print(f"Valid unit-rate observations: {len(df)}")
# Number penetrations independently within each animal. The first recording
# for each animal is P1, the second P2, etc.; different animals can have
# different numbers of penetrations.
penetration_table = (
    df[["animal", "penetration"]]
    .drop_duplicates()
    .sort_values(["animal", "penetration"])
    .reset_index(drop=True)
)
penetration_table["penetration_number"] = (
    penetration_table.groupby("animal").cumcount() + 1
)

# Merge the unique animal/penetration lookup back onto every unit-rate row.
# This avoids mapping through a non-unique MultiIndex (which raises
# pandas.errors.InvalidIndexError when each penetration has many units/rates).
df = df.merge(
    penetration_table,
    on=["animal", "penetration"],
    how="left",
    validate="many_to_one",
)
print("\nPenetrations per animal:")
for animal, group in df[["animal", "penetration", "penetration_number"]].drop_duplicates().groupby("animal"):
    mapping = ", ".join(
        f"P{int(row.penetration_number)}={row.penetration}"
        for row in group.sort_values("penetration_number").itertuples()
    )
    print(f"  Rat {animal}: {mapping}")
print("\nObservations per animal and rate:")
print(df.groupby(["rate", "animal"]).size().unstack(fill_value=0).to_string())

# Save the row-level data so the plotted/statistical observations are auditable.
df.to_csv(
    os.path.join(FIGURE_PATH, "ITD_ILD_index_by_animal_unit_data.csv"),
    index=False
)


# ---------------------------------------------------------------------
# 3. Signed-index plotting function: one 1x3 figure (one panel per rate).
# X-axis labels are sequential animal numbers (1, 2, ...), not recording dates.
# Okabe-Ito colour-blind-friendly palette: clear, distinct colours.
# ---------------------------------------------------------------------
PENETRATION_PALETTE = [
    "C0",  # blue
    "C1",  # orange
    "C2",  # green
    "C3",  # red
    "C4",  # purple
]

UNIT_COLOR = "C0"  # colour used when all units are plotted identically


def penetration_color(number):
    """Return a colour for penetration P1, P2, ... within each animal."""
    if number <= len(PENETRATION_PALETTE):
        return PENETRATION_PALETTE[number - 1]
    # Use tab20 for animals with more than five penetrations.
    return plt.get_cmap("tab20")((number - len(PENETRATION_PALETTE) - 1) % 20)



def format_p_value(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


def plot_by_animal(
    data, value_col, ylabel, filename, variance_by_rate, ylim=None,
    color_mode="penetration"
):
    """
    Plot units by animal.

    color_mode="single":      all unit points use UNIT_COLOR.
    color_mode="penetration": points use P1/P2/... colours within each animal.
    """
    if color_mode not in ("single", "penetration"):
        raise ValueError("color_mode must be 'single' or 'penetration'.")

    animals = sorted(data["animal"].unique())
    animal_labels = {animal: str(i + 1) for i, animal in enumerate(animals)}

    fig, axes = plt.subplots(
        1, len(STIMULUS_RATES),
        figsize=(13, 4),
        sharey=True,
        squeeze=False
    )
    axes = axes.ravel()
    rng = np.random.default_rng(42)

    for ax, rate in zip(axes, STIMULUS_RATES):
        sub = data.loc[data["rate"] == rate]
        plot_data = [
            sub.loc[sub["animal"] == animal, value_col].dropna().to_numpy()
            for animal in animals
        ]

        nonempty = [(i, vals) for i, vals in enumerate(plot_data) if len(vals)]
        if nonempty:
            positions = [i + 1 for i, _ in nonempty]
            values = [vals for _, vals in nonempty]

            ax.boxplot(
                values,
                positions=positions,
                widths=0.62,
                showfliers=False,
                patch_artist=True,
                boxprops=dict(facecolor="whitesmoke", edgecolor="black"),
                medianprops=dict(color="tab:orange", linewidth=1.8),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
            )

            # Scatter individual units. Jitter is applied separately per animal.
            for pos, animal in enumerate(animals, start=1):
                animal_sub = sub.loc[sub["animal"] == animal]

                if color_mode == "single":
                    vals = animal_sub[value_col].dropna().to_numpy()
                    if len(vals):
                        x = rng.normal(pos, 0.055, size=len(vals))
                        ax.scatter(
                            x, vals, s=22, alpha=0.8,
                            color=UNIT_COLOR,
                            edgecolors="black", linewidths=0.25, zorder=3
                        )
                else:
                    # Penetration numbering and colours reset within each animal.
                    for penetration_number, pen_sub in animal_sub.groupby(
                        "penetration_number", sort=True
                    ):
                        vals = pen_sub[value_col].dropna().to_numpy()
                        if len(vals) == 0:
                            continue
                        x = rng.normal(pos, 0.055, size=len(vals))
                        ax.scatter(
                            x, vals, s=22, alpha=0.8,
                            color=penetration_color(int(penetration_number)),
                            edgecolors="black", linewidths=0.25, zorder=3
                        )

        rate_label = "Single pulse" if rate == 1 else f"{rate} pps"
        (
            animal_share,
            penetration_share,
            animal_p,
            penetration_p,
        ) = variance_by_rate.get(rate, (np.nan,) * 4)
        animal_label = (
            f"R_rat = {animal_share:.3f}, p = {animal_p:.3g}"
            if np.isfinite(animal_share) and np.isfinite(animal_p)
            else (f"R_rat = {animal_share:.3f}, p = NA"
                  if np.isfinite(animal_share) else "R_rat = NA")
        )
        penetration_label = (
            f"R_pen = {penetration_share:.3f}, p = {penetration_p:.3g}"
            if np.isfinite(penetration_share) and np.isfinite(penetration_p)
            else (f"R_pen = {penetration_share:.3f}, p = NA"
                  if np.isfinite(penetration_share) else "R_pen = NA")
        )
        # Three title rows: stimulus rate, animal variance fraction,
        # and penetration variance fraction.
        
        animal_p_label = format_p_value(animal_p)
        penetration_p_label = format_p_value(penetration_p)

        ax.set_title(
            f"{rate_label}\n"
            f"R_rat = {animal_share:.3f} ({animal_p_label})\n"
            f"R_pen = {penetration_share:.3f} ({penetration_p_label})"
            )
        
        
        # ax.set_title(
        #     f"{rate_label}\n{animal_label}\n{penetration_label}"
        # )
        ax.set_xticks(np.arange(1, len(animals) + 1))
        ax.set_xticklabels([animal_labels[a] for a in animals])
        ax.set_xlabel("Rat")
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, zorder=1)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        if ylim is not None:
            ax.set_ylim(*ylim)

    axes[0].set_ylabel(ylabel)

    # Only show the P1/P2/... legend when points are coloured by penetration.
    if color_mode == "penetration":
        from matplotlib.lines import Line2D

        max_penetrations = int(data["penetration_number"].max())
        legend_handles = [
            Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=penetration_color(i + 1),
                markeredgecolor="black", markeredgewidth=0.45,
                markersize=7, label=f"P{i + 1}"
            )
            for i in range(max_penetrations)
        ]
        fig.tight_layout(rect=(0, 0, 0.84, 1))
        fig.legend(
            handles=legend_handles,
            title="Penetration\n(within each rat)",
            loc="center left",
            bbox_to_anchor=(0.83, 0.5),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            handletextpad=0.5,
            labelspacing=0.7,
        )
    else:
        fig.tight_layout()

    out = os.path.join(FIGURE_PATH, filename)
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"Saved {out}")
    # plt.close(fig)


# ---------------------------------------------------------------------
# 4. Three-level variance decomposition, separately for each stimulus rate
# ---------------------------------------------------------------------
def estimate_three_level_variance(sub, value_col):
    """
    Fit a nested random-intercept model:

        value ~ 1 + (1 | animal) + (1 | animal:penetration)

    statsmodels implementation:
      - groups=animal gives the animal random intercept;
      - vc_formula gives a separate random intercept for each penetration
        within each animal;
      - residual variance represents unit-level/unexplained variation.
    """
    required = ["animal", "penetration", value_col]
    d = sub[required].dropna().copy()
    d["animal"] = d["animal"].astype(str)
    d["penetration"] = d["penetration"].astype(str)
    d = d.rename(columns={value_col: "y"})

    n_animals = d["animal"].nunique()
    n_penetrations = d[["animal", "penetration"]].drop_duplicates().shape[0]

    result = {
        "n_observations": len(d),
        "n_animals": n_animals,
        "n_penetrations": n_penetrations,
        "animal_variance": np.nan,
        "penetration_variance": np.nan,
        "unit_residual_variance": np.nan,
        "total_variance": np.nan,
        "animal_variance_fraction": np.nan,
        "penetration_variance_fraction": np.nan,
        "unit_residual_variance_fraction": np.nan,
        "ICC_same_penetration": np.nan,
        "animal_variance_LRT_p": np.nan,
        "penetration_variance_LRT_p": np.nan,
        "animal_variance_LRT_stat": np.nan,
        "penetration_variance_LRT_stat": np.nan,
        "model": "three-level random-intercept MixedLM",
        "status": "not fitted",
    }

    if n_animals < 2:
        result["status"] = "fewer than 2 animals"
        return result
    if n_penetrations <= n_animals:
        result["status"] = "not enough penetrations to estimate nesting"
        return result

    # With groups=animal, C(penetration) is evaluated separately within
    # each animal, so penetration effects are nested within animal.
    fits = []
    last_error = None
    for method in ("lbfgs", "powell", "bfgs"):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = smf.mixedlm(
                    "y ~ 1",
                    data=d,
                    groups=d["animal"],
                    re_formula="1",
                    vc_formula={"penetration": "0 + C(penetration)"},
                )
                fit = model.fit(reml=True, method=method, disp=False)

            if fit.converged and np.isfinite(fit.llf):
                fits.append(fit)
        except Exception as exc:
            last_error = exc

    if not fits:
        result["status"] = (
            "three-level MixedLM did not converge"
            + (f": {last_error}" if last_error is not None else "")
        )
        return result

    fit = max(fits, key=lambda x: x.llf)

    try:
        animal_var = float(np.asarray(fit.cov_re)[0, 0])
        # There is one variance-component term in vc_formula.
        penetration_var = float(np.asarray(fit.vcomp).ravel()[0])
        unit_var = float(fit.scale)
    except (IndexError, TypeError, ValueError) as exc:
        result["status"] = f"could not extract variance components: {exc}"
        return result

    total_var = animal_var + penetration_var + unit_var
    if (
        not np.isfinite(total_var) or total_var <= 0
        or not np.all(np.isfinite([animal_var, penetration_var, unit_var]))
    ):
        result["status"] = "invalid or non-positive total variance"
        return result

    result.update({
        "animal_variance": animal_var,
        "penetration_variance": penetration_var,
        "unit_residual_variance": unit_var,
        "total_variance": total_var,
        "animal_variance_fraction": animal_var / total_var,
        "penetration_variance_fraction": penetration_var / total_var,
        "unit_residual_variance_fraction": unit_var / total_var,
        "ICC_same_penetration": (animal_var + penetration_var) / total_var,
        "status": "OK",
    })

    # Likelihood-ratio tests for each variance component.
    # Use ML (reml=False), because likelihoods from REML fits with
    # different random-effects structures should not be compared.
    # The null value of a variance component is on the boundary (zero),
    # so use the 50:50 mixture of chi-square(0) and chi-square(1).
    d["pen_id"] = d["animal"] + "_" + d["penetration"]

    def fit_ml(groups, vc_formula=None):
        model_kwargs = {
            "formula": "y ~ 1",
            "data": d,
            "groups": groups,
            "re_formula": "1",
        }
        if vc_formula is not None:
            model_kwargs["vc_formula"] = vc_formula
        model_ml = smf.mixedlm(**model_kwargs)
        return model_ml.fit(reml=False, method="lbfgs", disp=False)

    try:
        full_ml = fit_ml(
            d["animal"],
            vc_formula={"penetration": "0 + C(pen_id)"},
        )
        no_animal_ml = fit_ml(d["pen_id"])
        no_penetration_ml = fit_ml(d["animal"])

        lr_animal = max(0.0, 2.0 * (full_ml.llf - no_animal_ml.llf))
        lr_penetration = max(
            0.0, 2.0 * (full_ml.llf - no_penetration_ml.llf)
        )

        p_animal = (
            0.5 * chi2.sf(lr_animal, df=1) if lr_animal > 0 else 1.0
        )
        p_penetration = (
            0.5 * chi2.sf(lr_penetration, df=1)
            if lr_penetration > 0 else 1.0
        )

        result.update({
            "animal_variance_LRT_stat": lr_animal,
            "penetration_variance_LRT_stat": lr_penetration,
            "animal_variance_LRT_p": p_animal,
            "penetration_variance_LRT_p": p_penetration,
        })
    except Exception as exc:
        result["status"] = f"variance model OK; LRT failed: {exc}"

    return result




variance_rows = []
for rate in STIMULUS_RATES:
    rate_data = df.loc[df["rate"] == rate]
    stats_result = estimate_three_level_variance(rate_data, "signed_index")
    variance_rows.append({
        "rate_pps": rate,
        "index_type": "signed",
        **stats_result,
    })

variance_df = pd.DataFrame(variance_rows)
variance_path = os.path.join(
    FIGURE_PATH, "ITD_ILD_index_three_level_variance_by_rate.csv"
)
variance_df.to_csv(variance_path, index=False)

print("\nThree-level variance decomposition by rate")
display_columns = [
    "rate_pps",
    "n_observations",
    "n_animals",
    "n_penetrations",
    "animal_variance_fraction",
    "penetration_variance_fraction",
    "unit_residual_variance_fraction",
    "ICC_same_penetration",
    "animal_variance_LRT_p",
    "penetration_variance_LRT_p",
    "status",
]
print(variance_df[display_columns].round(4).to_string(index=False))
print(f"\nSaved {variance_path}")

# Rate -> (between-animal, between-penetration) variance fractions
# for the three-line plot titles.
variance_by_rate = {
    int(row.rate_pps): (
        row.animal_variance_fraction,
        row.penetration_variance_fraction,
        row.animal_variance_LRT_p,
        row.penetration_variance_LRT_p,
    )
    for row in variance_df.itertuples()
}


# Generate both colour styles:
# 1) all units in one colour;
# 2) units coloured by penetration (P1-Pn) within each animal.
plot_by_animal(
    df,
    value_col="signed_index",
    ylabel="CDI (ILD-dominated < 0 < ITD-dominated)",
    filename="ITD_ILD_signed_index_boxplot_by_animal_single_color.png",
    variance_by_rate=variance_by_rate,
    ylim=(-1.05, 1.05),
    color_mode="single",
)

plot_by_animal(
    df,
    value_col="signed_index",
    ylabel="CDI (ILD-dominated < 0 < ITD-dominated)",
    filename="ITD_ILD_signed_index_boxplot_by_animal_by_penetration.png",
    variance_by_rate=variance_by_rate,
    ylim=(-1.05, 1.05),
    color_mode="penetration",
)

print("\nFinished.")
