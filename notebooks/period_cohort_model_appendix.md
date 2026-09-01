---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

# Period–cohort models

[Run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/YouWouldChooseNow/blob/main/notebooks/period_cohort_model_appendix.ipynb)

This notebook is supporting material for [*You Would Choose Now: Measuring America's Progress Toward Fairness and Tolerance*](https://leanpub.com/ywcn).
It walks through the period–cohort models from the talk, using the same General Social Survey item as the time series model appendix: sexual relations between two adults of the same sex.

The outcome is the percent saying such relations are wrong to any degree (GSS response codes 1–3: always wrong, almost always wrong, or wrong only sometimes).
We start from weighted cohort × survey-year cells, look at the raw trajectories, then fit an additive period–cohort model with second-order random walks on cohort and period plus year-level overdispersion (no separate age effect: age is implied by survey year and birth cohort).
Finally we fit a separate time series model within each birth decade, so you can compare the shared period–cohort surface to decade-specific trends.

## Colab setup

On Google Colab, upgrade the sampling stack before importing PyMC and nutpie.

```python
# Colab ships an older PyMC; installing nutpie alone can pull ArviZ 1.x and break imports.
# Upgrade the sampling stack to get pymc>=6, arviz>=1.1, nutpie.
import sys

if "google.colab" in sys.modules:
    %pip install -q "pymc>=6" "arviz>=1.1" nutpie
```

## Imports

Load the libraries used throughout the notebook.

```python
import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.interpolate import interp1d
from scipy.special import expit

np.random.seed(17)
```

## Configuration

Parameter cell: outcome variable, paths, figure styling, sampling defaults, and RW priors.

```python tags=["parameters"]
VARIABLE = "homosex"
WEIGHT_COL = "wtsmerge"
GSS_FILENAME = "gss_extract_2024_r3.hdf"
GSS_HDF_KEY = "gss"

# Outcome coding by variable (extend this dict to reuse the notebook on other items)
SELECTED_VALUES = {
    "homosex": [1, 2, 3],  # always, almost always, wrong only sometimes
}
SELECTED_CODES = SELECTED_VALUES[VARIABLE]

DISPLAY_TITLE = "What about sexual relations between two adults of the same sex?"
OUTCOME_YLABEL = "Percent saying wrong (any degree)"
SOURCE_SUBTEXT = "Source: General Social Survey 1972-2024"
FACET_COHORT_TRAJECTORIES = "Cohort trajectories"
FACET_PERIOD_EFFECT = "Period effect"
FACET_COHORT_EFFECT = "Cohort effect"
AXIS_LABEL_PERCENT = "Percent"
YMAX = 100
GSS_FIG_YEAR_PAD = 1.5
GSS_FIG_COHORT_PAD = 2.0
GSS_FIG_COHORT_TRAJECTORY_XHI_PAD = 2.5
PERIOD_EFFECT_COLOR = "#5f3a8f"
COHORT_EFFECT_COLOR = "#b15928"
FIGSIZE = (9.0 * 0.75, 4.5 * 0.75)
FIG_DPI = 300
FIG_TITLE_PAD = 42
FIG_SUBTITLE_Y = 1.035

TARGET_ACCEPT = 0.95
DRAWS = 1000
TUNE = 1000
FORCE_RUN = False

SIGMA_C_SCALE = 0.125
SIGMA_T_SCALE = 0.125
SIGMA_SHORT_SCALE_PC = 0.2
SIGMA_SHORT_SCALE_TM = 0.5

# Skip birth decades with too few survey years for a stable RW2
MIN_OBSERVED_YEARS = 5

# GSS fielding → two-year nominal lattice: odds round up; these round down
GSS_NOMINAL_YEAR_EXCEPTIONS = {
    1993: 1992,
    2021: 2020,
}
```

## Nominal year lattice

Map fielding years onto the two-year nominal period grid used in the period–cohort model.

```python
def gss_nominal_year(year: int) -> int:
    y = int(year)
    return GSS_NOMINAL_YEAR_EXCEPTIONS.get(y, y + (y % 2))
```

## Output directories

Create local folders for saved figures and posterior NetCDF files.

```python
os.makedirs("figs", exist_ok=True)
os.makedirs("idata", exist_ok=True)
```

## Plot styling helpers

Small utilities for left-aligned titles, subtitles, source lines, and spine removal.

```python
def add_title(title, *, pad=FIG_TITLE_PAD):
    plt.title(title, loc="left", pad=pad)


def add_subtitle(subtitle):
    ax = plt.gca()
    ax.text(
        0,
        FIG_SUBTITLE_Y,
        subtitle,
        transform=ax.transAxes,
        color="0.2",
        ha="left",
        va="bottom",
        fontsize=9,
        linespacing=1.5,
        clip_on=False,
    )


def add_subtext(text, *, y=-0.18):
    ax = plt.gca()
    return plt.figtext(
        0,
        y,
        text,
        ha="left",
        va="top",
        fontsize=8,
        transform=ax.transAxes,
    )


def remove_spines():
    """Remove the spines of a plot but keep the ticks visible."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")
```

## Sampling helper

Load a saved posterior from disk when available; otherwise sample and write NetCDF.

```python
def load_idata_or_sample(model, filename, *, force_run=False, **sample_options):
    """Load NetCDF if present (unless ``force_run``), else ``pm.sample`` and save."""
    path = Path(filename)
    if path.is_file() and not force_run:
        return az.from_netcdf(path)
    with model:
        idata = pm.sample(**sample_options)
    idata.to_netcdf(path)
    return idata
```

## Axis limit helpers

```python
def xlim_year(years):
    s = pd.Series(years).dropna()
    return (float(s.min()) - GSS_FIG_YEAR_PAD, float(s.max()) + GSS_FIG_YEAR_PAD)


def xlim_cohort(cohorts):
    s = pd.Series(cohorts).dropna()
    return (float(s.min()) - GSS_FIG_COHORT_PAD, float(s.max()) + GSS_FIG_COHORT_PAD)
```

## RW2 helper

Shared second-order random walk: non-centered second differences, free initial slope with level fixed at 0, then a mean-centered path.

```python
def rw2(name, n, sigma, *, dims=None):
    """Mean-centered RW2 with a free initial slope.

    The initial level is fixed at zero because any constant offset is
    removed when the completed path is mean-centered. Second differences
    use a non-centered parameterization with scale ``sigma``. The initial
    slope is ``Normal(0, 0.2)``.

    Parameters
    ----------
    name : str
        Prefix for latent variables (e.g. ``"cohort"`` or ``"period"``).
    n : int
        Length of the latent path.
    sigma : tensor or float
        Scale of the second-difference innovations (usually an ``Exponential`` RV).
    dims : str or sequence of str, optional
        Coordinate dims passed to ``pm.Deterministic``.
    """
    z_delta = pm.Normal(f"{name}_z_delta", mu=0.0, sigma=1.0, shape=n - 2)
    delta = sigma * z_delta
    v0 = pm.Normal(f"{name}_slope_init", mu=0.0, sigma=0.2)
    # First differences (length n-1)
    rw1 = v0 + pm.math.concatenate(
        [pm.math.zeros(1), pm.math.cumsum(delta)]
    )
    # Cumulative sum of rw1 (length n), then mean-center
    raw = pm.math.concatenate(
        [pm.math.zeros(1), pm.math.cumsum(rw1)]
    )
    return pm.Deterministic(name, raw - pm.math.mean(raw), dims=dims)
```

## Period–cohort model builder

Additive period–cohort specification: RW2 cohort + RW2 period + year-level overdispersion, Binomial cells.

```python
def build_period_cohort_model(cells, cohorts, years_grid):
    """Additive period–cohort: RW2 cohort + RW2 period + year-level overdispersion, Binomial cells."""
    n_cohorts = len(cohorts)
    n_years = len(years_grid)
    coords = {
        "cohort_idx": cohorts,
        "year_idx": years_grid,
        "cell_idx": range(len(cells)),
    }
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        sigma_c = pm.Exponential("sigma_c", lam=1.0 / SIGMA_C_SCALE)
        sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
        cohort = rw2("cohort", n_cohorts, sigma_c, dims="cohort_idx")
        period = rw2("period", n_years, sigma_t, dims="year_idx")
        eta = pm.Deterministic(
            "eta",
            alpha + cohort[cells["cohort_idx"].values] + period[cells["year_idx"].values],
            dims="cell_idx",
        )
        sigma_short = pm.Exponential("sigma_short", lam=1.0 / SIGMA_SHORT_SCALE_PC)
        epsilon_z = pm.ZeroSumNormal("epsilon_z", sigma=1.0, dims="year_idx")
        epsilon = pm.Deterministic("epsilon", sigma_short * epsilon_z, dims="year_idx")
        p_obs = pm.math.sigmoid(eta + epsilon[cells["year_idx"].values])
        pm.Binomial(
            "y",
            n=cells["n"].values,
            p=p_obs,
            observed=cells["y"].values,
            dims="cell_idx",
        )
    return model
```

## Time series model builder

Time series model used for decade fits: RW2 period + year-level overdispersion on Kish yearly counts.

```python
def build_period_model(observed, years_grid):
    """Time series model: RW2 period + year-level overdispersion on Kish yearly counts."""
    n_years = len(years_grid)
    year_idx_obs = observed["year_idx"].to_numpy()
    p_w = observed["p_weighted"].to_numpy(dtype=float)
    n_eff = np.round(observed["n_eff"].to_numpy(dtype=float)).astype(int)
    y_eff = np.clip(np.round(p_w * n_eff).astype(int), 0, n_eff)
    coords = {"year_idx": years_grid}
    with pm.Model(coords=coords) as model:
        alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
        sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
        period = rw2("period", n_years, sigma_t, dims="year_idx")
        eta = pm.Deterministic("eta", alpha + period, dims="year_idx")
        support = pm.Deterministic("support", pm.math.sigmoid(eta), dims="year_idx")
        sigma_short = pm.HalfNormal("sigma_short", sigma=SIGMA_SHORT_SCALE_TM)
        epsilon_z = pm.ZeroSumNormal("epsilon_z", sigma=1.0, dims="year_idx")
        epsilon = pm.Deterministic("epsilon", sigma_short * epsilon_z, dims="year_idx")
        p_obs = pm.math.sigmoid(eta[year_idx_obs] + epsilon[year_idx_obs])
        pm.Binomial("y", n=n_eff, p=p_obs, observed=y_eff)
    return model
```

## Posterior summary and interpolation helpers

```python
def summarize_probability_draws(prob_samples, axis=(0, 1)):
    """Posterior mean and equal-tailed 84% interval (8th–92nd percentiles)."""
    mean = prob_samples.mean(axis=axis)
    lower = np.quantile(prob_samples, 0.08, axis=axis)
    upper = np.quantile(prob_samples, 0.92, axis=axis)
    return pd.DataFrame({"mean": mean, "lower": lower, "upper": upper})


def interpolate_1d_for_plot(x, y, step=0.5):
    """Smooth a 1D series for plotting (cubic / quadratic / linear by n points)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    n = len(x)
    if n < 2:
        return x, y
    kind = "linear" if n == 2 else ("quadratic" if n == 3 else "cubic")
    x_lo, x_hi = float(x.min()), float(x.max())
    x_fine = np.arange(x_lo, x_hi + step, step)
    x_fine = x_fine[(x_fine >= x_lo) & (x_fine <= x_hi)]
    f = interp1d(x, y, kind=kind, bounds_error=False, fill_value=(y[0], y[-1]))
    return x_fine, f(x_fine)


def interpolate_1d_band_for_plot(x, y_lower, y_mean, y_upper, step=0.5):
    x = np.asarray(x, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y_lower, y_mean, y_upper = y_lower[order], y_mean[order], y_upper[order]
    n = len(x)
    if n < 2:
        return x, y_lower, y_mean, y_upper
    kind = "linear" if n == 2 else ("quadratic" if n == 3 else "cubic")
    x_lo, x_hi = float(x.min()), float(x.max())
    x_fine = np.arange(x_lo, x_hi + step, step)
    x_fine = x_fine[(x_fine >= x_lo) & (x_fine <= x_hi)]
    fl = interp1d(x, y_lower, kind=kind, bounds_error=False, fill_value=(y_lower[0], y_lower[-1]))
    fm = interp1d(x, y_mean, kind=kind, bounds_error=False, fill_value=(y_mean[0], y_mean[-1]))
    fh = interp1d(x, y_upper, kind=kind, bounds_error=False, fill_value=(y_upper[0], y_upper[-1]))
    return x_fine, fl(x_fine), fm(x_fine), fh(x_fine)


def plot_probability_summary_band(ax, df, x_col, *, color, label=None):
    """Mean + percentile band from probability-scale summary (rendered as percent)."""
    x = df[x_col].values
    mean_pct = df["mean"].values * 100
    lower_pct = df["lower"].values * 100
    upper_pct = df["upper"].values * 100
    if len(x) >= 2:
        xf, lo, m, hi = interpolate_1d_band_for_plot(x, lower_pct, mean_pct, upper_pct)
        ax.fill_between(xf, lo, hi, color=color, alpha=0.25, linewidth=0, zorder=3)
        ax.plot(xf, m, color=color, linewidth=2, zorder=4, label=label)
    else:
        ax.fill_between(x, lower_pct, upper_pct, color=color, alpha=0.25, linewidth=0, zorder=3)
        ax.plot(x, mean_pct, color=color, linewidth=2, zorder=4, label=label)


def make_subtitle(facet_line: str) -> str:
    return f"{OUTCOME_YLABEL}\n{facet_line}"
```

## Trajectory plotting helper

One line per group; connect-the-dots unless ``interpolate`` (model surfaces).

```python
def plot_raw_trajectories(
    traj: pd.DataFrame,
    *,
    group_col: str,
    fig_stem: str,
    facet_line: str,
    linewidth: float = 0.8,
    interpolate: bool = False,
) -> Path:
    """One line per group; connect-the-dots unless ``interpolate`` (model surfaces)."""
    groups = sorted(traj[group_col].unique())
    cmap = plt.cm.plasma.reversed()
    norm = plt.Normalize(vmin=min(groups), vmax=max(groups))
    bbox_style = dict(boxstyle="round,pad=0.1", facecolor="white", linewidth=0)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for group in groups:
        mask = traj[group_col] == group
        years = traj.loc[mask, "nominal_year"].to_numpy()
        vals = traj.loc[mask, "p_raw"].to_numpy() * 100
        order = np.argsort(years)
        y_ord, v_ord = years[order], vals[order]
        if len(y_ord) == 0:
            continue
        if interpolate and len(y_ord) >= 2:
            y_plot, v_plot = interpolate_1d_for_plot(y_ord, v_ord)
            ax.plot(y_plot, v_plot, color=cmap(norm(group)), linewidth=linewidth, alpha=0.85)
            x_start, v_start = float(y_plot[0]), float(v_plot[0])
            x_end, v_end = float(y_plot[-1]), float(v_plot[-1])
        elif len(y_ord) == 1:
            y0, v0 = float(y_ord[0]), float(v_ord[0])
            ax.plot([y0 - 1, y0], [v0, v0], color=cmap(norm(group)), linewidth=linewidth, alpha=0.85)
            x_start, v_start = y0 - 1, v0
            x_end, v_end = y0, v0
        else:
            ax.plot(y_ord, v_ord, color=cmap(norm(group)), linewidth=linewidth, alpha=0.85)
            x_start, v_start = float(y_ord[0]), float(v_ord[0])
            x_end, v_end = float(y_ord[-1]), float(v_ord[-1])

        if group % 10 == 0:
            label = str(int(group))
            if group <= 1950:
                ax.annotate(
                    label,
                    xy=(x_end, v_end),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha="left",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox=bbox_style,
                )
            else:
                ax.annotate(
                    label,
                    xy=(x_start, v_start),
                    xytext=(-5, 0),
                    textcoords="offset points",
                    ha="right",
                    va="center",
                    fontsize=9,
                    color="black",
                    bbox=bbox_style,
                )

    ax.set_xlabel("Survey year")
    ax.set_ylabel(AXIS_LABEL_PERCENT)
    ax.set_xlim(fig_xlim)
    ax.set_ylim(0, YMAX)
    ax.grid(True, alpha=0.3)

    remove_spines()
    add_title(DISPLAY_TITLE)
    add_subtitle(make_subtitle(facet_line))
    subtext_obj = add_subtext(SOURCE_SUBTEXT)

    fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_{fig_stem}.png"
    fig.savefig(
        str(fig_path),
        dpi=FIG_DPI,
        bbox_inches="tight",
        bbox_extra_artists=[subtext_obj],
    )
    return fig_path
```

## Load data

Define a small helper that fetches the GSS extract only if it is not already on disk.

```python
from urllib.request import urlretrieve


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        local, _ = urlretrieve(url, filename=str(path))
        print("Downloaded " + local)
```

Pick a data directory for repo vs Colab layouts, download the HDF file, and load the GSS table.

```python
# Repo layout: run from notebooks/ → ../data. Colab / flat checkout → ./data
DATA_DIR = Path("..") / "data" if (Path("..") / "Makefile").is_file() else Path("data")
GSS_PATH = DATA_DIR / GSS_FILENAME

download(
    f"https://github.com/AllenDowney/YouWouldChooseNow/raw/main/data/{GSS_FILENAME}",
    GSS_PATH,
)

micro = pd.read_hdf(GSS_PATH, GSS_HDF_KEY).copy()
```

Code the binary outcome, map fielding years onto the nominal lattice, and keep respondents born in 1900 or later.

```python
micro["y"] = micro[VARIABLE].isin(SELECTED_CODES).astype(float)
micro.loc[micro[VARIABLE].isna(), "y"] = np.nan
micro = micro.dropna(subset=["cohort", "year", "y", WEIGHT_COL])
micro["cohort"] = micro["cohort"].round().astype(int)
micro["year"] = micro["year"].astype(int)
micro = micro.loc[micro["cohort"] >= 1900].copy()
micro["nominal_year"] = micro["year"].map(gss_nominal_year)
if micro["nominal_year"].isna().any():
    missing = sorted(micro.loc[micro["nominal_year"].isna(), "year"].unique())
    raise ValueError(f"Survey years could not be mapped to nominal years: {missing}")
micro["nominal_year"] = micro["nominal_year"].astype(int)
```

Compute weighted cell statistics for each birth-cohort × nominal-year pair.

```python
def _weighted_cell_stats(group: pd.DataFrame) -> pd.Series:
    y_vals = group["y"].to_numpy(dtype=float)
    weights = group[WEIGHT_COL].to_numpy(dtype=float)
    total_weight = float(weights.sum())
    if total_weight <= 0 or len(weights) == 0:
        return pd.Series({"y": 0.0, "n": 0.0, "sum_weight": 0.0, "p_weighted": 0.0})
    p_weighted = float((y_vals * weights).sum() / total_weight)
    n_eff = (total_weight**2) / float((weights**2).sum())
    return pd.Series(
        {
            "y": p_weighted * n_eff,
            "n": float(n_eff),
            "sum_weight": total_weight,
            "p_weighted": p_weighted,
        }
    )
```

Aggregate to cohort × nominal-year cells and build the even-year period lattice.

```python
cells = (
    micro.groupby(["cohort", "nominal_year"], group_keys=False)
    .apply(_weighted_cell_stats, include_groups=False)
    .reset_index()
)
cells["y"] = cells["y"].round().astype(int)
cells["n"] = cells["n"].round().astype(int)
cells["p_raw"] = cells["p_weighted"].astype(float)
cells["year"] = cells["nominal_year"]

cohorts = sorted(cells["cohort"].unique())
years_observed = sorted(cells["nominal_year"].unique())
nominal_year_min = int(min(years_observed))
nominal_year_max = int(max(years_observed))
# Even-year period lattice between first and last observed nominal year
y0 = nominal_year_min + (nominal_year_min % 2)
y1 = nominal_year_max - (nominal_year_max % 2)
years_grid = list(range(y0, y1 + 1, 2))

cohort_to_idx = {c: i for i, c in enumerate(cohorts)}
year_to_idx_grid = {y: i for i, y in enumerate(years_grid)}
cells["cohort_idx"] = cells["cohort"].map(cohort_to_idx)
cells["year_idx"] = cells["nominal_year"].map(year_to_idx_grid)

n_cohorts = len(cohorts)
n_years_grid = len(years_grid)
n_years_observed = len(years_observed)
n_respondents_used = len(micro)

fig_xlim = xlim_year(cells["nominal_year"])
fig_xlim = (fig_xlim[0], fig_xlim[1] + GSS_FIG_COHORT_TRAJECTORY_XHI_PAD)
```

Print coverage summaries and inspect cell sample sizes.

```python
print(f"Respondents used: {n_respondents_used:,}")
print(f"Cells: {len(cells):,}")
print(f"Cohorts: {cells['cohort'].min()}–{cells['cohort'].max()} ({n_cohorts})")
print(
    f"Nominal years observed: {n_years_observed} "
    f"({min(years_observed)}–{max(years_observed)}); "
    f"full grid {n_years_grid} nodes"
)
cells["n"].describe()
```

## Raw trajectories

One line per birth year: observed cell percent by nominal survey year. Connect the dots (no smooth interpolation). No model — noise and sparsity are the point.

```python
plot_raw_trajectories(
    cells,
    group_col="cohort",
    fig_stem="raw_cohort_trajectories",
    facet_line=f"{FACET_COHORT_TRAJECTORIES} (Weighted %)",
    linewidth=0.8,
)
```

Pool cells within each birth decade (weighted by cell $n$), then plot the same connect-the-dots view by decade.

```python
cells_dec = cells.copy()
cells_dec["decade"] = (cells_dec["cohort"] // 10) * 10
decade_cells = (
    cells_dec.groupby(["decade", "nominal_year"], as_index=False)
    .agg(y=("y", "sum"), n=("n", "sum"))
)
decade_cells["p_raw"] = decade_cells["y"].astype(float) / decade_cells["n"].astype(float)

plot_raw_trajectories(
    decade_cells,
    group_col="decade",
    fig_stem="raw_decade_trajectories",
    facet_line="Decade trajectories (weighted %)",
    linewidth=1.6,
)
```

Cohort × nominal year heatmap; empty cells stay blank so sparsity is visible.

```python
pivot = cells.pivot_table(
    index="cohort",
    columns="nominal_year",
    values="p_raw",
    aggfunc="mean",
)
pivot_pct = pivot * 100

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.imshow(
    pivot_pct.to_numpy(),
    aspect="auto",
    origin="upper",
    cmap="viridis",
    vmin=0,
    vmax=100,
    interpolation="nearest",
)
ax.set_xlabel("Survey year (nominal)")
ax.set_ylabel("Birth cohort")

year_cols = list(pivot_pct.columns)
cohort_idx = list(pivot_pct.index)
year_tick_pos = [i for i, y in enumerate(year_cols) if int(y) % 10 == 0]
ax.set_xticks(year_tick_pos)
ax.set_xticklabels([str(int(year_cols[i])) for i in year_tick_pos])
cohort_tick_pos = [i for i, c in enumerate(cohort_idx) if int(c) % 10 == 0]
ax.set_yticks(cohort_tick_pos)
ax.set_yticklabels([str(int(cohort_idx[i])) for i in cohort_tick_pos])

remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle("Weighted % (cohort × year)"))
subtext_obj = add_subtext(SOURCE_SUBTEXT)

fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_raw_heatmap.png"
fig.savefig(
    str(fig_path),
    dpi=FIG_DPI,
    bbox_inches="tight",
    bbox_extra_artists=[subtext_obj],
)
```

## Model 1: Additive period–cohort model

Fit the full additive specification used in the talk:

$$
\eta_{c,t} = \alpha + \mathrm{cohort}_c + \mathrm{period}_t,\quad
p_{c,t} = \mathrm{sigmoid}(\eta_{c,t} + \epsilon_t)
$$

with second-order random walks on both cohort and period. The model includes a year-level shock $\epsilon_t$, shared by all cohorts observed in year $t$. The shocks use a non-centered zero-sum normal prior, separating short-term period variation from the intercept and from the smooth RW2 period effect:

$$
z \sim \operatorname{ZeroSumNormal}(0,1),\quad
\epsilon_t = \sigma_{\mathrm{short}} z_t,\quad
\sum_t \epsilon_t = 0.
$$

The plotted trajectories below use the smooth surface $\mathrm{sigmoid}(\alpha + \mathrm{cohort}_c + \mathrm{period}_t)$ without $\epsilon$.

Define the model:

```python
model_pc = build_period_cohort_model(cells, cohorts, years_grid)
```

Visualize the model structure:

```python
model_pc.to_graphviz()
```

Sample or load the posterior:

```python
idata_path_pc = Path("idata") / f"model10_appendix_{VARIABLE}_rw2_overdisp.nc"

idata_pc = load_idata_or_sample(
    model_pc,
    str(idata_path_pc),
    force_run=FORCE_RUN,
    draws=DRAWS,
    tune=TUNE,
    target_accept=TARGET_ACCEPT,
    nuts_sampler="nutpie",
    nuts=dict(
        adaptation="low_rank",
        mass_matrix_eigval_cutoff=6.0,
        mass_matrix_gamma=1e-5,
    ),
)
```

Summarize the period–cohort fit:

```python
az.summary(
    idata_pc,
    var_names=["alpha", "sigma_c", "sigma_t", "sigma_short"],
    round_to=3,
)
print(
    "max |sum_t epsilon|:",
    float(np.abs(idata_pc.posterior["epsilon"].sum("year_idx")).max()),
)
```

Build posterior-mean cell probabilities $\mathrm{sigmoid}(\alpha + \mathrm{cohort}_c + \mathrm{period}_t)$ for plotting. For cohorts observed only once, add a ghost cell one year earlier: the logit midpoint between the model value at $t-2$ and the observed year $t$.

```python
alpha_mean = float(idata_pc.posterior["alpha"].mean(dim=("chain", "draw")).values)
cohort_mean = idata_pc.posterior["cohort"].mean(dim=("chain", "draw")).values
period_mean = idata_pc.posterior["period"].mean(dim=("chain", "draw")).values

model_cohorts = np.asarray(idata_pc.posterior.coords["cohort_idx"].values, dtype=int)
model_years = np.asarray(idata_pc.posterior.coords["year_idx"].values, dtype=int)
cohort_to_idx = {int(c): i for i, c in enumerate(model_cohorts)}
year_to_idx = {int(y): i for i, y in enumerate(model_years)}

traj_pc = cells.copy()
traj_pc["c_idx"] = traj_pc["cohort"].map(cohort_to_idx)
traj_pc["y_idx"] = traj_pc["nominal_year"].map(year_to_idx)
traj_pc = traj_pc.dropna(subset=["c_idx", "y_idx"]).copy()
traj_pc["c_idx"] = traj_pc["c_idx"].astype(int)
traj_pc["y_idx"] = traj_pc["y_idx"].astype(int)
traj_pc["p_raw"] = expit(
    alpha_mean + cohort_mean[traj_pc["c_idx"].to_numpy()] + period_mean[traj_pc["y_idx"].to_numpy()]
)

traj_pc_dec = traj_pc.copy()
traj_pc_dec["decade"] = (traj_pc_dec["cohort"] // 10) * 10
decade_pc = pd.DataFrame(
    [
        {
            "decade": int(decade),
            "nominal_year": int(year),
            "p_raw": float(np.average(g["p_raw"], weights=g["n"])),
        }
        for (decade, year), g in traj_pc_dec.groupby(["decade", "nominal_year"])
    ]
)

cohort_counts = traj_pc.groupby("cohort").size()
single_cohorts = cohort_counts[cohort_counts == 1].index.tolist()
ghost_rows = []
for cohort in single_cohorts:
    row = traj_pc.loc[traj_pc["cohort"] == cohort].iloc[0]
    obs_year = int(row["nominal_year"])
    ghost_year = obs_year - 1
    prev_year = obs_year - 2
    if prev_year not in year_to_idx:
        continue
    c_idx = int(row["c_idx"])
    eta_obs = alpha_mean + cohort_mean[c_idx] + period_mean[int(row["y_idx"])]
    eta_prev = alpha_mean + cohort_mean[c_idx] + period_mean[year_to_idx[prev_year]]
    eta_ghost = 0.5 * (eta_prev + eta_obs)
    ghost_rows.append(
        {
            "cohort": int(cohort),
            "nominal_year": ghost_year,
            "p_raw": float(expit(eta_ghost)),
            "n": np.nan,
            "c_idx": c_idx,
            "y_idx": np.nan,
        }
    )

if ghost_rows:
    traj_pc = pd.concat([traj_pc, pd.DataFrame(ghost_rows)], ignore_index=True)
```

Plot model cohort trajectories (one line per birth year, with light interpolation for plotting).

```python
plot_raw_trajectories(
    traj_pc,
    group_col="cohort",
    fig_stem="model10_cohort_trajectories",
    facet_line=f"{FACET_COHORT_TRAJECTORIES} (period–cohort model)",
    linewidth=0.8,
    interpolate=True,
)
```

Plot model decade trajectories.

```python
plot_raw_trajectories(
    decade_pc,
    group_col="decade",
    fig_stem="model10_decade_trajectories",
    facet_line="Decade trajectories (period–cohort model)",
    linewidth=1.6,
    interpolate=True,
)
```

Compute standardized period and cohort effect curves. Period curve uses a fixed reference cohort mix (weights proportional to summed survey weights in each cohort); cohort curve uses uniform weight on observed nominal years. Bands are equal-tailed 84% intervals on the probability scale. These curves use the smooth surface $\mathrm{sigmoid}(\alpha + \mathrm{cohort}_c + \mathrm{period}_t)$ and omit year-level $\epsilon$.

```python
alpha_samples = idata_pc.posterior["alpha"].values
cohort_samples = idata_pc.posterior["cohort"].values
period_samples = idata_pc.posterior["period"].values

ref_dist = (
    cells.loc[cells["cohort"].isin(model_cohorts)]
    .groupby("cohort", as_index=False)["sum_weight"]
    .sum()
    .rename(columns={"sum_weight": "w_ref"})
)
ref_dist["weight"] = ref_dist["w_ref"] / ref_dist["w_ref"].sum()
ref_dist = ref_dist.sort_values("cohort").reset_index(drop=True)
reference_cohort_idx = np.array([cohort_to_idx[int(c)] for c in ref_dist["cohort"]], dtype=int)

eta_standardized = (
    alpha_samples[:, :, None, None]
    + cohort_samples[:, :, reference_cohort_idx][:, :, :, None]
    + period_samples[:, :, None, :]
)
p_standardized = expit(eta_standardized)
p_standardized_period = np.tensordot(p_standardized, ref_dist["weight"].to_numpy(), axes=([2], [0]))

standardized_df = summarize_probability_draws(p_standardized_period)
standardized_df.insert(0, "nominal_year", model_years)

observed_years = np.array(
    sorted(set(cells["nominal_year"].astype(int)).intersection(set(model_years.tolist()))),
    dtype=int,
)
observed_year_idx = np.array([year_to_idx[int(y)] for y in observed_years], dtype=int)

eta_cohort_uniform_years = (
    alpha_samples[:, :, None, None]
    + cohort_samples[:, :, :, None]
    + period_samples[:, :, None, observed_year_idx]
)
p_cohort_uniform_years = expit(eta_cohort_uniform_years).mean(axis=3)

cohort_uniform_years_df = summarize_probability_draws(p_cohort_uniform_years)
cohort_uniform_years_df.insert(0, "cohort", model_cohorts)
```

Plot the standardized period effect.

```python
fig, ax = plt.subplots(figsize=FIGSIZE)
plot_probability_summary_band(ax, standardized_df, "nominal_year", color=PERIOD_EFFECT_COLOR)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, YMAX)
ax.set_xlim(fig_xlim)
ax.set_xlabel("Survey year")
ax.set_ylabel(AXIS_LABEL_PERCENT)
remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle(FACET_PERIOD_EFFECT))
subtext_obj = add_subtext(SOURCE_SUBTEXT)
fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_time_standardized.png"
fig.savefig(str(fig_path), dpi=FIG_DPI, bbox_inches="tight", bbox_extra_artists=[subtext_obj])
```

Plot the standardized cohort effect.

```python
fig_xlim_cohort = xlim_cohort(model_cohorts)
fig, ax = plt.subplots(figsize=FIGSIZE)
plot_probability_summary_band(ax, cohort_uniform_years_df, "cohort", color=COHORT_EFFECT_COLOR)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, YMAX)
ax.set_xlim(fig_xlim_cohort)
ax.set_xlabel("Birth cohort")
ax.set_ylabel(AXIS_LABEL_PERCENT)
remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle(FACET_COHORT_EFFECT))
subtext_obj = add_subtext(SOURCE_SUBTEXT)
fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_cohort_standardized.png"
fig.savefig(str(fig_path), dpi=FIG_DPI, bbox_inches="tight", bbox_extra_artists=[subtext_obj])
```

## Time series model + standardized period (attribution)

Overlay the national time series model trend (composition changing over time) with the standardized period curve (fixed cohort mix). Attribution below decomposes the **smooth** modeled national change $\mathrm{sigmoid}(\alpha + \mathrm{cohort} + \mathrm{period})$ — year-level $\epsilon$ is excluded — into a **period effect** vs a **cohort effect**, using summed survey weights for composition and a two-factor Shapley average so the split is not order-dependent. Percentage-point components are primary; share ratios are secondary (median of draw-level ratios when $|\Delta_{\mathrm{total}}|$ is safely away from zero). The full window is a synthetic counterfactual (1970s composition held into recent years); a post-1990 window is reported as sensitivity.

Requires the companion time series model idata at `idata/time_model_appendix_{VARIABLE}_time_model_extrap5.nc` (run the time series model appendix first if missing).

Build the smooth trend grid and composition weights from summed survey weights.

```python
TIME_MODEL_COLOR = "C0"
TIME_MODEL_IDATA = Path("idata") / f"time_model_appendix_{VARIABLE}_time_model_extrap5.nc"
SHARE_EPS = 1e-4  # |Δ_total| below this → skip share ratios

# Smooth trend grid (no ε)
p_grid = expit(
    alpha_samples[:, :, None, None]
    + cohort_samples[:, :, :, None]
    + period_samples[:, :, None, :]
)

# Composition w[year, cohort] from summed survey weights
w = np.zeros((len(model_years), len(model_cohorts)), dtype=float)
cohort_i = {int(c): i for i, c in enumerate(model_cohorts)}
year_i = {int(y): i for i, y in enumerate(model_years)}
for cohort, year, sw in cells[["cohort", "nominal_year", "sum_weight"]].itertuples(index=False):
    ci = cohort_i.get(int(cohort))
    yi = year_i.get(int(year))
    if ci is None or yi is None:
        continue
    w[yi, ci] += float(sw)
row_sums = w.sum(axis=1, keepdims=True)
with np.errstate(invalid="ignore", divide="ignore"):
    w = np.where(row_sums > 0, w / row_sums, 0.0)

years_with_data = sorted(int(y) for y in model_years if w[year_i[int(y)]].sum() > 0)
```

Define Shapley decomposition helpers.

```python
def marginal_at(p_g, weights, weight_year, period_year):
    """M(w_{weight_year}, period_year): composition × period on probability scale."""
    return np.tensordot(
        p_g[:, :, :, period_year],
        weights[weight_year],
        axes=([2], [0]),
    )


def shapley_window(p_g, weights, t0, t1):
    """Shapley Δ_period / Δ_cohort for one window; returns draw-level arrays."""
    i0, i1 = year_i[t0], year_i[t1]
    m00 = marginal_at(p_g, weights, i0, i0)
    m01 = marginal_at(p_g, weights, i0, i1)
    m10 = marginal_at(p_g, weights, i1, i0)
    m11 = marginal_at(p_g, weights, i1, i1)
    delta_total = (m11 - m00).reshape(-1)
    delta_period = (0.5 * ((m01 - m00) + (m11 - m10))).reshape(-1)
    delta_cohort = (0.5 * ((m10 - m00) + (m11 - m01))).reshape(-1)
    # Sequential orderings (sensitivity)
    delta_period_first = (m01 - m00).reshape(-1)
    delta_cohort_second = (m11 - m01).reshape(-1)
    delta_cohort_first = (m10 - m00).reshape(-1)
    delta_period_second = (m11 - m10).reshape(-1)
    return {
        "t0": t0,
        "t1": t1,
        "delta_total": delta_total,
        "delta_period": delta_period,
        "delta_cohort": delta_cohort,
        "delta_period_first": delta_period_first,
        "delta_cohort_second": delta_cohort_second,
        "delta_cohort_first": delta_cohort_first,
        "delta_period_second": delta_period_second,
    }


def report_decomposition(name, d):
    """Print pp components first; shares secondary if denominator is safe."""
    dt, dp, dc = d["delta_total"], d["delta_period"], d["delta_cohort"]
    print(f"=== {name} ({d['t0']}–{d['t1']}) — smooth modeled change (ε excluded) ===")
    print(
        f"  Δ_total {100 * dt.mean():+.2f} pp "
        f"[{100 * np.quantile(dt, 0.08):+.2f}, {100 * np.quantile(dt, 0.92):+.2f}]"
    )
    print(
        f"  period effect {100 * dp.mean():+.2f} pp "
        f"[{100 * np.quantile(dp, 0.08):+.2f}, {100 * np.quantile(dp, 0.92):+.2f}]"
    )
    print(
        f"  cohort effect {100 * dc.mean():+.2f} pp "
        f"[{100 * np.quantile(dc, 0.08):+.2f}, {100 * np.quantile(dc, 0.92):+.2f}]"
    )
    # Sensitivity: sequential orderings
    print(
        f"  (order: period-then-cohort) "
        f"period effect {100 * d['delta_period_first'].mean():+.2f} pp, "
        f"cohort effect {100 * d['delta_cohort_second'].mean():+.2f} pp"
    )
    print(
        f"  (order: cohort-then-period) "
        f"cohort effect {100 * d['delta_cohort_first'].mean():+.2f} pp, "
        f"period effect {100 * d['delta_period_second'].mean():+.2f} pp"
    )
    attr_line = ""
    if abs(dt.mean()) >= SHARE_EPS and np.all(np.abs(dt) >= SHARE_EPS):
        share_period = dp / dt
        share_cohort = dc / dt
        med_p, lo_p, hi_p = np.quantile(share_period, [0.5, 0.08, 0.92])
        med_c = float(np.median(share_cohort))
        ratio_of_means_p = float(dp.mean() / dt.mean())
        print(
            f"  shares (secondary): period effect median {100 * med_p:.0f}% "
            f"[{100 * lo_p:.0f}–{100 * hi_p:.0f}%]; "
            f"cohort effect median {100 * med_c:.0f}%; "
            f"ratio-of-means period effect {100 * ratio_of_means_p:.0f}%"
        )
        attr_line = (
            f"{d['t0']}–{d['t1']}: {100 * med_c:.0f}% cohort effect, "
            f"{100 * med_p:.0f}% period effect"
        )
    else:
        print("  shares omitted (|Δ_total| too small or crosses ~0 in draws)")
    return attr_line
```

Run the Shapley decomposition for the full overlap window and a post-1990 sensitivity window.

```python
from IPython.display import Markdown, display

# Full overlap window + post-1990 sensitivity
t0_full, t1_full = years_with_data[0], years_with_data[-1]
windows = [("full", t0_full, t1_full)]
t0_post = next((y for y in years_with_data if y >= 1990), None)
if t0_post is not None and t0_post < t1_full:
    windows.append(("post1990", t0_post, t1_full))

attr_line_full = ""
decomp_rows = []
for name, t0, t1 in windows:
    d = shapley_window(p_grid, w, t0, t1)
    line = report_decomposition(name, d)
    if name == "full":
        attr_line_full = line
    dt, dp, dc = d["delta_total"], d["delta_period"], d["delta_cohort"]
    period_share_pct = (
        f"{100 * float(np.median(dp / dt)):.0f}%"
        if abs(dt.mean()) >= SHARE_EPS and np.all(np.abs(dt) >= SHARE_EPS)
        else "—"
    )
    decomp_rows.append(
        {
            "Interval": f"{t0}–{t1}",
            "Δ total": f"{100 * dt.mean():.1f} pp",
            "Period": f"{100 * dp.mean():.1f} pp",
            "Cohort": f"{100 * dc.mean():.1f} pp",
            "Period share": period_share_pct,
        }
    )

decomp_summary = pd.DataFrame(decomp_rows)
display(
    Markdown(
        "Shapley decomposition of the smooth national change "
        "\n\n"
        + decomp_summary.to_markdown(index=False)
    )
)
```

Plot the standardized period curve with the national time series model overlay (full-window attribution in subtitle).

```python
fig, ax = plt.subplots(figsize=FIGSIZE)
facet = FACET_PERIOD_EFFECT
if TIME_MODEL_IDATA.is_file():
    idata_tm_national = az.from_netcdf(TIME_MODEL_IDATA)
    support = idata_tm_national.posterior["support"]
    tm_years = np.asarray(support.coords["year_idx"].values, dtype=int)
    tm_mean = support.mean(dim=("chain", "draw")).values * 100
    tm_lo = support.quantile(0.03, dim=("chain", "draw")).values * 100
    tm_hi = support.quantile(0.97, dim=("chain", "draw")).values * 100
    # Match observed data span (drop time series model extrapolation beyond fielded years)
    obs_lo, obs_hi = years_with_data[0], years_with_data[-1]
    in_obs = (tm_years >= obs_lo) & (tm_years <= obs_hi)
    tm_years = tm_years[in_obs]
    tm_mean = tm_mean[in_obs]
    tm_lo = tm_lo[in_obs]
    tm_hi = tm_hi[in_obs]
    if len(tm_years) >= 2:
        xf, lo, m, hi = interpolate_1d_band_for_plot(tm_years, tm_lo, tm_mean, tm_hi)
        ax.fill_between(xf, lo, hi, color=TIME_MODEL_COLOR, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(xf, m, color=TIME_MODEL_COLOR, linewidth=2, label="Time series model", zorder=2)
    elif len(tm_years) == 1:
        ax.fill_between(tm_years, tm_lo, tm_hi, color=TIME_MODEL_COLOR, alpha=0.22, linewidth=0, zorder=1)
        ax.plot(tm_years, tm_mean, color=TIME_MODEL_COLOR, linewidth=2, label="Time series model", zorder=2)
    facet = f"{FACET_PERIOD_EFFECT} · Time series model"
else:
    print(f"Missing {TIME_MODEL_IDATA}; plotting standardized period only.")

if attr_line_full:
    facet = f"{facet} · {attr_line_full}"

plot_probability_summary_band(
    ax,
    standardized_df,
    "nominal_year",
    color=PERIOD_EFFECT_COLOR,
    label="Standardized period",
)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, YMAX)
ax.set_xlim(fig_xlim)
ax.set_xlabel("Survey year")
ax.set_ylabel(AXIS_LABEL_PERCENT)
ax.legend(loc="best")
remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle(facet))
subtext_obj = add_subtext(SOURCE_SUBTEXT)
fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_time_standardized_time_series.png"
fig.savefig(str(fig_path), dpi=FIG_DPI, bbox_inches="tight", bbox_extra_artists=[subtext_obj])
```

## Model 2: Time series model by birth decade

For each decade of birth, fit the same time series model specification as in the companion appendix (RW2 + year-level overdispersion), using only respondents born in that decade. Survey years are calendar years here (not the nominal lattice). Decades with fewer than `MIN_OBSERVED_YEARS` waves are skipped.

Define the yearly aggregation helper for a birth-cohort window.

```python
def yearly_aggregates_for_cohort_range(micro_df: pd.DataFrame, cohort_lo: int, cohort_hi: int):
    """Kish yearly aggregates on calendar survey year for a birth-cohort window."""
    sub = micro_df[(micro_df["cohort"] >= cohort_lo) & (micro_df["cohort"] <= cohort_hi)].copy()
    if sub.empty:
        return None

    def _weighted_year_stats(group: pd.DataFrame) -> pd.Series:
        y_vals = group["y"].to_numpy(dtype=float)
        weights = group[WEIGHT_COL].to_numpy(dtype=float)
        total_weight = float(weights.sum())
        if total_weight <= 0 or len(weights) == 0:
            return pd.Series({"p_weighted": 0.0, "n_eff": 0.0})
        p_weighted = float((y_vals * weights).sum() / total_weight)
        n_eff = (total_weight**2) / float((weights**2).sum())
        return pd.Series({"p_weighted": p_weighted, "n_eff": float(n_eff)})

    by_year = (
        sub.groupby("year", group_keys=False)
        .apply(_weighted_year_stats, include_groups=False)
        .reset_index()
    )
    year_min = int(by_year["year"].min())
    year_last_data = int(by_year["year"].max())
    years_grid_tm = np.arange(year_min, year_last_data + 1, dtype=int)
    full = pd.DataFrame({"year": years_grid_tm}).merge(by_year, on="year", how="left")
    full["n_eff"] = full["n_eff"].fillna(0.0)
    full["p_weighted"] = full["p_weighted"].fillna(0.0)
    observed = full[full["n_eff"] > 0].copy()
    year_to_idx = {y: i for i, y in enumerate(years_grid_tm)}
    observed["year_idx"] = observed["year"].map(year_to_idx)
    return {
        "observed": observed,
        "years_grid": years_grid_tm,
        "n_observed": len(observed),
        "year_min": year_min,
        "year_last_data": year_last_data,
        "n_respondents": len(sub),
    }
```

Fit one time series model per birth decade.

```python
decades = sorted(((cells["cohort"] // 10) * 10).unique())
decade_fits = []

for decade in decades:
    decade = int(decade)
    idata_path = Path("idata") / f"model10_appendix_{VARIABLE}_timemodel_decade{decade}.nc"
    prep_tm = yearly_aggregates_for_cohort_range(micro, decade, decade + 9)
    if prep_tm is None:
        continue

    n_obs_years = int(prep_tm["n_observed"])
    print(
        f"Decade {decade}s: respondents={prep_tm['n_respondents']:,}, "
        f"observed years={n_obs_years} ({prep_tm['year_min']}–{prep_tm['year_last_data']})"
    )
    if n_obs_years < MIN_OBSERVED_YEARS:
        print(f"  Skip: need at least {MIN_OBSERVED_YEARS} observed years")
        continue

    model_tm = build_period_model(prep_tm["observed"], prep_tm["years_grid"])
    idata_tm = load_idata_or_sample(
        model_tm,
        str(idata_path),
        force_run=FORCE_RUN,
        draws=DRAWS,
        tune=TUNE,
        target_accept=TARGET_ACCEPT,
        nuts_sampler="nutpie",
        nuts=dict(
            adaptation="low_rank",
            mass_matrix_eigval_cutoff=6.0,
            mass_matrix_gamma=1e-5,
        ),
    )
    decade_fits.append({"decade": decade, "prep": prep_tm, "idata": idata_tm})

print(f"Fitted / loaded decades: {[d['decade'] for d in decade_fits]}")
```

Plot one latent-support curve per birth decade (posterior mean + light 94% band), with each decade's raw weighted percentages as open circles.

```python
if not decade_fits:
    raise RuntimeError("No decade time series models were fit or loaded; check MIN_OBSERVED_YEARS / data.")

cmap = plt.cm.plasma.reversed()
decades_fit = [d["decade"] for d in decade_fits]
norm = plt.Normalize(vmin=min(decades_fit), vmax=max(decades_fit))
bbox_style = dict(boxstyle="round,pad=0.1", facecolor="white", linewidth=0)

fig, ax = plt.subplots(figsize=FIGSIZE)
for fit in decade_fits:
    decade = fit["decade"]
    idata_tm = fit["idata"]
    prep_tm = fit["prep"]
    support = idata_tm.posterior["support"]
    years = np.asarray(support.coords["year_idx"].values, dtype=int)
    mean_pct = support.mean(dim=("chain", "draw")).values * 100
    lo_pct = support.quantile(0.03, dim=("chain", "draw")).values * 100
    hi_pct = support.quantile(0.97, dim=("chain", "draw")).values * 100
    color = cmap(norm(decade))

    ax.fill_between(years, lo_pct, hi_pct, color=color, alpha=0.12, linewidth=0)
    ax.plot(years, mean_pct, color=color, linewidth=1.8, alpha=0.95)

    obs = prep_tm["observed"]
    ax.scatter(
        obs["year"],
        obs["p_weighted"].values * 100,
        color=color,
        s=14,
        alpha=0.45,
        zorder=3,
    )

    x_end, v_end = float(years[-1]), float(mean_pct[-1])
    x_start, v_start = float(years[0]), float(mean_pct[0])
    if decade <= 1950:
        ax.annotate(
            str(decade),
            xy=(x_end, v_end),
            xytext=(5, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=9,
            color="black",
            bbox=bbox_style,
        )
    else:
        ax.annotate(
            str(decade),
            xy=(x_start, v_start),
            xytext=(-5, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=9,
            color="black",
            bbox=bbox_style,
        )

ax.set_xlabel("Survey year")
ax.set_ylabel(AXIS_LABEL_PERCENT)
ax.set_xlim(fig_xlim)
ax.set_ylim(0, YMAX)
ax.grid(True, alpha=0.3)

remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle("Time series model by birth decade"))
subtext_obj = add_subtext(SOURCE_SUBTEXT)

fig_path = Path("figs") / f"model10_appendix_{VARIABLE}_timemodel_by_decade.png"
fig.savefig(
    str(fig_path),
    dpi=FIG_DPI,
    bbox_inches="tight",
    bbox_extra_artists=[subtext_obj],
)
```

## The Readable Version

For the talk where I present this model, I use the following version of the period–cohort model, revised to be as readable as possible.

Readable form of the additive period–cohort model (same structure as ``build_period_cohort_model``), with ``rw2`` left factored out. Year-level $\varepsilon$ uses the same non-centered zero-sum construction as the fitted model.

```python
n_c = len(cohorts)
n_t = len(years_grid)
c_idx = cells["cohort_idx"].values
t_idx = cells["year_idx"].values
coords = {
    "cohort_idx": cohorts,
    "year_idx": years_grid,
    "cell_idx": range(len(cells)),
}

with pm.Model(coords=coords) as period_cohort_model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    sigma_c = pm.Exponential("sigma_c", lam=1.0 / SIGMA_C_SCALE)
    sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
    cohort = rw2("cohort", n_c, sigma_c, dims="cohort_idx")
    period = rw2("period", n_t, sigma_t, dims="year_idx")
    eta = alpha + cohort[c_idx] + period[t_idx]
    sigma_eps = pm.Exponential("sigma_eps", lam=1.0 / SIGMA_SHORT_SCALE_PC)
    eps_z = pm.ZeroSumNormal("eps_z", sigma=1.0, dims="year_idx")
    eps = pm.Deterministic("eps", sigma_eps * eps_z, dims="year_idx")
    p = pm.math.sigmoid(eta + eps[t_idx])
    pm.Binomial(
        "y",
        n=cells["n"].values,
        p=p,
        observed=cells["y"].values,
        dims="cell_idx",
    )
```

Visualize the model structure:

```python
period_cohort_model.to_graphviz()
```
