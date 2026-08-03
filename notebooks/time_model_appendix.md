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

# Period-only models

[Run this notebook on Colab](https://colab.research.google.com/github/AllenDowney/YouWouldChooseNow/blob/main/notebooks/time_model_appendix.ipynb)

This notebook walks through the period-only models from the talk, using as an example the General Social Survey item on sexual relations between two adults of the same sex. The outcome is the percent saying such relations are wrong to any degree (GSS response codes 1–3: always wrong, almost always wrong, or wrong only sometimes). Figure titles use a short stem of the survey question.

We start from yearly weighted proportions, then fit a sequence of Bayesian models on the logit scale: a linear trend, a first-order random walk, a second-order random walk (RW2), and finally RW2 with year-level overdispersion. Each fit uses a calendar-year grid that extends five years past the last observed survey year so we can see how the latent trend behaves under extrapolation (those future years have no likelihood contribution).


```python
# Colab ships an older PyMC; installing nutpie alone can pull ArviZ 1.x and break imports.
# Upgrade the sampling stack together (matches this project's env: pymc>=6, arviz>=1.1, nutpie).
import sys

if "google.colab" in sys.modules:
    %pip install -q "pymc>=6" "arviz>=1.1" nutpie
```

```python
import os
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
from scipy.stats import beta as beta_dist

np.random.seed(17)
```

## Configuration

```python tags=["parameters"]
VARIABLE = "homosex"
WEIGHT_COL = "wtsmerge"
GSS_FILENAME = "gss_extract_2024_r3.hdf"
GSS_HDF_KEY = "gss"
CI_LEVEL = 0.95

# Outcome coding by variable (extend this dict to reuse the notebook on other items)
SELECTED_VALUES = {
    "homosex": [1, 2, 3],  # always, almost always, wrong only sometimes
}
SELECTED_CODES = SELECTED_VALUES[VARIABLE]

# Short figure title (not the full GSS question text)
DISPLAY_TITLE = "What about sexual relations between two adults of the same sex?"
OUTCOME_YLABEL = "Percent saying wrong (any degree)"
SOURCE_SUBTEXT = "Source: General Social Survey 1972-2024"
FACET_TIME_SERIES = "Time series"
AXIS_LABEL_PERCENT = "Percent"
YMAX = 100  # percent axis for this outcome
GSS_FIG_YEAR_PAD = 1.5
# Compact figure size; high DPI for sharp display
FIGSIZE = (9.0 * 0.75, 4.5 * 0.75)
FIG_DPI = 300
FIG_TITLE_PAD = 42
FIG_SUBTITLE_Y = 1.035

# Extend the period grid beyond last data year (no likelihood on extrapolation years)
EXTRAPOLATE_YEARS = 5

# Sampling defaults
TARGET_ACCEPT = 0.9
DRAWS = 1000
TUNE = 1000
FORCE_RUN = False

# RW1 / RW2 innovation scale (yearly grid)
SIGMA_T_SCALE = 0.125
SIGMA_SHORT_SCALE = 0.5

# Figure styling
TREND_COLOR = "C0"
OBS_COLOR = "0.45"
```

## Notebook helpers

Small plotting and sampling utilities, plus the period RW2 used in later steps: non-centered second differences, free initial slope with level fixed at 0, then a mean-centered path.

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


def load_idata_or_sample(model, filename, *, force_run=False, **sample_options):
    """Load NetCDF if present (unless ``force_run``), else ``pm.sample`` and save."""
    path = Path(filename)
    if path.is_file() and not force_run:
        return az.from_netcdf(path)
    with model:
        idata = pm.sample(**sample_options)
    idata.to_netcdf(path)
    return idata


def rw2_period(K, sigma, *, dims=None):
    """Period RW2: non-centered Δ², free init slope, fixed init level 0, mean-centered path.

    The raw path starts at level 0 with free initial slope; mean-centering then
    removes any constant offset (``alpha`` carries the overall level). A free
    initial level would cancel under centering and is omitted.

    Parameters
    ----------
    K : int
        Length of the period grid (number of years on the latent path).
    sigma : tensor or float
        Scale of the second-difference innovations (usually an ``Exponential`` RV).
    dims : str or sequence of str, optional
        Coordinate dims passed to ``pm.Deterministic`` (e.g. ``"year_idx"``).
    """
    name = "period"
    z_delta2 = pm.Normal(f"{name}_z_delta2", mu=0.0, sigma=1.0, shape=K - 2)
    delta2 = sigma * z_delta2
    slope0 = pm.Normal(f"{name}_slope_init", mu=0.0, sigma=0.2)
    f_init = pm.math.stack([0.0, slope0])
    f_raw = pm.math.concatenate(
        [f_init, f_init[1] + pm.math.cumsum(f_init[1] - f_init[0] + pm.math.cumsum(delta2))]
    )
    f = f_raw - pm.math.mean(f_raw)
    return pm.Deterministic(name, f, dims=dims)


os.makedirs("figs", exist_ok=True)
os.makedirs("idata", exist_ok=True)
```

## Step 1 — Load data and yearly aggregates

Load the GSS extract, code the binary outcome from the selected response codes, and aggregate to survey year with a Kish effective sample size. The year grid includes five future years beyond the last wave with data; those nodes appear in the latent trend but do not enter the likelihood. For this notebook we treat codes 1–3 as the “wrong” outcomes (always, almost always, and wrong only sometimes).

```python
from urllib.request import urlretrieve

# Repo layout: run from notebooks/ → ../data. Colab / flat checkout → ./data
DATA_DIR = Path("..") / "data" if (Path("..") / "Makefile").is_file() else Path("data")
GSS_PATH = DATA_DIR / GSS_FILENAME


def download(url, path):
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        local, _ = urlretrieve(url, filename=str(path))
        print("Downloaded " + local)


download(
    f"https://github.com/AllenDowney/YouWouldChooseNow/raw/main/data/{GSS_FILENAME}",
    GSS_PATH,
)

gss = pd.read_hdf(GSS_PATH, GSS_HDF_KEY)
figure_xlim_year = (
    float(gss["year"].min()) - GSS_FIG_YEAR_PAD,
    float(gss["year"].max()) + GSS_FIG_YEAR_PAD,
)

gss["y"] = gss[VARIABLE].isin(SELECTED_CODES).astype(float)
gss.loc[gss[VARIABLE].isna(), "y"] = np.nan
gss = gss.dropna(subset=["y", "year", WEIGHT_COL])
gss["year"] = gss["year"].astype(int)


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
    gss.groupby("year", group_keys=False)
    .apply(_weighted_year_stats, include_groups=False)
    .reset_index()
)

year_min = int(by_year["year"].min())
year_last_data = int(by_year["year"].max())
year_max = year_last_data + EXTRAPOLATE_YEARS
years_grid = np.arange(year_min, year_max + 1, dtype=int)
n_years = len(years_grid)

full = pd.DataFrame({"year": years_grid}).merge(by_year, on="year", how="left")
full["n_eff"] = full["n_eff"].fillna(0.0)
full["p_weighted"] = full["p_weighted"].fillna(0.0)
observed = full[full["n_eff"] > 0].copy()
year_to_idx = {y: i for i, y in enumerate(years_grid)}
observed["year_idx"] = observed["year"].map(year_to_idx)
n_observed = len(observed)

year_idx_obs = observed["year_idx"].to_numpy()

# Year-level likelihood inputs: weighted proportion + Kish n_eff (rounded for Binomial)
p_weighted_obs = observed["p_weighted"].to_numpy(dtype=float)
n_eff_obs = observed["n_eff"].to_numpy(dtype=float)
n_eff_binom = np.round(n_eff_obs).astype(int)
y_eff = np.round(p_weighted_obs * n_eff_binom).astype(int)
y_eff = np.clip(y_eff, 0, n_eff_binom)

coords = {"year_idx": years_grid}
fig_xlim = (figure_xlim_year[0], float(year_max) + GSS_FIG_YEAR_PAD)

# Centered/scaled calendar year on the full grid (including extrapolation years)
year_scaled = (years_grid.astype(np.float64) - years_grid.mean()) / years_grid.std()
year_scaled_t = pm.math.as_tensor(year_scaled)

print(f"Respondents used: {len(gss):,}")
print(f"Observed survey years: {n_observed} ({year_min}–{year_last_data})")
print(f"Full grid: {n_years} years ({years_grid[0]}–{years_grid[-1]})")
by_year.head(10)
```

## Plotting helper

Shared figure for all fitted models: posterior mean of the latent support curve with a credible band, observed weighted percentages, and a shaded region for the extrapolation years.

```python
def make_subtitle(facet_line: str) -> str:
    return f"{OUTCOME_YLABEL}\n{facet_line}"


def plot_support_fit(
    idata: az.InferenceData,
    *,
    subtitle: str,
    fig_stem: str,
) -> Path:
    support_da = idata.posterior["support"]
    plot_years = np.asarray(support_da.coords["year_idx"].values, dtype=int)
    support_mean = support_da.mean(dim=("chain", "draw")).values
    support_low = support_da.quantile(0.03, dim=("chain", "draw")).values
    support_high = support_da.quantile(0.97, dim=("chain", "draw")).values

    raw_pct = observed["p_weighted"].values * 100
    support_low_pct = support_low * 100
    support_high_pct = support_high * 100
    support_mean_pct = support_mean * 100

    fig, ax = plt.subplots(figsize=FIGSIZE)
    if plot_years[-1] > year_last_data:
        ax.axvspan(
            year_last_data + 0.5,
            plot_years[-1] + 0.5,
            facecolor="0.94",
            edgecolor="none",
            zorder=0,
        )
    ax.fill_between(plot_years, support_low_pct, support_high_pct, alpha=0.3, color=TREND_COLOR, label="94% CI")
    ax.plot(plot_years, support_mean_pct, color=TREND_COLOR, label="Latent trend")
    ax.scatter(
        observed["year"],
        raw_pct,
        color=OBS_COLOR,
        s=12,
        alpha=0.4,
        label="Weighted %",
        zorder=3,
    )

    ax.set_xlabel("Survey year")
    ax.set_ylabel(AXIS_LABEL_PERCENT)
    ax.set_xlim(fig_xlim)
    ax.set_ylim(0, YMAX)
    ax.grid(True, alpha=0.3)

    remove_spines()
    add_title(DISPLAY_TITLE)
    add_subtitle(make_subtitle(subtitle))
    subtext_obj = add_subtext(SOURCE_SUBTEXT)
    ax.legend(loc="best")

    fig_path = Path("figs") / f"time_model_appendix_{VARIABLE}_{fig_stem}.png"
    fig.savefig(
        str(fig_path),
        dpi=FIG_DPI,
        bbox_inches="tight",
        bbox_extra_artists=[subtext_obj],
    )
    return fig_path
```

## Yearly weighted proportions and intervals

For each survey year we use the weighted proportion $p$, the Kish effective sample size $n_{\mathrm{eff}}$, and rounded effective counts $y = \mathrm{round}(p \cdot n_{\mathrm{eff}})$. Error bars are Jeffreys intervals for a binomial with those rounded counts—the same $(y, n_{\mathrm{eff}})$ pair that enters the models below.

```python
def jeffreys_interval(y: int, n: int, *, confidence_level: float) -> tuple[float, float, float]:
    """Jeffreys prior interval for a binomial proportion (aggregate y, n)."""
    if n <= 0:
        raise ValueError(f"jeffreys_interval requires n > 0, got n={n}")
    a = float(y) + 0.5
    b = float(n - y) + 0.5
    dist = beta_dist(a, b)
    lo = dist.ppf((1.0 - confidence_level) / 2.0)
    hi = dist.ppf(1.0 - (1.0 - confidence_level) / 2.0)
    p = a / (a + b)
    return p, lo, hi


yearly_obs = observed[["year", "p_weighted", "n_eff"]].copy()
yearly_obs["n_eff_round"] = n_eff_binom
yearly_obs["y_eff"] = y_eff
intervals = [
    jeffreys_interval(int(row.y_eff), int(row.n_eff_round), confidence_level=CI_LEVEL)
    for row in yearly_obs.itertuples(index=False)
]
yearly_obs["p_jeffreys"] = [t[0] for t in intervals]
yearly_obs["p_lo"] = [t[1] for t in intervals]
yearly_obs["p_hi"] = [t[2] for t in intervals]
yearly_obs["p_pct"] = yearly_obs["p_weighted"] * 100
yearly_obs["p_lo_pct"] = yearly_obs["p_lo"] * 100
yearly_obs["p_hi_pct"] = yearly_obs["p_hi"] * 100

yearly_obs[["year", "p_weighted", "n_eff", "n_eff_round", "y_eff", "p_lo", "p_hi"]]
```

## Plot observed proportions

Observed yearly percentages with Jeffreys intervals from the rounded Kish counts.

```python
yerr_lo = yearly_obs["p_pct"] - yearly_obs["p_lo_pct"]
yerr_hi = yearly_obs["p_hi_pct"] - yearly_obs["p_pct"]

fig, ax = plt.subplots(figsize=FIGSIZE)
ax.errorbar(
    yearly_obs["year"],
    yearly_obs["p_pct"],
    yerr=[yerr_lo, yerr_hi],
    fmt="o",
    color="C0",
    ecolor="C0",
    elinewidth=1.2,
    capsize=3,
    markersize=4,
    alpha=0.85,
    label=f"Weighted % ({CI_LEVEL:.0%} CI)",
)

ax.set_xlabel("Survey year")
ax.set_ylabel(AXIS_LABEL_PERCENT)
ax.set_xlim(fig_xlim)
ax.set_ylim(0, YMAX)
ax.grid(True, alpha=0.3)

remove_spines()
add_title(DISPLAY_TITLE)
add_subtitle(make_subtitle(FACET_TIME_SERIES))
subtext_obj = add_subtext(SOURCE_SUBTEXT)
ax.legend(loc="best")

fig_path = Path("figs") / f"time_model_appendix_{VARIABLE}_observed.png"
fig.savefig(
    str(fig_path),
    dpi=FIG_DPI,
    bbox_inches="tight",
    bbox_extra_artists=[subtext_obj],
)
```

## Step 2 — Bayesian linear-logit model

As a baseline, take the period effect to be a single global linear trend on the logit scale:

$$
\eta_t = \alpha + \mathrm{period}_t,\quad \mathrm{period}_t = \beta \cdot \tilde{t}
$$

where $\tilde{t}$ is survey year centered and scaled on the full grid (including extrapolation years). The likelihood is binomial in the rounded Kish counts: $y \sim \mathrm{Binomial}(n_{\mathrm{eff}}, p)$ with $p = \mathrm{sigmoid}(\eta)$ at observed years.

```python
with pm.Model(coords=coords) as model:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)
    period = pm.Deterministic("period", beta * year_scaled_t, dims="year_idx")
    eta = pm.Deterministic("eta", alpha + period, dims="year_idx")
    support = pm.Deterministic("support", pm.math.sigmoid(eta), dims="year_idx")

    eta_obs = eta[year_idx_obs]
    p_obs = pm.math.sigmoid(eta_obs)

    pm.Binomial(
        "y",
        n=n_eff_binom,
        p=p_obs,
        observed=y_eff,
    )
```

```python
idata_path = Path("idata") / f"time_model_appendix_{VARIABLE}_linear_logit_extrap{EXTRAPOLATE_YEARS}.nc"

idata = load_idata_or_sample(
    model,
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
az.summary(idata, var_names=["alpha", "beta"], round_to=4)
```

```python
plot_support_fit(idata, subtitle="Linear logit model", fig_stem="linear_logit")
```

## Step 3 — First-order random walk on the period effect

Replace the linear trend with a first-order Gaussian random walk: yearly innovations scaled by $\sigma_t$, initial level fixed at 0, then mean-centering for identifiability with the intercept:

$$
\mathrm{period}_0 = 0,\quad \mathrm{period}_t = \sum_{s=1}^{t} \delta_s,\quad
\mathrm{period} \leftarrow \mathrm{period} - \bar{\mathrm{period}}
$$

A free initial level would cancel under mean-centering (the same redundancy as $x_0$ in the RW2 construction below), so it is omitted; $\alpha$ carries the overall level. The walk continues through the extrapolation years, still with no data likelihood there.

```python
with pm.Model(coords=coords) as model_rw1:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
    period_innov = pm.Normal("period_innov", mu=0.0, sigma=sigma_t, shape=n_years - 1)
    period_raw = pm.math.concatenate(
        [pm.math.zeros((1,)), pm.math.cumsum(period_innov)]
    )
    period = pm.Deterministic("period", period_raw - pm.math.mean(period_raw), dims="year_idx")
    eta = pm.Deterministic("eta", alpha + period, dims="year_idx")
    support = pm.Deterministic("support", pm.math.sigmoid(eta), dims="year_idx")

    eta_obs = eta[year_idx_obs]
    p_obs = pm.math.sigmoid(eta_obs)

    pm.Binomial(
        "y",
        n=n_eff_binom,
        p=p_obs,
        observed=y_eff,
    )
```

```python
idata_path_rw1 = Path("idata") / f"time_model_appendix_{VARIABLE}_rw1_extrap{EXTRAPOLATE_YEARS}.nc"

idata_rw1 = load_idata_or_sample(
    model_rw1,
    str(idata_path_rw1),
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
az.summary(idata_rw1, var_names=["alpha", "sigma_t"], round_to=4)
```

```python
plot_support_fit(idata_rw1, subtitle="Random walk model", fig_stem="rw1")
```

## Step 4 — Second-order random walk (no overdispersion)

Next use a second-order random walk (RW2) for the period effect—smooth in level and slope—without year-level noise in the likelihood:

$$
\eta_t = \alpha + \mathrm{period}_t,\quad \mathrm{period} \sim \mathrm{RW2}
$$

The RW2 helper defined earlier builds this component.

```python
with pm.Model(coords=coords) as model_rw2:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
    period = rw2_period(n_years, sigma_t, dims="year_idx")
    eta = pm.Deterministic("eta", alpha + period, dims="year_idx")
    support = pm.Deterministic("support", pm.math.sigmoid(eta), dims="year_idx")

    eta_obs = eta[year_idx_obs]
    p_obs = pm.math.sigmoid(eta_obs)

    pm.Binomial(
        "y",
        n=n_eff_binom,
        p=p_obs,
        observed=y_eff,
    )
```

```python
idata_path_rw2 = Path("idata") / f"time_model_appendix_{VARIABLE}_rw2_extrap{EXTRAPOLATE_YEARS}.nc"

idata_rw2 = load_idata_or_sample(
    model_rw2,
    str(idata_path_rw2),
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
az.summary(idata_rw2, var_names=["alpha", "sigma_t"], round_to=4)
```

```python
plot_support_fit(idata_rw2, subtitle="RW2 model", fig_stem="rw2")
```

## Step 5 — RW2 with year-level overdispersion

The full period-only model adds mean-zero year-level deviations $\epsilon_t$ on the logit scale. The deviations use a non-centered zero-sum normal prior, so they represent short-term departures from the smooth RW2 trajectory without contributing an additional overall intercept. The plotted curve is $\mathrm{support}_t = \mathrm{sigmoid}(\eta_t)$, the latent RW2 trend; $\epsilon_t$ enters only the likelihood:

$$
\eta_t = \alpha + \mathrm{period}_t,\quad p_t = \mathrm{sigmoid}(\eta_t + \epsilon_t)\ \text{(observed years)}
$$

$$
z \sim \operatorname{ZeroSumNormal}(0,1),\quad
\epsilon_t = \sigma_{\mathrm{short}} z_t,\quad
\sum_t \epsilon_t = 0.
$$

```python
with pm.Model(coords=coords) as model_tm:
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    sigma_t = pm.Exponential("sigma_t", lam=1.0 / SIGMA_T_SCALE)
    period = rw2_period(n_years, sigma_t, dims="year_idx")
    eta = pm.Deterministic("eta", alpha + period, dims="year_idx")
    support = pm.Deterministic("support", pm.math.sigmoid(eta), dims="year_idx")

    sigma_short = pm.HalfNormal("sigma_short", sigma=SIGMA_SHORT_SCALE)
    epsilon_z = pm.ZeroSumNormal("epsilon_z", sigma=1.0, dims="year_idx")
    epsilon = pm.Deterministic("epsilon", sigma_short * epsilon_z, dims="year_idx")

    eta_obs = eta[year_idx_obs]
    p_obs = pm.math.sigmoid(eta_obs + epsilon[year_idx_obs])

    pm.Binomial(
        "y",
        n=n_eff_binom,
        p=p_obs,
        observed=y_eff,
    )
```

```python
idata_path_tm = Path("idata") / f"time_model_appendix_{VARIABLE}_time_model_extrap{EXTRAPOLATE_YEARS}.nc"

idata_tm = load_idata_or_sample(
    model_tm,
    str(idata_path_tm),
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
az.summary(idata_tm, var_names=["alpha", "sigma_t", "sigma_short"], round_to=4)
print(
    "max |sum_t epsilon|:",
    float(np.abs(idata_tm.posterior["epsilon"].sum("year_idx")).max()),
)
```

```python
plot_support_fit(idata_tm, subtitle="RW2 model + overdispersion", fig_stem="time_model")
```


### Simplified model

Same model as Step 5, written with the talk’s names: $\delta$ second differences (`delta`), $v_0$ initial slope (`v0`), `rw1` / `rw2` the integrated first-difference and level paths, and $\varepsilon$ year-level logit noise (`eps`). The initial level is fixed at 0: after mean-centering, a free $x_0$ would cancel exactly ($\mathrm{rw2}_{\mathrm{raw}}=x_0+g$ implies $\mathrm{rw2}=g-\bar g$), so $\alpha$ carries the overall level while $v_0$ remains (it controls the linear component). Year shocks use the same non-centered zero-sum construction as Step 5 (`eps_z` / `eps`). Scales are `sigma` / `sigma_eps` (talk: $\sigma$).

```python
with pm.Model(coords=coords) as period_model:
    n = n_years
    alpha = pm.Normal("alpha", mu=0.0, sigma=1.0)
    sigma = pm.Exponential("sigma", lam=1.0 / SIGMA_T_SCALE)
    sigma_eps = pm.HalfNormal("sigma_eps", sigma=SIGMA_SHORT_SCALE)
    v0 = pm.Normal("v0", mu=0.0, sigma=0.2)
    delta = pm.Normal("delta", mu=0.0, sigma=sigma, shape=n - 2)
    rw1 = v0 + pm.math.cumsum(delta)
    rw2_raw = pm.math.concatenate([
        pm.math.stack([0.0, v0]),
        v0 + pm.math.cumsum(rw1),
    ])
    rw2 = rw2_raw - pm.math.mean(rw2_raw)
    eps_z = pm.ZeroSumNormal("eps_z", sigma=1.0, dims="year_idx")
    eps = pm.Deterministic("eps", sigma_eps * eps_z, dims="year_idx")
    p = pm.math.sigmoid(alpha + rw2 + eps)
    pm.Binomial("y", n=n_eff_binom, p=p[year_idx_obs], observed=y_eff)
```
