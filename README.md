# You Would Choose Now

This repository contains data and technical appendices for [*You Would Choose Now: Measuring America's Progress Toward Fairness and Tolerance*](https://leanpub.com/ywcn).


## The Book

If you could choose any moment in history to be born, and you did not know ahead of time who you would be — what nationality, gender, race, religion, sexual orientation, or economic class — when would you choose?

Barack Obama posed this question in 2016, and it motivates the book.

[Allen Downey](https://allendowney.com), data scientist and author of [*Probably Overthinking It*](https://probablyoverthinking.it), uses decades of data from national surveys to see whether America has become fairer and more tolerant. He puts changing attitudes alongside the historical record of civil rights and legal protections — and asks, "Would you choose now?"

An early access edition is on [LeanPub](https://leanpub.com/ywcn). The introduction and a chapter about the decline of homophobia are available now (about 50 pages), with additional chapters planned monthly and a complete draft expected in early 2027.

Three editions are available in PDF and EPUB on LeanPub:

- [**Free Sample**](https://leanpub.com/ywcn) — introduction only
- [**Community Edition**](https://leanpub.com/ywcn) — introduction and first chapter (with email signup)
- [**Early Access Edition**](https://leanpub.com/ywcn) — all currently available chapters, plus updates through the final edition

This is a work in progress. Comments and suggestions are welcome on the [LeanPub forum](https://leanpub.com/forum/ywcn) or via [GitHub issues](https://github.com/AllenDowney/YouWouldChooseNow/issues).

## What's in this repo

### Run the appendices on Colab

Technical appendices that walk through the Bayesian models used in the book. Open either link to run on Google Colab (no local setup required):

- [**Time series models**](https://colab.research.google.com/github/AllenDowney/YouWouldChooseNow/blob/main/notebooks/time_series_model_appendix.ipynb) — national trend over survey years (linear logit → RW1 → RW2 → time series model with overdispersion), using GSS attitudes toward same-sex relations
- [**Period–cohort models**](https://colab.research.google.com/github/AllenDowney/YouWouldChooseNow/blob/main/notebooks/period_cohort_model_appendix.ipynb) — additive period–cohort surface and decade-specific time series models on the same outcome

Jupytext Markdown (`.md`) is the canonical notebook source; `.ipynb` files are generated for Colab.

### Other contents

| Path | Purpose |
|------|---------|
| [`notebooks/`](notebooks/) | Appendix notebooks (Jupytext `.md` + Colab `.ipynb`) |
| [`data/`](data/) | GSS extract used by the notebooks (`gss_extract_2024_r3.hdf`) |
| [`jb/`](jb/) | MyST site source (landing page with Colab links; `make build-local` in `jb/`) |
| [`scripts/`](scripts/) | Repo maintenance helpers |

## Local setup

To run notebooks locally instead of Colab:

```bash
make env
conda activate YouWouldChooseNow
make ipynb    # optional: regenerate .ipynb from .md
```

Notebooks expect the GSS HDF file under `data/` (included in this repo).
