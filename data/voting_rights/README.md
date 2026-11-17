# Data for the voting-rights technical appendix

Used by `notebooks/voting_rights_appendix.md`. The files here are built in the author's analysis repository (CultureWar) and copied to the public [YouWouldChooseNow](https://github.com/AllenDowney/YouWouldChooseNow) repository by `scripts/push_appendix_notebooks.sh` (CultureWar).

- `cps_turnout_policy.csv`: non-Hispanic Black and White turnout among citizens 18 and older, by state and year, 2000–2024.
  - Source: the CPS November Voting and Registration Supplement, via [IPUMS CPS](https://doi.org/10.18128/D030.V13.0).
  - Columns: year, key, value, group, rule, pct, se, n.
  - The group column is `Black`, `White` (both non-Hispanic) or `All` (all citizens 18+).
  - The rule column is either `census` (nonresponse counted as not voting, the Census Bureau's convention) or `dropped` (nonresponse left out).
  - Built by `scripts/report_voting_policy_turnout.py` (CultureWar).
- `ces_turnout_policy.csv`: the same from the [Cooperative Election Study](https://doi.org/10.7910/DVN/II2DB6), 2008–2024. Validated turnout among respondents matched to an active registration record.
- `state_election_measures.csv`: restrictions and expansions of voting access by state, 2000–2024, one row per measure, with the first and last federal general election it was in force for, `strength`, `id_type` (for ID laws: strict_photo, photo_affidavit, non_photo), sources and verification notes. Compiled from statutes, court opinions, state election offices, NCSL and the Brennan Center.
- `state_turnout_official.csv`: official turnout by state, 2000–2024: total ballots counted, voting-age population, noncitizen share, voting-eligible population, citizens of voting age (VAP × (1 − noncitizen share)), and turnout over CVAP and over VEP. From Michael McDonald's University of Florida Election Lab files, CC BY 4.0: ["1980-2022 General Election Turnout Rates (v1.2)"](https://election.lab.ufl.edu/dataset/1980-2022-general-election-turnout-rates-v1-2/) and ["2024 General Election Turnout Rates (v0.4)"](https://election.lab.ufl.edu/dataset/2024-general-election-turnout-rates-v0-4/) (estimates as of March 2026). Ballots are missing for some states before 2010. Built by `scripts/build_state_turnout.py` (CultureWar).
- `acs_covariates.csv`: county covariates from the ACS 5-year estimates for each election year: median income, median age, population, share non-Hispanic White, share non-Hispanic Black, and share with a BA or more. From the Census API.
- `replication.zip` (downloaded by the notebook, not stored): the Morris and Miller replication package, [doi:10.7910/DVN/XFNP4H](https://doi.org/10.7910/DVN/XFNP4H), CC0.
