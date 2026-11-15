# Data for the voting-rights technical appendix

Used by `notebooks/voting_rights_appendix.md`. The files here are built in the author's analysis repository (CultureWar) and copied to the public [YouWouldChooseNow](https://github.com/AllenDowney/YouWouldChooseNow) repository by `scripts/push_appendix_notebooks.sh` (CultureWar).

- `cps_turnout_policy.csv`: non-Hispanic Black and White turnout among citizens 18 and older, by state and year, 2000–2024.
  - Source: the CPS November Voting and Registration Supplement, via [IPUMS CPS](https://doi.org/10.18128/D030.V13.0).
  - Columns: year, key, value, group, rule, pct, se, n.
  - The rule column is either `census` (nonresponse counted as not voting, the Census Bureau's convention) or `dropped` (nonresponse left out).
  - Built by `scripts/report_voting_policy_turnout.py` (CultureWar).
- `ces_turnout_policy.csv`: the same from the [Cooperative Election Study](https://doi.org/10.7910/DVN/II2DB6), 2008–2024. Validated turnout among respondents matched to an active registration record.
- `state_election_measures.csv`: restrictions and expansions of voting access by state, 2000–2024, one row per measure, with the first and last federal general election it was in force for, `strength`, `id_type` (for ID laws: strict_photo, photo_affidavit, non_photo), sources and verification notes. Compiled from statutes, court opinions, state election offices, NCSL and the Brennan Center.
- `acs_covariates.csv`: county covariates from the ACS 5-year estimates for each election year: median income, median age, population, share non-Hispanic White, share non-Hispanic Black, and share with a BA or more. From the Census API.
- `replication.zip` (downloaded by the notebook, not stored): the Morris and Miller replication package, [doi:10.7910/DVN/XFNP4H](https://doi.org/10.7910/DVN/XFNP4H), CC0.
