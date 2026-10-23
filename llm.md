# Make_Stationary: deseasonalization benchmark for renewable-energy time series (MATLAB)

**Instructions for a language model using this software.** Read this file first; the machine-readable versions are `websemantic.json` (fields, units, bounds) and `ontology/` (RDF/Turtle and JSON-LD, QUDT units).

Software DOI (all versions): https://doi.org/10.5281/zenodo.21490743 · licence GPL-3.0 · language MATLAB · authors: Maklewa Agoundedemba, Cyril Voyant, Milan Despotovic, Luis Antonio Garcia-Gutierrez, Gilles Notton

Article: M. Agoundedemba, C. Voyant, M. Despotović, L. A. Garcia-Gutierrez, G. Notton. Does stationarization help? A frugal benchmark of deseasonalization for renewable energy forecasting (manuscript submitted; no result of it is reported here beyond the repository README).

## What it does
Tests whether removing the deterministic daily and annual cycle improves the final forecast of energy and weather series.

## Why
A more stationary residual is not, by itself, a more predictable one: the benchmark separates residual diagnostics from forecasting utility.

## Method
- Additive decomposition Y(t) = S(t) + R(t); forecast Ŷ(t+h) = Ŝ(t+h) + R̂(t+h)
- Phase features p(t) = [sin 2πτ_d/24, cos 2πτ_d/24, sin 2πτ_y/365, cos 2πτ_y/365]
- PROJECTION and PHASE: frugal phase-conditioned ELM operators for S; baselines: crossed Fourier regression, LOESS, ±10-day rolling median; RAW as control
- Residual forecasts by an ELM and a linear AR; scores nRMSE, nMAE, nMBE, R², NICE

## Entry points
- `Optimization/optimization_all_variables.m`: grid search of (L, m) per variable
- `Deseasonalization/main_deseasonalization.m`: deseasonalizes the 9 variables with the 6 methods
- `Forecasting/main_forecast_GHI_and_GHI_30min.m; Forecasting/main_forecasting_PV_WT_PT_WS_T2M.m`: forecasts and scores the residuals

## Inputs
| Field | Unit | Type | Bounds | Required | Default (proposal only) | Meaning |
|---|---|---|---|---|---|---|
| `variables` | as stated | set |  | yes |  | nine Corsican series 2021–2023 (GHI, PV, load, wind power, wind speed, temperature, at 30 min or 1 h), included in Data/ |

## Outputs
| Output | Unit | Meaning |
|---|---|---|
| `*_FULL.csv, Metrics.csv` | as stated | residuals and stationarity diagnostics (PACF, entropy, Lyapunov exponent) |
| `forecast metrics` | 1 (unit:UNITLESS) | nRMSE, nMAE, nMBE, R², NICE by method and horizon (1 to 6 h) |

## How to read the results
- Deseasonalization is a selective preprocessing option, to be kept only when it improves the final forecast.

## Rules
1. Never invent a value. Each input is given by the user, an exact unit conversion of a given value, or a documented default that the user explicitly accepts.
2. Use the canonical unit of each field; convert exactly and say so (for example 2 km -> 2000 m). Never mix a percentage and a fraction.
3. Ask when a request is ambiguous or a required input is missing; do not infer a number from a place name or a qualitative word.
4. Run the code unmodified. Report only what was actually computed, with quantity, unit and the input values used; never present a planned or unavailable run as a result.
5. State the scope and the limits below when you report a result.

## Scope
Supported: test whether a deseasonalization method helps the final forecast.
Out of scope (refuse or redirect): results of the submitted manuscript.

## Limits
- One region, three years, horizons up to 6 hours; the README lists three differences with the manuscript.

## Cite
Software: Maklewa Agoundedemba, Cyril Voyant, Milan Despotovic, Luis Antonio Garcia-Gutierrez, Gilles Notton. Make_Stationary: deseasonalization benchmark for renewable-energy time series (MATLAB). Zenodo. https://doi.org/10.5281/zenodo.21490743

Guide with the theory and runnable examples: https://cyrilvoyant.github.io/ · author: https://person.cyrilvoyant.com/ · generated 2026-10-23, documentation only.
