# Make_Stationary

MATLAB implementation of a lightweight deseasonalization operator for hourly PV and GHI time series.  
The method relies on a ridge-regularized Extreme Learning Machine (ELM) and a phase-conditioned projection to extract dominant cyclic components (diurnal and annual). The residual signal becomes substantially more stationary, quantified through PACF-based indicators.

## Features
- Deseasonalization of hourly PV/GHI data using a single-hidden-layer ELM.
- Phase-only model for extracting the seasonal component without multi-year averages or harmonic tuning.
- Stationarity diagnostics based on PACFsum.
- Works with any hourly cyclic time series of at least two consecutive years.
- Optional hyperparameter search for lag length (`LagH`) and hidden layer size (`m`).
- Optional visualisation (3-D surface / contour) of the PACF-based objective function.

## Files
- **Make_Stationary.m** — core operator (ELM training, projection, deseasonalization, metrics)  
- **script_demo.m** — minimal examples for PV (Corsica) and GHI (Ajaccio)  
- **LICENSE** — GPL-3.0  
- **README.md** — this document  

## Data Requirements
- Hourly time series covering ≥ 2 full years.
- PV power (MW) or GHI (W/m²). Negative values are clamped to zero.

## Quick Start
```matlab
T = readtable('Data.xlsx', 'PreserveVariableNames', true);
series = max(T.("Solaire photovoltaïque (MW)"), 0);

LagH  = 24;
m     = 500;
annee = 2;

[R_te2, ytrue_te, S_te2, R_te, S_te, h, S_raw, S_r1, S_r2] = ...
    Make_Stationary(series, annee, LagH, m);
