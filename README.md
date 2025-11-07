Make_Stationary

MATLAB implementation of a lightweight deseasonalization operator for hourly PV and GHI time series.
The method uses a ridge-regularized Extreme Learning Machine (ELM) and a phase-conditioned projection to isolate dominant cyclic components (day/night and seasonal cycles). The resulting residuals are more stationary, evaluated using PACF-based metrics.

Features:

Deseasonalization of hourly PV or GHI data using a single-hidden-layer ELM.

Phase-only projection to extract the seasonal trend without multi-year averaging or harmonic tuning.

Computation of stationarity indicators (including PACFsum).

Compatible with any hourly cyclic time series of at least two years.

Grid search for optimising lag length (LagH) and hidden layer size (m).

3D and contour visualisation of the PACF-based objective surface.

Files:

Make_Stationary.m Core operator (ELM training, projection, metrics)

script_demo.m Examples for PV (Corsica) and GHI (Ajaccio)

LICENSE GPL-3.0

README.txt This document

Data requirements:

Hourly time series with at least two consecutive years.

PV (MW) or GHI (W/m²). Negative values are clamped to zero.

Quick start:
T = readtable('Data.xlsx','PreserveVariableNames',true);
series = max(T.("Solaire photovoltaïque (MW)"),0);
LagH = 24;
m = 500;
annee = 2;
[desais, Raw, Seasonal_Trend, desais_Proj, Seasonal_Trend_Proj, ...
Lyap, Coeff_Statio_Raw, Coeff_Statio_Proj, Coeff_Statio] = ...
Make_Stationary(series, annee, LagH, m);

Hyperparameter search:
Includes an example evaluating PACFsum on a grid of LagH (5 to 50) and m (50 to 2000) with direct surface and contour plots.

Dependencies:

MATLAB R2021a+

Parallel Computing Toolbox (optional)

Citation:
If this code contributes to your work, please cite this repository or the associated manuscript when published.
