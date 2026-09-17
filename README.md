# VAE-DKL for Illiquid Corporate Bond Credit Spread Prediction

Code accompanying the MSc dissertation *"VAE-DKL Application on Corporate Bonds"* (Tawheed Alam, 2026).


## Overview
This repository contains the final implementations of the two core models developed in the dissertation:
- **V5** — the best-performing plain Deep Kernel Learning (DKL) model (RMSE 0.00323)
- **V10** — the best-performing VAE-DKL model (RMSE 0.00403–0.00444)


Both models predict credit spreads for illiquid corporate bonds using a neural feature extractor (deterministic for V5, variational for V10) jointly trained with a sparse variational Gaussian process (SVGP).

## Data
The dataset (~15M observations, 32,723 bonds, 2015–2025) combines OSBAP, CRSP/Compustat (via WRDS), and FRED. Due to data licensing (WRDS), the raw dataset is not included in this repository.
