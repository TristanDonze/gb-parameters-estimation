# Galactic Binary Parameter Estimation

This repository is the final stage of the work carried out during my internship at [L2IT](https://www.l2it.in2p3.fr/). It focuses on estimating Galactic binary parameters using MCMC and comparing results from real and reconstructed sources.

The project consists of four parts:

1. [Dataset generation](https://github.com/TristanDonze/gb-dataset-gen)
2. [Source counting](https://github.com/TristanDonze/gb-sources-counting)
3. [Source separation](https://github.com/TristanDonze/gb-sources-separation)
4. [Parameter estimation](https://github.com/TristanDonze/gb-parameters-estimation) (this repository)

## Installation with uv

With `uv` installed, run from the repository root:

```bash
uv sync --python 3.12
uv pip install attrs
```

The scripts use [Sylvain Marsat's version of ptemcee](https://github.com/SylvainMarsat/ptemcee), imported with `from ptemcee import ptemcee`. If the `ptemcee/` directory is missing, clone it into the repository root:

```bash
git clone https://github.com/SylvainMarsat/ptemcee.git ptemcee
```

The code loads this local directory directly; installing `ptemcee` from PyPI is unnecessary. `attrs` is its additional dependency. The commands below use `--no-sync` to preserve the environment after installation.

## Running MCMC

The input file `reconstructed_waveform.h5` must be present in the repository root. For each of the nine sources, two targets are analyzed in the A and E channels:

- `real`: real source + real noise (`source_real + residual_real`).
- `reconstructed`: reconstructed source + reconstructed noise (`source_reconstructed + residual_reconstructed`).

Run all 18 MCMC calculations sequentially:

```bash
uv run --no-sync python run_mcmc.py
```

Or run them in parallel on CPU:

```bash
uv run --no-sync python run_mcmc_parallelized.py
```

Settings are defined at the top of `run_mcmc.py`: 3,000 iterations, 64 walkers, and 10 temperatures by default. The parallel version uses the same settings, with up to nine processes (`MAX_WORKERS` in `run_mcmc_parallelized.py`).

Results are saved in `posteriors/` as `posteriors_snr_<low|medium|high>_quality_<1|2|3>_<real|reconstructed>.npz`. Each file includes samples, chains, log-likelihoods, and true parameter values. Rerunning a calculation overwrites its output file.

## Visualizing results

Run the notebook `visualize_posterior.ipynb` to visualize MCMC results. 
In the notebook, select an `.npz` file using `POSTERIOR_PATH`, adjust `BURN_IN`, and run the cells. The matching file (`real` or `reconstructed`) is loaded automatically to compare signals, chain evolution, and posterior distributions (*corner plots*). Keep `reconstructed_waveform.h5` in the repository root as well.