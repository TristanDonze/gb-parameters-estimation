"""Run the MCMC parameter estimation for all real and reconstructed sources."""

from pathlib import Path

import h5py
import numpy as np

from gbgpu.gbgpu import GBGPU
from lisatools.detector import EqualArmlengthOrbits
from lisatools.utils.constants import YRSID_SI
from ptemcee import ptemcee

from noise import AnalyticNoise


INPUT_FILE = Path("reconstructed_waveform.h5")
OUTPUT_DIRECTORY = Path("./posteriors")

N_SOURCES = 9
N_FREQUENCY_BINS = 128
N_ITER = 3_000
BURN_IN = 0
N_WALKERS = 64
N_TEMPERATURES = 10
T_MAX = 2.0e5

T_OBS = YRSID_SI
DT = 15.0
F0_CENTER = 0.004821699107149872
SNR_MIN = 10.0
SNR_MAX = 100.0

PARAMETER_NAMES = ("f0", "fdot", "beta", "lam", "amp", "iota")

PRIOR_LIMITS = {
    "f0": (0.004819671104859512, 0.004823727109440232),
    "fdot": (9.5e-17, 1.9e-15),
    "beta": (np.sin(-1.0), np.sin(0.15)),
    "lam": (0.0, 2.0 * np.pi),
    "amp": (5.0e-24, 1.8e-22),
    "iota": (np.cos(2.25), np.cos(0.9)),
}
PRIOR_RANGE = np.asarray([PRIOR_LIMITS[name] for name in PARAMETER_NAMES])


ORBITS = EqualArmlengthOrbits(force_backend="cpu")
GB = GBGPU(orbits=ORBITS, force_backend="cpu")


def create_one_waveform(f0_center, params):
    """Generate and whiten one Galactic-binary waveform in the A/E channels."""
    GB.run_wave(
        *params,
        N=N_FREQUENCY_BINS,
        dt=DT,
        T=T_OBS,
        oversample=None,
    )

    df = 1.0 / T_OBS
    f0_width = N_FREQUENCY_BINS * df
    f_min = f0_center - f0_width / 2.0
    k_min = int(np.round(f_min / df))
    frequencies = (np.arange(N_FREQUENCY_BINS) + k_min) * df

    noise = AnalyticNoise(frequencies, "MRDv1")
    asd_a = np.sqrt(noise.psd(option="A"))
    asd_e = np.sqrt(noise.psd(option="E"))

    start_indices = (GB.start_inds - k_min).astype(np.int32)
    columns = np.arange(N_FREQUENCY_BINS)
    source_columns = columns[None, :] - start_indices[:, None]
    valid = (source_columns >= 0) & (source_columns < N_FREQUENCY_BINS)
    rows = np.broadcast_to(
        np.arange(GB.A.shape[0])[:, None], source_columns.shape
    )

    waveform_a = np.zeros(source_columns.shape, dtype=np.complex128)
    waveform_e = np.zeros(source_columns.shape, dtype=np.complex128)
    waveform_a[valid] = GB.A[rows[valid], source_columns[valid]]
    waveform_e[valid] = GB.E[rows[valid], source_columns[valid]]

    whitening_factor = np.sqrt(4.0 * df)
    waveform_a *= whitening_factor / asd_a
    waveform_e *= whitening_factor / asd_e
    return waveform_a[0], waveform_e[0]


def log_likelihood_ae(target_a, target_e, model_a, model_e):
    """Return the log likelihood for already-whitened A/E data."""
    log_likelihood = 0.0
    for target, model in ((target_a, model_a), (target_e, model_e)):
        residual = target - model
        power = np.real(residual * np.conj(residual))
        log_likelihood -= 0.5 * np.sum((power[:-1] + power[1:]) / 2.0)
    return log_likelihood


def as_complex_channels(waveform):
    """Convert an array [A.real, A.imag, E.real, E.imag] to A/E channels."""
    channel_a = waveform[0] + 1j * waveform[1]
    channel_e = waveform[2] + 1j * waveform[3]
    return channel_a, channel_e


def make_target(source, residual):
    """Build the target from one source and its associated residual."""
    source_a, source_e = as_complex_channels(source)
    residual_a, residual_e = as_complex_channels(residual)
    return source_a + residual_a, source_e + residual_e


def source_parameters(data, index):
    """Return the sampled truth vector and the two fixed phase parameters."""
    truth = np.asarray(
        [
            data["f0"][index],
            data["fdot"][index],
            np.sin(data["beta"][index]),
            data["lam"][index],
            data["amp"][index],
            np.cos(data["iota"][index]),
        ],
        dtype=float,
    )
    return truth, float(data["phi0"][index]), float(data["psi"][index])


def initial_positions(truth):
    """Create the initial positions using the perturbations from the notebook."""
    epsilon = 0.1
    scales = np.asarray(
        [
            0.1e-6,
            epsilon * truth[1],
            epsilon,
            epsilon,
            epsilon * truth[4],
            epsilon,
        ]
    )
    positions = truth + scales * np.random.randn(
        N_TEMPERATURES, N_WALKERS, len(PARAMETER_NAMES)
    )

    # Keep every initial walker strictly inside the uniform prior.
    lower = PRIOR_RANGE[:, 0]
    upper = PRIOR_RANGE[:, 1]
    margin = np.finfo(float).eps * np.abs(upper - lower)
    return np.clip(positions, lower + margin, upper - margin)


def run_one_mcmc(target_a, target_e, truth, phi0, psi):
    """Run one MCMC and return the cold chain and its log likelihood."""
    ndim = len(PARAMETER_NAMES)

    def log_prior(parameters):
        if np.any(parameters < PRIOR_RANGE[:, 0]) or np.any(
            parameters > PRIOR_RANGE[:, 1]
        ):
            return -1e99
        return 0.0

    def log_likelihood(parameters):
        if log_prior(parameters) < 0.0:
            return -1e99

        f0, fdot, sin_beta, lam, amp, cos_iota = parameters
        waveform_parameters = np.asarray(
            [
                amp,
                f0,
                fdot,
                0.0,
                phi0,
                np.arccos(cos_iota),
                psi,
                lam,
                np.arcsin(sin_beta),
            ]
        )
        model_a, model_e = create_one_waveform(F0_CENTER, waveform_parameters)
        snr = np.sqrt(np.sum(np.abs(model_a) ** 2) + np.sum(np.abs(model_e) ** 2))
        if not SNR_MIN <= snr <= SNR_MAX:
            return -1e99
        return log_likelihood_ae(target_a, target_e, model_a, model_e)

    betas = ptemcee.sampler.make_ladder(
        ndim, ntemps=N_TEMPERATURES, Tmax=T_MAX
    )
    sampler = ptemcee.sampler.Sampler(
        N_WALKERS,
        ndim,
        log_likelihood,
        log_prior,
        betas=betas,
        adaptive=True,
    )
    chain = sampler.chain(initial_positions(truth))
    chain.run(N_ITER)

    # Temperature 0 is the cold chain. Keep the iteration and walker axes so
    # that trace plots and later burn-in choices remain possible.
    cold_chain = chain.x[BURN_IN:, 0, :, :]
    cold_log_likelihood = chain.logl[BURN_IN:, 0, :]
    return cold_chain, cold_log_likelihood


def load_input(path):
    """Load either supported HDF5 layout for the 18 MCMC runs."""
    keys = (
        "amp",
        "beta",
        "f0",
        "fdot",
        "iota",
        "lam",
        "phi0",
        "psi",
        "reconstruction_quality",
        "residual_real",
        "residual_reconstructed",
        "snr",
        "source_number",
        "source_real",
        "source_reconstructed",
    )
    with h5py.File(path, "r") as input_file:
        data = {key: input_file[key][:] for key in keys}
        if "element_index" in input_file:
            data["element_index"] = input_file["element_index"][:]
        elif "mixture_index" in input_file:
            data["element_index"] = input_file["mixture_index"][:]
        else:
            raise KeyError(
                "The HDF5 file must contain 'element_index' or 'mixture_index'."
            )

    if len(data["element_index"]) != N_SOURCES:
        raise ValueError(
            f"Expected {N_SOURCES} sources, found {len(data['element_index'])}."
        )
    return data


def snr_level(snr):
    """Map an SNR to the requested low, medium, or high category."""
    if 10.0 <= snr < 30.0:
        return "low"
    if 30.0 <= snr < 50.0:
        return "medium"
    if 50.0 <= snr <= 100.0:
        return "high"
    raise ValueError(f"SNR {snr} is outside the supported [10, 100] range.")


def posterior_filename(data, index, source_kind):
    """Build the explicit output name requested for one posterior."""
    level = snr_level(float(data["snr"][index]))
    quality = int(data["reconstruction_quality"][index])
    return f"posteriors_snr_{level}_quality_{quality}_{source_kind}.npz"


def run_and_save_posterior(data, index, source_kind):
    """Run and save one posterior, for sequential or parallel execution."""
    truth, phi0, psi = source_parameters(data, index)
    target_a, target_e = make_target(
        data[f"source_{source_kind}"][index],
        data[f"residual_{source_kind}"][index],
    )
    output_path = OUTPUT_DIRECTORY / posterior_filename(
        data, index, source_kind
    )

    chain, log_likelihood = run_one_mcmc(
        target_a, target_e, truth, phi0, psi
    )
    posterior = chain.reshape(-1, len(PARAMETER_NAMES))
    np.savez_compressed(
        output_path,
        posterior=posterior,
        chain=chain,
        log_likelihood=log_likelihood,
        truth=truth,
        parameter_names=np.asarray(PARAMETER_NAMES),
        element_index=int(data["element_index"][index]),
        source_number=int(data["source_number"][index]),
        source_kind=source_kind,
        burn_in=BURN_IN,
        n_iter=N_ITER,
        n_walkers=N_WALKERS,
    )
    return output_path, posterior.shape


def main():
    data = load_input(INPUT_FILE)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    for index in range(N_SOURCES):
        for source_kind in ("real", "reconstructed"):
            print(f"Running posterior {index + 1}/{N_SOURCES}: {source_kind}")
            output_path, shape = run_and_save_posterior(
                data, index, source_kind
            )
            print(f"Saved {output_path} with posterior shape {shape}")


if __name__ == "__main__":
    main()
