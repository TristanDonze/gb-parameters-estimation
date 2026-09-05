"""Create comparison figures and metrics for every posterior pair."""

import csv
from pathlib import Path

import corner
import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


POSTERIOR_DIRECTORY = Path("posterior")
OUTPUT_DIRECTORY = Path("posterior_figures")
H5_PATH = Path("reconstructed_waveform.h5")
BURN_IN = 1000
MAX_CORNER_SAMPLES = 30_000

BLUE = "#2563EB"
RED = "#DC2626"
QUALITY_COLORS = {1: "#E11D48", 2: "#F59E0B", 3: "#10B981"}
LABELS = (
    r"$f_0-f_{0,\rm true}$ [$\mu$Hz]",
    r"$\dot f$ [$10^{-15}$ Hz s$^{-1}$]",
    r"$\sin\beta$",
    r"$\lambda$ [rad]",
    r"$A$ [$10^{-22}$]",
    r"$\cos\iota$",
)


def complex_channels(waveform):
    return waveform[0] + 1j * waveform[1], waveform[2] + 1j * waveform[3]


def squared_norm(waveform):
    channel_a, channel_e = complex_channels(waveform)
    return float(np.vdot(channel_a, channel_a).real + np.vdot(channel_e, channel_e).real)


def waveform_metrics(reference, candidate):
    reference_a, reference_e = complex_channels(reference)
    candidate_a, candidate_e = complex_channels(candidate)
    overlap = (
        np.vdot(reference_a, candidate_a) + np.vdot(reference_e, candidate_e)
    ).real / np.sqrt(squared_norm(reference) * squared_norm(candidate))
    delta_norm = np.sqrt(squared_norm(reference - candidate))
    return float(overlap), float(delta_norm)


def transform_parameters(values, reference_f0):
    transformed = np.array(values, dtype=float, copy=True)
    transformed[..., 0] = (transformed[..., 0] - reference_f0) * 1e6
    transformed[..., 1] *= 1e15
    transformed[..., 4] *= 1e22
    return transformed


def snr_level(snr):
    """Map an SNR to the requested low, medium, or high category."""
    if 10.0 <= snr < 30.0:
        return "low"
    if 30.0 <= snr < 50.0:
        return "medium"
    if 50.0 <= snr <= 100.0:
        return "high"
    raise ValueError(f"SNR {snr} is outside the supported [10, 100] range.")


def load_posterior(level, quality, kind):
    path = POSTERIOR_DIRECTORY / (
        f"posteriors_snr_{level}_quality_{quality}_{kind}.npz"
    )
    with np.load(path) as archive:
        return {key: archive[key] for key in archive.files}


def posterior_samples(posterior):
    chain = posterior["chain"]
    if not 0 <= BURN_IN < len(chain):
        raise ValueError(f"Invalid burn-in {BURN_IN} for a chain of length {len(chain)}")
    return chain[BURN_IN:].reshape(-1, chain.shape[-1])


def thin_for_corner(samples):
    if len(samples) <= MAX_CORNER_SAMPLES:
        return samples
    indices = np.linspace(0, len(samples) - 1, MAX_CORNER_SAMPLES, dtype=int)
    return samples[indices]


def standardized_median_shifts(real_samples, reconstructed_samples):
    medians = []
    scales = []
    for samples in (real_samples, reconstructed_samples):
        medians.append(np.median(samples, axis=0))
        low, high = np.quantile(samples, (0.16, 0.84), axis=0)
        scales.append((high - low) / 2.0)
    denominator = np.sqrt(scales[0] ** 2 + scales[1] ** 2)
    return (medians[1] - medians[0]) / denominator


def common_ranges(real_samples, reconstructed_samples):
    combined = np.vstack((real_samples, reconstructed_samples))
    ranges = []
    for index in range(combined.shape[1]):
        low, high = np.quantile(combined[:, index], (0.002, 0.998))
        padding = 0.05 * (high - low) if high > low else 1.0
        ranges.append((low - padding, high + padding))
    return ranges


def make_corner_plot(record):
    real_samples = thin_for_corner(record["real_samples"])
    reconstructed_samples = thin_for_corner(record["reconstructed_samples"])
    ranges = common_ranges(real_samples, reconstructed_samples)

    figure = corner.corner(
        real_samples,
        labels=LABELS,
        truths=record["truth"],
        truth_color="#111827",
        color=BLUE,
        range=ranges,
        bins=32,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.393, 0.865),
        hist_kwargs={"linewidth": 1.6},
        contour_kwargs={"linewidths": 1.3},
        label_kwargs={"fontsize": 10},
    )
    corner.corner(
        reconstructed_samples,
        fig=figure,
        color=RED,
        range=ranges,
        bins=32,
        smooth=1.0,
        smooth1d=1.0,
        plot_datapoints=False,
        fill_contours=False,
        levels=(0.393, 0.865),
        hist_kwargs={"linewidth": 1.6},
        contour_kwargs={"linewidths": 1.3},
    )
    figure.legend(
        handles=(
            Line2D([0], [0], color=BLUE, lw=2, label="Réelle"),
            Line2D([0], [0], color=RED, lw=2, label="Reconstruite"),
            Line2D([0], [0], color="#111827", ls="--", label="Vraie"),
        ),
        loc="upper right",
        frameon=False,
    )
    figure.suptitle(
        f"Élément {record['element']} · source {record['source']} · "
        f"SNR {record['snr']:.1f} · qualité {record['quality']}",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )
    filename = f"corner_snr_{record['level']}_quality_{record['quality']}.png"
    figure.savefig(OUTPUT_DIRECTORY / filename, dpi=150, bbox_inches="tight")
    plt.close(figure)


def make_marginal_overview(records):
    figure, axes = plt.subplots(len(records), 6, figsize=(20, 24))
    for row, record in enumerate(records):
        for column, ax in enumerate(axes[row]):
            combined = np.concatenate(
                (
                    record["real_samples"][:, column],
                    record["reconstructed_samples"][:, column],
                )
            )
            low, high = np.quantile(combined, (0.002, 0.998))
            bins = np.linspace(low, high, 45)
            ax.hist(
                record["real_samples"][:, column], bins=bins, density=True,
                histtype="step", color=BLUE, linewidth=1.6,
            )
            ax.hist(
                record["reconstructed_samples"][:, column], bins=bins,
                density=True, histtype="step", color=RED, linewidth=1.6,
            )
            ax.axvline(record["truth"][column], color="#111827", ls="--", lw=1)
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LABELS[column], fontsize=11)
            if column == 0:
                ax.set_ylabel(
                    f"{record['element']}/{record['source']}\n"
                    f"ρ={record['snr']:.0f}, q={record['quality']}",
                    rotation=0,
                    ha="right",
                    va="center",
                )
    figure.legend(
        handles=(
            Line2D([0], [0], color=BLUE, lw=2, label="Réelle"),
            Line2D([0], [0], color=RED, lw=2, label="Reconstruite"),
            Line2D([0], [0], color="#111827", ls="--", label="Vraie"),
        ),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.979),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"Toutes les marginales · burn-in {BURN_IN}",
        fontsize=16,
        fontweight="bold",
        y=0.999,
    )
    figure.tight_layout(rect=(0.03, 0, 1, 0.95))
    figure.savefig(
        OUTPUT_DIRECTORY / "all_posterior_marginals.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_summary_plot(records):
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for quality in (1, 2, 3):
        selected = [record for record in records if record["quality"] == quality]
        snr = [record["snr"] for record in selected]
        axes[0].scatter(
            snr,
            [1.0 - record["source_overlap"] for record in selected],
            s=75,
            color=QUALITY_COLORS[quality],
            label=f"Qualité {quality}",
        )
        axes[1].scatter(
            snr,
            [record["max_standardized_shift"] for record in selected],
            s=75,
            color=QUALITY_COLORS[quality],
            label=f"Qualité {quality}",
        )
        for record in selected:
            label = f"{record['element']}/{record['source']}"
            axes[0].annotate(
                label,
                (record["snr"], 1.0 - record["source_overlap"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
            axes[1].annotate(
                label,
                (record["snr"], record["max_standardized_shift"]),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )
    axes[0].set(
        xlabel="SNR",
        ylabel="Mismatch complexe  $1-\\mathrm{overlap}$",
        title="Erreur de reconstruction de l’onde",
        yscale="log",
    )
    axes[1].axhline(1.0, color="#64748B", ls="--", lw=1.2)
    axes[1].set(
        xlabel="SNR",
        ylabel="Décalage marginal maximal  [$\\sigma$ combiné]",
        title="Séparation des deux postérieurs",
        yscale="log",
    )
    axes[0].legend(frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        OUTPUT_DIRECTORY / "reconstruction_summary.png",
        dpi=170,
        bbox_inches="tight",
    )
    plt.close(figure)


def save_metrics(records):
    fields = (
        "element",
        "source",
        "quality",
        "snr",
        "source_overlap",
        "source_delta_norm",
        "target_overlap",
        "target_delta_norm",
        "max_standardized_shift",
        "largest_shift_parameter",
    )
    with (OUTPUT_DIRECTORY / "posterior_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: record[field] for field in fields} for record in records)


def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(H5_PATH, "r") as h5_file:
        for index, (element, source) in enumerate(
            zip(h5_file["element_index"][:], h5_file["source_number"][:])
        ):
            element = int(element)
            source = int(source)
            quality = int(h5_file["reconstruction_quality"][index])
            level = snr_level(float(h5_file["snr"][index]))
            real = load_posterior(level, quality, "real")
            reconstructed = load_posterior(level, quality, "reconstructed")
            truth = transform_parameters(real["truth"], real["truth"][0])
            real_samples = transform_parameters(
                posterior_samples(real), real["truth"][0]
            )
            reconstructed_samples = transform_parameters(
                posterior_samples(reconstructed), real["truth"][0]
            )

            source_real = h5_file["source_real"][index]
            source_reconstructed = h5_file["source_reconstructed"][index]
            target_real = source_real + h5_file["residual_real"][index]
            target_reconstructed = (
                source_reconstructed + h5_file["residual_real"][index]
            )
            source_overlap, source_delta_norm = waveform_metrics(
                source_real, source_reconstructed
            )
            target_overlap, target_delta_norm = waveform_metrics(
                target_real, target_reconstructed
            )
            shifts = standardized_median_shifts(
                real_samples, reconstructed_samples
            )
            largest_index = int(np.argmax(np.abs(shifts)))
            record = {
                "element": element,
                "source": source,
                "quality": quality,
                "level": level,
                "snr": float(h5_file["snr"][index]),
                "source_overlap": source_overlap,
                "source_delta_norm": source_delta_norm,
                "target_overlap": target_overlap,
                "target_delta_norm": target_delta_norm,
                "max_standardized_shift": float(np.max(np.abs(shifts))),
                "largest_shift_parameter": str(real["parameter_names"][largest_index]),
                "truth": truth,
                "real_samples": real_samples,
                "reconstructed_samples": reconstructed_samples,
            }
            records.append(record)
            make_corner_plot(record)
            print(
                f"Created corner for {element}/{source}: "
                f"max shift {record['max_standardized_shift']:.2f} sigma"
            )

    make_marginal_overview(records)
    make_summary_plot(records)
    save_metrics(records)


if __name__ == "__main__":
    main()
