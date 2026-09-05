"""Run the 18 independent MCMC calculations on nine CPU processes."""

import os

# Each MCMC owns one CPU process. Prevent numerical libraries from creating
# extra threads inside every process and oversubscribing the machine.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import run_mcmc


MAX_WORKERS = 9
SOURCE_KINDS = ("real", "reconstructed")

_WORKER_DATA = None


def initialize_worker():
    """Load the small HDF5 input once in each worker process."""
    global _WORKER_DATA
    _WORKER_DATA = run_mcmc.load_input(run_mcmc.INPUT_FILE)


def run_job(index, source_kind):
    """Execute and save one independent MCMC inside a worker process."""
    if _WORKER_DATA is None:
        raise RuntimeError("Worker data were not initialized.")
    output_path, shape = run_mcmc.run_and_save_posterior(
        _WORKER_DATA, index, source_kind
    )
    return index, source_kind, str(output_path), shape


def main():
    run_mcmc.OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    jobs = [
        (index, source_kind)
        for index in range(run_mcmc.N_SOURCES)
        for source_kind in SOURCE_KINDS
    ]

    print(
        f"Running {len(jobs)} posteriors with {MAX_WORKERS} processes "
        f"and {run_mcmc.N_ITER} iterations."
    )
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=context,
        initializer=initialize_worker,
    ) as executor:
        futures = {
            executor.submit(run_job, index, source_kind): (index, source_kind)
            for index, source_kind in jobs
        }
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                index, source_kind, output_path, shape = future.result()
                print(
                    f"[{completed}/{len(jobs)}] Saved {output_path} "
                    f"for source {index + 1} ({source_kind}), shape {shape}."
                )
        except Exception:
            for future in futures:
                future.cancel()
            raise


if __name__ == "__main__":
    main()
