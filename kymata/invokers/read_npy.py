"""Small CLI helper for reading neuron selections from a .npy file.

Historically this script was called once per index (thousands of times) from Slurm
batch scripts. That is very slow because it reloads the .npy file each time.

This version supports bulk output and Slurm-style chunking in a single process.

Expected .npy format
--------------------
An array of shape (N, 2) where each row is (layer, neuron).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _load_pairs(path: str) -> np.ndarray:
    arr = np.load(path)
    arr = np.asarray(arr)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError(f"Expected array of shape (N, 2+) but got {arr.shape}")
    # Keep only first two columns.
    return arr[:, :2]


def _emit_single(arr: np.ndarray, index: int) -> None:
    layer, neuron = arr[index]
    # Space separated for easy shell parsing.
    print(f"{int(layer)} {int(neuron)}")


def _emit_range(arr: np.ndarray, start: int, end: int, fmt: str) -> None:
    sl = arr[start:end]
    if fmt == "pairs":
        # One pair per line: "<layer> <neuron>"
        for layer, neuron in sl:
            print(f"{int(layer)} {int(neuron)}")
    elif fmt == "bash":
        # Bash-friendly output.
        layers = " ".join(f"layer{int(layer)}" for layer, _ in sl)
        neurons = " ".join(str(int(neuron)) for _, neuron in sl)
        print("layers=" + layers)
        print("neurons=" + neurons)
    else:
        raise ValueError(fmt)


def _emit_chunk(arr: np.ndarray, task_id: int, n_jobs: int) -> None:
    total = int(arr.shape[0])
    chunk = (total + n_jobs - 1) // n_jobs
    start = task_id * chunk
    end = min(start + chunk, total)

    print(f"total={total}")
    print(f"start={start}")
    print(f"end={end}")

    if start >= total:
        print("layers=")
        print("neurons=")
        return

    sl = arr[start:end]
    layers = " ".join(f"layer{int(layer)}" for layer, _ in sl)
    neurons = " ".join(str(int(neuron)) for _, neuron in sl)
    print("layers=" + layers)
    print("neurons=" + neurons)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Read selections from (layer, neuron) .npy files")
    parser.add_argument("path", help="Path to the .npy file")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--index", type=int, help="Print a single pair (layer neuron) at index")
    mode.add_argument("--range", nargs=2, type=int, metavar=("START", "END"), help="Print pairs for [START, END)")
    mode.add_argument("--chunk", nargs=2, type=int, metavar=("TASK_ID", "N_JOBS"), help="Print bash vars for a chunk")

    parser.add_argument(
        "--format",
        choices=["pairs", "bash"],
        default="pairs",
        help="Output format for --range (ignored by --index/--chunk)",
    )

    args = parser.parse_args(argv)

    try:
        arr = _load_pairs(args.path)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    if args.index is not None:
        _emit_single(arr, args.index)
        return 0

    if args.range is not None:
        start, end = args.range
        _emit_range(arr, start, end, args.format)
        return 0

    task_id, n_jobs = args.chunk
    _emit_chunk(arr, task_id, n_jobs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
