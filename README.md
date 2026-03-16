# Tinnitus ART Generators

This repository contains Python and Rust implementations of an acoustic ripple tinnitus-treatment sound generator, along with benchmark notes comparing both ports to the original MATLAB workflow.

## Repository Layout

- `README.md` - project overview, usage, and runtime notes
- `PYTHON_RUST_MATLAB_COMPARISON.md` - Python, Rust, and MATLAB performance and equivalence notes
- `tinnitus-cross-frequency-therapy.pdf` - copy of the study paper
- `python/` - Python implementation and example audiogram inputs
- `rust/` - Rust implementation

## Study Reference

The paper discussed by this project is included at `tinnitus-cross-frequency-therapy.pdf`.

## Requirements

Python:

- Python `3.10+`
- `numpy`
- `numba` for the fastest Python backend

Rust:

- stable Rust toolchain
- `cargo`

Optional:

- `scipy` if you want to do your own waveform analysis outside the generator

## Python Generator

Main script:

- `python/generate_tinnitus_art.py`

Example audiogram inputs:

- `python/audiogram_example.json`
- `python/audiogram_example.csv`

Run from the repo root:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 4000 \
  --audiogram-file "python/audiogram_example.json" \
  --sub-id example_python_run
```

By default, output WAV files are written to `python/Participant_Sound_Files/`.

## Rust Generator

Rust package:

- `rust/Cargo.toml`
- `rust/src/main.rs`

Run from the repo root:

```bash
cargo run --manifest-path rust/Cargo.toml --release -- \
  --tinnitus-hz 4000 \
  --audiogram-file python/audiogram_example.json \
  --sub-id example_rust_run
```

By default, output WAV files are written to `rust/Participant_Sound_Files/`.

## Input Formats

Both implementations accept the same core inputs:

- tinnitus target frequency
- audiogram frequencies
- left-ear dB HL thresholds
- right-ear dB HL thresholds

You can provide audiogram data in either of these ways.

### Option 1: JSON or CSV file

JSON example:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 6000 \
  --audiogram-file "python/audiogram_example.json" \
  --sub-id example_json_run
```

CSV example:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 6000 \
  --audiogram-file "python/audiogram_example.csv" \
  --sub-id example_csv_run
```

CSV columns can be:

- `freq_khz,left_db_hl,right_db_hl`
- or `freq_hz,left_db_hl,right_db_hl`

### Option 2: Inline command-line values

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 4000 \
  --audiogram-freqs-khz 0.125,0.25,0.5,0.75,1,1.5,2,3,4,6,8 \
  --left-db-hl 10,10,10,10,10,15,15,20,20,25,30 \
  --right-db-hl 10,10,10,10,10,15,15,20,20,25,30 \
  --sub-id example_inline_run
```

## Useful Flags

Python:

- `--engine auto|numba|numpy` - choose the synthesis backend
- `--asymmetry-fraction 1.0|0.5|0.25|0.0` - control how much left/right audiogram asymmetry to keep
- `--output-dir PATH` - choose where output files are written
- `--progress-every N` - control progress logging frequency

Rust:

- `--asymmetry-fraction 1.0|0.5|0.25|0.0`
- `--output-dir PATH`
- `--threads N` - set Rayon worker count explicitly

## Performance Snapshot

Measured on:

- `MacBook Pro`
- `Apple M2 Max`
- `64 GB` memory

Representative full `900`-chunk / `60 min` run times on this machine:

- original MATLAB script: `588.51 s`
- Python `numba`: `18.98 s` on a repeated run
- Rust release build: `12.76 s` on a repeated run

The implementation comparison and deterministic equivalence checks are documented in `PYTHON_RUST_MATLAB_COMPARISON.md`.

## Cold vs Warm Start

In this repository, “cold” and “warm” refer to one-time setup costs that can affect the first run.

- `cold` means the implementation is being run without prior cached state for that benchmark
- `warm` means caches, compiled artifacts, or runtime-generated code are already available

For Python `numba`:

- the main warm-up cost is JIT compilation of the hot DSP functions
- for a full `900`-chunk run on this machine, the difference was negligible:
  - cold: `18.55 s`
  - warm: `19.07 s`

For Rust:

- there is no JIT at runtime
- the main difference is binary / OS cache behavior after the first run
- observed full `900`-chunk timings were:
  - first run: `20.41 s`
  - repeated run: `12.76 s`

So for short tests, warm-up effects can matter more. For full-length Python `numba` generation, the JIT cost is small relative to total runtime.

## Notes

- `--engine auto` in Python prefers `numba` when available
- the Python implementation is intended to be functionally equivalent to the MATLAB generator, not necessarily RNG-identical on ordinary randomized runs
- the Rust version accepts the same core treatment inputs as the Python version
- the example audiogram files are illustrative only; replace them with your own measured values for real use
