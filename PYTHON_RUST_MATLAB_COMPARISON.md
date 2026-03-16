# Python, Rust, and MATLAB Generator Comparison

This repository contains Python and Rust implementations of the acoustic ripple tinnitus-treatment sound generator, compared against the original MATLAB workflow.

Reference points in this repo:

- study paper: `tinnitus-cross-frequency-therapy.pdf`
- Python generator: `python/generate_tinnitus_art.py`
- Rust generator: `rust/src/main.rs`
- example audiogram inputs: `python/audiogram_example.json`, `python/audiogram_example.csv`

## Benchmark Environment

The timings in this document were captured on:

- `MacBook Pro`
- `Apple M2 Max`
- `64 GB` memory

## Runtime Comparison

Benchmarks below use the same representative full-length benchmark configuration across implementations. They are intended to compare implementations, not to prescribe a specific treatment target or audiogram profile.

| Implementation | Script / Engine | Benchmark basis | Cold / first run | Warm / repeated run | Relative speed vs MATLAB | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| MATLAB reference | Original `ART_Sound_Generation.m` as provided | Measured end-to-end at `900` chunks / full `60 min` file | `588.51 s` | n/a | `1.0x` | Original interactive workflow |
| Python fallback | `python/generate_tinnitus_art.py --engine numpy` | Measured end-to-end at `100` chunks, extrapolated to `900` chunks | not separately measured | estimated `237.69 s` (`3m 57.7s`) | about `2.5x` faster | Pure-NumPy fallback path |
| Python accelerated | `python/generate_tinnitus_art.py --engine numba` | Measured end-to-end at `900` chunks / full `60 min` file | `18.55 s` | `19.07 s` | about `30.8x` faster | Full-file cold and warm runs were effectively the same |
| Rust release | `cargo run --manifest-path rust/Cargo.toml --release -- ...` | Measured end-to-end at `900` chunks / full `60 min` file | `20.41 s` | `12.76 s` | about `46.1x` faster | First run vs repeated run differed noticeably |

## Cold vs Warm Start

In this document:

- `cold` means the benchmark includes first-run overheads or uncached state for that implementation
- `warm` means a repeated run after those costs have already been paid

Observed behavior:

- Python `numba`:
  - the main warm-up cost is JIT compilation of the hot DSP functions
  - for a full `900`-chunk run on this machine, the difference was negligible (`18.55 s` cold vs `19.07 s` warm)
- Rust:
  - there is no JIT at runtime
  - the main difference is process / binary / OS cache behavior on repeated runs
  - observed full-run timings were `20.41 s` first run vs `12.76 s` repeated run

Practical takeaway:

- for short tests, warm-up effects can matter more
- for full-length Python `numba` generation, JIT cost is small relative to total work
- on repeated full runs, the current Rust build is faster than the current Python `numba` build on this machine

## Similarity Sanity Checks

### Python vs MATLAB deterministic sample

To compare MATLAB and Python directly, a deterministic `4 s` stereo sample was generated in both implementations using the same fixed parameters:

- same tinnitus target frequency
- same audiogram-derived per-band corrections
- same modulation band
- same `f0`
- same modulation phase offsets
- same no-random-harmonic-phase setting

This avoids false differences caused only by MATLAB/Python RNG behavior.

| Metric | Result |
| --- | ---: |
| Sample rate | `44.1 kHz` in both |
| Duration | `4.0 s` in both |
| Max absolute PCM difference | `1` |
| Mean absolute PCM difference | `0.49` PCM counts |
| RMS PCM difference | `0.70` PCM counts |
| Left channel waveform correlation | `0.9999992547` |
| Right channel waveform correlation | `0.9999903416` |
| Left channel band-spectrum correlation | `0.9999999991` |
| Right channel band-spectrum correlation | `0.9999998612` |

### Rust vs Python deterministic sample

To compare Rust and Python directly, a deterministic `4 s` stereo sample was generated in both implementations using the same fixed parameters.

| Metric | Result |
| --- | ---: |
| Sample rate | `44.1 kHz` in both |
| Duration | `4.0 s` in both |
| Max absolute PCM difference | `1` |
| Mean absolute PCM difference | `0.44` PCM counts |
| RMS PCM difference | `0.66` PCM counts |
| Left channel waveform correlation | `0.9999987262` |
| Right channel waveform correlation | `0.9999839545` |

Interpretation:

- Python is functionally equivalent to the MATLAB implementation.
- Rust is functionally equivalent to the accelerated Python implementation.
- In practice, all three implementations are extremely close for matched deterministic inputs.
- Remaining tiny differences are at the level of quantization / floating-point rounding, not algorithmic mismatch.

## Why the Python Version Is Faster Than MATLAB

The speedup is not coming from changing the treatment logic. It comes from changing how the same logic is executed.

### 1. The heavy DSP loop runs in compiled machine code

The fastest Python mode uses `numba` to JIT-compile the inner synthesis loops.

- MATLAB builds large harmonic-time matrices and evaluates them in its own runtime.
- The Python `numba` path compiles the hot loop to native code and runs it directly.
- This removes most Python interpreter overhead while keeping the same signal math.

### 2. The Python generator avoids regenerating the full signal twice

An earlier Python approach, like the MATLAB workflow, effectively paid a heavy two-pass cost:

- once to find peak level
- once to write final output

The current Python version synthesizes each chunk once, stores it in a temporary float buffer, tracks the peak, and then writes the final WAVs from that buffer with the correct scale.

That preserves the same final normalization behavior while avoiding duplicated DSP work.

### 3. Repeated structural work is cached

The Python implementation caches items that do not need to be rebuilt every time:

- ramp windows
- double-ramp multipliers
- fixed-duration time vectors
- harmonic / band assignment plans for a given `f0` and correction set

### 4. The accelerated path computes outputs more directly

The current accelerated path avoids some of the heaviest temporary-array construction used in earlier vectorized approaches.

- It computes the left and right outputs directly.
- It applies equivalent modulation and ramping without storing as many large intermediate arrays.
- That reduces memory traffic and improves cache behavior.

### 5. Output writing is streamlined

The Python version writes through a simple temp-buffer-to-WAV flow.

- less chunk-management overhead
- less repeated file-update overhead
- fewer redundant copies in the hot path

## Why the Rust Version Is Faster Than MATLAB

The Rust speedup also preserves the same overall treatment logic while changing execution strategy.

### 1. Ahead-of-time compiled release binary

Rust runs as an optimized native executable built with:

- release mode
- thin LTO
- single codegen unit for the final build

That removes interpreter overhead and avoids runtime JIT dependency.

### 2. Parallel chunk synthesis with Rayon

Each `4 s` chunk is independent.

- the Rust implementation synthesizes chunks in parallel across CPU cores
- on this machine it used `12` Rayon worker threads during the benchmark
- this is the biggest practical reason repeated Rust runs outran the Python `numba` version

### 3. Precomputed harmonic plans per unique `f0`

The Rust implementation precomputes the expensive structural setup for each unique fundamental frequency used in the run:

- harmonic lists
- per-band intensities
- modulation masks
- modulation-frequency offsets

This avoids rebuilding the same plan over and over when multiple chunks share the same `f0`.

### 4. Direct inner-loop synthesis

Instead of materializing large intermediate matrices in the hot path, the Rust version computes output samples directly in tight loops.

- lower allocation pressure
- lower memory traffic
- better cache locality

### 5. Single post-synthesis scaling and direct WAV writing

Like the optimized Python version, the Rust implementation separates:

- chunk synthesis
- global peak determination
- final WAV emission

That avoids unnecessary recomputation while keeping the same clipping protection behavior.

## Bottom Line

- MATLAB remains the reference implementation for the study logic.
- Python reproduces that logic closely enough to serve as a practical replacement.
- Rust also matches the optimized Python implementation closely while running faster on repeated full-file benchmarks on this machine.

## Benchmark Notes

- MATLAB full-file time in the table is an actual measured run of `ART_Sound_Generation.m` with `900` chunks and interactive inputs supplied programmatically.
- Python `numba` times are actual measured `900`-chunk / `60 min` runs.
- Rust times are actual measured `900`-chunk / `60 min` release-build runs.
- Python `numpy` full-file time remains an extrapolation from a measured `100`-chunk run.
- Deterministic equivalence testing is more meaningful than comparing ordinary randomized full runs sample-for-sample, because the implementations do not share the same RNG sequence by default.
