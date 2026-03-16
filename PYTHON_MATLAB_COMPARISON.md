# Python vs MATLAB Tinnitus Generator

This repository contains a Python implementation of the acoustic ripple tinnitus-treatment sound generator described by the original MATLAB materials.

Reference points in this repo:

- study paper: `tinnitus-cross-frequency-therapy.pdf`
- Python generator: `python/generate_tinnitus_art.py`
- example audiogram inputs: `python/audiogram_example.json`, `python/audiogram_example.csv`

The Python generator accepts the same core treatment inputs as the MATLAB workflow:

- tinnitus target frequency
- audiogram frequencies
- left-ear dB HL thresholds
- right-ear dB HL thresholds

Those inputs can be supplied either inline on the command line or via `.json` / `.csv` audiogram files.

The Python implementation also supports:

- `numpy` fallback execution
- `numba` acceleration (`--engine auto` prefers it when available)
- left/right asymmetry scaling (`1.0`, `0.5`, `0.25`, `0.0`)

## Benchmark Environment

The timings in this document were captured on:

- `MacBook Pro`
- `Apple M2 Max`
- `64 GB` memory

## Comparison Table

Benchmarks below use the same representative full-length benchmark configuration across implementations. The table is intended to show implementation behavior, not to prescribe a treatment target or audiogram profile.

| Implementation | Script / Engine | Benchmark basis | Time to generate output | Relative speed vs MATLAB | Similarity to MATLAB |
| --- | --- | --- | ---: | ---: | --- |
| MATLAB reference | Original `ART_Sound_Generation.m` as provided | Measured end-to-end at `900` chunks / full `60 min` file | `588.51 s` (`9m 48.5s`) | `1.0x` | Baseline |
| Python fallback | `python/generate_tinnitus_art.py --engine numpy` | Measured end-to-end at `100` chunks, extrapolated to `900` chunks | `26.41 s` for `100` chunks; estimated `237.69 s` (`3m 57.7s`) for full file | about `2.5x` faster | Same DSP path as Python `numba`; deterministic sample matched the accelerated path bit-for-bit |
| Python accelerated | `python/generate_tinnitus_art.py --engine numba` | Measured end-to-end at `900` chunks / full `60 min` file | `33.40 s` | about `17.6x` faster | Deterministic comparison against MATLAB shows near-perfect agreement |

## Similarity Sanity Check

To compare MATLAB and Python directly, a deterministic `4 s` stereo sample was generated in both implementations using the same fixed parameters:

- same tinnitus target frequency
- same audiogram-derived per-band corrections
- same modulation band
- same `f0`
- same modulation phase offsets
- same no-random-harmonic-phase setting

This avoids false differences caused only by MATLAB/Python RNG behavior.

### Python vs MATLAB deterministic sample

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

Interpretation:

- The Python generator is functionally equivalent to the MATLAB implementation.
- The accelerated Python backend is not merely “similar sounding”; in the deterministic test it was essentially identical for practical purposes.
- Remaining tiny differences are at the level of quantization / floating-point rounding, not algorithmic mismatch.

## Why the Python Version Is Faster

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

This helps because the generator produces many short `4 s` chunks with the same overall structure.

### 4. The accelerated path computes outputs more directly

The current accelerated path avoids some of the heaviest temporary-array construction used in earlier vectorized approaches.

- It computes the left and right outputs directly.
- It applies equivalent modulation and ramping without storing as many large intermediate arrays.
- That reduces memory traffic and improves cache behavior.

This optimization was validated against MATLAB with the deterministic comparison above.

### 5. Output writing is streamlined

The Python version writes through a simple temp-buffer-to-WAV flow.

- less chunk-management overhead
- less repeated file-update overhead
- fewer redundant copies in the hot path

## Bottom Line

- MATLAB remains the reference implementation for the study logic.
- The Python implementation reproduces that logic closely enough to serve as a practical replacement.
- With `numba`, Python is dramatically faster while still matching MATLAB output behavior in the sanity check.

## Benchmark Notes

- MATLAB full-file time in the table is an actual measured run of `ART_Sound_Generation.m` with `900` chunks and interactive inputs supplied programmatically.
- Python `numba` full-file time in the table is an actual measured `900`-chunk / `60 min` run.
- Python `numpy` full-file time remains an extrapolation from a measured `100`-chunk run.
- Deterministic equivalence testing is more meaningful than comparing ordinary randomized full runs sample-for-sample, because MATLAB and Python do not share the same RNG sequence by default.
