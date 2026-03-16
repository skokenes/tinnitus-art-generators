# Tinnitus ART Python Port

This repository contains a Python implementation of the acoustic ripple tinnitus treatment sound generator, along with benchmarking notes comparing it to the original MATLAB workflow.

Files at the repo root:

- `README.md` - project overview and usage
- `PYTHON_MATLAB_COMPARISON.md` - performance and equivalence comparison
- `tinnitus-cross-frequency-therapy.pdf` - copy of the study paper

Python files live in `python/`:

- `python/generate_tinnitus_art.py` - main generator
- `python/audiogram_speros.json` - example JSON audiogram input
- `python/audiogram_speros.csv` - example CSV audiogram input

## Study Reference

The study paper is included here: `tinnitus-cross-frequency-therapy.pdf`

## Requirements

- Python 3.10+
- `numpy`
- `numba` for the fastest backend

Optional:

- `scipy` if you want to do your own waveform analysis outside the generator

## Run The Python Generator

From the repo root:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 3000 \
  --audiogram-file "python/audiogram_speros.json" \
  --sub-id my_3khz
```

That writes output WAV files into `python/Participant_Sound_Files/` by default.

## Input Options

You can provide audiogram data in either of these ways.

### Option 1: JSON or CSV file

JSON example:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 6000 \
  --audiogram-file "python/audiogram_speros.json" \
  --sub-id my_6khz
```

CSV example:

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 3000 \
  --audiogram-file "python/audiogram_speros.csv" \
  --sub-id my_3khz_csv
```

CSV columns can be:

- `freq_khz,left_db_hl,right_db_hl`
- or `freq_hz,left_db_hl,right_db_hl`

### Option 2: Inline command-line values

```bash
python "python/generate_tinnitus_art.py" \
  --tinnitus-hz 3000 \
  --audiogram-freqs-khz 0.125,0.25,0.5,0.75,1,1.5,2,3,4,6,8 \
  --left-db-hl 5,10,5,5,5,5,15,20,15,35,20 \
  --right-db-hl 10,10,15,10,5,5,5,5,5,5,15 \
  --sub-id my_3khz_inline
```

## Useful Flags

- `--engine auto|numba|numpy` - choose the synthesis backend
- `--asymmetry-fraction 1.0|0.5|0.25|0.0` - control how much left/right audiogram asymmetry to keep
- `--output-dir PATH` - choose where output files are written
- `--progress-every N` - control progress logging frequency

## Notes

- `--engine auto` prefers `numba` when available
- The Python implementation was benchmarked against the original MATLAB script and is documented in `PYTHON_MATLAB_COMPARISON.md`
- The Python version is intended to be functionally equivalent to the MATLAB generator, not necessarily RNG-identical on ordinary randomized runs
