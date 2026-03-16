#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

try:
    from numba import njit, prange

    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator

    def prange(*args):
        return range(*args)


SRATE = 44100
MOD_ASSIGN_MAT = np.array(
    [
        [1, 6],
        [1, 6],
        [1, 6],
        [2, 7],
        [3, 8],
        [4, 9],
        [5, 10],
        [6, 11],
        [7, 12],
        [8, 13],
        [9, 14],
        [10, 15],
        [11, 6],
        [12, 7],
        [13, 8],
        [14, 9],
        [15, 10],
        [16, 11],
        [16, 11],
        [16, 11],
    ],
    dtype=int,
)


@dataclass
class WavWriter:
    path: Path
    sample_rate: int
    num_channels: int
    handle: wave.Wave_write

    @classmethod
    def open(cls, path: Path, sample_rate: int, num_channels: int) -> "WavWriter":
        handle = wave.open(str(path), "wb")
        handle.setnchannels(num_channels)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        return cls(path=path, sample_rate=sample_rate, num_channels=num_channels, handle=handle)

    def write(self, samples: np.ndarray) -> None:
        samples = np.clip(samples, -1.0, 1.0)
        pcm = np.int16(samples * 32767.0)
        self.handle.writeframes(pcm.astype("<i2", copy=False).tobytes())

    def close(self) -> None:
        self.handle.close()


def parse_float_csv(text: str) -> np.ndarray:
    return np.array([float(part.strip()) for part in text.split(",") if part.strip()], dtype=float)


def load_audiogram_file(file_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".json":
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            freqs = data.get("freqs_khz", data.get("audiogram_freqs_khz"))
            left = data.get("left_db_hl")
            right = data.get("right_db_hl")
        else:
            raise ValueError("JSON audiogram file must contain an object with freqs_khz, left_db_hl, and right_db_hl.")
        if freqs is None or left is None or right is None:
            raise ValueError("JSON audiogram file must include freqs_khz, left_db_hl, and right_db_hl.")
        return np.array(freqs, dtype=float), np.array(left, dtype=float), np.array(right, dtype=float)

    if suffix == ".csv":
        freqs = []
        left = []
        right = []
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row:
                    continue
                if row.get("freq_khz"):
                    freq_khz = float(row["freq_khz"])
                elif row.get("freq_hz"):
                    freq_khz = float(row["freq_hz"]) / 1000.0
                else:
                    raise ValueError("CSV audiogram file must include freq_khz or freq_hz column.")
                if "left_db_hl" not in row or "right_db_hl" not in row:
                    raise ValueError("CSV audiogram file must include left_db_hl and right_db_hl columns.")
                freqs.append(freq_khz)
                left.append(float(row["left_db_hl"]))
                right.append(float(row["right_db_hl"]))
        if not freqs:
            raise ValueError("CSV audiogram file contained no data rows.")
        return np.array(freqs, dtype=float), np.array(left, dtype=float), np.array(right, dtype=float)

    raise ValueError("Unsupported audiogram file type. Use .json or .csv")


def resolve_audiogram_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if args.audiogram_file:
        return load_audiogram_file(args.audiogram_file)

    if not (args.audiogram_freqs_khz and args.left_db_hl and args.right_db_hl):
        raise ValueError("Provide either --audiogram-file or all of --audiogram-freqs-khz, --left-db-hl, and --right-db-hl.")

    return (
        parse_float_csv(args.audiogram_freqs_khz),
        parse_float_csv(args.left_db_hl),
        parse_float_csv(args.right_db_hl),
    )


@lru_cache(maxsize=32)
def ramp_window(srate: int, wdms: float) -> tuple[np.ndarray, int]:
    wds = round(2.0 * wdms / 1000.0 * srate)
    if wds % 2 != 0:
        wds += 1
    w = np.linspace(-(math.pi / 2.0), 1.5 * math.pi, wds)
    w = (np.sin(w) + 1.0) / 2.0
    half = round(wds / 2.0)
    return w, half


@lru_cache(maxsize=32)
def double_ramp_multiplier(srate: int, wdms: float, npts: int) -> np.ndarray:
    w, half = ramp_window(srate, wdms)
    mult = np.ones(npts, dtype=np.float64)
    mult[:half] = w[:half] * w[:half]
    mult[npts - half :] = w[half:] * w[half:]
    return mult


@lru_cache(maxsize=8)
def time_vector(duration_s: float) -> np.ndarray:
    return np.arange(1, int(round(duration_s * SRATE)) + 1, dtype=np.float64) / SRATE


@lru_cache(maxsize=256)
def stimulus_plan(
    f0: float,
    fbands_key: tuple[tuple[float, float], ...],
    fbanddb_l_key: tuple[float, ...],
    fbanddb_r_key: tuple[float, ...],
    fband_mod_key: tuple[int, ...],
    normfreq: bool,
    altharms: bool,
    altharm_depth: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fn_chunks = []
    fmod_chunks = []
    fintens_l_chunks = []
    fintens_r_chunks = []
    mod_band_set = set(fband_mod_key)
    for idx, frange_tmp in enumerate(fbands_key, start=1):
        start_h = math.ceil(frange_tmp[0] / f0)
        end_h = math.floor((frange_tmp[1] - 1.0) / f0)
        fntmp = np.arange(start_h, end_h + 1, dtype=np.int32)
        fn_chunks.append(fntmp)
        fmod_chunks.append(np.full(fntmp.shape, 1 if idx in mod_band_set else 0, dtype=np.int8))
        fintens_l_chunks.append(np.full(fntmp.shape, 10.0 ** (fbanddb_l_key[idx - 1] / 20.0), dtype=np.float64))
        fintens_r_chunks.append(np.full(fntmp.shape, 10.0 ** (fbanddb_r_key[idx - 1] / 20.0), dtype=np.float64))

    fn = np.concatenate(fn_chunks)
    fmod = np.concatenate(fmod_chunks)
    fintens_l = np.concatenate(fintens_l_chunks)
    fintens_r = np.concatenate(fintens_r_chunks)
    fi = fn.astype(np.float64) * f0

    if altharms:
        mask = (fn % 2) == 0
        fintens_l = fintens_l.copy()
        fintens_r = fintens_r.copy()
        fintens_l[mask] = altharm_depth
        fintens_r[mask] = altharm_depth

    if normfreq:
        fintens_l = fintens_l / fi
        fintens_r = fintens_r / fi

    mod_mask = fmod.astype(np.uint8)
    mod_f = np.zeros(fi.shape, dtype=np.float64)
    fmodind = np.where(mod_mask == 1)[0]
    if fmodind.size:
        mod_f_vals = np.log2(fi[fmodind] / np.min(fi[fmodind]))
        mod_f_vals = mod_f_vals - np.mean(mod_f_vals)
        mod_f[fmodind] = mod_f_vals
    return fi, fintens_l, fintens_r, mod_mask, mod_f


def apply_asymmetry_fraction(left: np.ndarray, right: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    mean = (left + right) / 2.0
    return mean + fraction * (left - mean), mean + fraction * (right - mean)


def build_db_corrections(
    pta_freqs_khz: np.ndarray,
    left_db_hl: np.ndarray,
    right_db_hl: np.ndarray,
    max_db_correction: float,
    f_centres_log: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    min_thresh = min(left_db_hl.min(), right_db_hl.min())
    left_mod = np.minimum(left_db_hl + min_thresh, max_db_correction)
    right_mod = np.minimum(right_db_hl + min_thresh, max_db_correction)
    log_pta_freqs = np.log2(pta_freqs_khz)

    if log_pta_freqs.min() > f_centres_log.min():
        log_pta_freqs = np.concatenate(([f_centres_log.min()], log_pta_freqs))
        left_mod = np.concatenate(([left_mod[0]], left_mod))
        right_mod = np.concatenate(([right_mod[0]], right_mod))

    if log_pta_freqs.max() < f_centres_log.max():
        log_pta_freqs = np.concatenate((log_pta_freqs, [f_centres_log.max()]))
        left_mod = np.concatenate((left_mod, [left_mod[-1]]))
        right_mod = np.concatenate((right_mod, [right_mod[-1]]))

    left_corr = np.interp(f_centres_log, log_pta_freqs, left_mod)
    right_corr = np.interp(f_centres_log, log_pta_freqs, right_mod)
    return left_corr, right_corr


def pick_mod_bands(target_hz: float, f_bounds_log: np.ndarray) -> np.ndarray:
    log_t_freq = math.log2(target_hz / 1000.0)
    if log_t_freq <= f_bounds_log.min():
        raise ValueError("Tinnitus frequency lower than bottom end of stimulus frequency range.")
    if log_t_freq >= f_bounds_log.max():
        raise ValueError("Tinnitus frequency higher than top end of stimulus frequency range.")
    t_centre_band = np.where(log_t_freq > f_bounds_log)[0].max()
    start_band = MOD_ASSIGN_MAT[t_centre_band, 0]
    return np.arange(start_band, start_band + 5, dtype=int)


def wind_ramp(srate: int, wdms: float, x: np.ndarray) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    npts = y.shape[-1]
    w, half = ramp_window(srate, wdms)
    y[..., :half] *= w[:half]
    y[..., npts - half :] *= w[half:]
    return y


if NUMBA_AVAILABLE:

    @njit(cache=True, fastmath=True, parallel=True)
    def synth_none_numba(fi, phase_vec, intens_l, intens_r, t):
        n_h = fi.shape[0]
        n_t = t.shape[0]
        out = np.empty((2, n_t), dtype=np.float64)
        twopi = 2.0 * math.pi
        for i in prange(n_t):
            ti = t[i]
            left = 0.0
            right = 0.0
            for h in range(n_h):
                val = math.sin(twopi * fi[h] * ti + phase_vec[h])
                left += intens_l[h] * val
                right += intens_r[h] * val
            out[0, i] = left
            out[1, i] = right
        return out

    @njit(cache=True, fastmath=True, parallel=True)
    def synth_amp_numba(fi, phase_vec, intens_l, intens_r, mod_mask, mod_f, t, phase_off, phase_s, tmr, smr_mean, smr_diff, scyc):
        n_h = fi.shape[0]
        n_t = t.shape[0]
        out = np.empty((2, n_t), dtype=np.float64)
        twopi = 2.0 * math.pi
        smrc = twopi * (1.0 / (scyc * 2.0))
        for i in prange(n_t):
            ti = t[i]
            s_i = smr_mean + smr_diff * math.sin(phase_s + smrc * ti)
            common = phase_off + tmr * ti
            left = 0.0
            right = 0.0
            for h in range(n_h):
                base = twopi * fi[h] * ti + phase_vec[h]
                val = math.sin(base)
                if mod_mask[h] != 0:
                    val *= 1.0 + math.sin(twopi * (common + mod_f[h] * s_i))
                left += intens_l[h] * val
                right += intens_r[h] * val
            out[0, i] = left
            out[1, i] = right
        return out

    @njit(cache=True, fastmath=True, parallel=True)
    def synth_phase_numba(fi, phase_vec, intens_l, intens_r, mod_mask, mod_f, t, phase_off, phase_s, tmr, smr_mean, smr_diff, scyc):
        n_h = fi.shape[0]
        n_t = t.shape[0]
        out = np.empty((2, n_t), dtype=np.float64)
        twopi = 2.0 * math.pi
        smrc = twopi * (1.0 / (scyc * 2.0))
        for i in prange(n_t):
            ti = t[i]
            s_i = smr_mean + smr_diff * math.sin(phase_s + smrc * ti)
            common = phase_off + tmr * ti
            left = 0.0
            right = 0.0
            for h in range(n_h):
                base = twopi * fi[h] * ti + phase_vec[h]
                if mod_mask[h] != 0:
                    phase_mod = math.pi * (1.0 + math.sin(twopi * (common + mod_f[h] * s_i))) / 2.0
                    val = math.sin(base + phase_mod)
                else:
                    val = math.sin(base)
                left += intens_l[h] * val
                right += intens_r[h] * val
            out[0, i] = left
            out[1, i] = right
        return out


def mod_ripple_stereo_numpy(
    rng: np.random.Generator,
    f0: float,
    dur: float,
    loud: float,
    ramp_seconds: float,
    tmr: float,
    smr: tuple[float, float],
    scyc: float,
    fbands: list[tuple[float, float]],
    fbanddb_l: np.ndarray,
    fbanddb_r: np.ndarray,
    fband_mod: np.ndarray,
    mod_type: str,
    rand_phase: bool = False,
    altharms: bool = False,
    altharm_depth: float = 0.0,
    normfreq: bool = False,
) -> np.ndarray:
    depth = 1.0
    phase_off = rng.random() * 2.0 * math.pi
    phase_s = rng.random() * 2.0 * math.pi
    t = time_vector(dur)
    fi, fintens_l, fintens_r, mod_mask, mod_f = stimulus_plan(
        float(f0),
        tuple((float(lo), float(hi)) for lo, hi in fbands),
        tuple(float(v) for v in fbanddb_l.tolist()),
        tuple(float(v) for v in fbanddb_r.tolist()),
        tuple(int(v) for v in fband_mod.tolist()),
        normfreq,
        altharms,
        float(altharm_depth),
    )
    fmodind = np.where(mod_mask != 0)[0]

    s = np.mean(smr) + abs(smr[1] - smr[0]) * np.sin(phase_s + 2.0 * math.pi * (1.0 / (scyc * 2.0)) * t)
    f = mod_f[fmodind]

    if rand_phase:
        phase_vec = rng.random(fi.size) * 2.0 * math.pi
    else:
        phase_vec = np.zeros(fi.size, dtype=np.float64)

    harmonic_phase = 2.0 * math.pi * fi[:, None] * t[None, :] + phase_vec[:, None]

    if mod_type == "phase":
        harmonic_mat = np.sin(harmonic_phase)
        phase_mod = math.pi * (
            1.0 + depth * np.sin(2.0 * math.pi * (phase_off + tmr * t[None, :] + f[:, None] * s[None, :]))
        ) / 2.0
        harmonic_mat[fmodind, :] = np.sin(harmonic_phase[fmodind, :] + phase_mod)
    elif mod_type == "amp":
        harmonic_mat = np.sin(harmonic_phase, out=harmonic_phase)
        amp_mod = 1.0 + depth * np.sin(2.0 * math.pi * (phase_off + tmr * t[None, :] + f[:, None] * s[None, :]))
        harmonic_mat[fmodind, :] *= amp_mod
    elif mod_type == "none":
        harmonic_mat = np.sin(harmonic_phase, out=harmonic_phase)
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")

    stim_pair = np.vstack((fintens_l, fintens_r)) @ harmonic_mat
    stim_scale = loud / (10.0 * np.mean([np.std(stim_pair[0]), np.std(stim_pair[1])]))
    ramp_mult = double_ramp_multiplier(SRATE, ramp_seconds * 1000.0, t.size)
    stim_pair *= stim_scale
    stim_pair *= ramp_mult[None, :]
    return stim_pair


def mod_ripple_stereo_numba(
    rng: np.random.Generator,
    f0: float,
    dur: float,
    loud: float,
    ramp_seconds: float,
    tmr: float,
    smr: tuple[float, float],
    scyc: float,
    fbands: list[tuple[float, float]],
    fbanddb_l: np.ndarray,
    fbanddb_r: np.ndarray,
    fband_mod: np.ndarray,
    mod_type: str,
    rand_phase: bool = False,
    altharms: bool = False,
    altharm_depth: float = 0.0,
    normfreq: bool = False,
) -> np.ndarray:
    if not NUMBA_AVAILABLE:
        raise RuntimeError("Numba backend requested but numba is not available.")

    phase_off = rng.random() * 2.0 * math.pi
    phase_s = rng.random() * 2.0 * math.pi
    t = time_vector(dur)
    fi, fintens_l, fintens_r, mod_mask, mod_f = stimulus_plan(
        float(f0),
        tuple((float(lo), float(hi)) for lo, hi in fbands),
        tuple(float(v) for v in fbanddb_l.tolist()),
        tuple(float(v) for v in fbanddb_r.tolist()),
        tuple(int(v) for v in fband_mod.tolist()),
        normfreq,
        altharms,
        float(altharm_depth),
    )
    if rand_phase:
        phase_vec = rng.random(fi.size) * 2.0 * math.pi
    else:
        phase_vec = np.zeros(fi.size, dtype=np.float64)

    smr_mean = float(np.mean(smr))
    smr_diff = float(abs(smr[1] - smr[0]))
    if mod_type == "phase":
        stim_pair = synth_phase_numba(fi, phase_vec, fintens_l, fintens_r, mod_mask, mod_f, t, phase_off, phase_s, tmr, smr_mean, smr_diff, scyc)
    elif mod_type == "amp":
        stim_pair = synth_amp_numba(fi, phase_vec, fintens_l, fintens_r, mod_mask, mod_f, t, phase_off, phase_s, tmr, smr_mean, smr_diff, scyc)
    elif mod_type == "none":
        stim_pair = synth_none_numba(fi, phase_vec, fintens_l, fintens_r, t)
    else:
        raise ValueError(f"Unsupported mod_type: {mod_type}")

    stim_scale = loud / (10.0 * np.mean([np.std(stim_pair[0]), np.std(stim_pair[1])]))
    ramp_mult = double_ramp_multiplier(SRATE, ramp_seconds * 1000.0, t.size)
    stim_pair *= stim_scale
    stim_pair *= ramp_mult[None, :]
    return stim_pair


def mod_ripple_stereo(
    rng: np.random.Generator,
    f0: float,
    dur: float,
    loud: float,
    ramp_seconds: float,
    tmr: float,
    smr: tuple[float, float],
    scyc: float,
    fbands: list[tuple[float, float]],
    fbanddb_l: np.ndarray,
    fbanddb_r: np.ndarray,
    fband_mod: np.ndarray,
    mod_type: str,
    rand_phase: bool = False,
    altharms: bool = False,
    altharm_depth: float = 0.0,
    normfreq: bool = False,
    engine: str = "auto",
) -> np.ndarray:
    backend = engine
    if backend == "auto":
        backend = "numba" if NUMBA_AVAILABLE else "numpy"
    if backend == "numba":
        return mod_ripple_stereo_numba(
            rng=rng,
            f0=f0,
            dur=dur,
            loud=loud,
            ramp_seconds=ramp_seconds,
            tmr=tmr,
            smr=smr,
            scyc=scyc,
            fbands=fbands,
            fbanddb_l=fbanddb_l,
            fbanddb_r=fbanddb_r,
            fband_mod=fband_mod,
            mod_type=mod_type,
            rand_phase=rand_phase,
            altharms=altharms,
            altharm_depth=altharm_depth,
            normfreq=normfreq,
        )
    return mod_ripple_stereo_numpy(
        rng=rng,
        f0=f0,
        dur=dur,
        loud=loud,
        ramp_seconds=ramp_seconds,
        tmr=tmr,
        smr=smr,
        scyc=scyc,
        fbands=fbands,
        fbanddb_l=fbanddb_l,
        fbanddb_r=fbanddb_r,
        fband_mod=fband_mod,
        mod_type=mod_type,
        rand_phase=rand_phase,
        altharms=altharms,
        altharm_depth=altharm_depth,
        normfreq=normfreq,
    )


def make_stimulus(
    rng: np.random.Generator,
    f0_range: tuple[int, int],
    dur_range: tuple[float, float],
    loud: float,
    ramp_prop: float,
    fbands: list[tuple[float, float]],
    db_corr_l: np.ndarray,
    db_corr_r: np.ndarray,
    mod_bands: np.ndarray,
    mod_type: str,
    engine: str,
) -> np.ndarray:
    dur_tmp = round(10.0 * (min(dur_range) + rng.random() * (max(dur_range) - min(dur_range)))) / 10.0
    f0_tmp = round(min(f0_range) + rng.random() * (max(f0_range) - min(f0_range)))
    return mod_ripple_stereo(
        rng=rng,
        f0=float(f0_tmp),
        dur=dur_tmp,
        loud=loud,
        ramp_seconds=dur_tmp * ramp_prop,
        tmr=1.0,
        smr=(3.0, 6.0),
        scyc=4.0,
        fbands=fbands,
        fbanddb_l=db_corr_l,
        fbanddb_r=db_corr_r,
        fband_mod=mod_bands,
        mod_type=mod_type,
        rand_phase=False,
        altharms=False,
        altharm_depth=0.0,
        normfreq=False,
        engine=engine,
    )


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    pta_freqs_khz, left_db_hl, right_db_hl = resolve_audiogram_inputs(args)

    if not (pta_freqs_khz.size == left_db_hl.size == right_db_hl.size):
        raise ValueError("Audiogram frequency, left threshold, and right threshold lists must be the same length.")

    left_db_hl, right_db_hl = apply_asymmetry_fraction(left_db_hl, right_db_hl, args.asymmetry_fraction)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    f_bounds_log = np.arange(0.0, 4.0 + 1e-9, 0.2)
    f_centres_log = f_bounds_log[:-1] + 0.1
    f_bounds = 1000.0 * (2.0 ** f_bounds_log)
    fbands = [(float(f_bounds[i]), float(f_bounds[i + 1])) for i in range(f_centres_log.size)]

    db_corr_l, db_corr_r = build_db_corrections(
        pta_freqs_khz=pta_freqs_khz,
        left_db_hl=left_db_hl,
        right_db_hl=right_db_hl,
        max_db_correction=args.max_db_correction,
        f_centres_log=f_centres_log,
    )
    mod_bands = pick_mod_bands(args.tinnitus_hz, f_bounds_log)

    print(f"Generating {args.sub_id} with tinnitus frequency {args.tinnitus_hz / 1000.0:.1f} kHz")
    print(f"Asymmetry fraction retained: {args.asymmetry_fraction:.2f}")
    print(f"Adjusted left thresholds (dB HL):  {left_db_hl.tolist()}")
    print(f"Adjusted right thresholds (dB HL): {right_db_hl.tolist()}")
    progress_every = max(1, int(args.progress_every))

    peak = 0.0
    total_frames = 0
    temp_handle = tempfile.NamedTemporaryFile(prefix=f"{args.sub_id}_", suffix=".f32", delete=False)
    temp_path = Path(temp_handle.name)
    rng = np.random.default_rng(np.random.MT19937(args.seed))
    try:
        for idx in range(args.n_per_file):
            stim_tmp = make_stimulus(
                rng=rng,
                f0_range=(args.f0_min_hz, args.f0_max_hz),
                dur_range=(args.stimulus_duration_s, args.stimulus_duration_s),
                loud=args.loud,
                ramp_prop=args.ramp_prop,
                fbands=fbands,
                db_corr_l=db_corr_l,
                db_corr_r=db_corr_r,
                mod_bands=mod_bands,
                mod_type=args.mod_type,
                engine=args.engine,
            )
            peak = max(peak, float(np.max(np.abs(stim_tmp))))
            frames = stim_tmp.shape[1]
            total_frames += frames
            temp_handle.write(np.asarray(stim_tmp.T, dtype=np.float32).tobytes())
            if (idx + 1) % progress_every == 0 or idx == 0 or (idx + 1) == args.n_per_file:
                print(f"Pass 1/2: stimulus {idx + 1} of {args.n_per_file}, peak {peak:.6f}")
        temp_handle.flush()
        temp_handle.close()

        scale = 1.0 if peak <= 1.0 else 1.0 / peak
        if scale < 1.0:
            print(f"Applying global downscale factor {scale:.6f} to prevent clipping.")
        else:
            print("No downscaling needed.")

        stereo_path = output_dir / f"{args.sub_id}_stereo.wav"
        mono_path = output_dir / f"{args.sub_id}_mono.wav"

        stereo_writer = WavWriter.open(stereo_path, SRATE, 2)
        mono_writer = WavWriter.open(mono_path, SRATE, 1)
        frames_per_chunk = max(1, int(args.stimulus_duration_s * SRATE * 4))
        with temp_path.open("rb") as reader:
            frames_written = 0
            chunk_idx = 0
            while frames_written < total_frames:
                frames_to_read = min(frames_per_chunk, total_frames - frames_written)
                raw = np.fromfile(reader, dtype=np.float32, count=frames_to_read * 2)
                if raw.size == 0:
                    break
                samples = raw.reshape(-1, 2).astype(np.float64, copy=False)
                samples *= scale
                stereo_writer.write(samples)
                mono_writer.write(np.mean(samples, axis=1, keepdims=True))
                frames_written += samples.shape[0]
                chunk_idx += 1
                if chunk_idx % progress_every == 0 or frames_written >= total_frames or chunk_idx == 1:
                    print(f"Pass 2/2: block {chunk_idx} written ({frames_written / SRATE:.1f}s total)")
        stereo_writer.close()
        mono_writer.close()

        print(f"Finished writing:\n  {stereo_path}\n  {mono_path}")
        return stereo_path, mono_path
    finally:
        try:
            temp_handle.close()
        except Exception:
            pass
        if temp_path.exists():
            os.remove(temp_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate tinnitus ART WAV files from audiogram and tinnitus frequency inputs.")
    parser.add_argument("--tinnitus-hz", type=float, required=True, help="Target tinnitus frequency in Hz, e.g. 3000 or 6000.")
    parser.add_argument("--audiogram-file", help="Path to a .json or .csv audiogram file.")
    parser.add_argument("--audiogram-freqs-khz", help="Comma-separated audiogram frequencies in kHz.")
    parser.add_argument("--left-db-hl", help="Comma-separated left-ear dB HL thresholds.")
    parser.add_argument("--right-db-hl", help="Comma-separated right-ear dB HL thresholds.")
    parser.add_argument("--sub-id", required=True, help="Output file prefix.")
    parser.add_argument("--output-dir", default=str(Path(__file__).resolve().parent / "Participant_Sound_Files"), help="Directory for generated WAV files.")
    parser.add_argument("--asymmetry-fraction", type=float, default=1.0, help="Fraction of original left/right asymmetry to retain. 1.0 keeps it all, 0.5 halves it, 0.0 averages both ears.")
    parser.add_argument("--mod-type", choices=["amp", "phase", "none"], default="amp", help="Modulation type.")
    parser.add_argument("--seed", type=int, default=6000, help="Random seed used for stimulus generation.")
    parser.add_argument("--n-per-file", type=int, default=900, help="Number of 4-second stimuli to concatenate.")
    parser.add_argument("--stimulus-duration-s", type=float, default=4.0, help="Duration of each stimulus chunk in seconds.")
    parser.add_argument("--loud", type=float, default=0.1, help="Base loudness scaling.")
    parser.add_argument("--ramp-prop", type=float, default=0.25, help="Onset/offset ramp as a proportion of stimulus duration.")
    parser.add_argument("--f0-min-hz", type=int, default=96, help="Minimum random fundamental frequency.")
    parser.add_argument("--f0-max-hz", type=int, default=256, help="Maximum random fundamental frequency.")
    parser.add_argument("--max-db-correction", type=float, default=50.0, help="Maximum per-band dB correction.")
    parser.add_argument("--progress-every", type=int, default=25, help="Print progress every N stimuli/blocks.")
    parser.add_argument("--engine", choices=["auto", "numpy", "numba"], default="auto", help="Synthesis engine. 'auto' prefers numba when available.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
