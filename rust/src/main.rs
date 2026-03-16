use anyhow::{Context, Result, bail};
use clap::{Parser, ValueEnum};
use hound::{SampleFormat, WavSpec, WavWriter};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde::Deserialize;
use std::collections::{BTreeSet, HashMap};
use std::f64::consts::{PI, TAU};
use std::fs;
use std::path::{Path, PathBuf};

const SRATE: u32 = 44_100;
const MOD_ASSIGN_MAT: [[usize; 2]; 20] = [
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
];

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum ModType {
    Amp,
    Phase,
    None,
}

#[derive(Parser, Debug)]
#[command(name = "tinnitus-art-rust")]
#[command(about = "Generate tinnitus ART WAV files from audiogram and tinnitus-frequency inputs")]
struct Args {
    #[arg(long)]
    tinnitus_hz: f64,

    #[arg(long)]
    audiogram_file: Option<PathBuf>,

    #[arg(long)]
    audiogram_freqs_khz: Option<String>,

    #[arg(long)]
    left_db_hl: Option<String>,

    #[arg(long)]
    right_db_hl: Option<String>,

    #[arg(long)]
    sub_id: String,

    #[arg(long, default_value = "rust/Participant_Sound_Files")]
    output_dir: PathBuf,

    #[arg(long, default_value_t = 1.0)]
    asymmetry_fraction: f64,

    #[arg(long, value_enum, default_value_t = ModType::Amp)]
    mod_type: ModType,

    #[arg(long, default_value_t = 3000)]
    seed: u64,

    #[arg(long, default_value_t = 900)]
    n_per_file: usize,

    #[arg(long, default_value_t = 4.0)]
    stimulus_duration_s: f64,

    #[arg(long, default_value_t = 0.1)]
    loud: f64,

    #[arg(long, default_value_t = 0.25)]
    ramp_prop: f64,

    #[arg(long, default_value_t = 96)]
    f0_min_hz: u16,

    #[arg(long, default_value_t = 256)]
    f0_max_hz: u16,

    #[arg(long, default_value_t = 50.0)]
    max_db_correction: f64,

    #[arg(long)]
    threads: Option<usize>,

    #[arg(long)]
    fixed_f0_hz: Option<u16>,

    #[arg(long)]
    fixed_phase_off: Option<f64>,

    #[arg(long)]
    fixed_phase_s: Option<f64>,
}

#[derive(Debug, Clone)]
struct Audiogram {
    freqs_khz: Vec<f64>,
    left_db_hl: Vec<f64>,
    right_db_hl: Vec<f64>,
}

#[derive(Deserialize)]
struct AudiogramJson {
    freqs_khz: Option<Vec<f64>>,
    audiogram_freqs_khz: Option<Vec<f64>>,
    left_db_hl: Vec<f64>,
    right_db_hl: Vec<f64>,
}

#[derive(Debug, Clone)]
struct ChunkParams {
    f0: u16,
    phase_off: f64,
    phase_s: f64,
}

#[derive(Debug, Clone)]
struct Plan {
    fi: Vec<f64>,
    intens_l: Vec<f64>,
    intens_r: Vec<f64>,
    mod_mask: Vec<bool>,
    mod_f: Vec<f64>,
}

#[derive(Debug)]
struct ChunkData {
    stereo: Vec<f32>,
    peak: f32,
}

#[derive(Debug)]
struct SharedConfig {
    t: Vec<f64>,
    ramp_multiplier: Vec<f64>,
    mod_type: ModType,
    loud: f64,
    tmr: f64,
    smr_mean: f64,
    smr_diff: f64,
    scyc: f64,
}

fn parse_float_csv(text: &str) -> Result<Vec<f64>> {
    text.split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| {
            part.parse::<f64>()
                .with_context(|| format!("Invalid float: {part}"))
        })
        .collect()
}

fn load_audiogram_file(path: &Path) -> Result<Audiogram> {
    let suffix = path
        .extension()
        .and_then(|ext| ext.to_str())
        .unwrap_or_default()
        .to_ascii_lowercase();

    match suffix.as_str() {
        "json" => {
            let data: AudiogramJson = serde_json::from_str(&fs::read_to_string(path)?)
                .with_context(|| {
                    format!("Failed to parse JSON audiogram file: {}", path.display())
                })?;
            let freqs_khz = data
                .freqs_khz
                .or(data.audiogram_freqs_khz)
                .context("JSON audiogram file must include freqs_khz or audiogram_freqs_khz")?;
            Ok(Audiogram {
                freqs_khz,
                left_db_hl: data.left_db_hl,
                right_db_hl: data.right_db_hl,
            })
        }
        "csv" => {
            let mut rdr = csv::Reader::from_path(path).with_context(|| {
                format!("Failed to open CSV audiogram file: {}", path.display())
            })?;
            let mut freqs_khz = Vec::new();
            let mut left_db_hl = Vec::new();
            let mut right_db_hl = Vec::new();

            for record in rdr.deserialize::<HashMap<String, String>>() {
                let row = record?;
                let freq_khz = if let Some(v) = row.get("freq_khz") {
                    v.parse::<f64>()?
                } else if let Some(v) = row.get("freq_hz") {
                    v.parse::<f64>()? / 1000.0
                } else {
                    bail!("CSV audiogram file must include freq_khz or freq_hz column")
                };
                freqs_khz.push(freq_khz);
                left_db_hl.push(
                    row.get("left_db_hl")
                        .context("CSV audiogram file must include left_db_hl column")?
                        .parse::<f64>()?,
                );
                right_db_hl.push(
                    row.get("right_db_hl")
                        .context("CSV audiogram file must include right_db_hl column")?
                        .parse::<f64>()?,
                );
            }

            if freqs_khz.is_empty() {
                bail!("CSV audiogram file contained no data rows")
            }

            Ok(Audiogram {
                freqs_khz,
                left_db_hl,
                right_db_hl,
            })
        }
        _ => bail!("Unsupported audiogram file type. Use .json or .csv"),
    }
}

fn resolve_audiogram_inputs(args: &Args) -> Result<Audiogram> {
    if let Some(path) = &args.audiogram_file {
        return load_audiogram_file(path);
    }

    let freqs_khz = parse_float_csv(
        args.audiogram_freqs_khz
            .as_deref()
            .context("Provide either --audiogram-file or all inline audiogram fields")?,
    )?;
    let left_db_hl = parse_float_csv(
        args.left_db_hl
            .as_deref()
            .context("Provide either --audiogram-file or all inline audiogram fields")?,
    )?;
    let right_db_hl = parse_float_csv(
        args.right_db_hl
            .as_deref()
            .context("Provide either --audiogram-file or all inline audiogram fields")?,
    )?;

    Ok(Audiogram {
        freqs_khz,
        left_db_hl,
        right_db_hl,
    })
}

fn apply_asymmetry_fraction(left: &[f64], right: &[f64], fraction: f64) -> (Vec<f64>, Vec<f64>) {
    let mut out_left = Vec::with_capacity(left.len());
    let mut out_right = Vec::with_capacity(right.len());
    for (&l, &r) in left.iter().zip(right.iter()) {
        let mean = (l + r) / 2.0;
        out_left.push(mean + fraction * (l - mean));
        out_right.push(mean + fraction * (r - mean));
    }
    (out_left, out_right)
}

fn lin_interp(xs: &[f64], ys: &[f64], xq: f64) -> f64 {
    if xq <= xs[0] {
        return ys[0];
    }
    if xq >= xs[xs.len() - 1] {
        return ys[ys.len() - 1];
    }
    let mut hi = 1;
    while xs[hi] < xq {
        hi += 1;
    }
    let lo = hi - 1;
    let t = (xq - xs[lo]) / (xs[hi] - xs[lo]);
    ys[lo] + t * (ys[hi] - ys[lo])
}

fn build_db_corrections(
    pta_freqs_khz: &[f64],
    left_db_hl: &[f64],
    right_db_hl: &[f64],
    max_db_correction: f64,
    f_centres_log: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let min_thresh = left_db_hl
        .iter()
        .chain(right_db_hl.iter())
        .copied()
        .fold(f64::INFINITY, f64::min);

    let mut left_mod: Vec<f64> = left_db_hl
        .iter()
        .map(|&v| (v + min_thresh).min(max_db_correction))
        .collect();
    let mut right_mod: Vec<f64> = right_db_hl
        .iter()
        .map(|&v| (v + min_thresh).min(max_db_correction))
        .collect();
    let mut log_pta_freqs: Vec<f64> = pta_freqs_khz.iter().map(|v| v.log2()).collect();

    if log_pta_freqs[0] > f_centres_log[0] {
        log_pta_freqs.insert(0, f_centres_log[0]);
        left_mod.insert(0, left_mod[0]);
        right_mod.insert(0, right_mod[0]);
    }

    if *log_pta_freqs.last().unwrap() < *f_centres_log.last().unwrap() {
        log_pta_freqs.push(*f_centres_log.last().unwrap());
        left_mod.push(*left_mod.last().unwrap());
        right_mod.push(*right_mod.last().unwrap());
    }

    let left_corr = f_centres_log
        .iter()
        .map(|&x| lin_interp(&log_pta_freqs, &left_mod, x))
        .collect();
    let right_corr = f_centres_log
        .iter()
        .map(|&x| lin_interp(&log_pta_freqs, &right_mod, x))
        .collect();
    (left_corr, right_corr)
}

fn pick_mod_bands(target_hz: f64, f_bounds_log: &[f64]) -> Result<Vec<usize>> {
    let log_t_freq = (target_hz / 1000.0).log2();
    if log_t_freq <= f_bounds_log[0] {
        bail!("Tinnitus frequency lower than bottom end of stimulus frequency range")
    }
    if log_t_freq >= *f_bounds_log.last().unwrap() {
        bail!("Tinnitus frequency higher than top end of stimulus frequency range")
    }
    let t_centre_band = f_bounds_log
        .iter()
        .enumerate()
        .filter(|(_, v)| log_t_freq > **v)
        .map(|(idx, _)| idx)
        .max()
        .unwrap();
    let start_band = MOD_ASSIGN_MAT[t_centre_band][0];
    Ok((start_band..start_band + 5).collect())
}

fn time_vector(duration_s: f64) -> Vec<f64> {
    let npts = (duration_s * SRATE as f64).round() as usize;
    (1..=npts).map(|i| i as f64 / SRATE as f64).collect()
}

fn double_ramp_multiplier(wdms: f64, npts: usize) -> Vec<f64> {
    let mut wds = (2.0 * wdms / 1000.0 * SRATE as f64).round() as usize;
    if wds % 2 != 0 {
        wds += 1;
    }
    let half = wds / 2;
    let mut w = Vec::with_capacity(wds);
    for i in 0..wds {
        let x = -(PI / 2.0) + (2.0 * PI * i as f64) / (wds.saturating_sub(1) as f64);
        w.push((x.sin() + 1.0) / 2.0);
    }
    let mut mult = vec![1.0f64; npts];
    for i in 0..half {
        mult[i] = w[i] * w[i];
    }
    for i in 0..half {
        mult[npts - half + i] = w[half + i] * w[half + i];
    }
    mult
}

fn build_plan(
    f0: u16,
    fbands: &[(f64, f64)],
    db_corr_l: &[f64],
    db_corr_r: &[f64],
    mod_bands: &[usize],
) -> Plan {
    let mod_set: BTreeSet<usize> = mod_bands.iter().copied().collect();
    let mut fi = Vec::new();
    let mut intens_l = Vec::new();
    let mut intens_r = Vec::new();
    let mut mod_mask = Vec::new();

    for (idx, &(lo, hi)) in fbands.iter().enumerate() {
        let band_num = idx + 1;
        let start_h = (lo / f0 as f64).ceil() as u16;
        let end_h = ((hi - 1.0) / f0 as f64).floor() as u16;
        let band_left = 10f64.powf(db_corr_l[idx] / 20.0);
        let band_right = 10f64.powf(db_corr_r[idx] / 20.0);
        for harmonic in start_h..=end_h {
            fi.push(harmonic as f64 * f0 as f64);
            intens_l.push(band_left);
            intens_r.push(band_right);
            mod_mask.push(mod_set.contains(&band_num));
        }
    }

    let mut mod_f = vec![0.0f64; fi.len()];
    let mod_indices: Vec<usize> = mod_mask
        .iter()
        .enumerate()
        .filter_map(|(idx, &flag)| flag.then_some(idx))
        .collect();
    if let Some(&first_idx) = mod_indices.first() {
        let min_fi = fi[first_idx];
        let mut vals: Vec<f64> = mod_indices
            .iter()
            .map(|&idx| (fi[idx] / min_fi).log2())
            .collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        for value in &mut vals {
            *value -= mean;
        }
        for (&idx, &value) in mod_indices.iter().zip(vals.iter()) {
            mod_f[idx] = value;
        }
    }

    Plan {
        fi,
        intens_l,
        intens_r,
        mod_mask,
        mod_f,
    }
}

fn stddev(values: &[f64]) -> f64 {
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let var = values
        .iter()
        .map(|v| {
            let d = *v - mean;
            d * d
        })
        .sum::<f64>()
        / values.len() as f64;
    var.sqrt()
}

fn synth_chunk(params: &ChunkParams, plan: &Plan, shared: &SharedConfig) -> ChunkData {
    let n_t = shared.t.len();
    let n_h = plan.fi.len();
    let smrc = TAU * (1.0 / (shared.scyc * 2.0));

    let samples: Vec<(f64, f64)> = (0..n_t)
        .map(|i| {
            let ti = shared.t[i];
            let s_i = shared.smr_mean + shared.smr_diff * (params.phase_s + smrc * ti).sin();
            let common = params.phase_off + shared.tmr * ti;
            let mut left = 0.0;
            let mut right = 0.0;
            for h in 0..n_h {
                let base = TAU * plan.fi[h] * ti;
                let mut val = match shared.mod_type {
                    ModType::Phase if plan.mod_mask[h] => {
                        let phase_mod =
                            PI * (1.0 + (TAU * (common + plan.mod_f[h] * s_i)).sin()) / 2.0;
                        (base + phase_mod).sin()
                    }
                    _ => base.sin(),
                };
                if matches!(shared.mod_type, ModType::Amp) && plan.mod_mask[h] {
                    val *= 1.0 + (TAU * (common + plan.mod_f[h] * s_i)).sin();
                }
                left += plan.intens_l[h] * val;
                right += plan.intens_r[h] * val;
            }
            (left, right)
        })
        .collect();

    let left_vec: Vec<f64> = samples.iter().map(|(l, _)| *l).collect();
    let right_vec: Vec<f64> = samples.iter().map(|(_, r)| *r).collect();
    let stim_scale = shared.loud / (10.0 * ((stddev(&left_vec) + stddev(&right_vec)) / 2.0));

    let mut peak = 0.0f32;
    let mut stereo = Vec::with_capacity(n_t * 2);
    for (i, (left, right)) in samples.into_iter().enumerate() {
        let l = (left * stim_scale * shared.ramp_multiplier[i]) as f32;
        let r = (right * stim_scale * shared.ramp_multiplier[i]) as f32;
        peak = peak.max(l.abs()).max(r.abs());
        stereo.push(l);
        stereo.push(r);
    }

    ChunkData { stereo, peak }
}

fn to_pcm16(sample: f32, global_scale: f32) -> i16 {
    let value = (sample * global_scale).clamp(-1.0, 1.0) * 32767.0;
    value as i16
}

fn write_outputs(
    stereo_path: &Path,
    mono_path: &Path,
    chunks: &[ChunkData],
    global_scale: f32,
) -> Result<()> {
    let spec_stereo = WavSpec {
        channels: 2,
        sample_rate: SRATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };
    let spec_mono = WavSpec {
        channels: 1,
        sample_rate: SRATE,
        bits_per_sample: 16,
        sample_format: SampleFormat::Int,
    };

    let mut stereo_writer = WavWriter::create(stereo_path, spec_stereo)
        .with_context(|| format!("Failed to create {}", stereo_path.display()))?;
    let mut mono_writer = WavWriter::create(mono_path, spec_mono)
        .with_context(|| format!("Failed to create {}", mono_path.display()))?;

    for chunk in chunks {
        for frame in chunk.stereo.chunks_exact(2) {
            let l = to_pcm16(frame[0], global_scale);
            let r = to_pcm16(frame[1], global_scale);
            stereo_writer.write_sample(l)?;
            stereo_writer.write_sample(r)?;
            let mono = (((frame[0] + frame[1]) * 0.5) * global_scale).clamp(-1.0, 1.0) * 32767.0;
            mono_writer.write_sample(mono as i16)?;
        }
    }

    stereo_writer.finalize()?;
    mono_writer.finalize()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(threads) = args.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global();
    }

    let audiogram = resolve_audiogram_inputs(&args)?;
    if !(audiogram.freqs_khz.len() == audiogram.left_db_hl.len()
        && audiogram.freqs_khz.len() == audiogram.right_db_hl.len())
    {
        bail!(
            "Audiogram frequency, left threshold, and right threshold lists must be the same length"
        )
    }

    let (left_db_hl, right_db_hl) = apply_asymmetry_fraction(
        &audiogram.left_db_hl,
        &audiogram.right_db_hl,
        args.asymmetry_fraction,
    );

    fs::create_dir_all(&args.output_dir)?;

    let f_bounds_log: Vec<f64> = (0..=20).map(|i| i as f64 * 0.2).collect();
    let f_centres_log: Vec<f64> = f_bounds_log[..f_bounds_log.len() - 1]
        .iter()
        .map(|v| v + 0.1)
        .collect();
    let f_bounds: Vec<f64> = f_bounds_log
        .iter()
        .map(|v| 1000.0 * 2f64.powf(*v))
        .collect();
    let fbands: Vec<(f64, f64)> = f_bounds.windows(2).map(|w| (w[0], w[1])).collect();
    let (db_corr_l, db_corr_r) = build_db_corrections(
        &audiogram.freqs_khz,
        &left_db_hl,
        &right_db_hl,
        args.max_db_correction,
        &f_centres_log,
    );
    let mod_bands = pick_mod_bands(args.tinnitus_hz, &f_bounds_log)?;

    println!(
        "Generating {} with tinnitus frequency {:.1} kHz",
        args.sub_id,
        args.tinnitus_hz / 1000.0
    );
    println!(
        "Asymmetry fraction retained: {:.2}",
        args.asymmetry_fraction
    );

    let mut rng = SmallRng::seed_from_u64(args.seed);
    let mut params = Vec::with_capacity(args.n_per_file);
    let duration_span = 0.0f64;
    let f0_span = f64::from(args.f0_max_hz.saturating_sub(args.f0_min_hz));
    let mut unique_f0 = BTreeSet::new();
    for _ in 0..args.n_per_file {
        let _dur = (10.0 * (args.stimulus_duration_s + rng.random::<f64>() * duration_span))
            .round()
            / 10.0;
        let f0 = args.fixed_f0_hz.unwrap_or_else(|| {
            (f64::from(args.f0_min_hz) + rng.random::<f64>() * f0_span).round() as u16
        });
        let phase_off = args
            .fixed_phase_off
            .unwrap_or_else(|| rng.random::<f64>() * TAU);
        let phase_s = args
            .fixed_phase_s
            .unwrap_or_else(|| rng.random::<f64>() * TAU);
        unique_f0.insert(f0);
        params.push(ChunkParams {
            f0,
            phase_off,
            phase_s,
        });
    }

    let mut plans = HashMap::new();
    for f0 in unique_f0 {
        plans.insert(
            f0,
            build_plan(f0, &fbands, &db_corr_l, &db_corr_r, &mod_bands),
        );
    }

    let shared = SharedConfig {
        t: time_vector(args.stimulus_duration_s),
        ramp_multiplier: double_ramp_multiplier(
            args.stimulus_duration_s * args.ramp_prop * 1000.0,
            (args.stimulus_duration_s * SRATE as f64).round() as usize,
        ),
        mod_type: args.mod_type,
        loud: args.loud,
        tmr: 1.0,
        smr_mean: 4.5,
        smr_diff: 3.0,
        scyc: 4.0,
    };

    println!(
        "Synthesizing {} chunks using {} Rayon threads",
        args.n_per_file,
        rayon::current_num_threads()
    );
    let shared_ref = &shared;
    let chunks: Vec<ChunkData> = params
        .par_iter()
        .map(|param| {
            let plan = plans.get(&param.f0).expect("missing plan for f0");
            synth_chunk(param, plan, shared_ref)
        })
        .collect();

    let peak = chunks.iter().map(|chunk| chunk.peak).fold(0.0f32, f32::max);
    let global_scale = if peak > 1.0 { 1.0 / peak } else { 1.0 };
    if global_scale < 1.0 {
        println!(
            "Applying global downscale factor {:.6} to prevent clipping",
            global_scale
        );
    } else {
        println!("No downscaling needed.");
    }

    let stereo_path = args.output_dir.join(format!("{}_stereo.wav", args.sub_id));
    let mono_path = args.output_dir.join(format!("{}_mono.wav", args.sub_id));
    write_outputs(&stereo_path, &mono_path, &chunks, global_scale)?;

    println!(
        "Finished writing:\n  {}\n  {}",
        stereo_path.display(),
        mono_path.display()
    );
    Ok(())
}
