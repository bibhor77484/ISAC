# 2-Tap Channel Analysis: SRS vs DMRS for Sensing

`2-tap_channel_analysis.ipynb` is a Monte Carlo simulation that compares 5G NR **SRS** (Sounding Reference Signal) and **DMRS** (Demodulation Reference Signal) as sensing pilots in an ISAC (Integrated Sensing and Communication) uplink. It does this on a controlled **2-tap channel**:

1. **LoS tap**: the direct UE → gNB path, taken from the ray-traced Outdoor dataset (strongest path of snapshot 0).
2. **Object tap**: a synthetic reflection whose AoA, excess delay (ToF) and Doppler are swept over a grid.

For every grid point the notebook builds the received OFDM signal on a 4-antenna array, estimates the channel from the pilots, and runs an iterative mDTrack-style estimator. It then reports the **RMSE of AoA, ToF and Doppler** and the **detection probability** of the object path.

The main questions it answers are:

- How much does the reduced DMRS bandwidth (12 or 25 PRBs out of 51) cost in AoA and ToF accuracy compared with full-band SRS?
- Can the denser DMRS time pattern help Doppler estimation, given that sparse SRS (20 ms period) aliases in Doppler?
- Does a **hybrid SRS + DMRS** symbol pattern get the best of both?

---

## Requirements

- Python 3 with `numpy`, `scipy` and `matplotlib` (developed with numpy 1.26, scipy 1.11, matplotlib 3.6)
- Local modules in [Functions/](Functions/):
  - `param_assoc.py` (`param_assoc`)
  - `kalman_filter.py` (`est_pos`, `kalman_filter_wo_doppler`, `kalman_filter_w_doppler`)

  These are imported but not used in this notebook's main analysis.
- Ray-tracing dataset in `Datasets/Outdoor_scenario/<Trajectory_N>/`:
  - `ISAC_ray_tracing_chan.npy`: per-snapshot path parameters `[AoA, Doppler, delay, pathgain]`
  - `ISAC_locations.npy`
  - `obj_params.npy`

> **Note:** The dataset paths in cell 4 are hard-coded absolute paths (`/home/bkumar2/...`). Change them if you run the notebook from another machine or location.

---

## System parameters (cell 4)

| Parameter | Value | Notes |
|---|---|---|
| Carrier `f_c` | 3.6192 GHz | |
| Bandwidth | 20 MHz (40 or 80 MHz also supported) | |
| Subcarrier spacing | 30 kHz | |
| FFT size / CP / `f_s` | 1024 / 72 / 30.72 MHz | for 20 MHz |
| SRS PRBs (`nPRB`) | 51 (full band) | |
| DMRS PRBs (`nPRB_dmrs`) | 25 (half) or 12 (quarter) | set per experiment |
| Comb (`K_TC`, `K_TC_dmrs`) | 2 | |
| Antennas | 4-element ULA, λ/2 spacing | |
| Oversampling `osf` | 8 | for delay resolution in the CAF |
| Tx power | 0.1 W | noise = kTB at 298 K |
| SRS period | 20 ms | |
| DMRS period | 2.5 ms | |
| Object path gain | 10000 (linear SNR) | fixed for all sweeps |

The ray-traced paths are sorted by path gain, AoAs are converted to the array's broadside convention, and the paths are truncated to the strongest `L = 25`. Only the strongest path is used as the LoS tap.

---

## Processing chain (common to all experiments)

1. **Pilot generation**: Zadoff–Chu sequences on a comb-2 grid over `12·nPRB` subcarriers (SRS) or `12·nPRB_dmrs` subcarriers (DMRS).
2. **Signal synthesis**: time-domain OFDM samples for the LoS and object taps, each with its own delay, phase, Doppler and array steering vector, plus unit-variance complex AWGN. For multi-symbol experiments, symbol *i* at time `sym_delays[i]` gets the object delay `τ − f_D·t_i / f_c`, so the Doppler shows up as a phase progression across symbols.
3. **Channel estimation**: FFT, then LS estimation on pilot subcarriers, then comb interpolation (`interpolate_channel_estimates`, a fixed-point FIR for K_TC = 2/4/8).
4. **Oversampled CIR**: each antenna's channel is multiplied by a full-band ZC sequence, zero-padded ×8 and IFFT'd.
5. **Iterative parameter estimation** (mDTrack-style, `resolve = 3` iterations, `L = 3` paths):
   - **AoA**: beam sweep over the spatial covariance (coarse 0.1 rad grid, then a ±7.5° fine search).
   - **ToF**: cross-ambiguity function (CAF) correlation against the reference pilot over delay bins.
   - **Doppler**: CAF over Doppler bins, using the inter-symbol phase `exp(-j2π f t_i)` across the stitched symbols.
   - **Amplitude** α is fitted. The reconstructed path is subtracted (successive cancellation), and in later iterations it is added back for refinement.
6. **Path association**:
   - `find_los` picks the earliest path within 7.5° of the UE AoA that has at least 50 % of the maximum amplitude.
   - The object path is the remaining path within 15° AoA and 50 ns ToF of the ground truth.
7. **Metrics**: RMSE over the detected trials, and detection rate in %.

---

## Experiments

Each sweep loops over `theta_space` (AoA, −75° to 75°), `tau_space` (object excess delay over LoS, in ns) and, where applicable, `Doppler_space`.

| Cell | Experiment | Symbols / timing | Estimation | Output file |
|---|---|---|---|---|
| 6 | AoA/ToF only, SRS (switch to DMRS by uncommenting the DMRS lines) | 1 symbol | single-shot | not saved; RMSE kept in memory (`rmse_tau`, `rmse_theta`) |
| 8 → 9 | SRS only, two-stage | 4 SRS @ [0, 20, 80, 200] ms | AoA/ToF from symbol 0, then Doppler from all 4 | `rmse_srs_51_2tapchan.npy` |
| 11 → 12 | SRS + DMRS, two-stage | 1 SRS for AoA/ToF; 4 DMRS (12 PRB) @ [0, 2.5, 10, 200] ms for Doppler | two-stage | `rmse_srs_51_dmrs_12_2tapchan.npy` |
| 15 → 16 | SRS only, joint | 4 SRS @ [0, 20, 80, 200] ms | joint AoA/ToF/Doppler on stitched symbols | `rmse_4Sym_srs_51_2tapchan_0Doppler.npy` (Doppler set to 0 here) |
| 18 → 19 | DMRS only, joint | 4 DMRS (12 PRB) @ [0, 2.5, 10, 200] ms | joint; coarse Doppler ±200 Hz, then ±20 Hz refine | `rmse_4Sym_dmrs_12_2tapchan.npy` |
| 23 → 24 | SRS only, 5 symbols, joint | 5 SRS @ [0, 20, 60, 160, 200] ms | joint | `rmse_5Sym_srs_51_2tapchan.npy` |
| 26 → 27 | SRS + DMRS hybrid, 5 symbols, joint | SRS @ 0, 80, 200 ms; DMRS @ 5, 20 ms | joint | `rmse_5Sym_srs_51_dmrs_12_2tapchan.npy` |

### Output format

Each saved `.npy` array has shape `[4, N_theta, N_tau, N_Doppler]`. Along the first axis:

| Index | Content |
|---|---|
| 0 | ToF RMSE (ns) |
| 1 | AoA RMSE (degrees) |
| 2 | Doppler RMSE (Hz) |
| 3 | Detection probability (%) |

Grid points with no detections give `NaN` (0/0). Cell 13 averages the Doppler RMSE with the NaNs zeroed.

Results files are written to the working directory. Earlier runs have been moved into [2-tap_channel_rmse/](2-tap_channel_rmse/) and [RMSE_data/](RMSE_data/). [2-tap_channel_rmse/readme.txt](2-tap_channel_rmse/readme.txt) lists the symbol timing and PRB setup of each archived file.

---

## Analysis and plotting cells

| Cell | Purpose |
|---|---|
| 20 | Doppler **array factor** `|Σ exp(-j2π f t_i)|` for a symbol-timing pattern, showing the alias "teeth" caused by sparse sampling. |
| 21 | `sidelobe_db()`: peak sidelobe level (dB) of the Doppler array factor for several candidate timing patterns. Use it to pick symbol timings. |
| 22 | 2-D delay–Doppler **CAF heatmap** with peak picking (`maximum_filter`) against the ground-truth taps. |
| 28 | Bar chart of Doppler RMSE: SRS (full BW) vs DMRS (half and quarter BW) at 15 and 30 Hz. |
| 29 | Line plots of ToF and AoA RMSE vs `tau_space` for the quarter, half and full bandwidth runs. |
| 30 | Three-panel bar chart (AoA RMSE, ToF RMSE, detection %) vs LoS–object ToF separation. Saves `results_bar_plot.png`. |
| 31 | Stem plot of an illustrative CIR (delay vs power per path). |
| 32 | Polar beam pattern of the 4-element ULA steered to 42°. |
| 33 | ToF-error CDFs: single symbol vs 4 symbols vs combined. |

The bar-chart values in cells 28 and 30 are typed in by hand from earlier runs. They are not loaded from the `.npy` files.

---

## How to run

1. Run cells 0 to 4 (imports, helper functions, system parameters, dataset load).
2. Run **one** experiment cell (6, 8, 11, 15, 18, 23 or 26), then the save cell right after it.
3. Run the plotting cells you need.

Runtime: every grid point synthesizes the signal with an explicit `N_ifft × N_ifft` sum and runs the CAF searches in Python loops. A full sweep can take a long time. Reduce `K`, `theta_space` or `tau_space` for quick checks.

Random seeds: `np.random.seed(100)` at import, and `np.random.seed(10)` at the start of each experiment.

---

## Known caveats

- **Shared variable names:** the experiment cells reuse `rmse_tau`, `rmse_theta`, `K`, `L`, `Xn`, `n` and others. Run them in order and save right after each one; running a different experiment overwrites the previous results.
- **`K` is overloaded:** cell 4 sets `K` first to Boltzmann's constant and then to 500 (number of snapshots). Each experiment then sets `K` to its Monte Carlo trial count (1 or 10). The noise variance is computed before the first overwrite, so it is correct.
- **Cells that won't run on their own:**
  - Cell 22 needs `time_gensym` to be 2-D (per-symbol), and it uses `f_idx`, `t_idx`, `f_peak` and `t_peak`, which are never defined.
  - Cell 29 needs `rmse_tau_12` and `rmse_tau_25` from earlier 1-D runs.
  - Cell 31 uses `colors[t]`, but its own `colors` definition is commented out.
  - Cell 33 needs `err_tau_1` and `err_tau_4`, which this notebook does not produce under those names.
- **Same-named files:** cells 9 and 16 both use the variable `rmse_srs_51`, and cells 24 and 27 do too. Their file names are different, though, so the saved results do not overwrite each other.
- **Fixed SNR and geometry:** the object SNR and the UE geometry (from snapshot 0 of `Trajectory_1`) are the same in every sweep. Only AoA, ToF and Doppler vary.
