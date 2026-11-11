# Andrew: clone ISAC, fetch bistatic-mmwave-sensing, reproduce the scene-visibility CIRs

This guide reproduces
`bistatic-mmwave-sensing/results/202611_scenevisibility/04_results_15dbsup_5ns_1p47ang.md`.
That run uses 15 dB LOS cancellation, 5 ns / 1.47° separation, a 3 GHz carrier and 80 MHz bandwidth.

The ray-traced CIRs (`cir/`, about 5 GB for all 20 configs) are **git-ignored**. A fresh clone does not have them, so you trace them first (step 3) and then run the visibility analysis on top (step 4).

## 1. Clone ISAC with the submodule

```bash
git clone --recurse-submodules -b dev/andrew_reflector git@github.com:bibhor77484/ISAC.git
cd ISAC
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init bistatic-mmwave-sensing
```

The submodule is pinned to a commit on its `andrew_sub6_human` branch. To move to the latest commit on that branch, run the following and commit the new pin in ISAC:

```bash
git submodule update --remote bistatic-mmwave-sensing
```

Both repos use SSH. If the clone fails with `Permission denied (publickey)`, point git at the key that has access, for example:

```bash
GIT_SSH_COMMAND="ssh -i ~/.ssh/id_ed25519_github" git clone --recurse-submodules ...
```

## 2. Python environment

The scripts expect the venv at `bistatic-mmwave-sensing/.venv`. They call it as `../../.venv/bin/python` from the experiment folder.

```bash
cd bistatic-mmwave-sensing
python3 -m venv .venv
./.venv/bin/pip install -r requirements.txt     # sionna-rt>=2.2, matplotlib, numpy
cd results/202611_scenevisibility
```

Tested with Python 3.12 and Sionna RT 2.2.0 on an NVIDIA GB10. Without a CUDA GPU, Mitsuba falls back to its LLVM backend, which is slower.

All remaining commands run from `bistatic-mmwave-sensing/results/202611_scenevisibility/`.

## 3. Ray-trace the CIRs

Each config in `configs/` is one scene, `{1..5}tx1rx_{reflector,human}[_rot90].json`:
- 1 to 5 Tx on an 11 m circle around the Rx
- a 1×4 λ/2 ULA Rx
- a metal plate or a human proxy, at yaw 0° or 90°

`objectgridscenes_cir.py` traces the empty scene, which is the LOS reference. It then moves the object over a 0.5 m grid of about 1500 positions and traces each one.

One config, to try it:

```bash
../../.venv/bin/python objectgridscenes_cir.py configs/2tx1rx_human.json    # ~2 min on the GB10
```

All 20 configs, which is what `04_…` uses:

```bash
for n in 1 2 3 4 5; do for o in reflector human reflector_rot90 human_rot90; do
  ../../.venv/bin/python objectgridscenes_cir.py configs/${n}tx1rx_$o.json
done; done
```

Finished positions are skipped, so you can re-run the same command to resume an interrupted sweep. Pass `--force` to re-trace.

For one scene without a grid sweep, `scene_sionna_cir.py` traces the object at the positions in the config (`configs/X.json`), or the empty scene with `--no-object`:

```bash
../../.venv/bin/python scene_sionna_cir.py configs/2tx1rx_human.json               # -> cir/2tx1rx_human/single/
../../.venv/bin/python scene_sionna_cir.py configs/2tx1rx_human.json --no-object   # -> cir/2tx1rx_human/empty/
```

### Where the CIRs end up

```
cir/<config>/
  empty/                      empty scene (LOS only)
  grid/x10.00_y14.00/ ...     one folder per object position (x, y in m)
  grid_index.json             grid axes and the list of (x, y, folder)
```

Each scene folder (`empty/`, `grid/xXX.XX_yYY.YY/`) contains:

| File | Contents |
|---|---|
| `cir_all.npz` | All Tx and Rx antennas stacked, shape `[num_tx, num_rx_ant(4), 1, num_paths]`: complex gain `a`, delay `tau` [s], `valid`, arrival angles `theta_r` / `phi_r`, per-bounce `interaction` and `objects` (which mesh each bounce hit), sampled `taps` (80 MHz), element positions `pos_world` / `pos_local` |
| `cir/txNN/ant_MM.npz` | The same for one Tx and one Rx element, plus `tx_position` |
| `meta.json` | Carrier, bandwidth, tap rate, Tx/Rx positions, Rx orientation, element positions, object type and position |

Load one position:

```python
import numpy as np, json
d = np.load("cir/2tx1rx_human/grid/x10.00_y14.00/cir_all.npz")
a, tau, valid = d["a"], d["tau"], d["valid"]      # [tx, rx_ant, 1, path]
meta = json.load(open("cir/2tx1rx_human/grid/x10.00_y14.00/meta.json"))
```

## 4. Run the 04 analysis (15 dB, 5 ns, 1.47°)

These are the commands from `04_results_15dbsup_5ns_1p47ang.md`. The full sweep took about 65 min.

```bash
F="--los-cancellation-db 15 --resolve-delay-ns 5 --resolve-angle-deg 1.47 --results-dir results_15dbsup_5ns_1p47ang"
for n in 1 2 3 4 5; do for o in reflector human reflector_rot90 human_rot90; do
  ../../.venv/bin/python visibility_heatmap.py configs/${n}tx1rx_$o.json $F
done; done
../../.venv/bin/python coverage_vs_ntx.py --results-dir results_15dbsup_5ns_1p47ang
../../.venv/bin/python pipeline_diagram.py $F

# delay-angle maps behind single cells (the examples in the md)
../../.venv/bin/python visibility_heatmap.py configs/2tx1rx_reflector.json $F --example 10,14
../../.venv/bin/python visibility_heatmap.py configs/2tx1rx_human.json     $F --example 10,14
../../.venv/bin/python visibility_heatmap.py configs/1tx1rx_human.json     $F --example 11.5,13
```

Outputs go to `results_15dbsup_5ns_1p47ang/<config>/`:
- `visibility_heatmap.png`
- `visibility_metrics.csv`, with every metric for each (position, Tx)
- `visibility_summary.json`
- `delay_angle_x*_y*.png` for the `--example` cells

They also produce `coverage_vs_ntx.png` and `pipeline_flow.png` at the top of that folder. Your numbers should match the tables in `04_…md`. For example, with 5 Tx at yaw 0°, the human is detected 31.3% of the time and the reflector 31.7%.

Other threshold sets use the same CIRs. Only the flags change:

| Run | Flags |
|---|---|
| `02_…` (original) | none, writes to `results/` |
| `03_…` | `--los-cancellation-db 30 --resolve-delay-ns 5 --resolve-angle-deg 1.47 --results-dir results_30dbsup_5ns_1p47ang` |
| `04_…` | as above, with 15 dB |

`--snr-det-min-db` changes the 13 dB detection-SNR threshold.
