# Numerical figure generation

This directory contains plotting code for selected dissertation result figures.
It ships no numerical results, datasets, checkpoints, fonts or artwork. Work 2
and Work 3 include the code to collect data, train models, run the retained
studies and export the numerical inputs used here. Additional numerical summaries are
produced by the reporting commands in the project READMEs.

Use Python 3.12 or newer in a separate plotting environment. From the repository
root:

```bash
python3 -m venv /path/to/figure-env
. /path/to/figure-env/bin/activate
python -m pip install -r reproduction/figures/requirements.txt
python reproduction/figures/build.py --help
```

Every builder takes `--input DIRECTORY` and `--output DIRECTORY`. Inputs and
outputs must be separate, non-nested locations. The builders read supplied
results; they do not train models or execute experiments. Output filenames are
fixed and are replaced when the same builder is run again in the same output
directory. Use a fresh output directory to preserve an earlier rendering.

## Work 2

Run the gravity sweep and all three adaptation arms using the commands in
[Work 2](../../paper_2/README.md). Export completed native traces:

```bash
python reproduction/figures/export_work2.py \
  --run /path/to/adaptation-zero \
  --run /path/to/adaptation-replay-only \
  --run /path/to/adaptation-target-plus-replay \
  --run /path/to/gravity-sweep \
  --output /path/to/work2-plot-inputs
python reproduction/figures/build.py work2-adaptation \
  --input /path/to/work2-plot-inputs --output /path/to/work2-figures
python reproduction/figures/build.py work2-gravity \
  --input /path/to/work2-plot-inputs --output /path/to/work2-figures
```

The exporter also accepts just the adaptation runs or just the gravity run. It
reads relative native trace locations under each supplied run directory and
recomputes returns from rewards. Stored absolute paths and stored metric values
are not used. The export directory must be fresh.

| Builder | Input files | Output |
| --- | --- | --- |
| `work2-adaptation` | `adaptation.csv`: method, arm, case_key, return_value, actual_gravity, bank_high, native_steps, assigned_steps, reset_seed, sampling_seed | `small_data_update.pdf` and `.png` |
| `work2-gravity` | `gravity_summary.csv`: actual_gravity, return_value. `gravity_series.csv`: actual_gravity, transition_index, action_time_s, outcome_time_s, reward, and posterior_000 through posterior_111 | `gravity_summary.pdf` and `.png` |

The adaptation plot uses both target resets, four controllers and three fitting
arms. The arms start from the same method-specific checkpoint; connecting their
means does not describe sequential training. The gravity plot shows all seven
fixed-bank returns and the 2x/5x posterior trajectories. Posterior sample `t` is
read from `beliefs[t + 1]`, after observing transition `t`; the initial prior is
not plotted as an outcome. Full 500-transition episodes are required. These
publication layouts retain their original axis limits; a changed outcome outside
those limits requires an intentional layout adjustment rather than silent
clipping.

## Work 3

Run the benchmark, forecast-selection comparison and development studies using
[Work 3](../../paper_3/README.md). Export from its native output tree, using the
Work 3 environment for this command:

```bash
python paper_3/export_figures.py all \
  --root /path/to/work3-outputs --output /path/to/work3-plot-inputs
```

The exporter accepts `benchmark`, `forecast` or `belief` in place of `all` when
only that study is available. Switch to the plotting environment to render:

```bash
python reproduction/figures/build.py work3-benchmark \
  --input /path/to/work3-plot-inputs --output /path/to/work3-figures
python reproduction/figures/build.py work3-forecast \
  --input /path/to/work3-plot-inputs/forecast --output /path/to/work3-figures
python reproduction/figures/build.py work3-belief \
  --input /path/to/work3-plot-inputs --output /path/to/work3-figures
```

| Builder | Input files | Output |
| --- | --- | --- |
| `work3-benchmark` | `benchmark.csv`: case_id, method, cost_per_day, eqi, oci, weighted_oci, integrated_cost, decision_mean_seconds | `benchmark.pdf` and `.png` |
| `work3-forecast` | `paired.csv`: case_id, weather, on_cost, off_cost, on_vs_off_percent. `cumulative_cost.csv`: case_id, day, on_integrated_cost, off_integrated_cost, on_vs_off_cumulative_percent | `forecast_control` and `forecast_control_traces`, each as PDF, PNG and SVG |
| `work3-belief` | `belief_startup.json`: particles `[9,5,8,2]`, weights `[9,5,8]`, truth `[9,2]` | `belief_evolution_zoom` as PDF, PNG and SVG |

The benchmark requires all seven controllers on the same six 14-day cases.
Mean cost changes are equal-case means of paired percentages relative to native
PI. The forecast comparison requires both executed variants on all six cases,
with 1,344 observed 15-minute boundaries per case; no invented zero-time ratio
is inserted. The belief illustration uses the first nine boundaries of the
storm development case, with coordinates reactor-5 dissolved oxygen and
reactor-2 nitrate. Its five stages are incoming particles, corrected particles,
selected forecasts, propagated particles, and public-channel-clamped particles.
The last two stages agree in these hidden coordinates. New evaluations save the
initial particles needed for the first stage; older logs without them cannot
reconstruct that boundary and are rejected by the exporter.

## Work 1 summary plots

The original IWAI26 experiment code remains outside this repository. The
`work1` builder accepts summaries produced by that project:

```bash
python reproduction/figures/build.py work1 \
  --input /path/to/work1-summaries --output /path/to/work1-figures
```

- `control.csv`: model, group, success_percent, ci95_low_percent,
  ci95_high_percent. Models are `diffusion`, `autoregressive`, `cvae`,
  `flow_matching`; groups are `zero_lateral_correct`, `all_correct`, `all_wrong`.
- `prediction.csv`: model, prediction_step, mean_euclidean_position_error_mm,
  ci95_low_mm, ci95_high_mm.
- `curves.csv`: panel (`A` or `B`), series, x, success_percent. Supported series
  are `CEM`, `Proposal model`, `Gated proposal model`, `Large-K CEM`, `CVAE`.

Outputs are `figure_2` and `figure_a1_ab`, each as PDF and PNG. These plot supplied
means and intervals; they do not recompute bootstrap intervals or run models.

Static environment illustrations, pose artwork, framework slides, the published
recovery figure and historical PDF crop/relabel scripts are not included.
Consequently this code-only package does not reconstruct all dissertation
artwork. It generates the numerical figures listed above from supplied results.
