# Work 2: locomotion

Code for the Chapter 4 and Appendix B controllers, data collection, training,
evaluation and numerical analysis. Models, datasets, saved results and figures
are not included. All source imports are local to this package or declared
Python dependencies. The retained earlier evidence and information-gain studies
are separate from the final controller.

## Install

Use Python 3.12 in a dedicated environment, from this directory:

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install .
python -m work2.run list
```

Dependencies are pinned in `pyproject.toml`. The commands below use a POSIX
shell. Training and source collection require an NVIDIA GPU with CUDA;
evaluation supports `--device cpu` or `cuda`. Fresh training and
different hardware can produce different trajectories and numerical results.
Use new output directories for independent runs. Installation starts no runs.

## Create teachers and datasets

The original source teachers use eight conditions with 1×/1.5× gravity. Train
each condition using the retained SAC recipe; this performs substantial training:

```sh
for condition in 0 1 2 3 4 5 6 7; do
  python -m work2.training.original_experts --condition "$condition" --output inputs/original --execute
done
python - <<'PY'
import json
from pathlib import Path
Path('inputs/teachers.json').write_text(json.dumps({str(c): f'original/experts/HalfCheetah-v5/condition_{c}/selected.zip' for c in range(8)}, indent=2))
PY
```

Teacher mapping paths resolve beside the JSON. The following commands use
`--regenerated-teachers` to record newly trained policy identities. Omit that
flag only when supplying the exact original policies identified in
`work2/collection/recipe.json`. Training and qualification are separate; failed
qualification stops collection instead of replacing or silently retraining a
policy. Keep each regenerated `selected.zip` beside its trainer's `result.json`:
collection requires a completed full-budget qualification receipt whose selected
policy hash, size and condition match. `--execute` is required for collector execution.

```sh
python -m work2.collection original-trunks --teachers inputs/teachers.json --regenerated-teachers --output inputs/original-trunks --execute
python -m work2.collection transfer --teachers inputs/teachers.json --regenerated-teachers --output inputs/base --execute
python -m work2.collection base --teachers inputs/teachers.json --regenerated-teachers --original-trunks inputs/original-trunks --transfer-result inputs/base/expert_transfer/result.json --output inputs/base --execute
for cell in g3_c4 g3_c5 g3_c6 g3_c7 g5_c4; do
  python -m work2.collection specialist --teachers inputs/teachers.json --regenerated-teachers --cell "$cell" --output inputs/specialists --execute
done
python -m work2.collection refresh --specialists inputs/specialists --normalization inputs/base/normalization.json --output inputs/refreshed --execute
python -m work2.collection target --teachers inputs/teachers.json --regenerated-teachers --base-data inputs/base --refreshed-data inputs/refreshed/data/gravity_3 --specialists inputs/specialists --output inputs/target --execute
```

The base corpus has 800,000 transitions at 1×/3× gravity. Refreshed data have
368,000 transitions from qualified 3× teachers. Target pools contain 1,000 expert
TRAIN transitions and 3,000 expert/random/weak DEV transitions at 5× gravity.
The original parent splits, reset/action seeds, proposal masks, physical-step
accounting and qualification thresholds remain fixed. Normalization is fitted
on source expert TRAIN data and remains frozen for replay and adaptation.

The generated corpus root contains `normalization.json`, `splits/R_index.json`,
`splits/paired_anchors.json`, and `data/HalfCheetah-v5/cell_0.h5` through
`cell_7.h5`. Refreshed data place their cell files directly under
`inputs/refreshed/data/gravity_3`. Index rows identify path, episode, length,
condition, parent, split, behavior and proposal eligibility. HDF5 episodes
contain observations `(N+1,17)`, actions `(N,6)`, rewards and termination masks;
branch groups also retain simulator states and history. The collectors create
the completion receipts required by training. TRAIN/CAL/DEV/TEST parent groups
must remain disjoint.

## Train the controllers

```sh
python -m work2.training.train_diffusion --root inputs/base --method shared --execute
python -m work2.training.train_diffusion --root inputs/base --method history --execute
python -m work2.training.train_dadp --root inputs/base --stage representation --execute
python -m work2.training.train_dadp --root inputs/base --stage policy --execute
python -m work2.training.replay --base-checkpoint inputs/base/models/shared_R/best.pt --original-data inputs/base --refreshed-data inputs/refreshed/data/gravity_3 --output runs/final-replay --regenerated-checkpoints
```

Base fitting uses the native DEV stopping/selection rules. The final shared model
adds exactly 5,000 replay updates. `--regenerated-checkpoints` allows a newly
selected base fit and labels its actual update count; without it, replay requires
the original selected 62,000-update base. Replay keeps the fixed final update
choice, with DEV losses recorded diagnostically.

Create the evaluation asset map; its paths resolve relative to the JSON file:

```sh
python - <<'PY'
from work2.io import save
save('inputs/assets.json', {'checkpoints': {
    'shared.pt': '../runs/final-replay/checkpoints/step_0005000.pt',
    'history.pt': 'base/models/history_R/best.pt',
    'dadp.pt': 'base/models/dadp_R/policy/selected.pt',
}})
PY
```

Checkpoints carry the model configuration, frozen normalization and EMA weights.
DADP's policy checkpoint includes its representation and value bounds.
`work2/required_assets.json` records the original checkpoint hashes; it contains
no weights or private filesystem locations.

For adaptation, create a manifest binding each model to its current identity:

```sh
python - <<'PY'
from pathlib import Path
from work2.io import read, save, identity
assets = Path('inputs/assets.json').resolve()
models = {}
for method in ('shared', 'history', 'dadp'):
    checkpoint = (assets.parent / read(assets)['checkpoints'][method + '.pt']).resolve()
    models[method] = {'checkpoint': str(checkpoint), 'identity': identity(checkpoint)}
save('inputs/adaptation.json', {'models': models})
PY
for method in shared history dadp; do
  for arm in replay_only expert_1000; do
    python -m work2.training.adapt --method "$method" --arm "$arm" --inputs inputs/adaptation.json --pools inputs/target/pools.json --output runs/adaptation
  done
done
```

The trainer writes each selected checkpoint to
`runs/adaptation/fits/METHOD/ARM/best.pt`. Register those six selected fits:

```sh
python - <<'PY'
from work2.io import read, save
assets = read('inputs/assets.json')
for method in ('shared', 'history', 'dadp'):
    for arm in ('replay_only', 'expert_1000'):
        assets['checkpoints'][f'{method}_{arm}.pt'] = f'../runs/adaptation/fits/{method}/{arm}/best.pt'
save('inputs/assets.json', assets)
PY
```

Adaptation compares fixed DEV losses at 0, 100 and 250 updates. Pool manifests
bind file hashes, frozen normalization and source/target parent ancestry.
Relative `path` and `checkpoint` fields resolve beside their manifest.

## Evaluate and analyze

```sh
python -m work2.support.expert_reference --data-root inputs/base --refreshed-data inputs/refreshed/data/gravity_3 --output runs/expert_reference.csv
for study in benchmark preference_a preference_b all_conditions gravity_sweep; do
  python -m work2.run --study "$study" --assets inputs/assets.json --output "runs/$study" --regenerated-checkpoints
  python -m work2.report --input "runs/$study" --expert-reference runs/expert_reference.csv --output "runs/$study-report.json"
done
for arm in zero replay_only expert_1000; do
  python -m work2.run --study adaptation --arm "$arm" --assets inputs/assets.json --output "runs/adaptation-$arm" --regenerated-checkpoints
done
python -m work2.report --input runs/adaptation-zero --input runs/adaptation-replay_only --input runs/adaptation-expert_1000 --output runs/adaptation-report.json
```

`--regenerated-checkpoints` labels new fits. Without it, evaluation enforces the
exact original checkpoint hashes. `--list` prints a study without execution;
`--case ID` and `--method NAME` select explicit subsets.

| Study | Protocol |
|---|---|
| `benchmark` | Four controllers, 120 paired 1,000-step cases, five change families |
| `preference_a`, `preference_b` | Final preference controller versus reward-only, 40 cases each |
| `all_conditions` | Selected 3,200-step eight-condition illustration |
| `gravity_sweep` | Seven actual gravity levels, fixed 1×/3× model bank, 500 steps each |
| `adaptation` | Two target and two source-retention resets, four controllers |

The adaptation commands use the six registered adapted checkpoints plus the
original three for `zero`. Repeating `work2.report --input` combines the three
arms without treating them as independent training repetitions.

Reporting recomputes metrics from generated traces and labels missing cases or
short episodes. The expert-reference generator derives eight stationary TRAIN
means using base 1× parents and refreshed 3× parents; omitting
`--expert-reference` leaves expert-relative percentages null.
References are not applied to off-bank gravity or adaptation. Outcome-selected
illustrations retain that interpretation; predicted alternatives are not
executed counterfactuals. The privileged reference receives evaluator truth;
the other controllers do not.

Export the adaptation and gravity traces for the numerical figure builders:

```sh
python ../reproduction/figures/export_work2.py --run runs/adaptation-zero --run runs/adaptation-replay_only --run runs/adaptation-expert_1000 --run runs/gravity_sweep --output runs/figure-inputs
```

Then follow the [figure-generation instructions](../reproduction/figures/README.md)
in the separate plotting environment. The supplied `work2-adaptation` and
`work2-gravity` builders consume these generated inputs. Other study summaries
are the JSON reports written above; static dissertation artwork is excluded.

## Calibration and supporting studies

```sh
python -m work2.support.calibrate --checkpoint runs/final-replay/checkpoints/step_0005000.pt --data-root inputs/base --output runs/calibration --regenerated-checkpoints
```

This recomputes the CAL-only evidence temperature using the fixed 256 transition
identifiers and probe/noise schedule. Controller configuration retains the
published coefficients for the fixed protocol; a newly trained model's
calibration can differ.

The earlier evidence and information comparisons use their own 1×/1.5× corpus
and 100,000-update shared checkpoint. Generate that corpus from the original
teachers trained above:

```sh
for stage in trunks branches finalize; do
  python -m work2.support.original_data "$stage" --root inputs/original --execute
done
python -m work2.training.original_shared --data-root inputs/original --output runs/earlier-shared
python -m work2.support.evidence_pilot --checkpoint runs/earlier-shared/checkpoints/step_0100000.pt --data-root inputs/original --output runs/evidence-pilot --regenerated-checkpoints
python -m work2.support.friction --original-data inputs/original --refreshed-data inputs/base --output runs/friction.json
python -m work2.support.information_run --study earlier --checkpoint runs/earlier-shared/checkpoints/step_0100000.pt --output runs/information-earlier --regenerated-checkpoints
python -m work2.support.information_run --study revised --checkpoint runs/earlier-shared/checkpoints/step_0100000.pt --output runs/information-revised --regenerated-checkpoints
python -m work2.support.information --study earlier --input runs/information-earlier/case_metrics.json --output runs/information-earlier-analysis.json
python -m work2.support.information --study revised --input runs/information-revised/case_metrics.json --output runs/information-revised-analysis.json
```

The two information comparisons keep their fixed earlier recipes and paired
120-case/24-case designs. Their bootstrap describes case variation conditional
on a trained model, not variation across training seeds. No saved numerical
results are bundled.

Original code is licensed under Apache-2.0; bundled components retain the
licenses described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
