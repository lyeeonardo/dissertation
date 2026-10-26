# Dissertation research code

Code for **Adaptive Control under Changing Hidden Dynamics through Generative
Active Inference**, by Yulin Li.

This repository contains the implementations used for the included studies,
along with code to collect training data, fit models, run evaluations, summarize
results, and generate numerical figures. Model weights, datasets, saved results,
figure assets, and the dissertation document are not distributed here.

| Component | Instructions |
| --- | --- |
| Work 2: locomotion | [Installation, data collection, training, adaptation, and evaluation](paper_2/README.md) |
| Work 3: wastewater treatment | [Installation, input preparation, training, and evaluation](paper_3/README.md) |
| Numerical figures | [Generate plots from your evaluation outputs](reproduction/figures/README.md) |

## Getting started

Use separate Python environments: **Python 3.12 for Work 2**, **Python 3.11 for
Work 3**, and **Python 3.12 or newer for plotting**. Dependency versions and
copyable commands are in the linked component READMEs. Store
generated inputs, checkpoints, and outputs in the documented runtime directories
or outside this checkout.

The reproduction sequence is:

1. **Create inputs.** Work 2 trains and qualifies SAC teachers, then collects the
   source, refreshed and target corpora. Work 3 downloads the official forcing
   records, simulates the behavior policies, and creates the training, validation
   and calibration splits.
2. **Train models.** Work 2 fits the shared, history and DADP controllers, performs
   the fixed replay stage, and trains the two adaptation arms. Work 3 fits the
   shared diffusion model and learned baselines, then calibrates the diagnostic.
3. **Evaluate.** Run the named studies in each README. Use their explicit
   regenerated-input options for newly trained checkpoints; strict reference
   mode checks the original input identities.
4. **Summarize and plot.** Run the reporting and figure-export commands on the
   generated traces, then use the plotting environment to render numerical
   figures. Saved publication result tables are not required.

Work 2 and Work 3 include all project source they import. Runtime inputs are
supplied explicitly or created by these commands; no code searches the author's
original work folders.

The Work 3 simulator requires the official BSM1 inflow records as external
benchmark inputs. Its collection code builds model-training datasets from those
records. Exact archived results require the original input and checkpoint
identities; newly trained models are independent repetitions and may produce
different results.

The original IWAI26 / Work 1 research implementation is outside this repository's
scope. The figure utilities include the retained Work 1 numerical plotting code
for separately supplied results. Static illustrations and manuscript layout
assets are not part of this code release.

## License

The original code is licensed under [Apache License 2.0](LICENSE).
Bundled third-party components retain their own licenses:

- Work 2 DADP/CleanDiffuser: [Apache-2.0 and attribution](paper_2/THIRD_PARTY_NOTICES.md).
- Work 3 BSM backend: [BSD-3-Clause](paper_3/BSM2_LICENSE).

The top-level license does not replace those notices or license external input
datasets and model weights supplied by a user.
