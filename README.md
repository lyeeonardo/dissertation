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

Use separate Python environments for Work 2 and Work 3; their Python and package
versions differ. Follow each component's README in order: prepare the required
inputs, train the models, evaluate, and summarize the resulting runs. Store
generated inputs, checkpoints, and outputs in the documented runtime directories
or outside this checkout.

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
