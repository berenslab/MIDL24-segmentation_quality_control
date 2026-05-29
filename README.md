### Efficiently correcting patch-based segmentation errors to control image-level performance
<i>Experiments for MIDL 2024 paper.</i>

## Installation
<sup>Requires Python >= 3.9</sup>

### From PyPI (recommended for use as a dependency)
Install the released package:

```bash
pip install segmentation_quality_control
```

PyPI wheels include the Python code only. Ensemble checkpoints (~360 MB) are not bundled; see [Model weights](#model-weights) below.

### From GitHub (development version)
Install the latest development version without cloning:

```bash
pip install git+https://github.com/berenslab/MIDL24-segmentation_quality_control
```

### Development setup with `uv` (recommended)
Clone the repository:

```bash
git clone https://github.com/berenslab/MIDL24-segmentation_quality_control.git
cd MIDL24-segmentation_quality_control
uv sync --extra dev
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

### Development setup without `uv`
Alternatively, use Python's built-in venv:

```bash
git clone https://github.com/berenslab/MIDL24-segmentation_quality_control.git
cd MIDL24-segmentation_quality_control
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Model weights
The FR-UNet ensemble uses five checkpoints: `FRUNet_0.pth`, ..., `FRUNet_4.pth`.

### Where weights are resolved
When loading models, the package resolves the checkpoint directory in this order:

1. Custom path: pass `models_dir=...` to `ensure_models_dir()`.
2. Local development checkout: if all five files are present in the repository's `trained/` folder, that directory is used. This applies to editable installs (`pip install -e .`) and cloned repositories.
3. User cache (OS default for PyPI and other installs):

| OS | Default cache directory |
|---|---|
| Linux | `~/.cache/segmentation_quality_control/trained/` |
| macOS | `~/Library/Caches/segmentation_quality_control/trained/` |
| Windows | `%LOCALAPPDATA%\segmentation_quality_control\trained\` |

### Automatic download
If weights are missing from the resolved location, they are downloaded over HTTPS from GitHub (`trained/`) into the cache directory:

```python
from segmentation_quality_control.utils import ensure_models_dir

models_dir = ensure_models_dir()  # downloads on first call if needed
```

If Python's https-download mechanism fails, the package falls back to using `git` when it is available on `PATH`.

### Manual download
If the automatic download fails, fetch the files directly:

```bash
mkdir -p ~/.cache/segmentation_quality_control/trained
wget -O ~/.cache/segmentation_quality_control/trained/FRUNet_0.pth \
  https://raw.githubusercontent.com/berenslab/MIDL24-segmentation_quality_control/make-package-without-bunch/trained/FRUNet_0.pth
# repeat for FRUNet_1.pth … FRUNet_4.pth, or use curl -L -o ...
```

Alternatively, use a sparse git checkout:

```bash
git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/berenslab/MIDL24-segmentation_quality_control.git \
  ~/tmp/segmentation_weights
cd ~/tmp/segmentation_weights
git sparse-checkout set trained
git checkout make-package-without-bunch
# then copy trained/FRUNet_*.pth into your cache or repo trained/ directory
```

### Checkpoint format
- Weights are plain PyTorch checkpoints (not bundled in PyPI wheels).
- Keys: `arch`, `epoch`, `state_dict`, `optimizer`, `config`
- Loading no longer requires the legacy `bunch` package.
- To save new checkpoints after retraining, use `segmentation_quality_control.utils.checkpoint.save_checkpoint`, to store them in the `bunch`-less format.

## Reproduce the paper results
Instructions to reproduce the experiments (development setup):

0. Install the package as above (`uv sync --extra dev` or `pip install -e ".[dev]"`).
1. Download the FIVES dataset (Jin et al., 2022) and register its path in `paths.yaml`.
2. Execute `0_pass_forward.ipynb` to pass the data through the ensemble and log the outputs. In a cloned repository, weights are read from `trained/`.
3. Calibrate with temperature scaling on the validation set with `1_calibrate.ipynb`.
4. Compute $\widehat{DSC}$ and $\widehat{DSC}_{\text{corr}}$ in `2_estimate_DSC.ipynb`.
5. Reproduce figures with the respective scripts.

## References
Jin, Kai, et al. "Fives: A fundus image dataset for artificial Intelligence based vessel segmentation." Scientific Data 9.1 (2022): 475.
