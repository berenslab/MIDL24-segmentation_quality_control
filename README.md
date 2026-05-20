### Efficiently correcting patch-based segmentation errors to control image-level performance 
<i> Experiments for MIDL 2024 submission. </i>


Instructions to Reproduce the Results:
  0. Install the package: `pip install -e .` (or `uv sync`). To also use the notebooks, install with the optional `dev` dependency: `pip install -e .[dev]` or `uv sync --extra dev`.
  1. Download FIVES dataset (Jin et al., 2022) and register its path in `paths.yaml`.
  2. Execute `0_pass_forward.ipynb` to pass the data through the ensemble and log the outputs. The models are already trained and their weights are stored in `trained/` as plain torch checkpoints.
  3. Calibrate with temperature scaling on the validation set with `1_calibrate.ipynb`
  4. Compute $\widehat{DSC}$ and $\widehat{DSC}_{\text{corr}}$ in `2_estimate_DSC.ipynb`
  5. Reproduce figures with respective scripts

Notes
- On loading weights
  - If you use this package as a dependency to another package, the weights will not be included. Download them manually from `trained/` to somewhere and point the scripts to that path.
- On changes of weight files
  - Weights are identical, but their format changed from `bunch` objects to standard dicts.
  - Keys: `arch`, `epoch`, `state_dict`, `optimizer`, `config`
  - Loading no longer requires the legacy `bunch` package. 
  - If you train your own checkpoints, save them without `bunch`, using `segmentation_quality_control.utils.checkpoint.save_checkpoint`.

References:
Jin, Kai, et al. "Fives: A fundus image dataset for artificial Intelligence based vessel segmentation." Scientific Data 9.1 (2022): 475.