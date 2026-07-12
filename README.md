# JAC-SCM full paper reproduction

PyTorch/scikit-learn reproduction of the complete model comparison in Guan et al., **A recognition method of soybean varieties based on near-infrared spectroscopy and improved deep learning model**, Food Chemistry 516 (2026), 149394.

## Reproduced models

Paper ablations:

- `lenet5`: three-layer spectral LeNet-5 baseline.
- `lenet5_ssc_jam`: LeNet-5 with SSC-JAM.
- `lenet5_lawd`: LeNet-5 trained with LAWD-AdamW.
- `jac_scm`: SSC-JAM plus LAWD-AdamW (the proposed model).

Table 3 baselines:

- Deep learning: `lstm`, `tcn`, `resnet`, `transformer`.
- Machine learning: `gbdt`, `bp`, `knn`, `nb`, `tree`, `svm`.

The pipeline implements MSC, CARS with Monte Carlo calibration/EDF/ARS/PLS-RMSECV, 114 selected variables, a stratified 3:2 calibration/prediction split, warmup plus cosine annealing warm restarts, early stopping, macro Precision/Recall/F1, accuracy, and confusion matrices.

## Data and commands

The CSV requires a header, one sample per row, spectral values in every column except the final class-label column. It accepts either the original 1845 variables or 114 already selected variables.

```powershell
# Proposed model
D:\Python\Anaconda3\python.exe train.py spectra.csv --model jac_scm

# All 14 experiment variants
D:\Python\Anaconda3\python.exe train.py spectra.csv --model all
```

For the included cigar-tobacco origin dataset, the loader automatically recognizes the first `origin` column as the label. Its 1555 spectral points are reduced to 272 by default (the lowest observed CARS-RMSECV region), and square-root inverse-frequency class weights compensate moderately for the imbalanced seven origins. The tobacco JAC-SCM retains the paper's convolution/SSC-JAM/LAWD core and adds BatchNorm, a compact Dropout classifier, label smoothing, and mild spectral perturbation to reduce small-sample overfitting.

```powershell
D:\Python\Anaconda3\python.exe train.py "dataset\20251203_535_产地_7.csv" --model jac_scm
```

Because this is the project's default dataset, the same training can be started directly with:

```powershell
python train.py
```

Use `--label-column`, `--features`, and `--cars-iterations` to override these adaptation settings.

For small-sample stability, keep the data split fixed while changing only the model seed, then ensemble probabilities:

```powershell
python train.py --seed 41 --split-seed 42 --output results_seed41
python train.py --seed 42 --split-seed 42 --output results_seed42
python train.py --seed 43 --split-seed 42 --output results_seed43
python ensemble.py results_seed41 results_seed42 results_seed43
```

Outputs are written to `results/`: `results_summary.csv`, confusion matrices, neural checkpoints, and training histories. When CARS is run, its indices and iteration/RMSECV trace are also saved.

## Paper mapping

The paper specifies JAC-SCM convolution channels `32/64/128`, kernels `7/4/4`, same padding, SSC-JAM dual pooling and two-layer spatial convolution, fusion weights `0.7/0.3`, LAWD learning-rate multipliers `1.2/1.0/0.9`, decay `1e-4/1e-4/5e-4`, 300 epochs, and early-stopping patience 20. These are implemented directly.

The prediction set is held out until final evaluation. Twenty percent of the paper's 300-sample calibration set is used for early stopping, avoiding selection on the prediction set.

## Reproduction limits

The publication does not provide its 500 raw spectra, exact 114 selected indices, source code, random seed, pooling dimensions, attention reduction ratio, batch size, base learning rate, warmup duration, restart period, or baseline-model hyperparameters. Defaults for these missing details are explicit in the code: max-pooling 2, reduction 8, batch size 32, LR `1e-3`, warmup 5, restart period 10, and seed 42.

Therefore this repository reproduces every published model and the documented experimental pipeline, but the reported `99.62%` cannot be independently regenerated without the authors' data and unpublished settings. The Table 3 baseline architectures are standard implementations because the paper only names them and gives no architecture definitions.
