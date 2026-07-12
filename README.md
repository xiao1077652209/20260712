# JAC-SCM paper reproduction

PyTorch implementation of **A recognition method of soybean varieties based on near-infrared spectroscopy and improved deep learning model** (Food Chemistry 516, 2026, 149394).

## Implemented pipeline

1. MSC baseline/scatter correction (paper Eq. 3).
2. CARS-style Monte Carlo/EDF/PLS feature selection to 114 variables.
3. JAC-SCM: same-padded 1-D convolutions with 32/64/128 filters and kernel sizes 7/4/4.
4. SSC-JAM channel and spatial branches, fused with paper weights 0.7/0.3.
5. LAWD-AdamW groups: attention `1.2*lr`, convolution `lr`, FC `0.9*lr`; weight decay `1e-4`, `1e-4`, `5e-4`.
6. Cosine annealing warm restarts, 300 epochs, early stopping patience 20.

## Run

The CSV must have one sample per row, spectral values in all columns except the last, and the class label in the last column. A header is required.

```powershell
D:\Python\Anaconda3\python.exe train.py spectra.csv
```

The paper uses 500 samples (10 classes, 50 each), stratified 3:2 into 300 training and 200 prediction samples. The script reproduces this split. It accepts either 1845 raw variables (then fits CARS on the training set) or 114 preselected variables.

## Reproduction boundary

The article does not provide the raw spectra, the exact 114 selected wavenumber indices, source code, random seed, pooling dimensions, attention reduction ratio, batch size, base learning rate, warm-restart period, or an independent validation split. Consequently, the reported 99.62% cannot be independently reproduced from the PDF alone. This implementation uses conventional, explicit defaults for those missing details: max-pooling by 2 after each convolution, reduction ratio 8, batch size 32, base LR `1e-3`, `T_0=10`, and seed 42. These are isolated in constructors/CLI arguments for adjustment.

The paper evaluates its prediction set during training while describing early stopping; `train.py` follows that reported workflow for comparability, although a separate validation set is recommended for unbiased research.
