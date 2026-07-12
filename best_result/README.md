# Best recorded JAC-SCM result

Dataset: `dataset/20251203_535_产地_7.csv`

Configuration: 272 CARS features, split seed 42, tobacco-adapted JAC-SCM with SSC-JAM, LAWD-AdamW, BatchNorm, Dropout, label smoothing, and mild spectral augmentation.

Metrics on the fixed 214-sample test split:

- Accuracy: 0.7336448598
- Macro precision: 0.7138441133
- Macro recall: 0.6953786651
- Macro F1: 0.7031503986

The checkpoint is intentionally excluded from Git. Run `python train.py` to reproduce the training pipeline. Small-sample randomness means a single rerun may differ.
