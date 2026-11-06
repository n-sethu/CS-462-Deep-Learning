# MNIST generation comparison

Classifier test accuracy: 98.79%. 3000 updates per model, seed 2026, 256 generated samples.

Lower MNIST classifier feature Frechet distance is better. This is not Inception FID; small single-seed runs cannot establish a universal winner. Raw losses across model families are not comparable.

| Model | Feature distance | Class entropy | NFE | Train seconds | Sample seconds |
|---|---:|---:|---:|---:|---:|
| rf_improved | 20.690 | 2.261 | 100 | 242.5 | 4.7 |
| rf_baseline | 23.945 | 2.267 | 100 | 247.6 | 4.8 |
| ddpm_improved | 24.890 | 2.286 | 1000 | 242.7 | 47.3 |
| ddpm_baseline | 28.554 | 2.269 | 1000 | 245.7 | 48.1 |

Comparison image: each consecutive block of 100 images follows execution order: ddpm_baseline, ddpm_improved, rf_baseline, rf_improved.

Training uses a fixed 55,000/5,000 split, no augmentation, identical architecture, initialization seed and batch indices. Classifier selection uses validation accuracy; the 10,000 official test images supply reference features. DDPM uses the full 1,000-step posterior chain; RF Euler and Heun use equal NFE. EMA decay ramps to 0.999. The improved variants bundle two changes, so these runs cannot attribute gains to either component alone.
