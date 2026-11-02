# VGG6 on CIFAR-10: activations, optimizers and hyperparameters

CS6886W Assignment 1. Trains the VGG6 network from the provided template on CIFAR-10 under different configurations and compares them.

- **Final model:** GELU + SGD (momentum 0.9), lr 0.05 with cosine schedule, weight decay 5e-4, batch 128, 100 epochs, seed 42
  **Best validation accuracy 88.74% (epoch 97), test top-1 accuracy 87.53%**
- **W&B project:** https://wandb.ai/varull-indian-institute-of-technology-madras/vgg6-cifar10
- **W&B sweep:** https://wandb.ai/varull-indian-institute-of-technology-madras/vgg6-cifar10/sweeps/rfoediqk

---

## Repository structure

| Path | Purpose |
|---|---|
| `src/data.py` | CIFAR-10 loading, normalization, augmentation (crop, flip, Cutout), 45k/5k/10k split |
| `src/model.py` | VGG6 (template architecture) with selectable activation |
| `src/train.py` | Training loop, 6 optimizers, cosine schedule, best-checkpoint saving, CSV + optional W&B logging |
| `src/evaluate.py` | Evaluation function; tests a saved checkpoint on the test set |
| `src/autoaugment.py` | Template's CIFAR10Policy (optional `--autoaugment`, not used in reported runs) |
| `src/plot.py` | Local train/val curves from the CSV logs |
| `src/upload_runs.py` | Uploads finished runs (CSV logs) to W&B without retraining |
| `sweep.yaml` | W&B sweep configuration (Bayesian search, maximise `best_val_acc`) |
| `colab_runner.ipynb` | Colab launcher (setup + commands only, no model code) |
| `checkpoints/best_model.pth` | Trained final model |

Each run writes `runs/<run_name>/metrics.csv` (one row per epoch), `summary.json` and `best_model.pth`.

---

## Environment

Results were produced on Google Colab with an NVIDIA T4 GPU.

python 3.13.16 | torch 2.11.0+cu130 | torchvision 0.26.0+cu130
GPU: Tesla T4

Local CPU setup used for development: Python 3.12.2, torch 2.14.1+cpu, torchvision 0.29.1+cpu, numpy 2.5.2, pandas 3.1.0, matplotlib 3.11.2.

**Colab:**
```
pip install -r requirements.txt
```
**Local (Windows, CPU):**
```
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

All commands are run from the repository root as modules (`python -m src.<name>`).

---

## Data

CIFAR-10 is downloaded automatically to `--data_dir` (default `./data`) on first use. If you already have `cifar-10-batches-py/`, place it inside the data folder and nothing is downloaded.

- **Split:** the 50,000 training images are split 45,000 train / 5,000 validation with a fixed seed (42). The 10,000 test images are used once, at the end of each run.
- **Normalization:** mean (0.4914, 0.4822, 0.4465), std (0.2023, 0.1994, 0.2010).
- **Training augmentation:** RandomCrop(32, padding=4) → RandomHorizontalFlip → ToTensor → Normalize → Cutout(16×16).
- **Validation/test:** ToTensor → Normalize only.

---

## Reproducing the results

### Final model (Q4)
```
python -m src.train --activation gelu --optimizer sgd --lr 0.05 --batch_size 128 --epochs 100 --scheduler cosine --weight_decay 5e-4
```
Takes about 35 minutes on a T4.

### Evaluate the submitted checkpoint
```
python -m src.evaluate --checkpoint checkpoints/best_model.pth
```
Expected: test top-1 accuracy 87.53%. The model's settings are read from the checkpoint.

### Template baseline
Defaults match the template (ReLU, Adam lr 0.001, batch 64, no schedule, no weight decay):
```
python -m src.train --epochs 50
```

### Q2 experiments (20 epochs, one change from the template setup each)
```
# Activations (Q2a)
python -m src.train --activation {relu|sigmoid|tanh|silu|gelu} --epochs 20

# Optimizers (Q2b)
python -m src.train --optimizer sgd      --lr 0.01  --epochs 20
python -m src.train --optimizer nesterov --lr 0.01  --epochs 20
python -m src.train --optimizer adam     --lr 0.001 --epochs 20
python -m src.train --optimizer adagrad  --lr 0.01  --epochs 20
python -m src.train --optimizer rmsprop  --lr 0.001 --epochs 20
python -m src.train --optimizer nadam    --lr 0.001 --epochs 20

# Batch size, learning rate, epochs (Q2c)
python -m src.train --batch_size {32|64|128|256} --epochs 20
python -m src.train --lr {0.0001|0.001|0.003|0.01} --epochs 20
python -m src.train --epochs {10|20|40}
```
On Colab add `--data_dir /content/data --out_dir runs` (see `colab_runner.ipynb`).

### W&B logging and sweep (Q3)
```
wandb login
python -m src.train ... --wandb                 # log a single run
python -m src.upload_runs --runs_dir runs       # upload already-finished runs from their CSV logs
wandb sweep sweep.yaml                          # prints a sweep ID
wandb agent <entity>/vgg6-cifar10/<sweep-id> --count 10
```
`sweep.yaml` passes `--data_dir /content/data`; change it if running outside Colab.

### Local plots
```
python -m src.plot --runs gelu_sgd_lr0.05_bs128_ep100 --out outputs/final_curves.png
```

---

## Results

Reference (template setup, 50 epochs): best val 85.38%, **test 84.48%**.
All Q2 runs: 20 epochs, seed 42; one setting changed from the template setup. GPU run-to-run variation is about 1–1.5%.

| Activation | Best val (%) | Test (%) |
|---|---|---|
| GELU | 81.16 | 80.25 |
| SiLU | 80.38 | 80.43 |
| ReLU | 79.80 | 78.42 |
| Tanh | 73.00 | 72.04 |
| Sigmoid | 57.30 | 57.34 |

| Optimizer (lr) | Best val (%) | Test (%) |
|---|---|---|
| Adam (0.001) | 79.80 | 78.42 |
| Nadam (0.001) | 78.64 | 78.74 |
| SGD + momentum (0.01) | 77.60 | 77.44 |
| Nesterov (0.01) | 77.04 | 76.02 |
| RMSprop (0.001) | 76.92 | 76.27 |
| Adagrad (0.01) | 73.16 | 72.32 |

| Batch size | Best val (%) | Learning rate | Best val (%) | Epochs | Best val (%) |
|---|---|---|---|---|---|
| 32 | 81.26 | 0.0001 | 70.40 | 10 | 75.16 |
| 64 | 79.80 | 0.001 | 79.80 | 20 | 79.80 |
| 128 | 78.68 | 0.003 | 80.70 | 40 | 83.44 |
| 256 | 76.72 | 0.01 | 82.18 | 50 | 85.38 |

Best W&B sweep run (20 epochs): GELU + Nadam, lr 0.05, cosine → 86.58% val / 85.43% test.

---

## How performance varies

**Activations.** ReLU, SiLU and GELU do not saturate for positive inputs, so gradients flow through all layers; the smooth variants (SiLU, GELU) were about 1% better, within noise. Tanh saturates for large inputs and learns more slowly. Sigmoid is worst: its derivative is at most 0.25 (vanishing gradients) and its outputs are not zero-centred; its validation accuracy was also very unstable. All activations use the template's Kaiming initialization, which suits ReLU-like units.

**Optimizers.** Adaptive methods (Adam, Nadam, RMSprop) converge fastest early on; Nadam had the lowest training loss. SGD with momentum starts slower but has the smoothest validation curve, and with a higher lr (0.05) and a cosine schedule it produced the best final model. Adagrad's step size keeps shrinking because it sums all past squared gradients, so it learns slowest. L2 weight decay worked well with SGD but caused severe underfitting with Adam (71.98% val in a 100-epoch run), because Adam rescales the decay term per parameter; decoupled weight decay (AdamW) avoids this.

**Batch size, learning rate, epochs.** At a fixed epoch budget, smaller batches did better because they make more updates per epoch. A too-small learning rate (0.0001) converges slowly; accuracy rose up to lr 0.01, with more epoch-to-epoch fluctuation at higher learning rates under a constant schedule. More epochs keep helping with diminishing returns. A cosine learning-rate schedule removes the late-training fluctuations and gave the best results.

---

## Reproducibility

- **Seed:** `--seed 42` seeds Python, NumPy and PyTorch (CPU and CUDA), the train/val split and the shuffle order. Non-default seeds and weight decay are added to the run name, so runs never overwrite each other.
- **Variation:** GPU convolution kernels are not bit-exact; two runs of the final command reached 88.88% and 88.74% best validation accuracy.
- **Checkpoint contents:** `model_state`, the full run configuration, the epoch, and its validation accuracy.

---

## Changes from the template (`vgg6.py`)

| Kept from the template | Added / changed |
|---|---|
| VGG6 architecture, layer config, weight initialization | Activation selectable by name |
| Normalization values, RandomCrop, HorizontalFlip, Cutout | Seeded 45k/5k train/validation split |
| Core training loop, CrossEntropyLoss, Adam lr 0.001 default | Six optimizers; all hyperparameters as command-line flags; optional cosine schedule and weight decay |
| `eval()` accuracy logic | Also returns loss; best-validation checkpoint; single final test evaluation |
| CIFAR10Policy (commented out in the template) | Optional `--autoaugment`; updated for Pillow ≥ 10 |
| — | Fix: `model.train()` is called every epoch (the template stayed in eval mode after its first `eval()` call) |
| — | Train accuracy measured during the training pass instead of a second full pass |
| — | Seed, CSV/JSON logging, W&B logging, upload script, sweep, modular files |

---

## Acknowledgment

The implementation builds on the template code provided with the assignment (`vgg6.py`). An AI assistant was used for help with code structure, debugging and W&B setup. [Adjust to course policy.]