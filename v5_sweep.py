import itertools
import csv
import os
import time
import torch
import numpy as np
import pandas as pd
from standardise import standardise_bond_data
from bond_dkl_v5 import (
    build_model, build_optimizer, build_mll, get_train_y_mean,
    train_one_epoch, evaluate, save_checkpoint, reinitialize_inducing_points_kmeans
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

result = standardise_bond_data()
feature_cols = [c for c in (result['continuous_cols'] + result['binary_cols']) if c != 'ytm']
train_y_mean = get_train_y_mean(result['train_path'])

# --- Preload into VRAM once (Gemini's proven-faster approach) ---
needed_cols = feature_cols + ['credit_spread']
train_df = pd.read_parquet(result['train_path'], columns=needed_cols)
test_df = pd.read_parquet(result['test_path'], columns=needed_cols)

X_train_full = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device).contiguous()
y_train_full = torch.log1p(torch.tensor(train_df['credit_spread'].values, dtype=torch.float32, device=device)).contiguous()
X_test_full = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device).contiguous()
y_test_full = torch.log1p(torch.tensor(test_df['credit_spread'].values, dtype=torch.float32, device=device)).contiguous()
del train_df, test_df

print(f"X_train_full: {X_train_full.shape}")
if device.type == 'cuda':
    print(f"GPU memory used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def fast_in_memory_batches(X, y, batch_size=4096, shuffle=True):
    n = X.size(0)
    if shuffle:
        perm = torch.randperm(n, device=X.device)
        X_curr, y_curr = X[perm], y[perm]
    else:
        X_curr, y_curr = X, y
    n_usable = (n // batch_size) * batch_size
    return zip(torch.split(X_curr[:n_usable], batch_size), torch.split(y_curr[:n_usable], batch_size))


# v5 ENCODER OVERHAUL SWEEP
HIDDEN_DIMS_OPTIONS = [
    [128],
    [128, 64],
    [128, 64, 32],
]
ACTIVATIONS = ['silu', 'relu', 'gelu', 'tanh']
LATENT_DIMS = [8, 16]

KERNEL_TYPE = 'matern'
NU = 0.5
NUM_INDUCING = 100  # prototyping run — v4's actual winner used 500;
                     # rerun the winning architecture at 500 to confirm

NN_LR = 0.001
GP_LR = 0.01
BATCH_SIZE = 4096
N_EPOCHS = 10
KMEANS_REINIT_AT_EPOCH = 2
USE_AMP = True

configs = list(itertools.product(HIDDEN_DIMS_OPTIONS, ACTIVATIONS, LATENT_DIMS))
print(f"Total configurations: {len(configs)}")

RESULTS_LOG = "v5_sweep_results.csv"
LOSS_LOG = "v5_sweep_losses.csv"
CHECKPOINT_DIR = "v5_checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

if not os.path.exists(RESULTS_LOG):
    with open(RESULTS_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(['run_id', 'hidden_dims', 'activation', 'latent_dim',
                                 'rmse', 'baseline_rmse', 'beats_baseline_meaningfully',
                                 'cumulative_train_seconds'])
if not os.path.exists(LOSS_LOG):
    with open(LOSS_LOG, 'w', newline='') as f:
        csv.writer(f).writerow(['run_id', 'epoch', 'loss', 'cumulative_seconds'])

#%%
for i, (hidden_dims, activation, latent_dim) in enumerate(configs):
    run_id = f"v5_{i:02d}"

    already_done = False
    if os.path.exists(RESULTS_LOG):
        with open(RESULTS_LOG) as f:
            reader = csv.reader(f)
            next(reader, None)
            already_done = any(row and row[0] == run_id for row in reader)
    if already_done:
        print(f"{run_id}: already done, skipping.")
        continue

    print(f"\n=== {run_id}: hidden={hidden_dims}, activation={activation}, latent={latent_dim} ===")

    model, likelihood = build_model(
        input_dim=len(feature_cols), latent_dim=latent_dim,
        num_inducing_points=NUM_INDUCING, hidden_dims=hidden_dims,
        kernel_type=KERNEL_TYPE, nu=NU, activation=activation, device=device,
    )
    optimizer = build_optimizer(model, likelihood, nn_lr=NN_LR, gp_lr=GP_LR)
    mll, _ = build_mll(model, likelihood, result['train_path'])

    cumulative_seconds = 0.0
    for epoch in range(1, N_EPOCHS + 1):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.time()

        train_loader = fast_in_memory_batches(X_train_full, y_train_full, batch_size=BATCH_SIZE, shuffle=True)
        loss = train_one_epoch(model, likelihood, optimizer, mll, train_loader, device=device, use_amp=USE_AMP)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        cumulative_seconds += time.time() - start
        print(f"  epoch {epoch}: loss={loss:.4f} ({cumulative_seconds:.0f}s so far)")

        with open(LOSS_LOG, 'a', newline='') as f:
            csv.writer(f).writerow([run_id, epoch, loss, cumulative_seconds])

        if epoch == KMEANS_REINIT_AT_EPOCH:
            kmeans_loader = fast_in_memory_batches(X_train_full, y_train_full, batch_size=BATCH_SIZE, shuffle=False)
            reinitialize_inducing_points_kmeans(model, kmeans_loader, latent_dim, NUM_INDUCING, device=device)

    test_loader = fast_in_memory_batches(X_test_full, y_test_full, batch_size=BATCH_SIZE, shuffle=False)
    results = evaluate(model, likelihood, test_loader, train_y_mean, device=device, target_inverse_transform=np.expm1)
    print(f"  FINAL: RMSE={results['rmse']:.4f} vs baseline {results['baseline_rmse']:.4f}")

    save_checkpoint(model, likelihood, os.path.join(CHECKPOINT_DIR, f"{run_id}.pt"))

    with open(RESULTS_LOG, 'a', newline='') as f:
        csv.writer(f).writerow([run_id, str(hidden_dims), activation, latent_dim,
                                 results['rmse'], results['baseline_rmse'],
                                 results['beats_baseline_meaningfully'], cumulative_seconds])

print("\nv5 sweep complete. Results in", RESULTS_LOG)
