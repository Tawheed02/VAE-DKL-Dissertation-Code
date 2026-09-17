import pandas as pd
import matplotlib.pyplot as plt
import torch
import numpy as np
import ast
from standardise import standardise_bond_data
from bond_dkl_v5 import build_model, load_checkpoint, predict
from plotting import plot_predictions_vs_actual, plot_residuals

RESULTS_LOG = "v5_sweep_results.csv"
LOSS_LOG = "v5_sweep_losses.csv"

results_df = pd.read_csv(RESULTS_LOG)
losses_df = pd.read_csv(LOSS_LOG)

print(f"Loaded {len(results_df)} completed configs")


# Ranked final results table
ranked = results_df.sort_values('rmse')
print("\n--- Ranked by RMSE (lower = better) ---")
print(ranked[['run_id', 'hidden_dims', 'activation', 'latent_dim',
              'rmse', 'baseline_rmse', 'cumulative_train_seconds']].to_string(index=False))

best = ranked.iloc[0]
print(f"\nBest config: {best['run_id']} "
      f"(hidden_dims={best['hidden_dims']}, activation={best['activation']}, latent_dim={best['latent_dim']}) "
      f"— RMSE {best['rmse']:.4f}")

# Loss curves: every run, overlaid
plt.figure(figsize=(12, 7))
for run_id in losses_df['run_id'].unique():
    run_losses = losses_df[losses_df['run_id'] == run_id].sort_values('epoch')
    cfg = results_df[results_df['run_id'] == run_id].iloc[0] if run_id in results_df['run_id'].values else None
    label = run_id if cfg is None else f"{run_id} ({cfg['activation']},ld={cfg['latent_dim']})"
    plt.plot(run_losses['epoch'], run_losses['loss'], label=label, alpha=0.8)

plt.xlabel("Epoch")
plt.ylabel("Loss (-ELBO)")
plt.title("v5 Training Loss — All Configs")
plt.legend(fontsize=6, loc='upper right', ncol=2)
plt.tight_layout()
plt.show()


# Bar chart: final RMSE, ranked
plt.figure(figsize=(12, 6))
labels = [f"{row.hidden_dims}\n{row.activation}\nld={row.latent_dim}" for row in ranked.itertuples()]
plt.bar(range(len(ranked)), ranked['rmse'])
plt.axhline(ranked['baseline_rmse'].iloc[0], color='red', linestyle='--', label='Mean-only baseline')
plt.xticks(range(len(ranked)), labels, rotation=90, fontsize=6)
plt.ylabel("RMSE")
plt.title("v5 Final RMSE — All Configs, Ranked")
plt.legend()
plt.tight_layout()
plt.show()


# Time cost vs accuracy trade-off
plt.figure(figsize=(10, 6))
plt.scatter(results_df['cumulative_train_seconds'], results_df['rmse'], s=80)
for row in results_df.itertuples():
    plt.annotate(row.run_id, (row.cumulative_train_seconds, row.rmse), fontsize=6, xytext=(5, 5), textcoords='offset points')
plt.xlabel("Training Time (seconds)")
plt.ylabel("RMSE")
plt.title("v5 Accuracy vs. Training Cost")
plt.tight_layout()
plt.show()


# Full diagnostic plots for the WINNING config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

result = standardise_bond_data()
feature_cols = [c for c in (result['continuous_cols'] + result['binary_cols']) if c != 'ytm']

# Rebuilding for VRAM
import pandas as pd
needed_cols = feature_cols + ['credit_spread']
test_df = pd.read_parquet(result['test_path'], columns=needed_cols)
X_test_full = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)
y_test_full = torch.log1p(torch.tensor(test_df['credit_spread'].values, dtype=torch.float32, device=device))

def fast_in_memory_batches(X, y, batch_size=4096):
    n = X.size(0)
    n_usable = (n // batch_size) * batch_size
    return zip(torch.split(X[:n_usable], batch_size), torch.split(y[:n_usable], batch_size))

test_loader = fast_in_memory_batches(X_test_full, y_test_full)

best_hidden_dims = ast.literal_eval(best['hidden_dims'])  # CSV stores it as a string like "[128, 64]"

model, likelihood = build_model(
    input_dim=len(feature_cols), latent_dim=int(best['latent_dim']),
    num_inducing_points=100, hidden_dims=best_hidden_dims,
    kernel_type='matern', nu=0.5, activation=best['activation'],
    device=device,
)
model, likelihood = load_checkpoint(model, likelihood, f"v5_checkpoints/{best['run_id']}.pt", device=device)

preds, actuals = predict(model, likelihood, test_loader, device=device)
preds_real = np.expm1(preds)
actuals_real = np.expm1(actuals)

plot_predictions_vs_actual(preds_real, actuals_real, title=f"v5 Winner ({best['run_id']}): Predicted vs Actual")
plot_residuals(preds_real, actuals_real, title=f"v5 Winner ({best['run_id']}): Residuals")
