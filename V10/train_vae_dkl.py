import numpy as np
import torch
import gpytorch
import matplotlib.pyplot as plt
from vae_dkl_model import VAE_DKL, kl_divergence


def build_vae_dkl(input_dim, latent_dim=6, num_inducing_points=100,
                   kernel_type='matern', nu=2.5, hidden_dims=None, device='cpu'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = VAE_DKL(
        input_dim=input_dim,
        latent_dim=latent_dim,
        num_inducing_points=num_inducing_points,
        kernel_type=kernel_type,
        nu=nu,
        hidden_dims=hidden_dims,
    ).to(device)
    return model, likelihood


def build_vae_dkl_optimizer(model, likelihood, nn_lr=0.001, gp_lr=0.01):
    """ as bond_dkl_v5.build_optimizer: the encoder
    gets the smaller, careful learning rate; GP-side parameters get the faster one."""
    return torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': nn_lr},
        {'params': model.covar_module.parameters(), 'lr': gp_lr},
        {'params': model.mean_module.parameters(), 'lr': gp_lr},
        {'params': model.variational_strategy.parameters(), 'lr': gp_lr},
        {'params': likelihood.parameters(), 'lr': gp_lr},
    ])


def train_vae_dkl_epoch(model, likelihood, optimizer, mll, train_loader,
                         beta=0.1, lambda_ls=0.0, penalty_mode='mean',
                         device='cpu', print_every=20):
    """
    One epoch. Loss = -GP marginal log-likelihood + beta * KL(q(z|x) || N(0,I))
    + lambda_ls * lengthscale_penalty.

    penalty_mode controls how the lengthscale penalty is computed:
      'mean' - mean(lengthscales). Only the average is penalized.
      'max'  - max(lengthscales). Targets whichever single dimension the kernel is currently
                # being laziest about.
      'quad' - mean(lengthscales ** 2). Punishes any very large
                lengthscale disproportionately harder than a large one.
    """
    model.train()
    likelihood.train()

    epoch_gp_loss = 0.0
    epoch_kl_loss = 0.0
    epoch_ls_mean = 0.0
    n_batches = 0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        output = model(x_batch)
        if hasattr(likelihood, 'set_z'):
            likelihood.set_z(model.last_z)
        gp_loss = -mll(output, y_batch)
        kl_loss = kl_divergence(model.last_mu, model.last_logvar)

        total_loss = gp_loss + beta * kl_loss

        ls_mean_val = 0.0
        if lambda_ls > 0:
            ls = model.covar_module.base_kernel.lengthscale
            if penalty_mode == 'mean':
                ls_penalty = ls.mean()
            elif penalty_mode == 'max':
                ls_penalty = ls.max()
            elif penalty_mode == 'quad':
                ls_penalty = ls.pow(2).mean()
            else:
                raise ValueError(f"Unknown penalty_mode: {penalty_mode}")
            total_loss = total_loss + lambda_ls * ls_penalty
            ls_mean_val = ls_penalty.item()

        total_loss.backward()
        optimizer.step()

        epoch_gp_loss += gp_loss.item()
        epoch_kl_loss += kl_loss.item()
        epoch_ls_mean += ls_mean_val
        n_batches += 1

        if n_batches % print_every == 0:
            print(f"    batch {n_batches}: gp_loss={gp_loss.item():.4f}  kl_loss={kl_loss.item():.4f}")

    return epoch_gp_loss / n_batches, epoch_kl_loss / n_batches, epoch_ls_mean / n_batches


def get_gp_hyperparams(model, likelihood):
    """
    Snapshot of the GP's learned hyperparameters at a point in training:
    """
    return {
        "lengthscales": model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten(),
        "outputscale": model.covar_module.outputscale.item(),
        "noise": likelihood.noise.item(),
        "mean_constant": model.mean_module.constant.item(),
    }


def plot_gp_evolution(gp_history, config_name, save_path=None):
    """
    gp_history: dict with keys 'epoch', 'lengthscales' (list of per-epoch
    arrays), 'outputscale', 'noise', 'mean_constant' (each a list, one
    entry per epoch)
    """

    epochs = gp_history['epoch']
    ls_array = np.array(gp_history['lengthscales'])  # shape (n_epochs, latent_dim)

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for dim in range(ls_array.shape[1]):
        axes[0].plot(epochs, ls_array[:, dim], marker='o', markersize=3, label=f'dim {dim}')
    axes[0].set_title('Lengthscale per dim, over training')
    axes[0].set_xlabel('Epoch')
    axes[0].legend(fontsize=7, ncol=2)
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, gp_history['outputscale'], marker='o', color='tab:purple')
    axes[1].set_title('Outputscale (signal variance)')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(alpha=0.3)

    axes[2].plot(epochs, gp_history['noise'], marker='o', color='tab:brown')
    axes[2].set_title('Likelihood noise')
    axes[2].set_xlabel('Epoch')
    axes[2].grid(alpha=0.3)

    noise_ratio = [n / o if o > 0 else float('nan')
                   for n, o in zip(gp_history['noise'], gp_history['outputscale'])]
    axes[3].plot(epochs, noise_ratio, marker='o', color='tab:red')
    axes[3].set_title('Noise / Outputscale ratio')
    axes[3].set_xlabel('Epoch')
    axes[3].grid(alpha=0.3)

    plt.suptitle(f'{config_name}: GP hyperparameter evolution')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)


def get_lengthscales(model):
    """
    Per-dimension ARD lengthscales from the trained kernel. Small
    lengthscale = kernel treats that dimension as informative (small
    differences matter). Large lengthscale = kernel has effectively
    learned to ignore that dimension
    """
    return model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()


def plot_lengthscales(lengthscales, config_name, save_path=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(range(len(lengthscales)), lengthscales, color='tab:cyan')
    ax.set_xlabel('Latent dimension')
    ax.set_ylabel('Lengthscale (lower = more relevant to the kernel)')
    ax.set_title(f'{config_name}: ARD lengthscales')
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)


def evaluate_latent_health(model, loader, device='cpu', max_samples=20000, verbose=True):
    """
    active/collapsed diagnostic - this is the number to compare against standalone runs
    to see if the GP's loss changes which
    dims activate.

    Returns (per_dim_std, n_active)
    """
    model.eval()
    mu_list = []
    collected = 0
    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            mu, _ = model.feature_extractor(x_batch)
            mu_list.append(mu.cpu().numpy())
            collected += x_batch.size(0)
            if collected >= max_samples:
                break
    mu_all = np.concatenate(mu_list, axis=0)[:max_samples]
    per_dim_std = mu_all.std(axis=0)
    n_active = int((per_dim_std > 0.05).sum())

    if verbose:
        print("\n[VAE-DKL Latent Health Diagnostic]")
        print("-" * 45)
        print(f"{'Dim':^5} | {'Std(mu)':^12} | {'Active?':^10}")
        print("-" * 45)
        for dim, s in enumerate(per_dim_std):
            active = "ACTIVE" if s > 0.05 else "COLLAPSED"
            print(f"{dim:^5} | {s:^12.4f} | {active:^10}")
        print("-" * 45)

    return per_dim_std, n_active


def evaluate_full(model, likelihood, test_loader, train_y_mean, device='cpu', tail_pct=0.10):
    """
    Returns predictions, actuals, predictive std (uncertainty), and RMSE broken out by region:
    """
    model.eval()
    likelihood.eval()

    all_preds, all_std, all_y = [], [], []
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, y_batch in test_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            output = model(x_batch)
            if hasattr(likelihood, 'set_z'):
                likelihood.set_z(model.last_z)
            preds = likelihood(output)
            all_preds.append(preds.mean.cpu())
            all_std.append(preds.stddev.cpu())
            all_y.append(y_batch.cpu())

    preds = torch.cat(all_preds).numpy()
    std = torch.cat(all_std).numpy()
    y = torch.cat(all_y).numpy()

    def _rmse(p, t):
        return float(((p - t) ** 2).mean() ** 0.5)

    order = np.argsort(y)
    n = len(y)
    n_tail = max(int(n * tail_pct), 1)
    low_idx = order[:n_tail]     # smallest actual credit_spread (tight/safe bonds)
    high_idx = order[-n_tail:]   # largest actual credit_spread (wide/risky bonds)

    rmse_overall = _rmse(preds, y)
    rmse_low_tail = _rmse(preds[low_idx], y[low_idx])
    rmse_high_tail = _rmse(preds[high_idx], y[high_idx])
    baseline_rmse = _rmse(train_y_mean, y)

    # Calibration: what fraction of actuals fall within +-1 predictive std of the prediction
    within_1std = float((np.abs(preds - y) <= std).mean())

    return {
        "preds": preds, "std": std, "y": y,
        "rmse_overall": rmse_overall,
        "rmse_low_tail": rmse_low_tail,
        "rmse_high_tail": rmse_high_tail,
        "baseline_rmse": baseline_rmse,
        "beats_baseline": rmse_overall < baseline_rmse,
        "within_1std_pct": within_1std,
    }


def plot_prediction_diagnostics(results, config_name, save_path=None):
    preds, std, y = results['preds'], results['std'], results['y']
    n = len(y)
    n_tail = max(int(n * 0.10), 1)
    order = np.argsort(y)
    tail_idx = np.concatenate([order[:n_tail], order[-n_tail:]])
    is_tail = np.zeros(n, dtype=bool)
    is_tail[tail_idx] = True

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y[~is_tail], preds[~is_tail], s=4, alpha=0.3, color='tab:blue', label='non-tail')
    axes[0].scatter(y[is_tail], preds[is_tail], s=6, alpha=0.5, color='tab:red', label='tail (top/bottom 10%)')
    lims = [min(y.min(), preds.min()), max(y.max(), preds.max())]
    axes[0].plot(lims, lims, 'k--', linewidth=1, label='perfect prediction')
    axes[0].set_xlabel('Actual credit_spread')
    axes[0].set_ylabel('Predicted credit_spread')
    axes[0].set_title(f'{config_name}: Predicted vs Actual')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    sort_order = np.argsort(y)
    y_sorted = y[sort_order]
    preds_sorted = preds[sort_order]
    std_sorted = std[sort_order]
    idx = np.arange(n)

    axes[1].plot(idx, y_sorted, color='black', linewidth=1, label='actual (sorted)')
    axes[1].plot(idx, preds_sorted, color='tab:blue', linewidth=0.8, alpha=0.7, label='predicted')
    axes[1].fill_between(idx, preds_sorted - std_sorted, preds_sorted + std_sorted,
                          color='tab:blue', alpha=0.2, label='+-1 std')
    axes[1].set_title(f'{config_name}: Sorted actual vs predicted (+ uncertainty)')
    axes[1].set_xlabel('Samples, sorted by actual value')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    plt.show()
    plt.close(fig)


def plot_sweep_summary(sweep_results, save_path=None):
    """
    Bar chart comparing all configs across RMSE and dims
    """
    names = [r['name'] for r in sweep_results]
    metrics = ['rmse_overall', 'rmse_low_tail', 'rmse_high_tail', 'n_active']
    titles = ['Overall RMSE', 'Low-tail RMSE', 'High-tail RMSE', 'Active latent dims']
    colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    for ax, metric, title, color in zip(axes, metrics, titles, colors):
        vals = [r[metric] for r in sweep_results]
        ax.bar(names, vals, color=color, alpha=0.8)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=60)
        ax.grid(alpha=0.3, axis='y')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved sweep summary to {save_path}")
    plt.show()


def plot_training_history(history, save_path=None):
    """
    history: dict covering keys 'epoch', 'gp_loss', 'kl_loss', 'n_active'
    , one entry per epoch """

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(history['epoch'], history['gp_loss'], marker='o', color='tab:blue')
    axes[0].set_title('GP loss (-ELBO)')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(alpha=0.3)

    axes[1].plot(history['epoch'], history['kl_loss'], marker='o', color='tab:orange')
    axes[1].set_title('KL loss')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(alpha=0.3)

    axes[2].plot(history['epoch'], history['n_active'], marker='o', color='tab:green')
    axes[2].set_title('Active latent dims')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylim(bottom=0)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
        print(f"Saved plot to {save_path}")
    plt.show()
