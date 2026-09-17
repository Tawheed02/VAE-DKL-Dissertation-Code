import torch
import gpytorch
import duckdb
from sklearn.cluster import KMeans


ACTIVATIONS = {
    "silu": torch.nn.SiLU,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
}


class FeatureExtractor(torch.nn.Module):
    def __init__(self, input_dim=6, latent_dim=1, hidden_dims=None, activation="silu"):
        """
        v5: activation function configurable (silu/relu/gelu/tanh), for encoder sweep.
         Defaults to silu, matching v4.
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]
        act_cls = ACTIVATIONS[activation]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(torch.nn.Linear(prev_dim, h))
            layers.append(act_cls())
            prev_dim = h
        layers.append(torch.nn.Linear(prev_dim, latent_dim))
        layers.append(torch.nn.LayerNorm(latent_dim))

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DKL(gpytorch.models.ApproximateGP):

    def __init__(self, input_dim, latent_dim, num_inducing_points=50,
                 hidden_dims=None, kernel_type='matern', nu=2.5, activation='silu'):
        inducing_points = torch.randn(num_inducing_points, latent_dim)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        self.feature_extractor = FeatureExtractor(input_dim, latent_dim, hidden_dims=hidden_dims, activation=activation)
        self.mean_module = gpytorch.means.ConstantMean()

        if kernel_type == 'matern':
            base_kernel = gpytorch.kernels.MaternKernel(nu=nu, ard_num_dims=latent_dim)
        elif kernel_type == 'rq':
            # Rational Quadratic: a mixture of RBF kernels at different

            base_kernel = gpytorch.kernels.RQKernel(ard_num_dims=latent_dim)
        else:
            raise ValueError(f"Unknown kernel_type: {kernel_type}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_z = self.mean_module(x)
        covar_z = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_z, covar_z)

    def __call__(self, x, **kwargs):
        z = self.feature_extractor(x)
        return super().__call__(z, **kwargs)


def build_model(input_dim=65, latent_dim=4, num_inducing_points=50, hidden_dims=None,
                 kernel_type='matern', nu=2.5, activation='silu', device='cpu'):
    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
    model = DKL(input_dim=input_dim, latent_dim=latent_dim, num_inducing_points=num_inducing_points,
                hidden_dims=hidden_dims, kernel_type=kernel_type, nu=nu, activation=activation).to(device)
    return model, likelihood


def build_optimizer(model, likelihood, nn_lr=0.001, gp_lr=0.01):
    """
    v4: split learning rate
    """
    optimizer = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'lr': nn_lr},
        {'params': model.covar_module.parameters(), 'lr': gp_lr},
        {'params': model.mean_module.parameters(), 'lr': gp_lr},
        {'params': model.variational_strategy.parameters(), 'lr': gp_lr},
        {'params': likelihood.parameters(), 'lr': gp_lr},
    ])
    return optimizer


def build_mll(model, likelihood, train_parquet_path):
    num_data = duckdb.query(f"SELECT COUNT(*) AS n FROM '{train_parquet_path}'").df()['n'][0]
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=int(num_data))
    return mll, int(num_data)


def get_train_y_mean(train_parquet_path):
    return duckdb.query(f"SELECT AVG(credit_spread) AS m FROM '{train_parquet_path}'").df()['m'][0]


def train_one_epoch(model, likelihood, optimizer, mll, train_loader, device='cpu',
                     print_every=50, use_amp=False):
    """
    use_amp: if True, wraps only the feature extractor's forward pass in
    mixed precision — Not the GP computation, that stays in full FP32.
    """
    model.train()
    likelihood.train()

    epoch_loss = 0.0
    n_batches = 0
    scaler = torch.cuda.amp.GradScaler() if (use_amp and device != 'cpu') else None

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()

        if use_amp and device != 'cpu':
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                z = model.feature_extractor(x_batch)
            # z cast back to FP32 before it touches the GP
            z = z.float()
            output = model.variational_strategy(z)
            loss = -mll(output, y_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()
        n_batches += 1

        if n_batches % print_every == 0:
            print(f"    batch {n_batches}: loss = {loss.item():.4f}")

    return epoch_loss / n_batches


def reinitialize_inducing_points_kmeans(model, sample_loader, latent_dim, num_inducing_points, device='cpu'):
    """
    v4: after warmup epochs with random inducing points, runs k-means on the resulting Z values
    from the extractor, and replaces the inducing point locations with the cluster centers
    """
    model.eval()
    z_samples = []
    with torch.no_grad():
        for x_batch, _ in sample_loader:
            x_batch = x_batch.to(device)
            z_batch = model.feature_extractor(x_batch)
            z_samples.append(z_batch.cpu())
            if len(z_samples) * x_batch.size(0) >= 50_000:  # cap sample size, don't need the full dataset
                break
    z_samples = torch.cat(z_samples).numpy()

    kmeans = KMeans(n_clusters=num_inducing_points, n_init=10, random_state=42)
    kmeans.fit(z_samples)
    new_inducing_points = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)

    model.variational_strategy.inducing_points.data = new_inducing_points
    model.train()
    print(f"  Reinitialized {num_inducing_points} inducing points via k-means.")
    return model


def predict(model, likelihood, loader, device='cpu', return_variance=False):
    """
    return_variance: if True, also returns the GP's predictive variance
    at each point — the model's own uncertainty estimate.
    """
    model.eval()
    likelihood.eval()

    all_preds = []
    all_var = []
    all_y = []

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = likelihood(model(x_batch))
            all_preds.append(preds.mean.cpu())
            all_y.append(y_batch.cpu())
            if return_variance:
                all_var.append(preds.variance.cpu())

    preds_out = torch.cat(all_preds).numpy()
    y_out = torch.cat(all_y).numpy()

    if return_variance:
        var_out = torch.cat(all_var).numpy()
        return preds_out, y_out, var_out
    return preds_out, y_out


def evaluate(model, likelihood, test_loader, train_y_mean, device='cpu', target_inverse_transform=None):
    mean_preds, y_test_all = predict(model, likelihood, test_loader, device)

    if target_inverse_transform is not None:
        mean_preds = target_inverse_transform(mean_preds)
        y_test_all = target_inverse_transform(y_test_all)

    rmse = ((mean_preds - y_test_all) ** 2).mean() ** 0.5
    baseline_rmse = ((train_y_mean - y_test_all) ** 2).mean() ** 0.5

    return {
        "rmse": rmse,
        "baseline_rmse": baseline_rmse,
        "target_std": y_test_all.std(),
        "beats_baseline_meaningfully": rmse < baseline_rmse * 0.9,
        "beats_baseline": rmse < baseline_rmse,
    }


def print_learned_hyperparameters(model, likelihood):
    print("--- Learned hyperparameters ---")
    print("outputscale:", model.covar_module.outputscale.item())
    print("noise:", likelihood.noise.item())
    print("mean constant:", model.mean_module.constant.item())


def save_checkpoint(model, likelihood, path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'likelihood_state_dict': likelihood.state_dict(),
    }, path)
    print(f"Saved checkpoint to {path}")


def load_checkpoint(model, likelihood, path, device='cpu'):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    likelihood.load_state_dict(checkpoint['likelihood_state_dict'])
    print(f"Loaded checkpoint from {path}")
    return model, likelihood