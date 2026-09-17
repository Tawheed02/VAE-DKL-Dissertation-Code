import torch
import gpytorch


class HeteroscedasticGaussianLikelihood(gpytorch.likelihoods.Likelihood):
    """
    Noise is a function of z: sigma_n^2(z) = softplus(noise_net(z)) + eps
    """
    def __init__(self, latent_dim, hidden=16):
        super().__init__()
        self.noise_net = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, hidden), torch.nn.ReLU(),
            torch.nn.Linear(hidden, 1), torch.nn.Softplus(),
        )
        self._current_z = None

    def set_z(self, z):
        self._current_z = z

    def current_noise(self):
        return self.noise_net(self._current_z).squeeze(-1) + 1e-4

    def forward(self, function_samples, *args, **kwargs):
        noise = self.current_noise()
        return torch.distributions.Normal(function_samples, noise.sqrt())

    def marginal(self, function_dist, *args, **kwargs):
        """
        Analytic predictive marginal: mean = GP mean, variance = GP variance
        + heteroscedastic noise(z).
        """
        mean = function_dist.mean
        var = function_dist.variance
        noise = self.current_noise()
        return torch.distributions.Normal(mean, (var + noise).sqrt())


class StudentTLikelihood(gpytorch.likelihoods.Likelihood):
    """
    Heavy-tailed likelihood: p(y|f) = StudentT(df, loc=f, scale=learned).
    """
    def __init__(self, df=4.0):
        super().__init__()
        self.df = df
        self.raw_scale = torch.nn.Parameter(torch.tensor(0.0))

    @property
    def scale(self):
        return torch.nn.functional.softplus(self.raw_scale) + 1e-4

    def forward(self, function_samples, *args, **kwargs):
        return torch.distributions.StudentT(df=self.df, loc=function_samples, scale=self.scale)

    def marginal(self, function_dist, *args, **kwargs):
        """
        Analytic predictive marginal, Normal approximation: mean = GP mean,
        variance = GP variance + variance of student t noise
        """
        mean = function_dist.mean
        var = function_dist.variance
        noise_var = (self.scale ** 2) * self.df / (self.df - 2)
        return torch.distributions.Normal(mean, (var + noise_var).sqrt())


def build_likelihood(variant, latent_dim, device='cpu'):
    """variant: 'gaussian' | 'heteroscedastic' | 'studentt'"""
    if variant == 'gaussian':
        return gpytorch.likelihoods.GaussianLikelihood().to(device)
    elif variant == 'heteroscedastic':
        return HeteroscedasticGaussianLikelihood(latent_dim).to(device)
    elif variant == 'studentt':
        return StudentTLikelihood().to(device)
    else:
        raise ValueError(f"Unknown likelihood variant: {variant}")
