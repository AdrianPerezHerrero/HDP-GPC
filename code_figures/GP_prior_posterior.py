import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np


# ==========================================
# MODEL DEFINITION
# ==========================================

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel:
            self.covar_module = kernel
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def build_and_train_gp(x, y, alpha_ini_bounds=(0.1, 0.9),
                       lengthscale_bounds=(1e-5, 2.0), verbose=True):
    """Train a standard GP model."""
    if verbose:
        print(f"\n--- Fitting GP ---")

    x_ = x.clone().detach()
    y_ = y.clone().detach()

    lik = gpytorch.likelihoods.GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(*alpha_ini_bounds)
    )
    gp = ExactGPModel(x_, y_, lik)
    gp.covar_module.base_kernel.raw_lengthscale_constraint = \
        gpytorch.constraints.Interval(*lengthscale_bounds)

    lik.train()
    gp.train()

    optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
    losses = []

    training_iter = 2000
    for i in range(training_iter):
        optimizer.zero_grad()
        output = gp(x_)
        loss = -mll(output, y_)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and i % 500 == 0:
            print(f'Iter {i}/{training_iter} - Loss: {loss.item():.4f}')

        if len(losses) > 1000:
            if np.isclose(np.sum(np.subtract(losses[-20:], losses[-21:-1])), 0, atol=1e-8):
                if verbose: print(f"Converged at iteration {i}")
                break

    gp.eval()
    lik.eval()
    return gp, lik


# ==========================================
# DATA LOADING
# ==========================================

# Style for dissertation quality
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 14,
    "font.size": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2
})

# Load or generate data
try:
    path = '../hdpgpc/data/mitbih/102.npy'
    print(f"Loading data from {path}...")
    raw_data = np.load(path)
    train_y = torch.from_numpy(raw_data[0, :, 0])
    train_y = (train_y - torch.mean(train_y)) / torch.std(train_y)
    train_x = torch.arange(0, train_y.shape[0]) / 250.0

    if len(train_x) > 500:
        mask = train_x < 2.0
        train_x = train_x[mask]
        train_y = train_y[mask]
        print("Subsetting to first 2 seconds for visualization clarity.")

except FileNotFoundError:
    print("\nWARNING: Data file not found. Generating synthetic ECG-like data.")
    t = torch.linspace(0, 2, 500)
    signal = 0.5 * torch.exp(-(t - 0.2) ** 2 / 0.005) + \
             -0.5 * torch.exp(-(t - 0.28) ** 2 / 0.001) + \
             3.0 * torch.exp(-(t - 0.3) ** 2 / 0.0005) + \
             -1.0 * torch.exp(-(t - 0.32) ** 2 / 0.001) + \
             0.8 * torch.exp(-(t - 0.5) ** 2 / 0.01)
    signal += 0.5 * torch.exp(-(t - 1.2) ** 2 / 0.005) + \
              -0.5 * torch.exp(-(t - 1.28) ** 2 / 0.001) + \
              3.0 * torch.exp(-(t - 1.3) ** 2 / 0.0005) + \
              -1.0 * torch.exp(-(t - 1.32) ** 2 / 0.001) + \
              0.8 * torch.exp(-(t - 1.5) ** 2 / 0.01)

    train_x = t
    train_y = signal + 0.05 * torch.randn_like(t)
    train_y = (train_y - train_y.mean()) / train_y.std()

print(f"Data Shapes -> X: {train_x.shape}, Y: {train_y.shape}")

train_x = train_x.float()
train_y = train_y.float()

# ==========================================
# TRAIN MODEL
# ==========================================

print("\n--- Training GP Model ---")
gp_trained, lik_trained = build_and_train_gp(train_x, train_y, verbose=True)

# Extract learned hyperparameters
learned_ls = gp_trained.covar_module.base_kernel.lengthscale.item()
learned_var = gp_trained.covar_module.outputscale.item()
learned_noise = lik_trained.noise.item()

print(f"\nLearned Hyperparameters:")
print(f"  Lengthscale (ℓ): {learned_ls:.4f}")
print(f"  Signal variance (σ²): {learned_var:.4f}")
print(f"  Noise variance (σ²_n): {learned_noise:.4f}")

# ==========================================
# GENERATE PRIOR AND POSTERIOR SAMPLES
# ==========================================

# Define sampling domain
sample_x = torch.linspace(train_x.min() - 0.1, train_x.max() + 0.1, 200)
num_samples = 5

# Set random seed for reproducibility
torch.manual_seed(42)

# --- PRIOR SAMPLES ---
print("\n--- Generating Prior Samples ---")
# Create GP with learned hyperparameters but no data conditioning
dummy_x = torch.linspace(1, 2, 2)
dummy_y = torch.zeros(2)
prior_lik = gpytorch.likelihoods.GaussianLikelihood()
prior_lik.noise = learned_noise
prior_gp = ExactGPModel(dummy_x, dummy_y, prior_lik)
prior_gp.covar_module.base_kernel.lengthscale = learned_ls
prior_gp.covar_module.outputscale = learned_var
prior_gp.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    prior_dist = prior_gp(sample_x)
    prior_samples = prior_dist.sample(sample_shape=torch.Size([num_samples]))
    prior_mean = prior_dist.mean
    prior_std = prior_dist.stddev

# --- POSTERIOR SAMPLES ---
print("--- Generating Posterior Samples ---")
gp_trained.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    posterior_dist = gp_trained(sample_x)
    posterior_samples = posterior_dist.sample(sample_shape=torch.Size([num_samples]))
    posterior_mean = posterior_dist.mean
    posterior_std = posterior_dist.stddev

# ==========================================
# PLOTTING
# ==========================================

fig, (ax_prior, ax_posterior) = plt.subplots(1, 2, figsize=(14, 5.5))

# Define color palettes
prior_colors = plt.cm.Blues(np.linspace(0.4, 0.8, num_samples))
posterior_colors = plt.cm.Reds(np.linspace(0.4, 0.8, num_samples))

# --- LEFT PANEL: PRIOR ---
# Plot prior mean and confidence interval
ax_prior.plot(sample_x.numpy(), prior_mean.numpy(), 'b-', linewidth=2.5,
              label='Prior Mean', zorder=5)
ax_prior.fill_between(sample_x.numpy(),
                      (prior_mean - 1.96 * prior_std).numpy(),
                      (prior_mean + 1.96 * prior_std).numpy(),
                      color='blue', alpha=0.15, label='95% CI', zorder=1)

# Plot prior samples
for i in range(num_samples):
    ax_prior.plot(sample_x.numpy(), prior_samples[i].numpy(),
                 color=prior_colors[i], linestyle='--', alpha=0.7,
                 linewidth=1.5)

ax_prior.set_title("(a) Prior Distribution", fontweight='bold', fontsize=15)
ax_prior.set_xlabel("Time (s)", fontsize=14)
ax_prior.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax_prior.grid(True, alpha=0.3)
ax_prior.set_xlim(sample_x.min(), sample_x.max())

# --- RIGHT PANEL: POSTERIOR ---
# Plot training data
ax_posterior.plot(train_x.numpy(), train_y.numpy(), 'ko', markersize=5,
                 label='Training Data', zorder=10, markeredgewidth=0.5,
                 markeredgecolor='white')

# Plot posterior mean and confidence interval
ax_posterior.plot(sample_x.numpy(), posterior_mean.numpy(), 'r-',
                 linewidth=2.5, label='Posterior Mean', zorder=5)
ax_posterior.fill_between(sample_x.numpy(),
                         (posterior_mean - 1.96 * posterior_std).numpy(),
                         (posterior_mean + 1.96 * posterior_std).numpy(),
                         color='red', alpha=0.15, label='95% CI', zorder=1)

# Plot posterior samples
for i in range(num_samples):
    ax_posterior.plot(sample_x.numpy(), posterior_samples[i].numpy(),
                     color=posterior_colors[i], linestyle='--', alpha=0.7,
                     linewidth=1.5)

ax_posterior.set_title("(b) Posterior Distribution",
                      fontweight='bold', fontsize=15)
ax_posterior.set_xlabel("Time (s)", fontsize=14)
ax_posterior.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax_posterior.grid(True, alpha=0.3)
ax_posterior.set_xlim(sample_x.min(), sample_x.max())

plt.tight_layout()

# Save figure
plt.savefig('../figures/gp_prior_posterior_comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('../figures/gp_prior_posterior_comparison.pdf', bbox_inches='tight')
print("\n" + "="*60)
print("SUCCESS: Figures saved as:")
print("  - gp_prior_posterior_comparison.png")
print("  - gp_prior_posterior_comparison.pdf")
print("="*60)
plt.show()