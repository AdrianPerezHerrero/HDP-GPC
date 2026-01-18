import torch
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy
import matplotlib.pyplot as plt
import numpy as np
import os


# ==========================================
# 1. MODEL DEFINITIONS (Exact & Projected)
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


class ProjectedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, base_points):
        super(ProjectedGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=base_points,
                                                                 likelihood=likelihood)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class VarProjectedGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super(VarProjectedGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


# ==========================================
# 2. TRAINING LOGIC
# ==========================================

def build_and_train_gp(x, y, base_points=None, reduced_points=False,
                       alpha_ini_bounds=(0.01, 0.1),
                       lengthscale_bounds=(1e-5, 2.0),  # Expanded bounds for small LS
                       verbose=True):
    """
    Constructs and trains the GP model using the specific optimizer grouping
    and convergence checks requested.
    """
    if verbose:
        print(f"\n--- Fitting GP (Reduced: {reduced_points}) ---")

    # Ensure tensors are detached copies
    x_ = x.clone().detach()
    y_ = y.clone().detach()

    # Handle Base Points
    if base_points is None:
        x_basis = x_.clone()
    else:
        x_basis = base_points.clone().detach()

    # --- Initialize Model ---
    use_projected = reduced_points or (not torch.equal(x_basis, x_))

    if use_projected:
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(*alpha_ini_bounds)
        )
        gp = ProjectedGPModel(x_, y_, lik, x_basis)
        gp.covar_module.base_kernel.base_kernel.raw_lengthscale_constraint = \
            gpytorch.constraints.Interval(*lengthscale_bounds)
        # gp = VarProjectedGPModel(x_basis)
        # gp.covar_module.base_kernel.raw_lengthscale_constraint = \
        #     gpytorch.constraints.Interval(*lengthscale_bounds)
    else:
        lik = gpytorch.likelihoods.GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(*alpha_ini_bounds)
        )
        gp = ExactGPModel(x_, y_, lik)
        gp.covar_module.base_kernel.raw_lengthscale_constraint = \
            gpytorch.constraints.Interval(*lengthscale_bounds)

    lik.train()
    gp.train()

    # --- Optimizer Groups ---
    training_iter = 4000

    if use_projected:
        # optimizer = torch.optim.Adam([
        #     {'params': gp.covar_module.inducing_points, 'lr': 0.0002},
        #     {'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.005},
        #     {'params': gp.likelihood.parameters(), 'lr': 0.005}
        # ])
        optimizer = torch.optim.Adam([
            {'params': gp.covar_module.inducing_points, 'lr': 0.001},
            {'params': gp.covar_module.base_kernel.parameters(), 'lr': 0.0005},
            {'params': gp.likelihood.parameters(), 'lr': 0.0005}
        ])
        training_iter = 20000
    else:
        optimizer = torch.optim.Adam(gp.parameters(), lr=0.1)
        training_iter = 2000

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, gp)
    losses = []

    # --- Training Loop ---
    for i in range(training_iter):
        optimizer.zero_grad()
        output = gp(x_)
        loss = -mll(output, y_)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if verbose and i % 500 == 0:
            print(f'Iter {i}/{training_iter} - Loss: {loss.item():.4f}')

        # Convergence Check (last 10 iterations stable)
        if len(losses) > 1000:
            if np.isclose(np.sum(np.subtract(losses[-20:], losses[-21:-1])), 0, atol=1e-8):
                if verbose: print(f"Converged at iteration {i}")
                break

    gp.eval()
    lik.eval()
    return gp, lik


def get_posterior(model, likelihood, test_x):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(test_x))
        return observed_pred.mean.detach().numpy(), observed_pred.variance.detach().numpy()


# ==========================================
# 3. DATA LOADING & MAIN EXECUTION
# ==========================================

# Style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "figure.figsize": (11, 14),
    "axes.grid": True,
    "grid.alpha": 0.3
})

# --- A. LOAD DATA ---
try:
    # Attempt to load from path specified
    path = '../hdpgpc/data/mitbih/102.npy'
    print(f"Loading data from {path}...")
    raw_data = np.load(path)
    train_y = torch.from_numpy(raw_data[0, :, 0])

    # Normalize
    train_y = (train_y - torch.mean(train_y)) / torch.std(train_y)

    # Create Time Axis (in seconds, assuming 250Hz)
    train_x = torch.arange(0, train_y.shape[0]) / 250.0

    # Create Test Axis
    test_x = torch.linspace(-0.1, train_x[-1] + 0.1, 300).double()

    # Subset for visualization if dataset is huge (e.g. take first 2 seconds)
    # Removing this line will use full data, but plotting might be dense.
    if len(train_x) > 500:
        mask = train_x < 2.0  # First 2 seconds
        train_x = train_x[mask]
        train_y = train_y[mask]
        test_x = torch.linspace(-0.1, 2.1, 300).double()
        print("Subsetting to first 2 seconds for visualization clarity.")

except FileNotFoundError:
    print("\nWARNING: Data file not found. Generating synthetic ECG-like data for demonstration.")
    # Synthetic Heartbeat
    t = torch.linspace(0, 2, 500)
    # Approximation of P-QRS-T complex
    signal = 0.5 * torch.exp(-(t - 0.2) ** 2 / 0.005) + \
             -0.5 * torch.exp(-(t - 0.28) ** 2 / 0.001) + \
             3.0 * torch.exp(-(t - 0.3) ** 2 / 0.0005) + \
             -1.0 * torch.exp(-(t - 0.32) ** 2 / 0.001) + \
             0.8 * torch.exp(-(t - 0.5) ** 2 / 0.01)
    # Repeat
    signal += 0.5 * torch.exp(-(t - 1.2) ** 2 / 0.005) + \
              -0.5 * torch.exp(-(t - 1.28) ** 2 / 0.001) + \
              3.0 * torch.exp(-(t - 1.3) ** 2 / 0.0005) + \
              -1.0 * torch.exp(-(t - 1.32) ** 2 / 0.001) + \
              0.8 * torch.exp(-(t - 1.5) ** 2 / 0.01)

    train_x = t
    train_y = signal + 0.05 * torch.randn_like(t)
    train_y = (train_y - train_y.mean()) / train_y.std()
    test_x = torch.linspace(-0.1, 2.1, 300).double()

print(f"Data Shapes -> X: {train_x.shape}, Y: {train_y.shape}")

# Ensure float32/double consistency
train_x = train_x.float()
train_y = train_y.float()
test_x = test_x.float()

fig = plt.figure(constrained_layout=True)
gs = fig.add_gridspec(3, 1, height_ratios=[1, 1.2, 1])

# --- FIGURE A: Standard GP ---
# Use the requested lengthscale target (1/250 = 0.004)
TARGET_LS = 1.0 / 250.0

print("\n--- Training Model A (Standard) ---")
gp_std, lik_std = build_and_train_gp(train_x, train_y, reduced_points=False)
mu, var = get_posterior(gp_std, lik_std, test_x)
std = np.sqrt(var)

ax1 = fig.add_subplot(gs[0])
ax1.plot(train_x.numpy(), train_y.numpy(), 'k.', alpha=0.5, markersize=5, label='MIT-BIH Data')
ax1.plot(test_x.numpy(), mu, 'b-', lw=1.5, label='Posterior Mean')
ax1.fill_between(test_x.numpy(), mu - 1.96 * std, mu + 1.96 * std, color='b', alpha=0.2, label='95% CI')
ax1.set_title("A. Exact Gaussian Process on MIT-BIH Data", fontweight='bold')
ax1.set_xlim(test_x.min(), test_x.max())
ax1.legend(loc='upper right')

# --- FIGURE B: Hyperparameters ---
# Visualizing effect of LS around 1/250 (0.004)
print("\n--- Generating Figure B (Hyperparameters) ---")
sub_gs = gs[1].subgridspec(1, 3)

# Define scenarios relative to the sampling rate (1/250)
configs = [
    (sub_gs[0], TARGET_LS * 0.2, 0.01, "Too Short (Noise)"),
    (sub_gs[1], TARGET_LS, 0.01, "Balanced (l ~ 1/250)"),
    (sub_gs[2], TARGET_LS * 10.0, 0.01, "Too Long (Smooth)")
]

for sub, ls, noise, title in configs:
    ax = fig.add_subplot(sub)

    # Create temporary model to force params
    t_lik = gpytorch.likelihoods.GaussianLikelihood()
    t_lik.noise = noise
    t_gp = ExactGPModel(train_x, train_y, t_lik)
    t_gp.eval()
    t_lik.eval()

    # Force Hyperparameters
    t_gp.covar_module.base_kernel.lengthscale = ls


    mu_h, var_h = get_posterior(t_gp, t_lik, test_x)
    std_h = np.sqrt(var_h)

    ax.plot(train_x.numpy(), train_y.numpy(), 'k.', alpha=0.2, markersize=3)
    ax.plot(test_x.numpy(), mu_h, 'r-', lw=1.5)
    ax.fill_between(test_x.numpy(), mu_h - 1.96 * std_h, mu_h + 1.96 * std_h, color='b', alpha=0.2, label='95% CI')
    ax.set_title(f"$l={ls:.4f}$\n{title}", fontsize=10)
    ax.set_xlim(test_x.min(), test_x.max())
    # Zoom Y slightly to see fit details
    ax.set_ylim(train_y.min() - 0.5, train_y.max() + 0.5)

# --- FIGURE C: Projected GP (Inducing Points) ---
print("\n--- Training Model C (Projected / Sparse) ---")

# Select Initial Inducing Points (random subset)
num_inducing = 15  # Sufficient for short segment
base_points_init = torch.linspace(train_x.min(), train_x.max(), num_inducing)

gp_proj, lik_proj = build_and_train_gp(
    train_x, train_y,
    base_points=base_points_init,
    reduced_points=True,
    verbose=True
)

mu_p, var_p = get_posterior(gp_proj, lik_proj, test_x)
std_p = np.sqrt(var_p)

ax3 = fig.add_subplot(gs[2])
ax3.plot(train_x.numpy(), train_y.numpy(), 'k.', alpha=0.2, label='Full Data')
ax3.plot(test_x.numpy(), mu_p, color='purple', lw=2, label='Projected Mean')
ax3.fill_between(test_x.numpy(), mu_p - 1.96 * std_p, mu_p + 1.96 * std_p, color='purple', alpha=0.2,
                 label='Uncertainty')

# Plot Optimized Inducing Locations
final_inducing = gp_proj.covar_module.inducing_points.detach().cpu()
with torch.no_grad():
    # Evaluate mean at inducing points just for plotting y-height
    ind_y = gp_proj(final_inducing).mean

ax3.scatter(final_inducing.numpy(), ind_y.numpy(),
            c='red', marker='D', s=40, zorder=10, label=f'Inducing Pts (M={num_inducing})')

ax3.set_title("C. Sparse GP Approximation (ProjectedGPModel)", fontweight='bold')
ax3.set_xlim(test_x.min(), test_x.max())
ax3.legend(loc='lower right', ncol=3)

# Save
plt.savefig('../figures/mitbih_gp_analysis.png', dpi=300, bbox_inches='tight')
print("\nSuccess. Figure saved to 'mitbih_gp_analysis.png'")
plt.show()