## A package to organize all models generated on pytorch and ease their usage.
import gpytorch
import torch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

dtype = torch.float32

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
        # self.base_covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.InducingPointKernel(self.base_covar_module, inducing_points=base_points, likelihood=likelihood)

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
class LinearExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None):
        super(LinearExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(1, bias=False)
        if kernel:
            self.covar_module = kernel
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = -self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class AlignmentGPModel(gpytorch.models.ExactGP):
    def __init__(self, fixed_x, fixed_model, final_y, likelihood, likelihood2, kernel=None):
        # super().__init__(fixed_model.train_inputs, fixed_model.train_targets, likelihood)
        # super().__init__(fixed_model.train_inputs[0].detach(), final_y, likelihood)
        super().__init__(fixed_x, self.monotonic_sequence(fixed_x), likelihood)
        super().__init__(self.monotonic_sequence(fixed_x), final_y, likelihood2)
        self.mean_module = GPMean(fixed_model)
        if kernel:
            self.covar_module = kernel
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # self.fixed_model = fixed_model
        # self.dict = fixed_model.state_dict()
    
    def forward(self, x):
        G = self.monotonic_sequence(x)
        mean_x = self.mean_module.forward(G)
        covar_x = self.covar_module(G)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
    def monotonic_sequence(self, x):
        N = x.shape[0]
        soft = torch.nn.functional.softmax(x,dim=0)
        soft = N*torch.cumsum(soft,dim=0)-1
        return soft
    
class AlignGPModel(gpytorch.models.ExactGP):
    def __init__(self, fixed_x, G_priors, fixed_model, final_y, likelihoods):
        w_gp = ExactGPModel(fixed_x, G_priors, likelihoods[0])
        # w_gp.eval()
        # w_gp.likelihood.eval()
        # fixed_model.eval()
        # fixed_model.likelihood.eval()
        G = w_gp(fixed_x).mean
        x_comb = torch.cat([fixed_x,G])
        y_comb = torch.cat([final_y,fixed_model(G).mean])
        super(AlignGPModel, self).__init__(x_comb,y_comb,likelihoods[1])
        self.G = G
        self.fixed_model = fixed_model
        self.x_comb = x_comb
        self.y_comb = y_comb
        self.w_gp = w_gp
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.mean_module = gpytorch.means.ConstantMean()
        
    def forward(self, x):
        mean_x = self.mean_module.forward(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        
                
    
class GPMean(gpytorch.means.Mean):
    
    def __init__(self, gp):
        super().__init__()
        self.gp = gp
        self.dict = gp.state_dict()
        
    def _reset_gp(self):
        self.gp.load_state_dict(self.dict)
        
    def forward(self, x):
        self.gp.eval()
        self.gp.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            with gpytorch.settings.detach_test_caches(False):
                return self.gp.likelihood(self.gp(x)).mean
        self._reset_gp()    
        
