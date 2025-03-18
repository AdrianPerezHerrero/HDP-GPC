# Class for easy plotting GP
import matplotlib.pyplot as plt
import torch
torch.set_default_dtype(torch.float64)
import pyro.contrib.gp as gp
import pyro.distributions as dist
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from matplotlib.colors import to_hex
from sklearn.manifold import MDS
import plotly.io as pio
from plotly.offline import plot
from plotly.express.colors import sample_colorscale
from matplotlib.ticker import MultipleLocator

pio.templates.default = "plotly_white"
pio.renderers.default = "browser"
# pio.renderers.default = "notebook"

color = {0: 'k', 1: 'b', 2: 'r', 3: 'g', 4: 'c', 5: 'purple', 6: 'darkgreen', 7: 'maroon',
         8: 'orange', 9: 'lightgreen', 10: 'magenta', 11: 'lightblue', 12: 'darkblue', 13: 'red',
         14: 'red', 15: 'goldenrod', 16: 'red'}
labels_trans = {'N': 1, 'V': 2, 'R': 3, '!': 4, 'F': 5, 'L': 6, 'A': 7, '/': 8, 'Q': 9, 'f': 10, 'E': 11,
                'J': 12, 'j': 13, 'e': 14, 'a': 15, 'S': 16}
config = {'responsive': False, 'displaylogo': False, 'modeBarButtonsToAdd':['drawline',
                                        'drawopenpath',
                                        'drawclosedpath',
                                        'drawcircle',
                                        'drawrect',
                                        'eraseshape'
                                       ], 'showLink': True,
                                        'plotlyServerURL': "https://chart-studio.plotly.com"}

def plot_gp_pyro(plot_observed_data=False, X=None, y=None, plot_predictions=False, n_prior_samples=0,
         model=None, kernel=None, n_test=500):
    
    ini = X[0]
    end = X[-1]
    plt.figure(figsize=(12, 6))
    if plot_observed_data:
        plt.plot(X.numpy(), y.numpy(), 'k.', markersize=2)
    if plot_predictions:
        Xtest = torch.linspace(ini, end, n_test)  # test inputs
        # compute predictive mean and variance
        with torch.no_grad():
            if type(model) == gp.models.VariationalSparseGP:
                mean, cov = model(Xtest, full_cov=True)
            else:
                mean, cov = model(Xtest, full_cov=True, noiseless=False)
        sd = cov.diag().sqrt()  # standard deviation at each input point x
        plt.plot(Xtest.numpy(), mean.numpy(), 'r', lw=2)  # plot the mean
        plt.fill_between(Xtest.numpy(),  # plot the two-sigma uncertainty about the mean
                         (mean - 2.0 * sd).numpy(),
                         (mean + 2.0 * sd).numpy(),
                         color='C0', alpha=0.3)
    if n_prior_samples > 0:  # plot samples from the GP prior
        Xtest = torch.linspace(ini, end, n_test)  # test inputs
        noise = (model.noise if type(model) != gp.models.VariationalSparseGP
                 else model.likelihood.variance)
        cov = kernel.forward(Xtest) + noise.expand(n_test).diag()
        samples = dist.MultivariateNormal(torch.zeros(n_test), covariance_matrix=cov)\
                      .sample(sample_shape=(n_prior_samples,))
        plt.plot(Xtest.numpy(), samples.numpy().T, lw=2, alpha=0.4)

    plt.xlim(ini,end)
    plt.show()
    

def plot_gp_gpytorch(train_x,train_y, test_x, observed_pred, title=None):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        if title is not None:
            ax.title.set_text(title)
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k--', markersize=4, alpha=0.7)
        # Plot predictive means as blue line
        if torch.cuda.is_available():
            ax.plot(test_x.numpy(), observed_pred.mean.cpu().numpy(), 'b')
        else:
            ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.cpu().numpy(), upper.cpu().numpy(), alpha=0.5)
        # ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])
        plt.show()
        
def plot_gp_region(train_x, mean, noise_matrix, title=None, num_fig=None):
    train_x = train_x.T[0]
    mean = mean.T[0]
    if num_fig is None:
        plt.figure(figsize=(12,6))
    else:
        plt.figure(num_fig, figsize=(12,6))
    if title is not None:
        plt.title(title)
    plt.plot(train_x, mean, 'b-')
    sd = np.sqrt(np.diag(noise_matrix))
    plt.fill_between(train_x, mean-2.0*sd, mean+2.0*sd, color='cyan', alpha=0.4)
    plt.show()

def plot_ecg(x, ecg, ax=None, title=None, num_fig=None, save=None, end_beats=None):
    def setup_ecg_grid(ax):
        ax.set_ylim(np.min(ecg) * 1.3, np.max(ecg) * 1.3)
        ax.set_xlim(0.0, 10.0)
        #ax.set_aspect(0.25)
        # Major grid
        ax.grid(which='major', linestyle='-', linewidth='0.3', color='red')
        # Minor grid
        ax.grid(which='minor', linestyle='-', linewidth='0.1', color='red')
        # Set major and minor tick locators for x and y axis
        ax.xaxis.set_major_locator(MultipleLocator(0.2))  # 200 ms major grid
        ax.xaxis.set_minor_locator(MultipleLocator(0.04))  # 40 ms minor grid
        ax.yaxis.set_major_locator(MultipleLocator(0.5))  # 0.5 mV major grid
        # ax.yaxis.set_minor_locator(MultipleLocator(0.1))
        ax.tick_params(axis='y', which='major', labelsize=4)
        ax.tick_params(axis='y', which='minor', labelsize=4)
        ax.tick_params(axis='x', which='both', labelbottom=False)
        # Tranforming time steps into seconds for STT !!

    x = x / 500.0  # Back to analog sample 250Hz
    if end_beats is not None:
        end_beats = end_beats / 500.0
    # ecg = ecg.T[0]/500 #Back to analog sample 200ADC
    if ax is None:
        if num_fig is None:
            fig, ax = plt.subplots(figsize=(11.7, 1.53))
        else:
            fig, ax = plt.subplots(num_fig, figsize=(11.7, 1.53))
    if title is not None:
        ax.title(title)
    # Adding red grid
    setup_ecg_grid(ax)
    # Plot figure
    ax.plot(x, ecg, 'b-', linewidth=0.5)
    if end_beats is not None:
        ax.vlines(end_beats, np.min(ecg) * 1.3, np.max(ecg) * 1.3, colors='k', linestyles='--', linewidth=0.5)
    if save is not None:
        # plt.savefig(save + ".pdf", dpi=150)
        # plt.savefig(save + ".eps", dpi=150)
        if not plt.rcParams["text.usetex"]:
            plt.savefig(save + ".png", dpi=350)
    else:
        return ax

def plot_grid_ecg(annotations, data, time_indexes, N_0=0, save=None, figsize=None):
    dim = (1, len(time_indexes))
    figsize = (25, 6) if figsize is None else figsize
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    for i, j in enumerate(time_indexes):
        ind = annotations[j+N_0]
        x = np.atleast_2d(np.arange(ind - 87, ind + len(data[j+N_0])-87)).T
        ecg = data[j+N_0]
        plot_ecg(x, ecg, axs[i])
    if save is not None:
        # plt.savefig(save + ".pdf", dpi=150)
        # plt.savefig(save + ".eps", dpi=150)
        if not plt.rcParams["text.usetex"]:
            plt.savefig(save + ".png", dpi=350)
    else:
        plt.show()




def plot_grid(train_x, means, Sigma, Gamma=None, titles=None, dim=None, x_examples=None, y_examples=None, figsize=None):
    if dim is None:
        dim = (1, len(means))
    figsize = (10, 6) if figsize is None else figsize
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    for i, _ in enumerate(means):
        train_x_ = train_x.T[0]
        mean_ = means[i].T[0]
        r = int(np.floor(i / dim[1]))
        c = i % dim[1]
        if dim[0] == 1:
            if titles is not None:
                axs[c].set_title(titles[i])
        else:
            if titles is not None:
                axs[r][c].set_title(titles[i])
        if dim[0] == 1:
            axs[c].plot(train_x_, mean_, 'k-', linewidth=0.5)
        else:
            axs[r][c].plot(train_x_, mean_, 'k-', linewidth=0.5)
        if not (x_examples is None and y_examples is None):
            x_ = x_examples[i].T[0]
            y_ = y_examples[i].T[0]
            if dim[0] == 1:
                axs[c].plot(x_, y_, 'b.', markersize=0.3)
            else:
                axs[r][c].plot(x_, y_, 'b.', markersize=0.3)
        if not Gamma is None:
            sd_g = np.sqrt(np.diag(Gamma[i]))
            if dim[0] == 1:
                axs[c].fill_between(train_x_, mean_ - 2.0 * sd_g, mean_ + 2.0 * sd_g, color='darkblue',
                                    edgecolor='w', alpha=0.1)
            else:
                axs[r][c].fill_between(train_x_, mean_ - 2.0 * sd_g, mean_ + 2.0 * sd_g, color='darkblue',
                                       edgecolor='w', alpha=0.1)
        sd = np.sqrt(np.diag(Sigma[i]))
        if dim[0] == 1:
            axs[c].fill_between(train_x_, mean_ - 2.0 * sd, mean_ + 2.0 * sd, color='cyan', edgecolor='w', alpha=0.2)
        else:
            axs[r][c].fill_between(train_x_, mean_ - 2.0 * sd, mean_ + 2.0 * sd, color='cyan', edgecolor='w', alpha=0.2)
    for ax in fig.get_axes():
        ax.set_ylim(-240,100)
        ax.label_outer()
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.set_aspect(0.25 * 200)
    return fig

def plot_comparison(train_x, sw_gp, selected_gpmodels, time_instant=[-1], save=None, dim=None, examples=False, figsize=None, ld=None):
    means = []
    Sigma = []
    Gamma = []
    titles = []
    if examples:
        x_examples = []
        y_examples = []
    else:
        x_examples = None
        y_examples = None
    for t_ins in time_instant:
        for g in selected_gpmodels:
            if ld is None:
                gp = sw_gp.gpmodels[g]
            else:
                gp = sw_gp.gpmodels[ld][g]
            x_ = gp.x_basis.T[0]
            x_s = torch.atleast_2d(torch.arange(min(x_), max(x_), 0.01)).T
            mean_, Sig_ = gp.observe(x_s, t_ins)
            mean_lat, Gam_ = gp.step_forward_last(x_s)
            means.append(mean_)
            Sigma.append(Sig_)
            Gamma.append(Gam_)
            titles.append("Model of cluster "+ str(g) +" on t = " + str(t_ins))
            #titles = None
            if examples:
                x_examples.append(gp.x_train[t_ins - 1])
                y_examples.append(gp.y_train[t_ins - 1])
    fig = plot_grid(x_s, means, Sigma, Gamma, titles, dim=dim, x_examples=x_examples, y_examples=y_examples, figsize=figsize)
    if save is not None:
        #plt.savefig(save + ".pdf", dpi=150)
        #plt.savefig(save + ".eps", dpi=150)
        if not plt.rcParams["text.usetex"]:
            plt.savefig(save + ".png", dpi=150)
    else:
        fig.show()


def print_hyperparams(model, cuda = False):
    print("\n")
    for param_name, param in model.named_parameters():
        if torch.cuda.is_available() and cuda:
            par = param.detach().cpu().numpy()
        else:
            par = param.detach().numpy()
        print(f'Parameter name: {param_name:42} value = {par}')
    # print('covar_module.base_kernel.lenghtscale.item = ',model.covar_module.base_kernel.lengthscale.item())

def print_results(sw_gp, labels, N_0, error=False, purity=False):
    main_model = ["None"] * len(sw_gp.gpmodels[0])
    for i, _ in enumerate(sw_gp.gpmodels[0]):
        cont = np.unique([labels[j + N_0] for j in sw_gp.gpmodels[0][i].indexes], return_counts=True)
        sr = '['
        for j in range(len(cont[0])):
            sr = sr + str(cont[0][j]) + '-' + str(cont[1][j]) + ','
        if len(cont[0]) != 0:
            sr = sr[:-1]
        sr = sr + ']'
        mm = ''
        if len(cont[1]) > 0:
            main_model[i] = cont[0][np.argmax(cont[1])]
            mm = ": MainModel: " + str(main_model[i])
        print('Model', (i + 1), mm, ':', sr)
    err = np.zeros(len(sw_gp.gpmodels[0]))
    for m, gp in enumerate(sw_gp.gpmodels[0]):
        for i in gp.indexes:
            if labels[i + N_0] != main_model[m]:
                err[m] = err[m] + 1
        if purity:
            print('Model', (m + 1), ': Purity: ', 1 - err[m]/len(gp.indexes))
    print(f"Classification error: {int(err.sum())} / {sw_gp.T} -- {(int(err.sum()) / sw_gp.T):.5f}")
    if purity:
        print(f"Classification purity: {sw_gp.T - int(err.sum())}/{sw_gp.T} -- {(1 - err.sum() / sw_gp.T):.5f}")
    if purity:
        return main_model, int(err.sum()), sw_gp.T - int(err.sum())
    if error:
        return main_model, int(err.sum())
    else:
        return main_model

def plot_models(sw_gp, selected_gpmodels, main_model, labels, N_0, save=None, lead=0, step=0.1, plot_latent=False):
    num_models = len(selected_gpmodels)
    num_cols = int(np.ceil(np.sqrt(num_models)))
    num_rows = int(np.ceil(num_models / num_cols))
    titles = ()
    for i in selected_gpmodels:
        titles = titles + ("ECG Model " + str(i + 1) + " - " + str(main_model[i]),)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles, shared_yaxes='all',
                        horizontal_spacing = 0.05, vertical_spacing = 0.05)
    names = {}
    n = 0

    def col_fun(lab):
        if type(labels[0]) is np.int32:
            return to_hex(color.get(lab, 'b'))
        else:
            return to_hex(color.get(labels_trans.get(lab, 0), 'b'))
    for i, m in enumerate(selected_gpmodels):
        t = len(sw_gp.gpmodels[lead][m].y_train)
        # col = sample_colorscale('jet', list(t))
        r = int(np.floor(i / num_cols) + 1)
        c = i % num_cols + 1
        for j_, d in enumerate(sw_gp.gpmodels[lead][m].y_train):
            j = sw_gp.gpmodels[lead][m].indexes[j_]
            x_t = sw_gp.gpmodels[lead][m].x_train[j_].T[0]
            if type(d) is torch.Tensor:
                d = d.detach()
            if torch.cuda.is_available():
                d = d.cpu()
                x_t = x_t.cpu()
            x_t = list(x_t)
            fig.add_trace(go.Scatter(x=x_t, y=d.T[0], opacity=max(0.1, 0.5/(np.log(t-j_+1)+1)),
                                     line=dict(color=col_fun(labels[j + N_0]), width=1.2),
                                     mode='lines', name=' [' + str(m + 1) + '] - ' + str(j)), row=r, col=c)
    for i, m in enumerate(selected_gpmodels):
        gp = sw_gp.gpmodels[lead][m]
        x_b = gp.x_basis.T[0]
        x_ = torch.arange(min(x_b), max(x_b), step, dtype=torch.float64)
        if torch.cuda.is_available():
            x_ = x_.cpu()
            x_b = x_b.cpu()
        x = list(x_)
        x_b = list(x_b)
        x_b_r = x_b[::-1]
        x_r = x[::-1]
        # Gam_ = gp.Gamma[-1]
        mean_, Sig_ = gp.observe_last(torch.atleast_2d(x_).T)
        mean_l, Sig_l = gp.step_forward_last(torch.atleast_2d(x_).T)
        C_ = gp.C[-1]
        A_ = gp.A[-1]
        mean_latent = gp.f_star_sm[-1].T[0]
        # mean_latent_C = torch.linalg.multi_dot([C_, A_, gp.f_star_sm[-1]]).T[0]
        #mean_latent_C, Sig_lat_C = gp.observe_last(gp.x_basis)
        mean_latent_C, Sig_lat_C = gp.step_forward_last(gp.x_basis)
        Sig_lat = gp.Sigma[-1]
        mean_latent_C = mean_latent_C.T[0]
        if torch.cuda.is_available():
            Sig_ = Sig_.cpu()
            Sig_l = Sig_l.cpu()
            Sig_lat = Sig_lat.cpu()
            Sig_lat_C = Sig_lat_C.cpu()
            # Gam_ = Gam_.cpu()
            mean_ = mean_.cpu()
            mean_l = mean_l.cpu()
            mean_latent = mean_latent.cpu()
            mean_latent_C = mean_latent_C.cpu()
            C_ = C_.cpu()
        noise_ob = np.sqrt(np.diag(Sig_))
        noise_ob_l = np.sqrt(np.diag(Sig_l))
        noise_lat = 1.9 * np.sqrt(np.diag(Sig_lat))
        noise_lat_C = 1.9 * np.sqrt(np.diag(Sig_lat_C))
        #noise_lat = np.sqrt(np.diag(Sig_lat))
        #noise_lat_C = np.sqrt(np.diag(Sig_lat_C))

        mean = mean_.T[0]
        mean_l = mean_l.T[0]
        lower_ob = list(mean - 1.9 * noise_ob)
        upper_ob = list(mean + 1.9 * noise_ob)
        lower_ob_l = list(mean_l - 1.9 * noise_ob_l)
        upper_ob_l = list(mean_l + 1.9 * noise_ob_l)
        lower_in = list(mean_latent - 1.9 * noise_lat)
        upper_in = list(mean_latent + 1.9 * noise_lat)
        lower_ob = lower_ob[::-1]
        lower_ob_l = lower_ob_l[::-1]
        lower_in = lower_in[::-1]
        r = int(np.floor(i / num_cols) + 1)
        c = i % num_cols + 1
        fig.add_trace(go.Scatter(x=x, y=mean,
                                 line=dict(color='black', width=2), mode='lines',
                                 name='GP [' + str(m + 1) + ']'), row=r, col=c)
        fig.add_trace(go.Scatter(x=x_b, y=mean_latent, opacity=0.8,
                                 line=dict(color='grey', width=1.5), mode='lines+markers', error_y=dict(
                                            type='data', # value of error bar given in data coordinates
                                            array=noise_lat,
                                            visible=True),
                                 name='Mean [' + str(m + 1) + ']'), row=r, col=c)
        fig.add_trace(go.Scatter(x=x_b, y=mean_latent_C, opacity=0.5,
                                 line=dict(color='grey', width=1.5), mode='lines+markers', error_y=dict(
                                            type='data', # value of error bar given in data coordinates
                                            array=noise_lat_C,
                                            visible=True),
                                 name='Mean C [' + str(m + 1) + ']'), row=r, col=c)
        fig.add_trace(go.Scatter(x=x + x_r, y=upper_ob + lower_ob, opacity=0.3, fill='toself',
                                 fillcolor=col_fun(main_model[m]), mode='none',
                                 name='Var [' + str(m + 1) + ']'), row=r, col=c)
        if plot_latent:
            fig.add_trace(go.Scatter(x=x + x_r, y=upper_ob_l + lower_ob_l, opacity=0.2, fill='toself',
                                    fillcolor=col_fun(main_model[m]), mode='none',
                                    name='Var_l [' + str(m + 1) + ']'), row=r, col=c)
    fig.update_traces()
    fig.update_layout(
        dragmode='zoom',
        newshape_line_color='cyan')
    if save is not None:
        plot(fig, auto_open=False, filename=save, config=config)
    else:
        fig.show(config=config)



def plot_partial_models(sw_gp, selected_gpmodels, main_model, labels, N_0, time_instant=[-1], save=None):
    num_models = len(time_instant)
    num_cols = int(np.ceil(np.sqrt(num_models)))
    num_rows = int(np.ceil(num_models / num_cols))
    titles = ()
    for i in time_instant:
        titles = titles + ("ECG Model instant: " + str(i),)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles, shared_yaxes='all',
                        horizontal_spacing = 0.05, vertical_spacing = 0.05)
    names = {}
    n = 0
    for i, t_inst in enumerate(time_instant):
        for m in selected_gpmodels:
            t = len(sw_gp.gpmodels[m].y_train[:t_inst])
            # col = sample_colorscale('jet', list(t))
            r = int(np.floor(i / num_cols) + 1)
            c = i % num_cols + 1
            for j_, d in enumerate(sw_gp.gpmodels[m].y_train):
                j = sw_gp.gpmodels[m].indexes[j_]
                if j < t_inst:
                    x_t = sw_gp.gpmodels[m].x_train[j_].T[0]
                    if type(d) is torch.Tensor:
                        d = d.detach()
                    if torch.cuda.is_available():
                        d = d.cpu()
                        x_t = x_t.cpu()
                    x_t = list(x_t)
                    fig.add_trace(go.Scatter(x=x_t, y=d.T[0], opacity=max(0.05, 0.1/(np.log(t-j_+1)+1)),
                                             line=dict(color=to_hex(color[labels_trans[labels[j + N_0]]]), width=1.2),
                                             mode='lines', name=' [' + str(m + 1) + '] - ' + str(j)), row=r, col=c)
    for i, t_inst in enumerate(time_instant):
        for m in selected_gpmodels:
            gp = sw_gp.gpmodels[m]
            x_b = gp.x_basis.T[0]
            x_ = torch.arange(min(x_b), max(x_b), 0.1, dtype=torch.float64)
            if torch.cuda.is_available():
                x_ = x_.cpu()
                x_b = x_b.cpu()
            x = list(x_)
            x_b = list(x_b)
            x_b_r = x_b[::-1]
            x_r = x[::-1]
            # Gam_ = gp.Gamma[-1]
            mean_, Sig_ = gp.observe(torch.atleast_2d(x_).T, t_inst)
            # mean_, Sig_ = gp.step_forward_last(torch.atleast_2d(x_).T)
            if len(gp.C) <= t_inst:
                C_ = gp.C[-1]
                A_ = gp.A[-1]
                Sig_lat = gp.Sigma[-1]
            else:
                C_ = gp.C[t_inst]
                A_ = gp.A[t_inst]
                Sig_lat = gp.Sigma[t_inst]
            mean_latent = gp.f_star_sm[t_inst].T[0]
            # mean_latent_C = torch.linalg.multi_dot([C_, A_, gp.f_star_sm[-1]]).T[0]
            mean_latent_C, Sig_lat_C = gp.observe(gp.x_basis, t_inst)
            # mean_latent_C, Sig_lat_C = gp.step_forward_last(gp.x_basis)
            # Sig_lat = gp.Sigma[-1]
            mean_latent_C = mean_latent_C.T[0]
            if torch.cuda.is_available():
                Sig_ = Sig_.cpu()
                Sig_lat = Sig_lat.cpu()
                Sig_lat_C = Sig_lat_C.cpu()
                # Gam_ = Gam_.cpu()
                mean_ = mean_.cpu()
                mean_latent = mean_latent.cpu()
                mean_latent_C = mean_latent_C.cpu()
                C_ = C_.cpu()
            noise_ob = np.sqrt(np.diag(Sig_))
            noise_lat = 1.9 * np.sqrt(np.diag(Sig_lat))
            noise_lat_C = 1.9 * np.sqrt(np.diag(Sig_lat_C))

            mean = mean_.T[0]
            lower_ob = list(mean - 1.9 * noise_ob)
            upper_ob = list(mean + 1.9 * noise_ob)
            lower_in = list(mean_latent - 1.9 * noise_lat)
            upper_in = list(mean_latent + 1.9 * noise_lat)
            lower_ob = lower_ob[::-1]
            lower_in = lower_in[::-1]
            r = int(np.floor(i / num_cols) + 1)
            c = i % num_cols + 1
            fig.add_trace(go.Scatter(x=x, y=mean,
                                     line=dict(color='black', width=2), mode='lines',
                                     name='GP [' + str(m + 1) + ']'), row=r, col=c)
            fig.add_trace(go.Scatter(x=x_b, y=mean_latent, opacity=0.8,
                                     line=dict(color='grey', width=1.5), mode='lines+markers', error_y=dict(
                                                type='data', # value of error bar given in data coordinates
                                                array=noise_lat,
                                                visible=True),
                                     name='Mean [' + str(m + 1) + ']'), row=r, col=c)
            fig.add_trace(go.Scatter(x=x_b, y=mean_latent_C, opacity=0.5,
                                     line=dict(color='grey', width=1.5), mode='lines+markers', error_y=dict(
                                                type='data', # value of error bar given in data coordinates
                                                array=noise_lat_C,
                                                visible=True),
                                     name='Mean C [' + str(m + 1) + ']'), row=r, col=c)
            fig.add_trace(go.Scatter(x=x + x_r, y=upper_ob + lower_ob, opacity=0.2, fill='toself',
                                     fillcolor=to_hex(color.get(labels_trans[main_model[m]], 'b')), mode='none',
                                     name='Var [' + str(m + 1) + ']'), row=r, col=c)
    fig.update_traces()
    fig.update_layout(
        dragmode='zoom',
        newshape_line_color='cyan')
    if save is not None:
        plot(fig, auto_open=False, filename=save, config=config)
    else:
        fig.show(config=config)

def plot_warp(sw_gp, selected_gpmodels, main_model, labels, N_0, save=None):
    num_models = len(selected_gpmodels)
    num_cols = int(np.ceil(np.sqrt(num_models)))
    num_rows = int(np.ceil(num_models / num_cols))
    titles = ()
    for i in selected_gpmodels:
        titles = titles + ("ECG Model " + str(i + 1) + " - " + str(main_model[i]),)
    fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=titles,
                        horizontal_spacing = 0.05, vertical_spacing = 0.05)

    def col_fun(lab):
        if type(labels[0]) is np.int32:
            return to_hex(color.get(lab, 'b'))
        else:
            return to_hex(color.get(labels_trans.get(lab, 0), 'b'))
    for i, m in enumerate(selected_gpmodels):
        t = len(sw_gp.gpmodels[m].indexes[1:])
        r = int(np.floor(i / num_cols) + 1)
        c = i % num_cols + 1
        p = 1
        for jx, j in enumerate(sw_gp.gpmodels[m].indexes[1:]):
            x_w_ = sw_gp.x_w[j][m].T[0]
            x_t = sw_gp.gpmodels[m].x_train[p].T[0]
            p = p + 1
            if type(x_w_) is torch.Tensor:
                x_w_ = x_w_.detach()
            if torch.cuda.is_available():
                x_w_ = x_w_.cpu()
                x_t = x_t.cpu()
            x_t = list(x_t)
            fig.add_trace(go.Scatter(x=x_t, y=x_w_, opacity=max(0.1, 0.5/(np.log(t-jx+1)+1)),
                                     line=dict(color=col_fun(labels[j + N_0]), width=1.2),
                                     mode='lines', name=' [' + str(m + 1) + '] - ' + str(j)), row=r, col=c)
    for i, m in enumerate(selected_gpmodels):
        gp = sw_gp.wp_sys[m].warp_gp
        x_ = gp.x_basis.T[0]
        if torch.cuda.is_available():
            x_ = x_.cpu()
        x = list(x_)
        x_r = x[::-1]
        Sig_ = gp.Sigma[-1]
        mean_ = gp.f_star_sm[-1]
        C_ = gp.C[-1]
        if torch.cuda.is_available():
            Sig_ = Sig_.cpu()
            mean_ = mean_.cpu()
            C_ = C_.cpu()
        noise = np.sqrt(np.diag(Sig_))
        mean = torch.matmul(C_, mean_.T[0])
        lower = list(mean - 2.0 * noise)
        upper = list(mean + 2.0 * noise)
        lower = lower[::-1]
        r = int(np.floor(i / num_cols) + 1)
        c = i % num_cols + 1
        fig.add_trace(go.Scatter(x=x, y=mean,
                                 line=dict(color='black', width=2), mode='lines', name='Mean [' + str(m + 1) + ']'),
                      row=r, col=c)
        fig.add_trace(go.Scatter(x=x + x_r, y=upper + lower, opacity=0.2, fill='toself',
                                 fillcolor=col_fun(main_model[m]), mode='none',
                                 name='Var [' + str(m + 1) + ']'), row=r, col=c)
    fig.update_traces()
    fig.update_layout(
        dragmode='zoom',
        newshape_line_color='cyan')
    if save is not None:
        plot(fig, auto_open=False, filename=save, config=config)
    else:
        fig.show(config=config)

def plot_MDS(sw_gp, main_model, labels, N_0, lead=0, save=None):
    # %% MDS compute and plot
    print("Compute distance matrix.")
    t_ini = sw_gp.M
    x_bas = sw_gp.cond_to_torch(sw_gp.x_basis[0])
    last_T1 = 0
    # KL_dist = np.zeros((N-t_ini,N-t_ini))
    KL_dist = np.zeros((sw_gp.T, sw_gp.T))
    for m, gp1 in enumerate(sw_gp.gpmodels[lead]):
        print("Compute model " + str(m + 1))
        for i, ind1 in enumerate(gp1.indexes):
            for gp2 in sw_gp.gpmodels[lead]:
                # Computing map of MDS transformation of obtained models
                for j, ind2 in enumerate(gp2.indexes):
                    if ind1 < ind2:
                        KL_dist[ind1, ind2] = gp1.KL_divergence(i, gp2, j, smoothed=False, x_bas=x_bas)
    for i in range(sw_gp.T):
        for j in range(i, sw_gp.T):
            KL_dist[j, i] = KL_dist[i, j]

    print("Calculando MDS para representación")
    mds = MDS(dissimilarity='precomputed')
    models_transformed = mds.fit_transform(KL_dist)

    def col_fun(lab):
        if type(labels[0]) is np.int32:
            return to_hex(color.get(lab, 'b'))
        else:
            return to_hex(color.get(labels_trans.get(lab, 0), 'b'))

    col = ["black"] * sw_gp.T
    col2 = ["black"] * sw_gp.T
    indexes_col = []
    for m in range(sw_gp.M):
        # for i in sw_gp.gpmodels[m].indexes[1:]:
        for i in sw_gp.gpmodels[lead][m].indexes:
            indexes_col.append(i)
            col[i] = col_fun(labels[i + N_0])
            col2[i] = col_fun(main_model[m])
    n = sw_gp.T
    t = np.linspace(0, 1, n)
    c = sample_colorscale('amp', list(t))
    # c.reverse()
    fig = make_subplots(rows=1, cols=2, subplot_titles=("MDS with real labels", "MDS with our labels"))
    for i, _ in enumerate(models_transformed[:, 0]):
        fig.add_trace(
            go.Scatter(x=np.array(models_transformed[i, 0]), y=np.array(models_transformed[i, 1]), mode='markers',
                       marker=dict(color=col[i])), row=1, col=1)
        fig.add_trace(
            go.Scatter(x=np.array(models_transformed[i, 0]), y=np.array(models_transformed[i, 1]), mode='markers',
                       marker=dict(color=col2[i])), row=1, col=2)
    fig.add_trace(
        go.Scatter(x=np.array(models_transformed[:, 0]), y=np.array(models_transformed[:, 1]), opacity=0.5, mode='lines+markers',
                   marker=dict(color=c, symbol="arrow", angleref="previous", size=10)), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=np.array(models_transformed[:, 0]), y=np.array(models_transformed[:, 1]), opacity=0.5, mode='lines+markers',
                   marker=dict(color=c, symbol="arrow", angleref="previous", size=10)), row=1, col=2)
    fig.update_traces()
    fig.update_layout(
        dragmode='zoom',
        newshape_line_color='cyan')
    if save is not None:
        plot(fig, auto_open=False, filename=save, config=config)
    else:
        fig.show(config=config)
        fig.show()


def plot_MDS_plotly(sw_gp, main_model, labels, N_0, lead=0, save=None):
    # %% MDS compute and plot
    print("Compute distance matrix.")
    t_ini = sw_gp.M
    x_bas = sw_gp.cond_to_torch(sw_gp.x_basis[0])
    last_T1 = 0
    # KL_dist = np.zeros((N-t_ini,N-t_ini))
    KL_dist = np.zeros((sw_gp.T, sw_gp.T))
    for m, gp1 in enumerate(sw_gp.gpmodels[lead]):
        print("Compute model " + str(m + 1))
        for i, ind1 in enumerate(gp1.indexes):
            for gp2 in sw_gp.gpmodels[lead]:
                # Computing map of MDS transformation of obtained models
                for j, ind2 in enumerate(gp2.indexes):
                    if ind1 < ind2:
                        KL_dist[ind1, ind2] = gp1.KL_divergence(i, gp2, j, smoothed=False, x_bas=x_bas)
    for i in range(sw_gp.T):
        for j in range(i, sw_gp.T):
            KL_dist[j, i] = KL_dist[i, j]

    print("Calculando MDS para representación")
    mds = MDS(dissimilarity='precomputed')
    models_transformed = mds.fit_transform(KL_dist)

    def col_fun(lab):
        if type(labels[0]) is np.int32:
            return to_hex(color.get(lab, 'b'))
        else:
            return to_hex(color.get(labels_trans.get(lab, 0), 'b'))

    col = ["black"] * sw_gp.T
    col2 = ["black"] * sw_gp.T
    indexes_col = []
    for m in range(sw_gp.M):
        # for i in sw_gp.gpmodels[m].indexes[1:]:
        for i in sw_gp.gpmodels[lead][m].indexes:
            indexes_col.append(i)
            col[i] = col_fun(labels[i + N_0])
            col2[i] = col_fun(main_model[m])
    n = sw_gp.T
    t = np.linspace(0, 1, n)
    # c.reverse()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].set_title('MDS with real labels')
    axs[1].set_title('MDS with our labels')

    for i, _ in enumerate(models_transformed[:, 0]):
        axs[0].scatter(models_transformed[i, 0], models_transformed[i, 1], c=col[i], alpha=0.3, s=0.1)
        axs[1].scatter(models_transformed[i, 0], models_transformed[i, 1], c=col2[i], alpha=0.3, s=0.1)

    fig.tight_layout()

    if save is not None:
        plt.savefig(save, format='png')
    else:
        plt.show()


def plot_models_plotly(sw_gp, selected_gpmodels, main_model, labels, N_0, title=None, save=None, lead=0, step=0.1, plot_latent=False, ticks=False, yscale=False):
    num_models = len(selected_gpmodels)
    num_cols = int(np.ceil(np.sqrt(num_models)))
    num_rows = int(np.ceil(num_models / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(25, 20), squeeze=False)
    axes = axes.flatten()

    def col_fun(lab):
        if type(labels[0]) is np.int32:
            return to_hex(color.get(lab, 'b'))
        else:
            return to_hex(color.get(labels_trans.get(lab, 0), 'b'))

    y_max = 0.0
    for i, m in enumerate(selected_gpmodels):
        ax = axes[i]
        gp = sw_gp.gpmodels[lead][m]

        # Plot training data
        for j_, d in enumerate(gp.y_train):
            j = gp.indexes[j_]
            x_t = gp.x_train[j_].T[0]
            d = sw_gp.y_train[j,:,[lead]]
            if y_max < np.max(d.numpy()):
                y_max = np.max(d.numpy())
            if isinstance(d, torch.Tensor):
                d = d.detach().cpu()
                x_t = x_t.cpu()
            ax.plot(x_t, d.T[0], alpha=max(0.07, 0.5 / (np.log(len(gp.y_train) - j_ + 1) + 1)),
                    color=col_fun(labels[j + N_0]), linewidth=1.2)

        # Mean and variance
        x_b = gp.x_basis.T[0]
        if (x_b[1] - x_b[0]) == step:
            x_ = torch.clone(x_b)
        else:
            x_ = torch.arange(min(x_b), max(x_b), step, dtype=torch.float64).cpu()

        mean_, Sig_ = gp.observe_last(torch.atleast_2d(x_).T)
        #mean_l, Sig_l = gp.step_forward_last(torch.atleast_2d(x_).T)

        noise_ob = np.sqrt(np.diag(Sig_.cpu()))
        mean = mean_.cpu().T[0]
        neg_lim = mean - 1.9 * noise_ob
        neg_lim[neg_lim < 0.0] = 0.0
        ax.plot(x_, mean, color='black', linewidth=2, label=f'Emission GP mean [{m + 1}]')
        ax.fill_between(x_, neg_lim, mean + 1.9 * noise_ob, color=col_fun(main_model[i]), alpha=0.3)

        # Latent mean
        mean_latent = gp.f_star_sm[-1].cpu().T[0]
        noise_lat = 1.9 * np.sqrt(np.diag(gp.Gamma[-1].cpu()))
        neg_lim = mean_latent - noise_lat
        neg_lim[neg_lim < 0.0] = 0.0
        # ax.plot(x_b.cpu(), mean_latent, color='grey', linewidth=1.5, label=f'Latent GP Mean [{m + 1}]')
        ax.fill_between(x_b.cpu(), neg_lim, mean_latent+noise_lat, color=col_fun(main_model[i]), alpha=0.22)

        #Experimental mean
        mean = np.mean(np.array(gp.y_train), axis=0)
        ax.plot(x_b.cpu(), mean, color='black', linewidth=2, label=f'Experimental mean [{m + 1}]', linestyle='--')

        if title is None:
            ax.set_title(f"ECG CLUSTER {m + 1} ({main_model[m]})")
        else:
            ax.set_title(title)
        #ax.grid(True)
    if yscale:
        for ax in fig.get_axes():
            ax.set_ylim(np.min(sw_gp.y_train.numpy())-0.5, y_max + 0.5)
            ax.set_xticks(np.arange(0.0,0.5,0.1))
    if not ticks:
        for ax in fig.get_axes():
        #     ax.label_outer()
        #     ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.set_xticks([])
        #     ax.set_yticks([])
    # Hide unused subplots
    for j in range(len(axes)):
        if j >= num_models:
            axes[j].axis('off')

    fig.tight_layout()
    if save:
        plt.savefig(save)
    else:
        plt.show()

import matplotlib.animation as animation

class FunctionEvolutionVisualizer:
    def __init__(self, gp, initial_step, num_steps):
        """
        Initialize the visualizer with required parameters.

        Parameters:
        - gp: GP_model.
        - initial_step: Initial time step for animation.
        - num_steps: Number of time steps to simulate.
        """
        self.gp = gp
        self.num_steps = num_steps
        self.initial_step = initial_step

        # Initialize figure and lines for animation
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.lines = {
            "prediction": self.ax.plot([], [], label="Predicted Mean", color="red")[0],
            "observation": self.ax.plot([], [], label="Observation", color="blue")[0],
            "correction": self.ax.plot([], [], label="Corrected Mean", color="green")[0],
        }

        # Add fill_between for variance band (initially empty)
        self.variance_band = None

        # Set plot properties
        self.ax.legend()
        self.ax.set_xlim(0.0, 0.5)
        self.ax.set_ylim(0.0, 7.0)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Energy")
        self.ax.set_title("Function Evolution Over Time")

    def init_animation(self):
        """Initialize the animation by clearing all lines."""
        for line in self.lines.values():
            line.set_data([], [])

        # Clear the variance band if it exists
        if self.variance_band is not None:
            self.variance_band.remove()
            self.variance_band = None

        return list(self.lines.values())

    def animate(self, i):
        """
        Animation function for each frame.

        Parameters:
        - i: Current frame index.

        Steps:
        1. Show prediction.
        2. Show observation.
        3. Show correction.

        Each step is shown with a delay of half a second (500 ms).
        """
        # Determine which logical step (sub-frame) we are in
        step = i % 4  # Cycle through 0 (prediction), 1 (observation), 2 (correction)
        j = self.initial_step + i // 4  # Determine the actual time step

        if j >= len(self.gp.x_train):  # Stop if we exceed available data
            return list(self.lines.values())

        x_ = self.gp.x_train[j].T[0]

        if step == 0:  # First correction step
            if i == 0:
                # Initialize the current function for new animation
                self.current_function = self.gp.f_star[j].T[0].numpy().copy()
                self.lines["correction"].set_data(x_, self.current_function)
            # Update the title dynamically with the current time step
            self.ax.set_title(f"Function Evolution Over Time - Time Step: {j}")

            # Clear observation and correction temporarily
            self.lines["observation"].set_data([], [])
            self.lines["prediction"].set_data([], [])
            # Clear observation and prediction temporarily
            if self.variance_band is not None:
                self.variance_band.remove()
                self.variance_band = None

        elif step == 1:  # Prediction step

            predicted_f = np.linalg.matrix_power(self.gp.A[j].numpy(),20) @ self.current_function
            self.lines["prediction"].set_data(x_, predicted_f)

            variance = np.sqrt(torch.diag(self.gp.Gamma[j]).numpy()) * 1.96
            # Update variance band dynamically
            if self.variance_band is not None:
                self.variance_band.remove()  # Remove the old band
            lower_bound = predicted_f - variance
            upper_bound = predicted_f + variance
            self.variance_band = self.ax.fill_between(x_, lower_bound, upper_bound, color="red", alpha=0.2)

        elif step == 2:  # Observation step

            observed_f = self.gp.y_train[j].T[0].numpy()
            self.lines["observation"].set_data(x_, observed_f)

        elif step == 3:

            corrected_f = self.gp.f_star[j + 1].T[0].numpy().copy()
            self.lines["correction"].set_data(x_, corrected_f)

            # Update current function for the next iteration
            self.current_function = corrected_f

        return list(self.lines.values())

    def create_animation(self, output_filename="function_evolution.gif"):
        """
        Create and save the animation as a GIF.

        Parameters:
          - output_filename: Name of the output GIF file.

        Returns:
          None
          (The GIF will be saved to disk.)
        """

        ani = animation.FuncAnimation(
            self.fig,
            func=self.animate,
            init_func=self.init_animation,
            frames=self.num_steps,
            interval=400,
            blit=False,
            repeat=False,
        )

        ani.save(output_filename, writer="pillow")








