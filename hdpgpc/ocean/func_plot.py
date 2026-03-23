import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wavespectra import specarray
import numpy as np

def plot_cluster_spectrum_and_timeline(df, efth_ordered, n_clusters, norm=True):
    efth_max = efth_ordered.max().item()
    efth_min = efth_ordered.min().item()

    fig = plt.figure(figsize=(20, 20))
    gs = gridspec.GridSpec(n_clusters, 2, width_ratios=[0.1 ,1], hspace = 0.15,)

    for i in range(n_clusters):
        ax = plt.subplot(gs[i, 0], projection='polar')
        plt.sca(ax)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)

        da = efth_ordered.isel(cluster=n_clusters-1-i)
        nspecs = len(df[df['cluster'] == n_clusters-1-i])
        spec = specarray.SpecArray(da)

        spec.plot(
            kind="contourf",
            cmap='Spectral_r',
            add_colorbar=False,
            normalised=norm,
            show_theta_labels=False,
            show_radii_labels=False,
            vmin=efth_min,
            vmax=efth_max,
        )
        ax.set_title(f"Cluster {n_clusters-1-i} - {nspecs}", fontsize=15, x=-1, y = 0.38)

    # Panel de dispersión
    ax_scatter = plt.subplot(gs[:, -1])

    def rand_jitter(arr):
        stdev = 0.5
        return arr + np.random.randn(len(arr)) * stdev

    x_jittered = rand_jitter(df['day'].values)
    #y_jittered = rand_jitter(df['cluster'].values)

    ax_scatter.scatter(x_jittered, df['cluster'].values, s=15, color='black', alpha=0.5)

    month_start_days = df.groupby('month')['day'].min()
    month_labels_presentes = ['Xan', 'Feb', 'Mar', 'Abr', 'Mai', 'Xuñ', 'Xul', 'Ago', 'Set', 'Out', 'Nov', 'Dec']
    month_labels_filtrados = [month_labels_presentes[m - 1] for m in month_start_days.index]

    ax_scatter.set_xticks(month_start_days.values)
    ax_scatter.set_xticklabels(month_labels_filtrados)

    ax_scatter.set_yticks([])
    ax_scatter.set_xlabel("Mes")
    ax_scatter.set_title("Distribución diaria dos espectros por cluster")
    ax_scatter.grid(True, linestyle="--", alpha=0.3)
    
import numpy as np

def classify_partition_portilla(part, freq, dir, wspd, wdir, g=9.81):
    if part.sum() == 0:
        return "null"

    ipeak = np.unravel_index(np.argmax(part, axis=None), part.shape)
    f_peak = freq[ipeak[0]]
    theta_peak = dir[ipeak[1]]
    cp = g / (2 * np.pi * f_peak)
    Ueff = wspd * np.cos(np.radians(theta_peak - wdir))
    waveage = Ueff / cp
    
    if Ueff <= 0:
        return "swell"
    
    beta_min = cp / Ueff
    
    print(beta_min)

    if beta_min <= 1.3:
        return "wind sea"
    elif beta_min <= 2.0:
        return "old wind sea"
    else:
        return "swell"