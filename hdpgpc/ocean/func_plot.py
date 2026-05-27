import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from wavespectra import specarray
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import xarray as xr
import builtins
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter


def plot_cluster_spectrum_and_timeline(
        ds_subset,
        df,
        efth_ordered,
        windowed_data_no_direct,
        freq,
        n_clusters=None,
        norm=True,
        clusters_to_plot=None,
        ncols=1,
        include_direction=True,
        output_path=None,
):
    """
    Plot the notebook-style cluster summary.

    Parameters
    ----------
    ncols : int
        Number of cluster blocks per row. Clusters are arranged in
        column-major order:
            first column: 0, 1, 2, ...
            second column: next clusters, etc.
    include_direction : bool
        If True, each cluster block includes the direction-integrated spectrum.
        If False, only the polar mean spectrum and the frequency spectrum are
        shown.
    output_path : str or Path, optional
        If provided, save the figure to this path (e.g. "figure.pdf").
    """
    # If not specified, plot all clusters
    if clusters_to_plot is None:
        if n_clusters is None:
            clusters_to_plot = list(efth_ordered.cluster.values)
        else:
            clusters_to_plot = list(range(n_clusters))

    # Always use increasing cluster order
    clusters_to_plot = sorted([int(c) for c in list(clusters_to_plot)])
    n_plot = len(clusters_to_plot)
    ncols = max(1, int(ncols))
    nrows = int(np.ceil(n_plot / ncols)) if n_plot > 0 else 1

    # Determine the last occupied row in each column
    last_row_by_col = {}
    for panel in range(n_plot):
        row_tmp = panel % nrows
        col_tmp = panel // nrows
        last_row_by_col[col_tmp] = max(last_row_by_col.get(col_tmp, -1), row_tmp)

    # Modified colormap: white at the minimum
    base_cmap = plt.get_cmap("Spectral_r")
    colors = base_cmap(np.linspace(0.05, 1, 256))
    colors[0] = [1, 1, 1, 1]
    custom_cmap = LinearSegmentedColormap.from_list("custom_spectral", colors)

    efth_max = efth_ordered.max().item()
    y_max_freq = np.max(windowed_data_no_direct)
    y_max_dir = None
    if include_direction:
        y_max_dir = np.max(ds_subset.efth.integrate(coord="freq").values)

    # Figure sizing: smaller width because title no longer has its own left cell
    cell_width = 14.5 if include_direction else 10.5
    cell_height = 4.9
    fig = plt.figure(figsize=(cell_width * ncols, cell_height * nrows))
    outer_gs = gridspec.GridSpec(nrows, ncols, wspace=0.08, hspace=0.35)

    for panel, cluster_id in enumerate(clusters_to_plot):
        # Column-major order
        row = panel % nrows
        col = panel // nrows

        indices = df.index[df["cluster"] == cluster_id].to_numpy()
        nspecs = len(indices)

        # =========================
        # Inner layout per cluster
        # row 0 -> title
        # row 1 -> polar | frequency | direction(optional)
        # =========================
        if include_direction:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 3,
                subplot_spec=outer_gs[row, col],
                height_ratios=[0.18, 1.0],
                width_ratios=[1.0, 1.45, 1.45],
                hspace=0.05,
                wspace=0.12,
            )
        else:
            inner_gs = gridspec.GridSpecFromSubplotSpec(
                2, 2,
                subplot_spec=outer_gs[row, col],
                height_ratios=[0.18, 1.0],
                width_ratios=[1.0, 1.65],
                hspace=0.05,
                wspace=0.12,
            )

        # =========================
        # 0) Title spanning whole block
        # =========================
        if include_direction:
            ax_title = plt.subplot(inner_gs[0, :])
        else:
            ax_title = plt.subplot(inner_gs[0, :])

        ax_title.axis("off")
        ax_title.text(
            0.18,
            0.50,
            f"Cluster {cluster_id} ({nspecs})",
            ha="center",
            va="bottom",
            fontsize=22 if ncols > 1 else 24,
            fontweight="bold",
            transform=ax_title.transAxes,
        )

        # =========================
        # 1) Mean polar spectrum
        # =========================
        ax = plt.subplot(inner_gs[1, 0], projection="polar")
        plt.sca(ax)
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_thetagrids([0, 90, 180, 270], labels=["N", "E", "S", "W"])

        da = efth_ordered.sel(cluster=cluster_id).squeeze(drop=True)
        spec = specarray.SpecArray(da)

        show_theta = (row == 0)
        spec.plot(
            kind="contourf",
            cmap=custom_cmap,
            add_colorbar=False,
            normalised=norm,
            show_theta_labels=show_theta,
            show_radii_labels=False,
            levels=100,
            vmin=0.00086,
            vmax=efth_max,
        )

        # Remove automatic title such as "time = 0"
        ax.set_title("")

        for label in ax.get_xticklabels():
            label.set_fontsize(16 if ncols > 1 else 20)
            label.set_fontweight("bold")

        ax.xaxis.set_tick_params(pad=0)


        # =========================
        # 2) Frequency spectra
        # =========================
        ax = plt.subplot(inner_gs[1, 1])
        ax.tick_params(axis="both", labelsize=10 if ncols > 1 else 20)

        for j in indices:
            ax.plot(freq, windowed_data_no_direct[j], color="dodgerblue", alpha=0.2)

        if len(indices) > 0:
            ax.plot(
                freq,
                np.mean(windowed_data_no_direct[indices], axis=0),
                linewidth=1.5,
                color="red",
            )

        ax.set_ylim(-0.5, y_max_freq)
        ax.grid(alpha=0.15)

        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.set_yticklabels([])
        ax.set_ylabel("")

        # if row != last_row_by_col[col]:
        #     ax.set_xticklabels([])
        #     ax.set_xlabel("")
        # else:
        #     ax.set_xlabel(
        #         "Frecuency (Hz)",
        #         fontsize=12 if ncols > 1 else 25,
        #         fontweight="bold",
        #     )

        # =========================
        # 3) Direction spectra
        # =========================
        if include_direction:
            ax = plt.subplot(inner_gs[1, 2])
            ax.tick_params(axis="both", labelsize=10 if ncols > 1 else 20)

            da_dir = (
                ds_subset.where(ds_subset.cluster_label == cluster_id, drop=True)
                .efth
                .integrate(coord="freq")
            )

            if da_dir.sizes.get("time", 0) > 0:
                for j in range(len(da_dir.time)):
                    ax.plot(da_dir.dir.values, da_dir.values[j], color="dodgerblue", alpha=0.2)

                ax.plot(
                    da_dir.dir.values,
                    np.mean(da_dir.values, axis=0),
                    linewidth=1.5,
                    color="red",
                )

            # Exponential-style y ticks in units of 1e-4
            tick_step = 2.5e-4
            y_top = np.ceil(y_max_dir / tick_step) * tick_step

            ax.set_ylim(0, y_top)
            ax.set_yticks(np.arange(0, y_top + 0.5 * tick_step, tick_step))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y / 1e-4:g}"))

            ax.text(
                0.02,
                1.02,
                r"$\times 10^{-4}$",
                transform=ax.transAxes,
                fontsize=9 if ncols > 1 else 16,
                ha="left",
                va="bottom",
            )

            ax.grid(alpha=0.15)

            ax.set_xticklabels([])
            ax.set_xlabel("")
            ax.set_yticklabels([])
            ax.set_ylabel("")

            # if row != last_row_by_col[col]:
            #     ax.set_xticklabels([])
            #     ax.set_xlabel("")
            # else:
            #     ax.set_xlabel(
            #         "Direction (º)",
            #         fontsize=12 if ncols > 1 else 25,
            #         fontweight="bold",
            #     )

    # Hide unused cluster cells
    occupied = {
        (panel % nrows, panel // nrows)
        for panel in range(n_plot)
    }

    for row in range(nrows):
        for col in range(ncols):
            if (row, col) not in occupied:
                ax = plt.subplot(outer_gs[row, col])
                ax.axis("off")

    plt.tight_layout()

    if output_path is not None:
        fig.savefig(
            output_path,
            format="pdf" if str(output_path).lower().endswith(".pdf") else None,
            bbox_inches="tight",
            dpi=300,
        )


from wavespectra.partition import specpart
from wavespectra.core.utils import (
    set_spec_attributes,
    regrid_spec,
    smooth_spec,
    check_same_coordinates,
    D2R,
    celerity,
    is_overlap,
    waveage,
)

import numpy as np


def hs_func(spec, freq, dir):
    df = np.diff(freq).mean()
    dtheta = np.deg2rad(np.diff(dir).mean())
    m0 = np.sum(spec) * df * dtheta
    return 4 * np.sqrt(m0)


def get_partition_centroid(part, freq, dir):
    energy = part.sum()
    if energy == 0:
        return 0.0, 0.0
    fx = np.sum(part * freq[:, None]) / energy
    dx = np.sum(part * dir[None, :]) / energy
    return fx, dx


def find_closest_partition(i, centroids, valid_idxs):
    fi, di = centroids[i]
    min_dist = float("inf")
    closest = None
    for j in valid_idxs:
        if j == i:
            continue
        fj, dj = centroids[j]
        dist = (fi - fj) ** 2 + (di - dj) ** 2
        if dist < min_dist:
            min_dist = dist
            closest = j
    return closest


def np_portilla(
        spectrum,
        spectrum_smooth,
        freq,
        dir,
        wspd,
        wdir,
        dpt,
        swells=None,
        max_parts=10,
        ihmax=500,
        threshold_energy=0.02,
        combine_energy=0.05,
        max_filter_iter=1,
        freq_window=3,
        dir_window=3
):
    total_energy = spectrum.sum()
    filtered = spectrum_smooth.copy()

    for _ in range(max_filter_iter):
        watershed_map = specpart.partition(filtered.values.astype(np.float32), ihmax)
        nparts = watershed_map.max()

        parts = []
        centroids = []
        energies = []

        # Paso 1: gardar todas as particións (sen filtrar)
        for ipart in range(nparts):
            part = np.where(watershed_map == ipart + 1, spectrum, 0.0)
            energy = part.sum()
            parts.append(part)
            centroids.append(get_partition_centroid(part, freq, dir))
            energies.append(energy)

        # Paso 2: combinar particións de baixa enerxía (< threshold_energy)
        valid_idxs = list(range(len(parts)))
        for i in valid_idxs[:]:
            if energies[i] / total_energy < threshold_energy:
                j = find_closest_partition(i, centroids, valid_idxs)
                if j is not None:
                    parts[j] += parts[i]
                    energies[j] += energies[i]
                    valid_idxs.remove(i)

        if len(valid_idxs) <= max_parts:
            break
        else:
            filtered = smooth_spec(filtered, freq_window, dir_window)

    # Paso 5: combinar particións de baixa enerxía (< combine_energy)
    for i in valid_idxs[:]:
        if energies[i] / total_energy < combine_energy:
            j = find_closest_partition(i, centroids, valid_idxs)
            if j is not None:
                parts[j] += parts[i]
                energies[j] += energies[i]
                valid_idxs.remove(i)

    final_parts = [parts[i] for i in valid_idxs]
    labels = []
    classified_parts = []
    fpeaks = []

    for part in final_parts:
        label, pico = classify_partition_portilla(part, freq, dir, wspd, wdir)
        if label != "null":
            classified_parts.append(part)
            labels.append(label)
            fpeaks.append(pico)

    # Separar mar de fondo e mar de vento antigo
    swell_fondo = [(p, l) for p, l in zip(classified_parts, labels) if l == "Mar de fondo"]
    swell_antigo = [(p, l) for p, l in zip(classified_parts, labels) if l == "Mar de transición"]

    # Ordenar cada grupo por Hs de maior a menor
    swell_fondo_sorted = sorted(swell_fondo, key=lambda x: -hs_func(x[0], freq, dir))
    swell_antigo_sorted = sorted(swell_antigo, key=lambda x: -hs_func(x[0], freq, dir))

    # Unir mantendo a orde conceptual
    swell_parts = [p for p, _ in swell_fondo_sorted + swell_antigo_sorted]
    swell_labels = [l for _, l in swell_fondo_sorted + swell_antigo_sorted]

    windsea = [p for p, l in zip(classified_parts, labels) if l == "Mar de vento"]
    windsea_part = np.sum(windsea, axis=0) if windsea else np.zeros_like(spectrum)
    # windsea_part = windsea[0] if windsea else np.zeros_like(spectrum)

    if swells is not None:
        if len(swell_parts) > swells:
            swell_parts = swell_parts[:swells]
            swell_labels = swell_labels[:swells]
        elif len(swell_parts) < swells:
            for _ in range(swells - len(swell_parts)):
                swell_parts.append(np.zeros_like(spectrum))
                swell_labels.append("null")

    all_parts = []
    all_labels = []

    # Engadir primeiro o mar de fondo (swell)
    all_parts.extend(swell_parts)
    all_labels.extend(swell_labels)

    # Logo o mar de vento (se existe)
    if windsea:
        all_parts.append(windsea_part)
        all_labels.append("Mar de vento")
    return np.array(all_parts), all_labels, sorted(fpeaks)


def ptm_portilla(
        partition,
        wspd,
        wdir,
        dpt,
        smooth=True,
        freq_window=3,
        dir_window=3,
        ihmax=10000
):
    check_same_coordinates(wspd, wdir, dpt)

    if smooth:
        dset_smooth = smooth_spec(partition.dset, freq_window, dir_window)
    else:
        dset_smooth = partition.dset

    partitions, labels, picos = np_portilla(
        spectrum=partition.dset.values,
        spectrum_smooth=dset_smooth,
        freq=partition.dset.freq.values,
        dir=partition.dset.dir.values,
        wspd=float(wspd),
        wdir=float(wdir),
        dpt=float(dpt),
        swells=None,
        ihmax=ihmax,
    )

    da = xr.DataArray(
        partitions,
        coords={
            "part": np.arange(len(partitions)),
            "freq": partition.dset.freq,
            "dir": partition.dset.dir,
            "label": ("part", labels),
        },
        dims=("part", "freq", "dir"),
        name="efth"
    )

    return da.to_dataset(), picos


import numpy as np


def classify_partition_portilla(part, freq, dir, wspd, wdir, g=9.81):
    '''if part.sum() == 0:
        return "null"

    ipeak = np.unravel_index(np.argmax(part, axis=None), part.shape)
    f_peak = freq[ipeak[0]]
    print(f'Peak freq: {f_peak}')
    theta_peak = dir[ipeak[1]]
    cp = g / (2 * np.pi * f_peak)
    Ueff = wspd * np.cos(np.radians(theta_peak - wdir))
    waveage = Ueff / cp
    print(f'Angle: {np.cos(np.radians(theta_peak - wdir))}')

    if np.cos(np.radians(theta_peak - wdir)) <= 0:
        #return "swell"

    beta_min = cp / Ueff

    print(beta_min)

    if beta_min < 1.3 and beta_min > 0:
        print("WIND")
        return "wind sea"
    elif beta_min >= 1.3 and beta_min <= 2.0:
        print("OLD")
        return "old wind sea"
    else:
        print("SWELL")
        return "swell"'''
    if part.sum() == 0:
        return "null"

    # ---------- PARÁMETROS MEDIOS ----------
    ipeak = np.unravel_index(np.argmax(part, axis=None), part.shape)
    f_peak = freq[ipeak[0]]
    part_energy = part.sum()
    f_mean = (part.sum(axis=1) * freq).sum() / part_energy
    theta_mean = (part.sum(axis=0) * dir).sum() / part_energy
    cp_mean = g / (2 * np.pi * f_mean)
    angle_mean = np.cos(np.radians(theta_mean - wdir))
    Ueff_mean = wspd * angle_mean
    beta_min = cp_mean / Ueff_mean if Ueff_mean > 0 else np.inf

    if angle_mean <= 0:
        return "Mar de fondo", f_peak

    if beta_min <= 1.3 and beta_min > 0:
        return "Mar de vento", f_peak
    elif beta_min > 1.3 and beta_min <= 2.0:
        return "Mar de transición", f_peak
    else:
        return "Mar de fondo", f_peak
    if part.sum() == 0:
        return "null"


'''def plot_spectra_for_cluster_series(ds_cluster_means, ds_subset):
    # Colormap modificado: branco no mínimo
    base_cmap = plt.get_cmap('Spectral_r')
    colors = base_cmap(np.linspace(0.05, 1, 256))  # Cortamos a parte inferior
    colors[0] = [1, 1, 1, 1]  # Branco no primeiro valor
    custom_cmap = LinearSegmentedColormap.from_list("custom_spectral", colors)
    n_clusters = ds_cluster_means.sizes['cluster']

    fig = plt.figure(figsize=(30, 25))

    # Primeiro determinamos o número máximo de particións
    max_parts = 0
    dsparts = []
    for n in range(n_clusters):
        dspart, pico = ptm_portilla(
            partition=ds_cluster_means.isel(cluster=n).spec.partition,
            wspd=float(ds_cluster_means.wspd.isel(cluster=n).values),
            wdir=float(ds_cluster_means.wdir.isel(cluster=n).values),
            dpt=33.0,
            smooth=True
        )
        n_parts = len(dspart.part)
        dsparts.append(dspart)
        max_parts = builtins.max(max_parts, n_parts)

    # Crear grid global
    outer_gs = gridspec.GridSpec(n_clusters, 1, hspace=0.7)

    all_efths = []
    all_ef1d_max = []
    picos = []

    for n in range(n_clusters):
        dspart, pico = ptm_portilla(
            partition=ds_cluster_means.isel(cluster=n).spec.partition,
            wspd=float(ds_cluster_means.wspd.isel(cluster=n).values),
            wdir=float(ds_cluster_means.wdir.isel(cluster=n).values),
            dpt=33.0,
            smooth=True
        )
        all_efths.append(dspart.efth)
        ef1d = dspart.spec.oned()
        all_ef1d_max.append(ef1d.max().item())
        dsparts.append(dspart)
        picos.append(pico)

    # Concatenar espectros e obter valores globais
    efth_concat = xr.concat(all_efths, dim="part")
    vmin = efth_concat.min().item()
    vmax = efth_concat.max().item()
    ymax = builtins.max(all_ef1d_max)

    for n in range(n_clusters):
        dspart = dsparts[n_clusters-1-n]
        pico = picos[n_clusters-1-n]
        ef_1d = dspart.spec.oned()
        ef_dir = dspart.efth.integrate(coord='freq').values
        freq = dspart.spec.freq.values
        labels = dspart.label.values
        nparts = dspart.part.size

        # Grid interno: 2 columnas por partición, ata o máximo
        gs = gridspec.GridSpecFromSubplotSpec(1, 3 * max_parts, subplot_spec=outer_gs[n], wspace=0.5, width_ratios=[1, 1, 1] * max_parts)

        for i in range(nparts):
            da = dspart.isel(part=i)
            spec = da.spec
            picopart = pico[i]

            fmin, fmax = detectar_banda_activa(ef_1d[0], freq)
            ef_dir = dspart.isel(part=i).efth.integrate(coord='freq').values
            dmin, dmax = detectar_banda_activa_dir(ef_dir, ds_subset.dir.values)

            cluster_id = n_clusters - 1 - n
            mask_cluster = ds_subset.cluster_label == cluster_id 
            mask_cluster = ds_subset.isel(time=mask_cluster)

            # Gráfico polar
            ax_polar = fig.add_subplot(gs[0, 3 * i], projection='polar')
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            spec.plot(
                kind="contourf",
                cmap=custom_cmap,
                add_colorbar=False,
                normalised=False,
                show_theta_labels=False,
                show_radii_labels=False,
                levels=100,
                vmax=vmax,
                vmin=0.00086
            )
            ax_polar.set_title(labels[i], fontsize=18, pad=10, fontweight='bold')
            ax_polar.tick_params(axis='y', colors='black')
            for label in ax_polar.get_yticklabels():
                label.set_color('black')
                label.set_fontweight('bold')
                label.set_fontsize(12)

            if i == 0:
                ax_polar.text(-1.6, 0.5, f"Cluster {n_clusters - 1 - n}", fontsize=18, ha='left', va='center', fontweight='bold', transform=ax_polar.transAxes)

            if np.isclose(fmax, 0.485):
                dir_mask = (mask_cluster.dir >= builtins.min(mask_cluster.dir)) & (mask_cluster.dir <= builtins.max(mask_cluster.dir))
                freq_mask = (mask_cluster.freq >= builtins.min(mask_cluster.freq)) & (mask_cluster.freq <= builtins.max(mask_cluster.freq))
            else:
                # Dúas particións → usar cortes distintos segundo i
                if i == 0:
                    dir_mask = (mask_cluster.dir >= dmin) & (mask_cluster.dir <= dmax)
                    freq_mask = (mask_cluster.freq >= fmin) & (mask_cluster.freq <= fmax)
                elif i == 1:
                    dir_mask = (mask_cluster.dir >= dmin) & (mask_cluster.dir <= dmax)
                    freq_mask = (mask_cluster.freq > fmax)  # a partir de fmax cara arriba
                else:
                    freq_mask = None  # por seguridade 

            # Gráfico unidimensional
            ax_line = fig.add_subplot(gs[0, 3 * i + 1])
            ax_line.tick_params(axis='both', labelsize=15)
            # Cálculo só se hai máscara válida
            if freq_mask is not None:

                masked = mask_cluster.spec.oned()
                masked.loc[:, ~freq_mask] = 0.0
                ef1d_all = masked

                for j in range(len(ef1d_all)):
                    ax_line.plot(freq, ef1d_all[j], color='dodgerblue', alpha=0.2)

                ax_line.plot(freq, ef_1d[i], color='red', linewidth=1.5)
                ax_line.text(
                    1.1, 1.2,
                    f"Frecuencia pico = {picopart:.3f} Hz",
                    ha='right', va='top',
                    transform=ax_line.transAxes,
                    fontsize=15, fontweight='bold'
                )
                ax_line.set_ylim(0, ymax+3)
            else:
                # Fallback en caso raro
                ax_line.plot(freq, ef1d_part, color='navy', linewidth=1)
                ax_line.fill_between(freq, ef1d_part, color='cornflowerblue', alpha=1)

                fig.savefig("parts.png", dpi=100, bbox_inches="tight")
                plt.show()


            # Gráfico unidimensional
            ax_line = fig.add_subplot(gs[0, 3 * i + 2])
            ax_line.tick_params(axis='both', labelsize=15)
            mask_cluster = ds_subset.isel(time=ds_subset.cluster_label == cluster_id)

            # Máscara de frecuencia e dirección
            #freq_mask = (mask_cluster.freq >= fmin) & (mask_cluster.freq <= fmax)
#            dir_mask = (mask_cluster.dir >= dmin) & (mask_cluster.dir <= dmax)

            # —— FILTRAR EN FRECUENCIA ANTES DE INTEGRAR ——
            # Poñer a cero as frecuencias fóra do rango da partición
            efth_filtrado_freq = mask_cluster.efth.where(freq_mask, 0.0)

            # Agora integrar en frecuencia → queda E(theta)
            ef_theta_all = efth_filtrado_freq.integrate(coord='freq')  # dims: time x dir

            # Aplicar máscara de dirección (poñer a 0 onde non hai enerxía na banda angular)
            ef_theta_masked = ef_theta_all.where(dir_mask, 0.0)

            # Extraer datos como array (tempo x dirección)
            valores = ef_theta_masked.values

            for j in range(len(ef1d_all)):
                ax_line.plot(mask_cluster.dir.values, valores[j], color='dodgerblue', alpha=0.2)

            ax_line.plot(mask_cluster.dir.values, np.mean(valores, axis=0), color='red', linewidth=1.5)

            idx_max = np.argmax(np.mean(valores, axis=0))
            direccion_dominante = mask_cluster.dir.values[idx_max]

            ax_line.text(
                    1.4, 1.2,
                    f"Dirección dominante = {direccion_dominante:.0f}º",
                    ha='right', va='top',
                    transform=ax_line.transAxes,
                    fontsize=15, fontweight='bold'
                )


            ax_line.set_ylim(0, np.max(ds_subset.efth.integrate(coord='freq').values)*1.1)'''

import ipywidgets as widgets
from IPython.display import display


def plot_spectra_for_cluster_series(ds_cluster_means, ds_subset, clusters_to_plot=None):
    # Colormap modificado: branco no mínimo
    base_cmap = plt.get_cmap('Spectral_r')
    colors = base_cmap(np.linspace(0.05, 1, 256))  # Cortamos a parte inferior
    colors[0] = [1, 1, 1, 1]  # Branco no primeiro valor
    custom_cmap = LinearSegmentedColormap.from_list("custom_spectral", colors)

    if clusters_to_plot is None:
        clusters_to_plot = list(ds_cluster_means.cluster.values)

    n_plot = len(clusters_to_plot)

    fig = plt.figure(figsize=(30, 10))

    # Primeiro determinamos o número máximo de particións
    max_parts = 0
    dsparts = []
    for cluster_id in clusters_to_plot:
        dspart, pico = ptm_portilla(
            partition=ds_cluster_means.sel(cluster=cluster_id).spec.partition,
            wspd=float(ds_cluster_means.wspd.sel(cluster=cluster_id).values),
            wdir=float(ds_cluster_means.wdir.sel(cluster=cluster_id).values),
            dpt=33.0,
            smooth=True
        )
        n_parts = len(dspart.part)
        dsparts.append(dspart)
        max_parts = builtins.max(max_parts, n_parts)

    # Crear grid global
    outer_gs = gridspec.GridSpec(n_plot, 1, hspace=0.7)

    all_efths = []
    all_ef1d_max = []
    picos = []

    for cluster_id in clusters_to_plot:
        dspart, pico = ptm_portilla(
            partition=ds_cluster_means.sel(cluster=cluster_id).spec.partition,
            wspd=float(ds_cluster_means.wspd.sel(cluster=cluster_id).values),
            wdir=float(ds_cluster_means.wdir.sel(cluster=cluster_id).values),
            dpt=33.0,
            smooth=True
        )
        all_efths.append(dspart.efth)
        ef1d = dspart.spec.oned()
        all_ef1d_max.append(ef1d.max().item())
        dsparts.append(dspart)
        picos.append(pico)

    # Concatenar espectros e obter valores globais
    efth_concat = xr.concat(all_efths, dim="part")
    vmin = efth_concat.min().item()
    vmax = efth_concat.max().item()
    ymax = builtins.max(all_ef1d_max)

    for n in range(n_plot):
        dspart = dsparts[n_plot - 1 - n]
        pico = picos[n_plot - 1 - n]
        cluster_id = clusters_to_plot[n_plot - 1 - n]

        ef_1d = dspart.spec.oned()
        ef_dir = dspart.efth.integrate(coord='freq').values
        freq = dspart.spec.freq.values
        labels = dspart.label.values
        nparts = dspart.part.size

        # Grid interno: 2 columnas por partición, ata o máximo
        gs = gridspec.GridSpecFromSubplotSpec(
            1, 3 * max_parts,
            subplot_spec=outer_gs[n],
            wspace=0.5,
            width_ratios=[1, 1, 1] * max_parts
        )

        for i in range(nparts):
            da = dspart.isel(part=i)
            spec = da.spec
            picopart = pico[i]

            fmin, fmax = detectar_banda_activa(ef_1d[0], freq)
            ef_dir = dspart.isel(part=i).efth.integrate(coord='freq').values
            dmin, dmax = detectar_banda_activa_dir(ef_dir, ds_subset.dir.values)

            mask_cluster = ds_subset.cluster_label == cluster_id
            mask_cluster = ds_subset.isel(time=mask_cluster)

            # Gráfico polar
            ax_polar = fig.add_subplot(gs[0, 3 * i], projection='polar')
            ax_polar._cluster_id = cluster_id
            ax_polar.set_theta_zero_location('N')
            ax_polar.set_theta_direction(-1)
            spec.plot(
                kind="contourf",
                cmap=custom_cmap,
                add_colorbar=False,
                normalised=False,
                show_theta_labels=False,
                show_radii_labels=False,
                levels=100,
                vmax=0.025,
                vmin=0.00086
            )
            ax_polar.set_title(labels[i], fontsize=25, pad=10, fontweight='bold')
            ax_polar.tick_params(axis='y', colors='black')
            for label in ax_polar.get_yticklabels():
                label.set_color('black')
                label.set_fontweight('bold')
                label.set_fontsize(12)

            if i == 0:
                ax_polar.text(
                    -1, 0.2, f"Cluster {cluster_id}",
                    fontsize=20, ha='left', va='center',
                    fontweight='bold', transform=ax_polar.transAxes
                )

            if np.isclose(fmax, 0.485):
                dir_mask = (mask_cluster.dir >= builtins.min(mask_cluster.dir)) & (
                            mask_cluster.dir <= builtins.max(mask_cluster.dir))
                freq_mask = (mask_cluster.freq >= builtins.min(mask_cluster.freq)) & (
                            mask_cluster.freq <= builtins.max(mask_cluster.freq))
            else:
                # Dúas particións → usar cortes distintos segundo i
                if i == 0:
                    dir_mask = (mask_cluster.dir >= dmin) & (mask_cluster.dir <= dmax)
                    freq_mask = (mask_cluster.freq >= fmin) & (mask_cluster.freq <= fmax)
                elif i == 1:
                    dir_mask = (mask_cluster.dir >= dmin) & (mask_cluster.dir <= dmax)
                    freq_mask = (mask_cluster.freq > fmax)  # a partir de fmax cara arriba
                else:
                    freq_mask = None  # por seguridade

            # Gráfico unidimensional
            ax_line = fig.add_subplot(gs[0, 3 * i + 1])
            ax_line._cluster_id = cluster_id
            ax_line.tick_params(axis='both', labelsize=15)

            if freq_mask is not None:
                masked = mask_cluster.spec.oned()
                masked.loc[:, ~freq_mask] = 0.0
                ef1d_all = masked

                for j in range(len(ef1d_all)):
                    ax_line.plot(freq, ef1d_all[j], color='dodgerblue', alpha=0.2)

                ax_line.plot(freq, ef_1d[i], color='red', linewidth=1.5)
                ax_line.text(
                    1.2, -0.15,
                    f"Frecuencia pico = {picopart:.3f} Hz",
                    ha='right', va='top',
                    transform=ax_line.transAxes,
                    fontsize=20, fontweight='bold'
                )
                ax_line.set_ylim(0, ymax + 3)
            else:
                ax_line.plot(freq, ef1d_part, color='navy', linewidth=1)
                ax_line.fill_between(freq, ef1d_part, color='cornflowerblue', alpha=1)

                fig.savefig("parts.png", dpi=100, bbox_inches="tight")
                plt.show()

            # Gráfico unidimensional
            ax_line = fig.add_subplot(gs[0, 3 * i + 2])
            ax_line._cluster_id = cluster_id
            ax_line.tick_params(axis='both', labelsize=15)
            mask_cluster = ds_subset.isel(time=ds_subset.cluster_label == cluster_id)

            # —— FILTRAR EN FRECUENCIA ANTES DE INTEGRAR ——
            efth_filtrado_freq = mask_cluster.efth.where(freq_mask, 0.0)

            # Agora integrar en frecuencia → queda E(theta)
            ef_theta_all = efth_filtrado_freq.integrate(coord='freq')  # dims: time x dir

            # Aplicar máscara de dirección
            ef_theta_masked = ef_theta_all.where(dir_mask, 0.0)

            # Extraer datos como array (tempo x dirección)
            valores = ef_theta_masked.values

            for j in range(len(ef1d_all)):
                ax_line.plot(mask_cluster.dir.values, valores[j], color='dodgerblue', alpha=0.2)

            ax_line.plot(mask_cluster.dir.values, np.mean(valores, axis=0), color='red', linewidth=1.5)

            idx_max = np.argmax(np.mean(valores, axis=0))
            direccion_dominante = mask_cluster.dir.values[idx_max]

            ax_line.text(
                1.5, -0.15,
                f"Dirección dominante = {direccion_dominante:.0f}º",
                ha='right', va='top',
                transform=ax_line.transAxes,
                fontsize=20, fontweight='bold'
            )

            ax_line.set_ylim(0, np.max(ds_subset.efth.integrate(coord='freq').values) * 1.1)


def detectar_banda_activa(ef1d_part, freq, umbral=1e-6, min_gap=2):
    # Constrúe máscara binaria onde hai enerxía
    mask = (ef1d_part > umbral).astype(int).values

    # Detectar bloques de 1s separados por ceros longos
    start = None
    end = None
    inside = False
    gap = 0

    for i, val in enumerate(mask):
        if val == 1:
            if not inside:
                start = i
                inside = True
            gap = 0
            end = i
        elif inside:
            gap += 1
            if gap >= min_gap:
                break  # cortar se hai varios ceros seguidos

    if start is not None and end is not None:
        return freq[start], freq[end]
    else:
        return None, None  # sen zona válida


def detectar_banda_activa_dir(ef_theta, dirs, umbral=1e-3, min_gap=2, percentil=0.80):
    energia_total = ef_theta.sum()
    cumsum = np.cumsum(ef_theta)
    cumsum_norm = cumsum / energia_total

    for i_start in range(len(cumsum)):
        for i_end in range(i_start + 1, len(cumsum)):
            if cumsum_norm[i_end] - cumsum_norm[i_start] >= percentil:
                return dirs[i_start], dirs[i_end]
    return None, None


import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from plotly.offline import plot


def plot_spectral_grid_mean(cluster_means, frequencies, directions):
    """
    Plots 3D spectral energy distributions using cluster mean tensors.

    Parameters:
        cluster_means (list of np.ndarray): List of K tensors representing cluster means.
        frequencies (np.ndarray): Frequency array corresponding to the clusters.
        directions (np.ndarray): Direction array corresponding to the clusters.
    """
    num_clusters = len(cluster_means)
    cols = min(2, num_clusters)
    rows = (num_clusters // cols) + (num_clusters % cols > 0)

    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=[f'Cluster {i__ + 1}' for i__ in range(num_clusters)],
        specs=[[{'type': 'surface'}] * cols for _ in range(rows)]
    )

    # Ensure directions include both 0° and 360°
    # Add 360° to the directions array
    directions_rad = np.radians(directions)  # Convert to radians
    print(len(directions_rad))

    # Extend cluster_means by repeating the values for 0° at 360°

    # Generate meshgrid for plotting
    Freq, Theta = np.meshgrid(frequencies, directions_rad)
    X = Freq * np.cos(Theta)  # Use Freq as radial and Theta as angular
    Y = Freq * np.sin(Theta)

    # Loop through each cluster mean tensor and plot
    for idx, cluster_mean in enumerate(cluster_means):
        row, col = divmod(idx, cols)

        Z = cluster_mean.T  # Transpose for correct orientation

        surface = go.Surface(
            x=X,
            y=-Y,
            z=Z,
            colorscale='Spectral_r',
            opacity=1,
            showlegend=False,
            showscale=False
        )
        fig.add_trace(surface, row=row + 1, col=col + 1)

        # Add circles and directional labels
        max_frequency = np.max(frequencies)
        circle_radii = [0.75 * max_frequency, 0.5 * max_frequency, 0.25 * max_frequency]

        for freq in circle_radii:
            circle_x = freq * np.cos(np.linspace(0, 2 * np.pi, 100))
            circle_y = freq * np.sin(np.linspace(0, 2 * np.pi, 100))
            fig.add_trace(go.Scatter3d(
                x=circle_x,
                y=circle_y,
                z=np.zeros_like(circle_x),
                mode='lines',
                line=dict(color='black', width=0.5, dash='dash'),
                showlegend=False
            ), row=row + 1, col=col + 1)

        border_x = max_frequency * np.cos(np.linspace(0, 2 * np.pi, 100))
        border_y = max_frequency * np.sin(np.linspace(0, 2 * np.pi, 100))
        fig.add_trace(go.Scatter3d(
            x=border_x,
            y=border_y,
            z=np.zeros_like(border_x),
            mode='lines',
            line=dict(color='black', width=1),
            showlegend=False
        ), row=row + 1, col=col + 1)

        for angle in range(0, 360, 45):
            rad = np.radians(-angle)
            x = max_frequency * np.cos(rad)
            y = max_frequency * np.sin(rad)

            fig.add_trace(go.Scatter3d(
                x=[0, x],
                y=[0, y],
                z=[0, 0],
                mode='lines',
                line=dict(color='black', width=0.5, dash='dash'),
                showlegend=False
            ), row=row + 1, col=col + 1)
            fig.add_trace(go.Scatter3d(
                x=[x],
                y=[y],
                z=[0],
                mode='text',
                text=[f"{angle}°"],
                showlegend=False,
                textposition='bottom center'
            ), row=row + 1, col=col + 1)

    fig.update_layout(
        height=800 * rows,
        width=600 * cols,
        title_text='Cluster Mean Spectral Energy Distributions',
        showlegend=False,
    )

    fixed_camera = dict(
        eye=dict(x=1.25, y=1.25, z=0.7),
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0)
    )

    for i in range(1, num_clusters + 1):
        fig.update_scenes(
            xaxis=dict(visible=False, showgrid=False, zeroline=False),
            yaxis=dict(visible=False, showgrid=False, zeroline=False),
            zaxis=dict(visible=False, showgrid=False, zeroline=False),
            camera=fixed_camera,
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            zaxis_range=[0, np.max(cluster_means)],
            bgcolor='rgba(0,0,0,0)',
            row=(i - 1) // cols + 1,
            col=(i - 1) % cols + 1
        )

    fig.show()


def plot_single_spectrum(spectrum, frequencies, directions, title='', zmin=0, zmax=0.02):
    """
    Plots a single 2D wave spectrum in polar coordinates (3D surface).

    Parameters:
        spectrum (np.ndarray): 2D array with shape (n_freqs, n_dirs), spectral energy.
        frequencies (np.ndarray): 1D array of frequencies (length n_freqs).
        directions (np.ndarray): 1D array of directions in degrees (length n_dirs).
        title (str): Title for the plot.
    """

    base_cmap = plt.get_cmap('Spectral_r')
    colors = base_cmap(np.linspace(0.05, 1, 256))
    colors[0] = [1, 1, 1, 1]  # Branco
    custom_cmap = LinearSegmentedColormap.from_list("custom_spectral", colors)

    plotly_colorscale = matplotlib_to_plotly(custom_cmap)

    # Convert directions to radians
    directions_ext = np.append(directions, 360)

    spectrum_ext = np.concatenate([spectrum, spectrum[:, [0]]], axis=1)
    directions_rad = np.radians(directions_ext)

    # Meshgrid for polar coordinates
    Freq, Theta = np.meshgrid(frequencies, directions_rad)
    X = Freq * np.cos(Theta)
    Y = Freq * np.sin(Theta)
    Z = spectrum_ext.T  # Transpose for plotting

    fig = go.Figure()

    # Add the 3D surface

    fig.add_trace(go.Surface(
        x=X, y=-Y, z=Z,
        colorscale=plotly_colorscale,
        opacity=1,
        cmin=zmin,  # ← aquí
        cmax=zmax,
        showscale=False
    ))

    radius = max(frequencies)

    # X e Y do círculo
    theta = np.linspace(0, 2 * np.pi, 300)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    # Engadir o círculo en z=0
    fig.add_trace(go.Scatter3d(
        x=circle_x,
        y=-circle_y,  # manter orientación habitual
        z=np.zeros_like(circle_x),
        mode='lines',
        line=dict(color='black', width=1),
        showlegend=False
    ))

    # Layout
    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=600,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False, range=[0, zmax]),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=0.4),
            camera=dict(eye=dict(x=1.25, y=1.25, z=0.7)),
            bgcolor='rgba(0,0,0,0)'
        )
    )

    fig.show()


import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgba


def matplotlib_to_plotly(cmap, n_colors=256):
    colors = [cmap(i / (n_colors - 1)) for i in range(n_colors)]

    return [
        [i / (n_colors - 1), f'rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})']
        for i, (r, g, b, a) in enumerate(colors)
    ]
