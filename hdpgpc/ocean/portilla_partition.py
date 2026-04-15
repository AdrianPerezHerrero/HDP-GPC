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
        dist = (fi - fj)**2 + (di - dj)**2
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

    for part in final_parts:
        label = classify_partition_portilla(part, freq, dir, wspd, wdir)
        if label != "null":
            classified_parts.append(part)
            labels.append(label)

    swell_parts = [p for p, l in zip(classified_parts, labels) if l != "wind sea"]
    swell_labels = [l for l in labels if l != "wind sea"]
    hs_vals = [-hs_func(p, freq, dir) for p in swell_parts]
    isort = np.argsort(hs_vals)
    swell_parts = list(np.array(swell_parts)[isort])
    swell_labels = list(np.array(swell_labels)[isort])

    print(labels)
    windsea = [p for p, l in zip(classified_parts, labels) if l == "wind sea"]
    windsea_part = np.sum(windsea, axis=0) if windsea else np.zeros_like(spectrum)
    #windsea_part = windsea[0] if windsea else np.zeros_like(spectrum)

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

    if windsea:  # só se hai wind sea
        all_parts.append(windsea_part)
        all_labels.append("wind sea")

    all_parts.extend(swell_parts)
    all_labels.extend(swell_labels)
    return np.array(all_parts), all_labels



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

    partitions, labels = np_portilla(
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

    return da.to_dataset()

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

    # ---------- PARÁMETROS DO PICO ----------
    ipeak = np.unravel_index(np.argmax(part, axis=None), part.shape)
    f_peak = freq[ipeak[0]]
    theta_peak = dir[ipeak[1]]
    cp_peak = g / (2 * np.pi * f_peak)
    angle_peak = np.cos(np.radians(theta_peak - wdir))
    Ueff_peak = wspd * angle_peak
    beta_peak = cp_peak / Ueff_peak if Ueff_peak > 0 else np.inf

    # ---------- PARÁMETROS MEDIOS ----------
    part_energy = part.sum()
    f_mean = (part.sum(axis=1) * freq).sum() / part_energy
    theta_mean = (part.sum(axis=0) * dir).sum() / part_energy
    cp_mean = g / (2 * np.pi * f_mean)
    angle_mean = np.cos(np.radians(theta_mean - wdir))
    Ueff_mean = wspd * angle_mean
    beta_mean = cp_mean / Ueff_mean if Ueff_mean > 0 else np.inf

    # ---------- CLASIFICACIÓN ----------
    if angle_peak <= 0 and angle_mean <= 0:
        return "swell"

    if 0 < beta_peak < 1.3:
        return "wind sea"
    elif 1.3 <= beta_peak <= 2.0:
        return "old wind sea"
    elif beta_peak > 2.0:
        if 1.3 <= beta_mean <= 2.0 and angle_mean > 0:
            return "mixed sea"
        else:
            return "swell"
    else:
        return "null"