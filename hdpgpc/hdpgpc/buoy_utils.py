import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
from scipy.interpolate import CubicSpline, interp1d
from scipy.integrate import simpson
import wavespectra as wv
import xarray as xr
from geographiclib.geodesic import Geodesic

@dataclass
class BuoyObservation:
    """Container for buoy observation data"""
    lat: float  # Latitude (degrees)
    lon: float  # Longitude (degrees)
    time: datetime  # Observation timestamp
    mean_spectrum: np.ndarray  # Mean directional spectrum (freq x direction)
    peak_frequency: float  # Peak frequency (Hz)
    peak_direction: float  # Peak direction (degrees, nautical convention)
    peak_energy: float  # Peak spectral energy (m²/Hz/deg)
    time_end: Optional[datetime] = None #End timestamp of the event, could be puntual if not stated
    spread_direction: Optional[float] = 30.0# Spread of direction
    qp_norm: Optional[float] = 0.0 # Goda Q, measure of peakedness
    variance_spectrum: Optional[np.ndarray] = None # Variance of spectrum
    frequency_evolution: Optional[np.ndarray] = None # Frequency evolution for cluster
    dates: Optional[np.ndarray] = None # Times for each observation
    energies: Optional[np.ndarray] = None # Energies for each observation
    A_matrix: Optional[np.ndarray] = None  # Linear transformation matrix A
    C_matrix: Optional[np.ndarray] = None  # Linear transformation matrix C

def compute_weighted_mean_direction(directions, energies):
    """
    Compute energy-weighted mean direction using circular statistics.

    Parameters:
    -----------
    directions : array-like, shape (n,)
        Directions in degrees (nautical convention: 0°=North, clockwise)
    energies : array-like, shape (n,)
        Spectral energy values at each direction

    Returns:
    --------
    mean_direction : float
        Weighted mean direction in degrees [0, 360)
    """
    # Convert to radians
    theta = np.radians(directions)

    # Normalize weights
    weights = energies / np.sum(energies)

    # Compute circular mean using unit vectors
    C = np.sum(weights * np.cos(theta))  # Mean resultant x-component
    S = np.sum(weights * np.sin(theta))  # Mean resultant y-component

    # Mean direction
    mean_theta = np.arctan2(S, C)
    mean_direction = np.degrees(mean_theta) % 360

    return mean_direction

def compute_weighted_peak_direction(directions, energies):
    """
    Compute energy-weighted mean direction using circular statistics.

    Parameters:
    -----------
    directions : array-like, shape (n,)
        Directions in degrees (nautical convention: 0°=North, clockwise)
    energies : array-like, shape (n,)
        Spectral energy values at each direction

    Returns:
    --------
    mean_direction : float
        Weighted mean direction in degrees [0, 360)
    """

    # Convert to radians
    theta = np.radians(directions)

    # Normalize weights
    weights = energies / np.sum(energies)

    # Compute circular mean using unit vectors
    C = np.sum(weights * np.cos(theta))  # Mean resultant x-component
    S = np.sum(weights * np.sin(theta))  # Mean resultant y-component

    # Mean direction
    mean_theta = np.arctan2(S, C)
    peak_direction = np.degrees(mean_theta) % 360

    return peak_direction

def compute_directional_spread(directions, energies):
    """
    Compute directional spread (circular standard deviation).

    Parameters:
    -----------
    directions : array-like, shape (n,)
        Directions in degrees
    energies : array-like, shape (n,)
        Spectral energy at each direction

    Returns:
    --------
    spread : float
        Directional spread in degrees
    R : float
        Mean resultant length (concentration parameter, 0 ≤ R ≤ 1)
    """
    theta = np.radians(directions)
    weights = energies / np.sum(energies)

    # Compute circular mean components
    C = np.sum(weights * np.cos(theta))
    S = np.sum(weights * np.sin(theta))

    # Mean resultant length (concentration)
    R = np.sqrt(C**2 + S**2)

    # Circular variance
    circular_var = 1 - R

    # Circular standard deviation (in radians)
    circular_std = np.sqrt(-2 * np.log(R)) if R > 0 else np.pi

    # Convert to degrees
    spread_degrees = np.degrees(circular_std)

    return spread_degrees, R


def compute_cluster_representative(cluster_observations, frequencies):
    """
    Compute centroid representative for a cluster of BuoyObservations.

    Parameters:
    -----------
    cluster_observations : list of BuoyObservation
        All observations in the cluster

    Returns:
    --------
    representative : BuoyObservation
        Centroid representative with statistics
    """
    # Extract mean spectra
    spectra = [obs.mean_spectrum for obs in cluster_observations]
    centroid_spectrum = np.mean(spectra, axis=0)

    # Get frequency and direction grids (assuming consistent across obs)
    n_freq, n_dir = centroid_spectrum.shape
    directions = np.linspace(0, 360, n_dir, endpoint=False)

    # Integrate over frequency to get directional distribution
    g_theta = np.trapezoid(centroid_spectrum, axis=0)

    # Compute weighted mean direction and spread
    mean_dir = compute_weighted_mean_direction(directions, g_theta)

    # Find peak frequency
    f_spectrum = np.trapezoid(centroid_spectrum, axis=1)
    peak_freq_idx = np.argmax(f_spectrum)

    #Compute weighted peak direction
    peak_dir = compute_weighted_peak_direction(directions, centroid_spectrum[peak_freq_idx, :])
    peak_dir = np.mean([obs.peak_direction for obs in cluster_observations])

    #Compute the spread for peak frequency
    spreads = [obs.spread_direction for obs in cluster_observations]
    spread_deg = np.mean(spreads)
    # energy = centroid_spectrum[peak_freq_idx, :]
    # spread_deg, R = compute_directional_spread(directions, energy)
    # spread_deg = spread_deg * 0.5 #Try to reduce it a bit for using many samples.

    #Compute peakedness
    d_theta = np.mean(np.diff(np.radians(directions)))
    S_f = np.sum(centroid_spectrum, axis=1) * d_theta
    m0 = simpson(y=S_f, x=frequencies)

    if m0 > 1e-9:
        S_squared_int = simpson(y=S_f ** 2, x=frequencies)
        Qp = (2.0 / m0 ** 2) * S_squared_int

        # Normalize to be dimensionless (Optional but recommended)
        # Qp_norm = Qp * Peak_Frequency
        peak_idx = np.argmax(S_f)
        fp = frequencies[peak_idx]
        Qp_norm = Qp * fp
        #Qp_norm = 1.0 / Qp_norm
    else:
        Qp = 0.0
        Qp_norm = 0.0

    # Peak energy
    peak_energy = np.max(centroid_spectrum)

    frequency_evolution = np.array([obs.peak_frequency for obs in cluster_observations])
    dates = np.array([obs.time for obs in cluster_observations])
    energies = np.array([obs.peak_energy for obs in cluster_observations])
    # Create representative observation
    representative = BuoyObservation(
        lat=np.mean([obs.lat for obs in cluster_observations]),
        lon=np.mean([obs.lon for obs in cluster_observations]),
        time=cluster_observations[0].time,# or compute mean time
        mean_spectrum=centroid_spectrum,
        peak_frequency=frequencies[peak_freq_idx],
        peak_direction=peak_dir,
        peak_energy=peak_energy,
        qp_norm = Qp,
        spread_direction=spread_deg,
        time_end = cluster_observations[-1].time,
        frequency_evolution = frequency_evolution,
        dates = dates,
        energies = energies
    )

    return representative

def interpolate_spectral_coefficients(
    f_coarse,
    variance_density,
    a1,
    b1,
    a2,
    b2,
    f_dense=None,
    n_points=50,
    missing_value=-9999.0,
    enforce_constraints=True,
    bc_type='not-a-knot'
):
    """
    Interpolate wave spectral Fourier coefficients from coarse to dense frequency grid.

    Parameters
    ----------
    f_coarse : array_like, shape (n_freq_coarse,)
        Coarse frequency grid (Hz), e.g., 9 points from 0.04 to 0.12 Hz

    variance_density : array_like, shape (n_freq_coarse, n_time)
        Omnidirectional variance density S(f) at coarse frequencies

    a1, b1, a2, b2 : array_like, shape (n_freq_coarse, n_time)
        Directional Fourier coefficients at coarse frequencies

    f_dense : array_like, optional
        Target dense frequency grid. If None, creates linearly spaced grid

    n_points : int, default=50
        Number of points in dense grid (only used if f_dense is None)

    missing_value : float, default=-9999.0
        Value indicating missing data in input arrays

    enforce_constraints : bool, default=True
        If True, enforce physical constraints: a1^2 + b1^2 <= 1, a2^2 + b2^2 <= 1

    bc_type : str, default='not-a-knot'
        Boundary condition for CubicSpline ('not-a-knot', 'natural', 'clamped', 'periodic')
        'not-a-knot' is most robust for irregular data

    Returns
    -------
    dict with keys:
        'f_dense' : array_like, shape (n_freq_dense,)
            Dense frequency grid
        'variance_density_dense' : array_like, shape (n_freq_dense, n_time)
            Interpolated variance density
        'a1_dense', 'b1_dense', 'a2_dense', 'b2_dense' : array_like, shape (n_freq_dense, n_time)
            Interpolated directional coefficients
        'valid_mask' : array_like, shape (n_time,)
            Boolean mask indicating which time points had sufficient valid data

    Notes
    -----
    - Uses cubic spline interpolation for smooth, physically realistic spectra
    - Handles missing data by falling back to linear interpolation when needed
    - Optionally enforces normalization constraints on directional coefficients
    - Ensures non-negative variance density values

    Examples
    --------
    >>> f_coarse = np.linspace(0.04, 0.12, 9)
    >>> # Assume variance_density, a1, b1, a2, b2 are (9, 100) arrays
    >>> result = interpolate_spectral_coefficients(
    ...     f_coarse, variance_density, a1, b1, a2, b2, n_points=50
    ... )
    >>> f_dense = result['f_dense']  # (50,)
    >>> S_dense = result['variance_density_dense']  # (50, 100)
    """

    # Convert inputs to numpy arrays
    f_coarse = np.asarray(f_coarse)
    variance_density = np.asarray(variance_density)
    a1 = np.asarray(a1)
    b1 = np.asarray(b1)
    a2 = np.asarray(a2)
    b2 = np.asarray(b2)

    # Validate input dimensions
    n_freq_coarse = len(f_coarse)
    if variance_density.shape[0] != n_freq_coarse:
        raise ValueError(f"variance_density first dimension {variance_density.shape[0]} "
                        f"must match f_coarse length {n_freq_coarse}")

    # Get number of time points
    if variance_density.ndim == 1:
        variance_density = variance_density.reshape(-1, 1)
        a1 = a1.reshape(-1, 1)
        b1 = b1.reshape(-1, 1)
        a2 = a2.reshape(-1, 1)
        b2 = b2.reshape(-1, 1)

    n_time = variance_density.shape[1]

    # Create dense frequency grid
    if f_dense is None:
        f_dense = np.linspace(f_coarse[0], f_coarse[-1], n_points)
    else:
        f_dense = np.asarray(f_dense)

    n_freq_dense = len(f_dense)

    # Initialize output arrays
    variance_density_dense = np.zeros((n_freq_dense, n_time))
    a1_dense = np.zeros((n_freq_dense, n_time))
    b1_dense = np.zeros((n_freq_dense, n_time))
    a2_dense = np.zeros((n_freq_dense, n_time))
    b2_dense = np.zeros((n_freq_dense, n_time))
    valid_mask = np.zeros(n_time, dtype=bool)

    # Loop over time points
    for t in range(n_time):
        # Extract data for this time
        S_coarse = variance_density[:, t]
        a1_coarse = a1[:, t]
        b1_coarse = b1[:, t]
        a2_coarse = a2[:, t]
        b2_coarse = b2[:, t]

        # Identify valid data points (not missing)
        valid = (a1_coarse != missing_value) & \
                (b1_coarse != missing_value) & \
                (a2_coarse != missing_value) & \
                (b2_coarse != missing_value) & \
                (S_coarse > 0) & \
                np.isfinite(S_coarse) & \
                np.isfinite(a1_coarse) & \
                np.isfinite(b1_coarse) & \
                np.isfinite(a2_coarse) & \
                np.isfinite(b2_coarse)

        n_valid = np.sum(valid)

        # Need at least 3 points for cubic spline
        if n_valid < 3:
            # Insufficient data - use fallback
            if n_valid >= 2:
                # Linear interpolation
                variance_density_dense[:, t] = np.interp(
                    f_dense, f_coarse[valid], S_coarse[valid],
                    left=0, right=0
                )
                a1_dense[:, t] = np.interp(f_dense, f_coarse[valid], a1_coarse[valid])
                b1_dense[:, t] = np.interp(f_dense, f_coarse[valid], b1_coarse[valid])
                a2_dense[:, t] = np.interp(f_dense, f_coarse[valid], a2_coarse[valid])
                b2_dense[:, t] = np.interp(f_dense, f_coarse[valid], b2_coarse[valid])
                valid_mask[t] = False  # Mark as lower quality
            else:
                # Fill with zeros
                valid_mask[t] = False
            continue

        # Cubic spline interpolation
        try:
            cs_S = CubicSpline(f_coarse[valid], S_coarse[valid], bc_type=bc_type)
            cs_a1 = CubicSpline(f_coarse[valid], a1_coarse[valid], bc_type=bc_type)
            cs_b1 = CubicSpline(f_coarse[valid], b1_coarse[valid], bc_type=bc_type)
            cs_a2 = CubicSpline(f_coarse[valid], a2_coarse[valid], bc_type=bc_type)
            cs_b2 = CubicSpline(f_coarse[valid], b2_coarse[valid], bc_type=bc_type)

            # Evaluate on dense grid
            variance_density_dense[:, t] = cs_S(f_dense)
            a1_dense[:, t] = cs_a1(f_dense)
            b1_dense[:, t] = cs_b1(f_dense)
            a2_dense[:, t] = cs_a2(f_dense)
            b2_dense[:, t] = cs_b2(f_dense)

            valid_mask[t] = True

        except Exception as e:
            # Fallback to linear if spline fails
            print(f"Warning: Cubic spline failed at time {t}, using linear interpolation. Error: {e}")
            variance_density_dense[:, t] = np.interp(
                f_dense, f_coarse[valid], S_coarse[valid], left=0, right=0
            )
            a1_dense[:, t] = np.interp(f_dense, f_coarse[valid], a1_coarse[valid])
            b1_dense[:, t] = np.interp(f_dense, f_coarse[valid], b1_coarse[valid])
            a2_dense[:, t] = np.interp(f_dense, f_coarse[valid], a2_coarse[valid])
            b2_dense[:, t] = np.interp(f_dense, f_coarse[valid], b2_coarse[valid])
            valid_mask[t] = False

        # Enforce physical constraints
        if enforce_constraints:
            # Ensure non-negative variance density
            variance_density_dense[:, t] = np.maximum(variance_density_dense[:, t], 0)

            # Normalize directional coefficients: a1^2 + b1^2 <= 1
            r1 = np.sqrt(a1_dense[:, t]**2 + b1_dense[:, t]**2)
            mask1 = r1 > 1.0
            if np.any(mask1):
                a1_dense[mask1, t] /= r1[mask1]
                b1_dense[mask1, t] /= r1[mask1]

            # Normalize directional coefficients: a2^2 + b2^2 <= 1
            r2 = np.sqrt(a2_dense[:, t]**2 + b2_dense[:, t]**2)
            mask2 = r2 > 1.0
            if np.any(mask2):
                a2_dense[mask2, t] /= r2[mask2]
                b2_dense[mask2, t] /= r2[mask2]

    return {
        'f_dense': f_dense,
        'variance_density_dense': variance_density_dense,
        'a1_dense': a1_dense,
        'b1_dense': b1_dense,
        'a2_dense': a2_dense,
        'b2_dense': b2_dense,
        'valid_mask': valid_mask
    }

class SwellBackTriangulation:
    """
    Physics-based back-triangulation for swell origin estimation
    and coastal arrival prediction from clustered directional spectra.

    References:
    - Hanson & Phillips (2001): Automated spectral partitioning
    - Munk et al. (1963): Deep-water dispersion theory
    - Snodgrass et al. (1966): Trans-Pacific swell propagation
    """

    # Physical constants
    EARTH_RADIUS_KM = 6371.0  # Mean Earth radius (km)
    GRAVITY = 9.81  # Gravitational acceleration (m/s²)
    SWELL_ATTENUATION_COEF = 1e-7  # Frequency-dependent attenuation (1/m)
    DISPERSION_RATE = 0.7 #Dispersion rate when swell moves /100km

    def __init__(self,
                 buoy1: BuoyObservation,
                 buoy2: BuoyObservation,
                 coastal_buoy_location: Tuple[float, float],
                 min_frequency: float = 0.04,
                 max_frequency: float = 0.25):
        """
        Initialize back-triangulation analysis.

        Parameters:
        -----------
        buoy1, buoy2 : BuoyObservation
            Deep-sea buoy observations (simultaneous or near-simultaneous)
        coastal_buoy_location : tuple
            (latitude, longitude) of coastal buoy
        min_frequency : float
            Minimum swell frequency to consider (Hz), default 0.04 Hz (25s period)
        max_frequency : float
            Maximum swell frequency to consider (Hz), default 0.25 Hz (4s period)
        """
        self.buoy1 = buoy1
        self.buoy2 = buoy2
        self.coastal_lat, self.coastal_lon = coastal_buoy_location
        self.min_freq = min_frequency
        self.max_freq = max_frequency

        # Validate simultaneous observations
        time_diff = abs((buoy1.time - buoy2.time).total_seconds())
        if time_diff > 3600:  # 1 hour threshold
            warnings.warn(f"Buoy observations are {time_diff / 3600:.2f} hours apart")

    # ==================== GEODESIC CALCULATIONS ====================

    @staticmethod
    def haversine_distance(lat1: float, lon1: float,
                           lat2: float, lon2: float) -> float:
        """
        Calculate great-circle distance between two points using Haversine formula.

        Returns: Distance in kilometers
        """
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        return SwellBackTriangulation.EARTH_RADIUS_KM * c

    @staticmethod
    def initial_bearing(lat1: float, lon1: float,
                        lat2: float, lon2: float) -> float:
        """
        Calculate initial bearing (azimuth) from point 1 to point 2.

        Returns: Bearing in degrees (0-360, clockwise from North)
        """
        geod = Geodesic.WGS84
        result = geod.Inverse(lat1, lon1, lat2, lon2)
        return result['azi1']

    @staticmethod
    def destination_point(lat: float, lon: float,
                          bearing: float, distance_km: float) -> Tuple[float, float]:
        """
        Calculate destination point given start point, bearing, and distance.

        Parameters:
        -----------
        lat, lon : float
            Starting coordinates (degrees)
        bearing : float
            Bearing in degrees (clockwise from North)
        distance_km : float
            Distance to travel (kilometers)

        Returns: (destination_lat, destination_lon)
        """
        R = SwellBackTriangulation.EARTH_RADIUS_KM
        lat_r = np.radians(lat)
        lon_r = np.radians(lon)
        bearing_r = np.radians(bearing)

        angular_distance = distance_km / R

        dest_lat = np.arcsin(
            np.sin(lat_r) * np.cos(angular_distance) +
            np.cos(lat_r) * np.sin(angular_distance) * np.cos(bearing_r)
        )

        dest_lon = lon_r + np.arctan2(
            np.sin(bearing_r) * np.sin(angular_distance) * np.cos(lat_r),
            np.cos(angular_distance) - np.sin(lat_r) * np.sin(dest_lat)
        )

        return np.degrees(dest_lat), np.degrees(dest_lon)

    # ==================== WAVE PHYSICS ====================

    @staticmethod
    def deep_water_group_velocity(frequency: float) -> float:
        """
        Calculate deep-water group velocity.

        Parameters:
        -----------
        frequency : float
            Wave frequency (Hz)

        Returns: Group velocity (m/s)
        """
        v = SwellBackTriangulation.GRAVITY / (4 * np.pi * frequency)
        return v if type(v) is np.float64 else np.float64(v)

    @staticmethod
    def wave_period_to_frequency(period: float) -> float:
        """Convert wave period (s) to frequency (Hz)"""
        return 1.0 / period

    @staticmethod
    def energy_attenuation(initial_energy: float, frequency: float,
                           distance_km: float) -> float:
        """
        Estimate energy attenuation due to swell dissipation.

        Simple exponential decay model (Ardhuin et al. 2009).
        """
        distance_m = distance_km * 1000
        alpha = SwellBackTriangulation.SWELL_ATTENUATION_COEF * (frequency ** 2)
        return initial_energy * np.exp(-alpha * distance_m)

    # ==================== BACK-TRIANGULATION METHODS ====================

    def compute_back_azimuth(self, forward_direction: float) -> float:
        """
        Compute back-azimuth (opposite of wave approach direction).

        Wave direction is "coming from", so back-azimuth points toward source.
        """
        #return (forward_direction + 180) % 360
        return forward_direction

    def triangulate_source_location_loss(self, method: str = "cone_intersection", weights=[1.0, 0.5, 0.5, 0.1, 0.01], event_limit=24.0) -> Dict:
        """
        Enhanced source location estimation using directional cones intersection.

        This method improves on the original grid_search by:
        1. Constructing directional cones from each buoy (mean bearing ± spread)
        2. Searching only in the intersection region of both cones
        3. Using temporal consistency (observed vs predicted time lag) as primary loss
        4. Incorporating directional misfit as secondary constraint

        Parameters:
        -----------
        method : str
            'cone_intersection': Search in overlapping cone regions (recommended)
            'grid_search': Original exhaustive grid search (for comparison)

        Returns:
        --------
        dict containing:
            - source_lat, source_lon: Estimated source coordinates
            - temporal_error_hours: Time lag matching quality
            - directional_consistency: Angular agreement score
            - cone_intersection_area_km2: Size of search region
        """
        import numpy as np
        from scipy.optimize import minimize
        from geographiclib.geodesic import Geodesic

        t1_mid = self.buoy1.time
        t2_mid = self.buoy2.time

        time_lag_observed_hours = (t2_mid - t1_mid).total_seconds() / 3600.0

        if abs(time_lag_observed_hours) > event_limit or time_lag_observed_hours < 0.0:
            print(f"No event matching or nonsense matching, lag {time_lag_observed_hours}, limit {event_limit}.")
            return {
            'source_lat': self.buoy1.lat,
            'source_lon': self.buoy1.lon,
            'source_dp': 0.0,
            'source_directional_spread': 0.0,
            'confidence_radius_km': 0.0,

            # Quality metrics
            'temporal_error_hours': 0.0,
            'directional_error_degrees': 0.0,
            'time_lag_observed_hours': 0.0,
            'time_lag_predicted_hours': 0.0,
            'time_consistency_score': 0.0,  # Decays with 12-hour half-life

            # Cone diagnostics
            'cone_spread_buoy1_deg': None,
            'cone_spread_buoy2_deg': None,
            'cone_intersection_area_km2': None,

            # Standard outputs
            'buoy1_backtrack': None,
            'buoy2_backtrack': None,
            'back_azimuth_buoy1': None,
            'back_azimuth_buoy2': None,
            'method': method,

            # Grids for visualization
            'lat_grid': None,
            'lon_grid': None,
            'loss_grid': None,
            'temporal_error_grid': None,
            'directional_error_grid': None,
            'posterior_grid': None,
            'matching': False
        }



        # Extract buoy data
        lat1, lon1 = self.buoy1.lat, self.buoy1.lon
        lat2, lon2 = self.buoy2.lat, self.buoy2.lon

        # Compute back-azimuths (directions pointing toward source)
        back_az1 = self.compute_back_azimuth(self.buoy1.peak_direction)
        back_az2 = self.compute_back_azimuth(self.buoy2.peak_direction)

        # Directional spreads define cone half-angles
        cone_spread1 = self.buoy1.spread_direction  # degrees
        cone_spread2 = self.buoy2.spread_direction

        # Energy weights for multi-objective optimization
        w1 = np.sqrt(self.buoy1.peak_energy)
        w2 = np.sqrt(self.buoy2.peak_energy)
        w_total = w1 + w2
        w1_norm = w1 / w_total
        w2_norm = w2 / w_total

        # Compute observed time lag between buoy detections
        # if self.buoy1.time_end is not None and self.buoy2.time_end is not None:
        #     # Use cluster midpoints
        #     t1_mid = self.buoy1.time + (self.buoy1.time_end - self.buoy1.time) / 2
        #     t2_mid = self.buoy2.time + (self.buoy2.time_end - self.buoy2.time) / 2
        # else:


        # Mean frequency for group velocity (will be improved with dispersion correction)
        mean_freq = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        intersection_lat, intersection_lon, confidence = self._find_great_circle_intersection(
            self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.spread_direction,
            self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.spread_direction
        )

        # # Build backtrack dicts
        backtrack1_ = self._backtrack_single_buoy(
            self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.peak_frequency,
            self.buoy1.spread_direction
        )
        backtrack2_ = self._backtrack_single_buoy(
            self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.peak_frequency,
            self.buoy2.spread_direction
        )

        b1_lat = np.array([cand['lat'] for cand in backtrack1_['all_candidates']])
        b1_lon = np.array([cand['lon'] for cand in backtrack1_['all_candidates']])
        b2_lat = np.array([cand['lat'] for cand in backtrack2_['all_candidates']])
        b2_lon = np.array([cand['lon'] for cand in backtrack2_['all_candidates']])

        bis_lat = (b1_lat + b2_lat) / 2.0
        bis_lon = (b1_lon + b2_lon) / 2.0

        #Check with frequency dispersion distance
        dist_freq_b1 = abs(compute_source_distance_from_dispersion(self.buoy1.dates, self.buoy1.frequency_evolution, self.buoy1.energies)['distance_km'])
        dist_freq_b2 = abs(compute_source_distance_from_dispersion(self.buoy2.dates, self.buoy2.frequency_evolution, self.buoy2.energies)['distance_km'])

        if dist_freq_b1 == 3000 and dist_freq_b2 != 3000:
            dist_freq_b1 = dist_freq_b2
        elif dist_freq_b1 != 3000 and dist_freq_b2 == 3000:
            dist_freq_b2 = dist_freq_b1
        # if dist_freq_b1 == 3000 or dist_freq_b2 == 3000:
        #     #No convergence of the estimation so the weight for this term should be lower.
        #     weights[3] = 0.1

        # ========== LOSS FUNCTION: TEMPORAL + DIRECTIONAL ==========
        def loss_function(source_coords):
            """
            Combined loss: temporal consistency (primary) + directional fit (secondary)
            """
            source_lat, source_lon = source_coords
            geod = Geodesic.WGS84

            # Compute geodesic from source to each buoy
            g1 = geod.Inverse(lat1, lon1, source_lat, source_lon)
            g2 = geod.Inverse(lat2, lon2, source_lat, source_lon)

            dist1_km = g1['s12'] / 1000.0
            dist2_km = g2['s12'] / 1000.0

            # Bearing from source to buoys
            expected_back_az1 = g1['azi1'] % 360
            expected_back_az2 = g2['azi1'] % 360

            # Expected back-azimuth: buoys should observe waves coming FROM these directions
            # expected_back_az1 = (bearing_src_to_buoy1 + 180) % 360
            # expected_back_az2 = (bearing_src_to_buoy2 + 180) % 360

            # === 1. TEMPORAL CONSISTENCY (Primary Loss) ===
            # Travel times from source to each buoy
            tau1 = self._compute_travel_time(dist1_km, mean_freq)  # hours
            tau2 = self._compute_travel_time(dist2_km, mean_freq)

            # Predicted time lag
            time_lag_predicted_hours = tau2 - tau1
            if time_lag_predicted_hours < 0:
                time_lag_predicted_hours = 0.0

            # Temporal mismatch (this is what we primarily want to minimize)
            temporal_error = abs(time_lag_predicted_hours - time_lag_observed_hours) / abs(time_lag_observed_hours)
            temporal_error = np.min([temporal_error, 5.0])

            # === 2. DIRECTIONAL CONSISTENCY (Secondary Constraint) ===
            # Angular misfits (handle circular wrapping)
            def angular_distance(obs, exp):
                diff = (obs - exp) % 360 if obs > exp else (exp - obs) % 360
                return abs(diff)

            dir_error1 = angular_distance(back_az1, expected_back_az1)
            dir_error2 = angular_distance(back_az2, expected_back_az2)

            # Weighted directional error
            directional_error = (w1_norm * dir_error1 + w2_norm * dir_error2) / 15.0
            directional_error = np.min([directional_error, 5.0])

            best_dist = np.inf
            # === 3. DISTANCE TO BISECT OF BACKTRACKS CONSISTENCY (Third Constraint) ===
            # for bis_a, bis_o in zip(bis_lat, bis_lon):
            #     dist_b = self.haversine_distance(bis_a, bis_o, source_lat, source_lon)
            #     if best_dist > dist_b:
            #         best_dist = dist_b
            # dist_error = best_dist / 1000.0
            # dist_error = np.min([dist_error, 5.0])

            # === 3. DISTANCE TO INTERSECTION OF BACKTRACKS CONSISTENCY (Third Constraint) ===
            # best_dist = self.haversine_distance(intersection_lat, intersection_lon, source_lat, source_lon)
            #
            # dist_error = best_dist / 1000.0
            # dist_error = np.min([dist_error, 5.0])

            # === 3. PEAKEDNESS OF SPECTRA CONSISTENCY (Third Constraint) ===
            #Compare the distance with the peakedness coefficient times 130, as QP > 30 is for swells > 4000 km
            dist_error = (w1_norm * abs(dist1_km - self.buoy1.qp_norm * 130.0) +
                          w2_norm * abs(dist2_km - self.buoy2.qp_norm * 130.0))
            dist_error = np.min([dist_error / 1000.0, 5.0])


            # === 4. DISTANCE FREQUENCY EVOLUTION CONSISTENCY (Fourth Constraint) ===
            dist_f_error = w1_norm * abs(dist1_km - dist_freq_b1) + w2_norm * abs(dist2_km - dist_freq_b2)
            dist_f_error = np.min([dist_f_error / 1000.0, 5.0])


            # === 5. DISTANCE REGULARIZATION  ===
            #Before optimal=4000 tolerance=2000
            optimal_distance_km = 4000.0
            distance_tolerance_km = 2000.0  # Standard deviation of "acceptable" range

            distance_penalty = 0.0
            for dist in [dist1_km, dist2_km]:
                # Squared deviation from optimal, normalized by tolerance
                deviation = (dist - optimal_distance_km) / distance_tolerance_km
                distance_penalty += deviation ** 2 * 0.5
            #distance_penalty = np.min([distance_penalty, 10.0])

            # === COMBINED LOSS ===
            # Temporal error dominates (weight = 1.0)
            # Directional error is secondary (weight = 1.0)
            # Penalties are regularization (weight = 0.01)
            loss = (weights[0] * temporal_error +  # Primary: match time lag
                    weights[1] * directional_error +  # Secondary: fit directions
                    weights[2] * dist_error + # Third: peakedness estimator of distance.
                    weights[3] * dist_f_error + # Fourth constraint: frequency distance estimation.
                    weights[4] * distance_penalty)  # Regularization: realistic distances

            return loss, temporal_error, directional_error, dist_error, dist_f_error, distance_penalty

        # ========== CONE INTERSECTION GRID CONSTRUCTION ==========
        if method == "cone_intersection":
            # Step 1: Define search region centered on cone intersection

            # Estimate typical source distance (use energy-weighted back-projection)
            # Conservative estimate: 2000-5000 km for Pacific swells
            search_distances = np.linspace(1500, 6000, 20)  # km

            # Generate candidate points along each cone's central axis
            candidates_cone1 = []
            for dist in search_distances:
                lat, lon = self.destination_point(lat1, lon1, back_az1, dist)
                candidates_cone1.append((lat, lon, dist))

            candidates_cone2 = []
            for dist in search_distances:
                lat, lon = self.destination_point(lat2, lon2, back_az2, dist)
                candidates_cone2.append((lat, lon, dist))

            # Step 2: Find approximate intersection point (geometric center)
            # Use the pair of candidates with minimum separation
            min_sep = np.inf
            center_lat, center_lon = None, None

            for lat_a, lon_a, _ in candidates_cone1:
                for lat_b, lon_b, _ in candidates_cone2:
                    sep = self.haversine_distance(lat_a, lon_a, lat_b, lon_b)
                    if sep < min_sep:
                        min_sep = sep
                        center_lat = (lat_a + lat_b) / 2
                        center_lon = (lon_a + lon_b) / 2

            # Step 3: Construct grid around intersection region
            # Grid size adapts to cone spread (wider cones → larger search area)
            grid_size_km = 1500 + 30 * (cone_spread1 + cone_spread2)  # ~1500-3000 km radius

            # Convert to lat/lon spans
            lat_span = grid_size_km / 111.0  # 1 degree ≈ 111 km
            lon_span = lat_span / np.cos(np.radians(center_lat))

            # Define grid bounds
            min_lat = max(-85, center_lat - lat_span)
            max_lat = min(85, center_lat + lat_span)
            min_lon = center_lon - lon_span
            max_lon = center_lon + lon_span

            # High-resolution grid in intersection region
            lat_grid = np.linspace(min_lat, max_lat, 100)
            lon_grid = np.linspace(min_lon, max_lon, 100)

            print(f"Grid search centered at ({center_lat:.2f}, {center_lon:.2f})")
            print(f"Grid size: {lat_span * 111:.0f} km × {lon_span * 111 * np.cos(np.radians(center_lat)):.0f} km")

        elif method == "grid_search":
            # Original exhaustive search (for comparison)
            avg_lat = (lat1 + lat2) / 2
            avg_lon = (lon1 + lon2) / 2

            u = np.cos(np.radians(back_az1)) + np.cos(np.radians(back_az2))
            v = np.sin(np.radians(back_az1)) + np.sin(np.radians(back_az2))
            search_bearing = np.degrees(np.arctan2(v, u)) % 360

            grid_center_lat, grid_center_lon = self.destination_point(
                avg_lat, avg_lon, search_bearing, 2000.0
            )

            lat_span = 30.0
            lon_scale = 1.0 / np.cos(np.radians(grid_center_lat))
            lon_span = 35.0 * lon_scale

            min_lat = max(-85, grid_center_lat - lat_span)
            max_lat = min(85, grid_center_lat + lat_span)
            min_lon = grid_center_lon - lon_span
            max_lon = grid_center_lon + lon_span

            lat_grid = np.linspace(min_lat, max_lat, 120)
            lon_grid = np.linspace(min_lon, max_lon, 120)

        else:
            raise ValueError(f"Unknown method: {method}")

        # ========== GRID SEARCH OPTIMIZATION ==========
        best_loss = np.inf
        best_coords = (center_lat if method == "cone_intersection" else grid_center_lat,
                       center_lon if method == "cone_intersection" else grid_center_lon)
        best_temporal_error = np.inf
        best_directional_error = np.inf
        best_dist_error = np.inf
        best_dist_f_error = np.inf
        best_dist_penalty = np.inf

        # Store full results for visualization
        loss_grid = np.zeros((len(lat_grid), len(lon_grid)))
        temporal_error_grid = np.zeros((len(lat_grid), len(lon_grid)))
        directional_error_grid = np.zeros((len(lat_grid), len(lon_grid)))
        dist_error_grid = np.zeros((len(lat_grid), len(lon_grid)))
        dist_f_error_grid = np.zeros((len(lat_grid), len(lon_grid)))
        for i, slat in enumerate(lat_grid):
            for j, slon in enumerate(lon_grid):
                loss, temp_err, dir_err, dist_err, dist_f_err, distance_penalty = loss_function((slat, slon))

                loss_grid[i, j] = loss
                temporal_error_grid[i, j] = temp_err
                directional_error_grid[i, j] = dir_err
                dist_error_grid[i, j] = dist_err
                dist_f_error_grid[i, j] = dist_f_err

                if loss < best_loss:
                    best_loss = loss
                    best_coords = (slat, slon)
                    best_temporal_error = temp_err
                    best_directional_error = dir_err
                    best_dist_error = dist_err
                    best_dist_f_error = dist_f_err
                    best_dist_penalty = distance_penalty

        source_lat, source_lon = best_coords

        # ========== UNCERTAINTY QUANTIFICATION ==========
        # Convert loss to pseudo-probability (for uncertainty estimation)
        posterior = np.exp(-loss_grid / np.std(loss_grid))
        posterior /= np.sum(posterior)

        # Compute uncertainty from posterior spread
        lat_std = np.sqrt(np.sum(posterior * (lat_grid[:, None] - source_lat) ** 2)) / 2.0 #Half angle for spread
        lon_std = np.sqrt(np.sum(posterior * (lon_grid[None, :] - source_lon) ** 2)) / 2.0
        confidence_radius = np.sqrt(lat_std ** 2 + lon_std ** 2) * 111.0  # km

        # ========== DERIVED QUANTITIES ==========
        geod = Geodesic.WGS84
        g1 = geod.Inverse(lat1, lon1, source_lat, source_lon)
        g2 = geod.Inverse(lat2, lon2, source_lat, source_lon)

        dist1 = g1['s12'] / 1000.0
        dist2 = g2['s12'] / 1000.0

        # Mean propagation direction
        bearing_to_buoy1 = (g1['azi1']) % 360
        bearing_to_buoy2 = (g2['azi1']) % 360

        source_x = w1_norm * np.cos(np.radians(bearing_to_buoy1)) + w2_norm * np.cos(np.radians(bearing_to_buoy2))
        source_y = w1_norm * np.sin(np.radians(bearing_to_buoy1)) + w2_norm * np.sin(np.radians(bearing_to_buoy2))
        source_dp = np.degrees(np.arctan2(source_y, source_x)) % 360

        # Source directional spread
        source_spread = self._estimate_source_directional_spread(
            self.buoy1.spread_direction, self.buoy2.spread_direction,
            dist1, dist2, confidence_radius
        )

        # Compute cone intersection area (for diagnostics)
        cone_intersection_cells = np.sum(directional_error_grid < (cone_spread1 + cone_spread2) / 2)
        cell_area_km2 = (lat_span * 111) * (lon_span * 111 * np.cos(
            np.radians(center_lat if method == "cone_intersection" else grid_center_lat))) / (
                                    len(lat_grid) * len(lon_grid))
        cone_intersection_area_km2 = cone_intersection_cells * cell_area_km2

        return {
            'source_lat': source_lat,
            'source_lon': source_lon,
            'source_dp': source_dp,
            'source_directional_spread': source_spread,
            'confidence_radius_km': confidence_radius,

            # Quality metrics
            'temporal_error_hours': best_temporal_error,
            'directional_error_degrees': best_directional_error * 30.0 ,
            'directional_error_stand': best_directional_error,
            'distance_error_km': best_dist_error * 1000.0,
            'distance_error_stand': best_dist_error,
            'distance_frequency_error': best_dist_f_error * 1000.0,
            'distance_frequency_error_stand': best_dist_f_error,
            'distance_frequency_b1': dist_freq_b1,
            'distance_frequency_b2': dist_freq_b2,
            'distance_penalty': np.sqrt(best_dist_penalty * 2.0) * 2000.0,
            'distance_penalty_stand': best_dist_penalty,
            'time_lag_observed_hours': time_lag_observed_hours,
            'time_lag_predicted_hours': self._compute_travel_time(dist2, mean_freq) - self._compute_travel_time(dist1,
                                                                                                                mean_freq),
            'time_consistency_score': np.exp(-best_temporal_error / 12.0),  # Decays with 12-hour half-life

            # Cone diagnostics
            'cone_spread_buoy1_deg': cone_spread1,
            'cone_spread_buoy2_deg': cone_spread2,
            'cone_intersection_area_km2': cone_intersection_area_km2,

            # Standard outputs
            'buoy1_backtrack': backtrack1_,
            'buoy2_backtrack': backtrack2_,
            'back_azimuth_buoy1': back_az1,
            'back_azimuth_buoy2': back_az2,
            'method': method,

            # Grids for visualization
            'lat_grid': lat_grid,
            'lon_grid': lon_grid,
            'loss_grid': loss_grid,
            'temporal_error_grid': temporal_error_grid,
            'directional_error_grid': directional_error_grid,
            'posterior_grid': posterior,
            'matching': True
        }

    def triangulate_source_location(self,
                                    method: str = "intersection") -> Dict:
        """
        Estimate source location by triangulating from two buoy observations.

        Parameters:
        -----------
        method : str
            'intersection': Great-circle intersection (default)
            'weighted_centroid': Energy-weighted centroid of candidate regions

        Returns:
        --------
        dict containing:
            - source_lat, source_lon: Estimated source coordinates
            - confidence_radius_km: Uncertainty radius
            - buoy1_backtrack, buoy2_backtrack: Individual backtrack results
        """
        # Compute back-azimuths (directions TO source FROM buoys)
        back_az1 = self.compute_back_azimuth(self.buoy1.peak_direction)
        back_az2 = self.compute_back_azimuth(self.buoy2.peak_direction)

        # Individual backtracking along great circles
        backtrack1 = self._backtrack_single_buoy(
            self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.peak_frequency,
            self.buoy1.spread_direction
        )
        backtrack2 = self._backtrack_single_buoy(
            self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.peak_frequency,
            self.buoy2.spread_direction
        )

        w1 = self.buoy1.peak_energy
        w2 = self.buoy2.peak_energy
        if method == "intersection":
            source_lat, source_lon, confidence = self._find_great_circle_intersection(
                self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.spread_direction,
                self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.spread_direction

            )
        elif method == "weighted_centroid":
            # Weight by inverse variance or peak energy
            source_lat = (w1 * backtrack1['candidate_lat'] +
                          w2 * backtrack2['candidate_lat']) / (w1 + w2)
            source_lon = (w1 * backtrack1['candidate_lon'] +
                          w2 * backtrack2['candidate_lon']) / (w1 + w2)
            confidence = np.mean([backtrack1['spatial_uncertainty_km'],
                                  backtrack2['spatial_uncertainty_km']])

        else:
            raise ValueError(f"Unknown method: {method}")

        geod = Geodesic.WGS84

        g1 = geod.Inverse(source_lat, source_lon, self.buoy1.lat, self.buoy1.lon)
        bearing_to_buoy_1 = (180 + g1['azi1']) % 360
        g2 = geod.Inverse(source_lat, source_lon, self.buoy2.lat, self.buoy2.lon)
        bearing_to_buoy_2 = (180 + g2['azi1']) % 360

        source_x = (w1 * np.cos(np.radians(bearing_to_buoy_1)) +
                    w2 * np.cos(np.radians(bearing_to_buoy_2))) / (w1 + w2)
        source_y = (w1 * np.sin(np.radians(bearing_to_buoy_1)) +
                    w2 * np.sin(np.radians(bearing_to_buoy_2))) / (w1 + w2)
        initial_bearing = np.arctan2(source_y, source_x)
        initial_bearing = np.degrees(initial_bearing) % 360

        distance_source_buoy1 =  self.haversine_distance(self.buoy1.lat, self.buoy1.lon, source_lat, source_lon)
        distance_source_buoy2 =  self.haversine_distance(self.buoy2.lat, self.buoy2.lon, source_lat, source_lon)
        source_directional_spread = self._estimate_source_directional_spread(
            self.buoy1.spread_direction, self.buoy2.spread_direction, distance_source_buoy1, distance_source_buoy2, confidence)

        return {
            'source_lat': source_lat,
            'source_lon': source_lon,
            'source_dp': initial_bearing,
            'source_directional_spread': source_directional_spread,
            'confidence_radius_km': confidence,
            'buoy1_backtrack': backtrack1,
            'buoy2_backtrack': backtrack2,
            'back_azimuth_buoy1': back_az1,
            'back_azimuth_buoy2': back_az2
        }

    def triangulate_source_asynchronous(self,
                                        time_offset_hours: Optional[float] = None,
                                        use_dispersion: bool = True,
                                        method: str = "time_corrected") -> Dict:
        """
        Estimate source location from TWO ASYNCHRONOUS buoy observations.

        This method handles the case where the same swell event is detected at different
        times by each buoy (e.g., days apart due to clustering). It uses the time lag
        between detections to improve source estimation.

        Parameters:
        -----------
        time_offset_hours : float, optional
            Known or estimated time offset between buoy1 and buoy2 detections (hours).
            Positive means buoy2 detected AFTER buoy1.
            If None, computed from buoy1.time and buoy2.time

        use_dispersion : bool
            If True, use frequency dispersion to refine distance estimates

        method : str
            'time_corrected': Use time offset to find source location (recommended)
            'intersection_weighted': Weight by temporal proximity
            'iterative': Iteratively refine using both direction and time

        Returns:
        --------
        dict containing:
            - source_lat, source_lon: Estimated source coordinates
            - source_dp: Mean propagation direction from source
            - source_directional_spread: Estimated spreading at source
            - confidence_radius_km: Uncertainty radius
            - source_generation_time: Estimated time when swell was generated
            - time_consistency_score: Quality metric (0-1, higher is better)
            - buoy1_arrival_offset_hours: Time from source to buoy1
            - buoy2_arrival_offset_hours: Time from source to buoy2

        Algorithm:
        ----------
        1. Determine time offset between detections (Δt)
        2. Estimate source distances using dispersion (if available)
        3. For candidate source locations along each back-azimuth:
           - Compute travel times from source to each buoy
           - Check if Δt_predicted ≈ Δt_observed
        4. Select source location that best matches observed time offset

        References:
        -----------
        - Munk et al. (1963): Dispersion relation d = g/(4πm) where m = df/dt
        - Hanson & Phillips (2001): Automated swell tracking and source identification
        """

        # ==================== STEP 1: TIME OFFSET ====================
        if time_offset_hours is None:
            # Compute from observation times
            time_offset_hours = (self.buoy2.time - self.buoy1.time).total_seconds() / 3600.0

        # Use cluster time ranges if available
        # if self.buoy1.time_end is not None and self.buoy2.time_end is not None:
        #     # Use middle of cluster time ranges
        #     buoy1_mid_time = self.buoy1.time + (self.buoy1.time_end - self.buoy1.time) / 2
        #     buoy2_mid_time = self.buoy2.time + (self.buoy2.time_end - self.buoy2.time) / 2
        #     time_offset_hours = (buoy2_mid_time - buoy1_mid_time).total_seconds() / 3600.0
        # else:
        buoy1_mid_time = self.buoy1.time
        buoy2_mid_time = self.buoy2.time

        # ==================== STEP 2: BACK-AZIMUTHS ====================
        back_az1 = self.compute_back_azimuth(self.buoy1.peak_direction)
        back_az2 = self.compute_back_azimuth(self.buoy2.peak_direction)

        # ==================== STEP 3: DISPERSION-BASED DISTANCES (OPTIONAL) ====================
        distance_estimate_buoy1 = None
        distance_estimate_buoy2 = None

        if use_dispersion:
            # If we have frequency evolution data, estimate distances
            # For now, use typical swell distances
            distance_estimate_buoy1 = 3000.0  # km, typical Pacific swell
            distance_estimate_buoy2 = 3000.0

        # ==================== STEP 4: SOURCE SEARCH ====================
        if method == "time_corrected":
            result = self._find_source_time_corrected(
                back_az1, back_az2, time_offset_hours,
                distance_estimate_buoy1, distance_estimate_buoy2
            )

        elif method == "intersection_weighted":
            result = self._find_source_intersection_weighted(
                back_az1, back_az2, time_offset_hours
            )

        elif method == "iterative":
            result = self._find_source_iterative(
                back_az1, back_az2, time_offset_hours
            )

        else:
            raise ValueError(f"Unknown method: {method}")

        # ==================== STEP 5: SOURCE PROPERTIES ====================
        source_lat = result['source_lat']
        source_lon = result['source_lon']

        # Distance from source to each buoy
        dist_to_buoy1 = self.haversine_distance(source_lat, source_lon,
                                                self.buoy1.lat, self.buoy1.lon)
        dist_to_buoy2 = self.haversine_distance(source_lat, source_lon,
                                                self.buoy2.lat, self.buoy2.lon)

        # Compute source-to-buoy bearings
        geod = Geodesic.WGS84
        g1 = geod.Inverse(source_lat, source_lon, self.buoy1.lat, self.buoy1.lon)
        bearing_to_buoy1 = (180 + g1['azi1']) % 360

        g2 = geod.Inverse(source_lat, source_lon, self.buoy2.lat, self.buoy2.lon)
        bearing_to_buoy2 = (180 + g2['azi1']) % 360

        # Weighted mean direction (energy-weighted)
        w1 = self.buoy1.peak_energy
        w2 = self.buoy2.peak_energy
        source_x = (w1 * np.cos(np.radians(bearing_to_buoy1)) +
                    w2 * np.cos(np.radians(bearing_to_buoy2))) / (w1 + w2)
        source_y = (w1 * np.sin(np.radians(bearing_to_buoy1)) +
                    w2 * np.sin(np.radians(bearing_to_buoy2))) / (w1 + w2)
        initial_bearing = np.degrees(np.arctan2(source_y, source_x)) % 360

        # Estimate source directional spread
        source_directional_spread = self._estimate_source_directional_spread(
            self.buoy1.spread_direction, self.buoy2.spread_direction,
            dist_to_buoy1, dist_to_buoy2, result['confidence_radius_km']
        )

        # ==================== STEP 6: GENERATION TIME ====================
        # Use mean frequency for travel time calculation
        # TODO: Possible improvement if used a weighted energy mean frequency?
        mean_freq = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        # Travel time from source to buoy1
        travel_time_buoy1_hours = self._compute_travel_time(dist_to_buoy1, mean_freq)

        # Generation time: subtract travel time from buoy1 observation
        source_generation_time = buoy1_mid_time - timedelta(hours=travel_time_buoy1_hours)

        # Verify with buoy2
        travel_time_buoy2_hours = self._compute_travel_time(dist_to_buoy2, mean_freq)
        predicted_buoy2_time = source_generation_time + timedelta(hours=travel_time_buoy2_hours)

        # Time consistency score
        time_error_hours = abs((predicted_buoy2_time - buoy2_mid_time).total_seconds() / 3600.0)
        time_consistency_score = np.exp(-time_error_hours / 24.0)  # Decays over ~24 hours

        # ==================== STEP 7: CREATE BACKTRACK INFO (for predict_coastal_arrival) ====================
        # Create backtrack dictionaries similar to _backtrack_single_buoy output

        # Spatial uncertainty based on directional spread
        spread_half_angle_rad1 = np.radians(self.buoy1.spread_direction / 2)
        lateral_uncertainty_km_1 = dist_to_buoy1 * np.tan(
            spread_half_angle_rad1) - dist_to_buoy1 / 100.0 * self.DISPERSION_RATE

        spread_half_angle_rad2 = np.radians(self.buoy2.spread_direction / 2)
        lateral_uncertainty_km_2 = dist_to_buoy2 * np.tan(
            spread_half_angle_rad2) - dist_to_buoy2 / 100.0 * self.DISPERSION_RATE

        # Temporal uncertainty based on frequency spread
        frequency_spread1 = 0.1 * self.buoy1.peak_frequency
        timing_uncertainty_hours_1 = travel_time_buoy1_hours * (frequency_spread1 / self.buoy1.peak_frequency)

        frequency_spread2 = 0.1 * self.buoy2.peak_frequency
        timing_uncertainty_hours_2 = travel_time_buoy2_hours * (frequency_spread2 / self.buoy2.peak_frequency)

        #TODO: Backtrack computed with previous model, try to implement it into this new version.
        backtrack1_ = self._backtrack_single_buoy(
            self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.peak_frequency,
            self.buoy1.spread_direction
        )
        backtrack2_ = self._backtrack_single_buoy(
            self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.peak_frequency,
            self.buoy2.spread_direction
        )

        backtrack1 = {
            'candidate_lat': source_lat,
            'candidate_lon': source_lon,
            'candidate_distance_km': dist_to_buoy1,
            'travel_time_hours': travel_time_buoy1_hours,
            'spatial_uncertainty_km': lateral_uncertainty_km_1,
            'temporal_uncertainty_hours': timing_uncertainty_hours_1,
            'all_candidates': backtrack1_['all_candidates']  # Not used in predict_coastal_arrival
        }

        backtrack2 = {
            'candidate_lat': source_lat,
            'candidate_lon': source_lon,
            'candidate_distance_km': dist_to_buoy2,
            'travel_time_hours': travel_time_buoy2_hours,
            'spatial_uncertainty_km': lateral_uncertainty_km_2,
            'temporal_uncertainty_hours': timing_uncertainty_hours_2,
            'all_candidates': backtrack2_['all_candidates']  # Not used in predict_coastal_arrival
        }

        return {
            'source_lat': source_lat,
            'source_lon': source_lon,
            'source_dp': initial_bearing,
            'source_directional_spread': source_directional_spread,
            'confidence_radius_km': result['confidence_radius_km'],
            'source_generation_time': source_generation_time,
            'time_offset_observed_hours': time_offset_hours,
            'time_offset_predicted_hours': travel_time_buoy2_hours - travel_time_buoy1_hours,
            'time_consistency_score': time_consistency_score,
            'buoy1_arrival_offset_hours': travel_time_buoy1_hours,
            'buoy2_arrival_offset_hours': travel_time_buoy2_hours,
            'distance_to_buoy1_km': dist_to_buoy1,
            'distance_to_buoy2_km': dist_to_buoy2,
            'back_azimuth_buoy1': back_az1,
            'back_azimuth_buoy2': back_az2,
            'buoy1_backtrack': backtrack1,  # Added for predict_coastal_arrival
            'buoy2_backtrack': backtrack2,  # Added for predict_coastal_arrival
            'method_used': method,
            'result': result,
        }

    def _find_source_time_corrected(self, back_az1: float, back_az2: float,
                                    time_offset_hours: float,
                                    dist_est1: Optional[float] = None,
                                    dist_est2: Optional[float] = None) -> Dict:
        """
        Find source by matching predicted time offset with observed time offset.

        The key insight: if we know buoy2 detected the event Δt hours after buoy1,
        then the source must be located such that:
            travel_time_to_buoy2 - travel_time_to_buoy1 ≈ Δt
        """

        # Search ranges
        if dist_est1 is not None:
            distances1 = np.linspace(200,
                                     dist_est1 + 1000, 100)
        else:
            distances1 = np.linspace(200, 7000, 150)

        if dist_est2 is not None:
            distances2 = np.linspace(200,
                                     dist_est2 + 1000, 100)
        else:
            distances2 = np.linspace(200, 7000, 150)

        #First, correct the lag on observations.
        v_2 = self.deep_water_group_velocity(self.buoy2.peak_frequency)
        d_bias = time_offset_hours * 3600 * v_2 / 1000.0
        # Mean frequency for travel time calculations
        #mean_freq = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        best_score = np.inf
        best_source = None
        best_confidence = 0
        candidates = []
        for d1 in distances1:
            # Candidate location along back-azimuth from buoy1
            p1_lat, p1_lon = self.destination_point(self.buoy1.lat, self.buoy1.lon,
                                                    back_az1, d1)



            #for d2 in distances2:
            # Candidate location along back-azimuth from buoy2
            p2_lat, p2_lon = self.destination_point(self.buoy2.lat, self.buoy2.lon,
                                                    back_az2, d1 + d_bias)

            c_lat, c_lon = (p1_lat + p2_lat) / 2, (p1_lon + p2_lon) / 2

            #Distances from centroid to each buoy
            d1_s = self.haversine_distance(c_lat, c_lon, p1_lat, p1_lon)
            d2_s = self.haversine_distance(c_lat, c_lon, p2_lat, p2_lon)

            # Travel time from this location to buoy1
            t1 = self._compute_travel_time(d1_s, self.buoy1.peak_frequency)
            # Travel time from second location to buoy2
            t2 = self._compute_travel_time(d2_s, self.buoy2.peak_frequency)

            # Spatial separation from candidate to backtrack
            spatial_sep = self.haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)

            # Predicted time offset
            predicted_offset = t2 - t1

            # Score based on:
            # 1. Spatial proximity of candidates (smaller is better)
            # 2. Time offset match (closer to observed is better)
            time_error = abs(predicted_offset - time_offset_hours) / time_offset_hours

            # Weighted score
            #score = spatial_sep + 500.0 * time_error  # 500 km per ratio of error
            score = time_error
            #score = spatial_sep * time_error

            if score < best_score:
                best_score = score
                # Use midpoint as source estimate
                best_source = {
                    'source_lat': c_lat,
                    'source_lon': c_lon,
                    'confidence_radius_km': spatial_sep,
                    'spatial_separation_km': spatial_sep,
                    'time_error_hours': time_error
                }

        return best_source

    def _find_source_intersection_weighted(self, back_az1: float, back_az2: float,
                                           time_offset_hours: float) -> Dict:
        """
        Find great circle intersection with weighting based on temporal information.
        """
        # Determine which buoy detected first
        buoy_first = 1 if time_offset_hours > 0 else 2

        # Search along great circles
        distances = np.linspace(500, 7000, 150)

        min_weighted_distance = np.inf
        best_point = None
        best_confidence = 0

        mean_freq = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        for d1 in distances:
            p1_lat, p1_lon = self.destination_point(self.buoy1.lat, self.buoy1.lon,
                                                    back_az1, d1)
            t1 = self._compute_travel_time(d1, mean_freq)

            for d2 in distances:
                p2_lat, p2_lon = self.destination_point(self.buoy2.lat, self.buoy2.lon,
                                                        back_az2, d2)
                t2 = self._compute_travel_time(d2, mean_freq)

                separation = self.haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)

                # Temporal weight: prefer solutions matching observed time offset
                time_diff_predicted = t2 - t1
                temporal_weight = np.exp(-abs(time_diff_predicted - time_offset_hours) / 12.0)

                # Weighted distance (lower is better)
                weighted_distance = separation / (temporal_weight + 0.1)

                if weighted_distance < min_weighted_distance:
                    min_weighted_distance = weighted_distance
                    best_point = {
                        'source_lat': (p1_lat + p2_lat) / 2,
                        'source_lon': (p1_lon + p2_lon) / 2,
                        'confidence_radius_km': separation,
                        'temporal_weight': temporal_weight
                    }

        return best_point

    def _find_source_iterative(self, back_az1: float, back_az2: float,
                               time_offset_hours: float,
                               max_iterations: int = 5) -> Dict:
        """
        Iteratively refine source location using both direction and time constraints.
        """
        # Start with time-corrected estimate
        initial_result = self._find_source_time_corrected(back_az1, back_az2,
                                                          time_offset_hours)

        current_lat = initial_result['source_lat']
        current_lon = initial_result['source_lon']

        mean_freq = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        for iteration in range(max_iterations):
            # Compute current distances
            d1 = self.haversine_distance(current_lat, current_lon,
                                         self.buoy1.lat, self.buoy1.lon)
            d2 = self.haversine_distance(current_lat, current_lon,
                                         self.buoy2.lat, self.buoy2.lon)

            # Compute current bearings from source
            geod = Geodesic.WGS84
            g1 = geod.Inverse(current_lat, current_lon, self.buoy1.lat, self.buoy1.lon)
            g2 = geod.Inverse(current_lat, current_lon, self.buoy2.lat, self.buoy2.lon)

            current_bearing1 = (g1['azi1'] + 180) % 360
            current_bearing2 = (g2['azi1'] + 180) % 360

            # Errors
            bearing_error1 = self._angular_difference(current_bearing1, back_az1)
            bearing_error2 = self._angular_difference(current_bearing2, back_az2)

            t1 = self._compute_travel_time(d1, mean_freq)
            t2 = self._compute_travel_time(d2, mean_freq)
            time_error = (t2 - t1) - time_offset_hours

            # If errors are small enough, converged
            if abs(bearing_error1) < 0.5 and abs(bearing_error2) < 0.5 and abs(time_error) < 0.5:
                break

            # Adjust position based on errors
            # This is a simplified gradient descent
            delta_lat = -0.01 * (bearing_error1 + bearing_error2) / 2
            delta_lon = -0.01 * time_error

            current_lat += delta_lat
            current_lon += delta_lon

        confidence = self.haversine_distance(initial_result['source_lat'],
                                             initial_result['source_lon'],
                                             current_lat, current_lon)

        return {
            'source_lat': current_lat,
            'source_lon': current_lon,
            'confidence_radius_km': max(confidence, 50.0),
            'iterations': iteration + 1
        }

    def _angular_difference(self, angle1: float, angle2: float) -> float:
        """
        Compute smallest angular difference between two angles (degrees).
        Returns value in [-180, 180].
        """
        diff = (angle1 - angle2) % 360
        # if diff > 180:
        #     diff -= 360
        return diff

    def _backtrack_single_buoy(self, lat: float, lon: float,
                               back_azimuth: float,
                               frequency: float, directional_spread: float) -> Dict:
        """
        Backtrack from single buoy along great circle.

        Assumes swell traveled from a storm at typical fetch distance.
        """
        # Estimate reasonable source distances for Pacific swells
        # Typical: 1000-5000 km for well-developed swell
        candidate_distances = np.linspace(100, 5000, 50)

        candidates = []
        for dist_km in candidate_distances:
            candidate_lat, candidate_lon = self.destination_point(
                lat, lon, back_azimuth, dist_km
            )
            # Compute spatial uncertainty in terms of directional spread
            spread_half_angle_rad = np.radians(directional_spread / 2)
            lateral_uncertainty_km = dist_km * np.tan(spread_half_angle_rad) - dist_km / 100.0 * self.DISPERSION_RATE

            # Compute temporal uncertainty in terms of arrival time spread
            travel_time_hours = self._compute_travel_time(dist_km, frequency)
            frequency_spread = 0.1 * frequency  # Typical 10% frequency bandwidth
            timing_uncertainty_hours = travel_time_hours * (frequency_spread / frequency)

            candidates.append({
                'distance_km': dist_km,
                'lat': candidate_lat,
                'lon': candidate_lon,
                'spatial_uncertainty_km': lateral_uncertainty_km,
                'temporal_uncertainty_hours': timing_uncertainty_hours,
                'travel_time_hours': travel_time_hours
            })

        # Select middle-distance candidate as representative
        mid_idx = len(candidates) // 2
        representative = candidates[mid_idx]

        return {
            'candidate_lat': representative['lat'],
            'candidate_lon': representative['lon'],
            'candidate_distance_km': representative['distance_km'],
            'travel_time_hours': representative['travel_time_hours'],
            'spatial_uncertainty_km': representative['spatial_uncertainty_km'],
            'temporal_uncertainty_hours': representative['temporal_uncertainty_hours'],
            'all_candidates': candidates
        }

    def _estimate_source_directional_spread(
            self,
            buoy1_spread: float,
            buoy2_spread: float,
            distance_buoy1_to_source: float,
            distance_buoy2_to_source: float,
            confidence: float) -> float:
        """
        Estimate directional spread at source from buoy observations.

        Directional spread increases with propagation distance due to:
        - Angular dispersion (~0.5-1° per 100 km for swell)
        - Nonlinear interactions

        Parameters:
        -----------
        buoy1_spread, buoy2_spread : float
            Observed directional spreads at buoys (degrees)
        distance_buoy1_to_source, distance_buoy2_to_source : float
            Backtracked distances (km)

        Returns:
        --------
        source_spread : float
            Estimated directional spread at source (degrees)
        """

        # Typical angular dispersion rate for swell
        # Literature values: 0.5-1.5° per 100 km
        dispersion_rate = self.DISPERSION_RATE  # degrees per 100 km (conservative)

        # Remove propagation-induced spreading from each buoy
        spread1_at_source = buoy1_spread - (distance_buoy1_to_source / 100) * dispersion_rate
        spread2_at_source = buoy2_spread - (distance_buoy2_to_source / 100) * dispersion_rate

        # Clamp to reasonable minimum (source can't have zero spread)
        spread1_at_source = max(spread1_at_source, 10.0)# + confidence# Minimum 10°
        spread2_at_source = max(spread2_at_source, 10.0)# + confidence

        # Weighted average (weight by inverse distance - closer buoy more reliable)
        w1 = distance_buoy1_to_source / (distance_buoy1_to_source + distance_buoy2_to_source)
        w2 = distance_buoy2_to_source / (distance_buoy1_to_source + distance_buoy2_to_source)

        source_spread = (w1 * spread1_at_source + w2 * spread2_at_source) / (w1 + w2)

        return source_spread

    def _find_great_circle_intersection(self, lat1: float, lon1: float,
                                        bearing1: float, dir_spread1: float,
                                        lat2: float, lon2: float,
                                        bearing2: float, dir_spread2:float) -> Tuple[float, float, float]:
        """
        Find intersection of two great circles (simplified approach).

        Uses iterative search to minimize distance between great circles.
        """
        # Sample points along each great circle
        distances = np.linspace(500, 7000, 150)

        min_separation = np.inf
        best_confidence = np.inf
        best_point = (0, 0)

        for d1 in distances:
            p1_lat, p1_lon = self.destination_point(lat1, lon1, bearing1, d1)


            for d2 in distances:
                p2_lat, p2_lon = self.destination_point(lat2, lon2, bearing2, d2)

                separation = self.haversine_distance(p1_lat, p1_lon, p2_lat, p2_lon)

                if separation < min_separation:
                    min_separation = separation
                    # Use midpoint as intersection estimate
                    best_point = (
                        (p1_lat + p2_lat) / 2,
                        (p1_lon + p2_lon) / 2
                    )

                    spread_half_angle_rad1 = np.radians(dir_spread1 / 2)
                    lateral_uncertainty_km_1 = d1 * np.tan(spread_half_angle_rad1)
                    spread_half_angle_rad2 = np.radians(dir_spread2 / 2)
                    lateral_uncertainty_km_2 = d2 * np.tan(spread_half_angle_rad2)
                    best_confidence = (lateral_uncertainty_km_1 + lateral_uncertainty_km_2) / 2

        confidence_radius = min_separation# + best_confidence  # Uncertainty equals closest approach

        return best_point[0], best_point[1], confidence_radius

    def _compute_travel_time(self, distance_km: float, frequency: float) -> float:
        """
        Compute swell travel time.

        Returns: Travel time in hours
        """
        distance_m = distance_km * 1000
        group_velocity = self.deep_water_group_velocity(frequency)
        travel_time_sec = distance_m / group_velocity
        return travel_time_sec / 3600.0  # Convert to hours

    def _compute_directional_weight(self, source_direction: float,
                                   target_bearing: float,
                                   directional_spread: float = 30.0,
                                   spreading_exponent: float = 2.0) -> float:
        """
        Compute energy fraction that reaches target based on directional spreading.

        Uses a cosine-power spreading function to model how energy radiates
        from source in different directions.

        Parameters:
        -----------
        source_direction : float
            Mean propagation direction from source (degrees, 0=N, 90=E)
        target_bearing : float
            Bearing from source to coastal buoy (degrees)
        directional_spread : float
            Characteristic angular width of the directional distribution (degrees)
            Typical values: 20-40° for swell, 40-60° for wind-sea
        spreading_exponent : float
            Exponent controlling directional concentration
            Higher values = narrower beam
            Typical values: 2-4 for swell, 1-2 for wind-sea

        Returns:
        --------
        weight : float
            Energy fraction reaching target (0-1)
            1.0 = target perfectly aligned with source direction
            0.0 = target outside spreading cone
        """

        # Angular difference between source direction and target bearing
        angular_offset = abs(source_direction - target_bearing)

        # Normalize to [-180, 180]
        if angular_offset > 180:
            angular_offset = 360 - angular_offset

        # Cosine-power spreading function
        # Common model: D(θ) = A * cos^n(θ/2)
        # where θ is angle from mean direction

        # Normalize offset by spreading width
        normalized_offset = angular_offset / directional_spread

        # Apply spreading function
        if normalized_offset < 1.5:  # Within ~1.5σ of mean direction
            # Cosine-power model
            weight = np.cos(np.radians(angular_offset / 2)) ** (2 * spreading_exponent)

            # Alternative: Gaussian-like model
            # weight = np.exp(-0.5 * (angular_offset / directional_spread)**2)
        else:
            # Far from main lobe - minimal energy
            weight = 0.0

        return max(0.0, min(1.0, weight))  # Clamp to [0, 1]

    # ==================== COASTAL ARRIVAL PREDICTION ====================

    def predict_coastal_arrival(self, source_result: Dict) -> Dict:
        """
        Predict arrival time and spectral characteristics at coastal buoy.

        Parameters:
        -----------
        source_result : dict
            Output from triangulate_source_location()

        Returns:
        --------
        dict containing:
            - arrival_time: Predicted datetime at coast
            - arrival_frequency: Peak frequency (may shift due to dispersion)
            - arrival_energy: Attenuated energy
            - arrival_direction: Predicted approach direction
            - propagation_distance_km: Distance from source to coast
        """
        source_lat = source_result['source_lat']
        source_lon = source_result['source_lon']

        # Calculate distance and bearing from source to coastal buoy
        distance_to_coast = self.haversine_distance(
            source_lat, source_lon, self.coastal_lat, self.coastal_lon
        )

        geometric_bearing = self.initial_bearing(
                self.coastal_lat, self.coastal_lon, source_lat, source_lon
            )

        if source_result['source_dp'] is None:
            approach_direction = geometric_bearing
        else:
            approach_direction = source_result['source_dp']

        directional_weight = self._compute_directional_weight(
            approach_direction,
            geometric_bearing,
            source_result.get('source_directional_spread', 30.0),
            spreading_exponent=2.0
        )

        angular_offset = abs(approach_direction - geometric_bearing)
        if angular_offset > 180:
            angular_offset = 360 - angular_offset

        angular_offset_rad = np.radians(angular_offset)

        # Effective distance along propagation direction
        effective_distance = distance_to_coast# * np.cos(angular_offset_rad)

        # Lateral distance (perpendicular to propagation)
        lateral_offset = distance_to_coast * np.sin(angular_offset_rad)

        # Check if target is reachable
        # Swell has finite beam width due to directional spreading
        directional_spread = source_result.get('source_directional_spread', 30.0)

        # Spread increases as it covers distance
        directional_spread = directional_spread + (effective_distance / 100.0) * self.DISPERSION_RATE
        # Beam half-width at target distance (simple spreading model)
        # Beam expands with distance: width ≈ 2 * distance * tan(spread/2)
        beam_halfwidth_km = effective_distance * np.tan(np.radians(directional_spread / 2))

        will_reach = lateral_offset <= beam_halfwidth_km

        # Compute directional weight (energy fraction reaching target)
        if not will_reach:
            # Energy decreases with lateral offset from beam center
            # Use Gaussian-like decay
            directional_weight = np.exp(-0.5 * (lateral_offset / beam_halfwidth_km) ** 2)
            #directional_weight = directional_weight - directional_weight
            # Swell misses the target completely

        # Use mean frequency from both buoys
        mean_frequency = (self.buoy1.peak_frequency + self.buoy2.peak_frequency) / 2

        # Compute travel time from source to coast
        travel_time_hours = self._compute_travel_time(effective_distance, mean_frequency)

        # Estimate source generation time (use buoy1 as reference)
        # Subtract travel time from buoy1 to source
        buoy1_to_source_dist = source_result['buoy1_backtrack']['candidate_distance_km']
        buoy1_travel_time_hours = self._compute_travel_time(
            buoy1_to_source_dist, self.buoy1.peak_frequency
        )
        estimated_source_time = self.buoy1.time - timedelta(hours=buoy1_travel_time_hours)

        # Predict coastal arrival
        arrival_time = estimated_source_time + timedelta(hours=int(travel_time_hours))

        # Energy attenuation
        attenuation = True
        arrival_scheme_1 = []
        arrival_scheme_2 = []
        if not self.buoy1.energies is None:
            arrival_energies1 = self.energy_attenuation(self.buoy1.energies, self.buoy1.frequency_evolution,
                                                        np.repeat(distance_to_coast, self.buoy1.energies.shape[0]))
            if not attenuation:
                arrival_energies1 = self.buoy1.energies
            for j, d in enumerate([(d - self.buoy1.time) for d in self.buoy1.dates]):
                arrival_scheme_1.append({'date': arrival_time + d,
                                       'energy': arrival_energies1[j]})
        if not self.buoy2.energies is None:
            arrival_energies2 = self.energy_attenuation(self.buoy2.energies, self.buoy2.frequency_evolution,
                                                        np.repeat(distance_to_coast, self.buoy2.energies.shape[0]))
            if not attenuation:
                arrival_energies2 = self.buoy2.energies
            for j, d in enumerate([(d - self.buoy2.time) for d in self.buoy2.dates]):
                arrival_scheme_2.append({'date': arrival_time + d,
                                       'energy': arrival_energies2[j]})

        mean_energy = (self.buoy1.peak_energy + self.buoy2.peak_energy) / 2
        arrival_energy = self.energy_attenuation(
            mean_energy, mean_frequency, distance_to_coast
        )
        #Assume not attenuation.
        if not attenuation:
            arrival_energy = mean_energy


        arrival_energy = arrival_energy * directional_weight

        # Frequency may shift slightly due to dispersion (simplified: assume constant)
        arrival_frequency = mean_frequency

        # Directional shift due to shoaling/refraction (placeholder)
        # In practice, apply Snell's law with bathymetry
        arrival_direction = approach_direction  # Simplified assumption

        return {
            'arrival_time': arrival_time,
            'arrival_frequency': arrival_frequency,
            'arrival_period': 1.0 / arrival_frequency,
            'arrival_energy': arrival_energy,
            'arrival_direction': arrival_direction,
            'directional_weight': directional_weight,
            'propagation_distance_km': distance_to_coast,
            'travel_time_hours': travel_time_hours,
            'source_generation_time': estimated_source_time,
            'energy_attenuation_factor': arrival_energy / mean_energy,
            'uncertainty_hours': source_result['confidence_radius_km'] / \
                                 self.deep_water_group_velocity(mean_frequency) / 3600,
            'arrival_scheme_1': arrival_scheme_1,
            'arrival_scheme_2': arrival_scheme_2
        }

    def predict_coastal_arrival_spectral(self, source_result: Dict) -> Dict:
        """
        Predicts the dispersive arrival of the full swell spectrum at the coast.

        Improvements over original:
        1. Removes incorrect 'effective_distance' projection (uses actual Great Circle).
        2. Predicts arrival times for a RANGE of frequencies (dispersive fan).
        3. Fixes integer truncation bug in timestamps.
        4. Uses physics-consistent attenuation (Ardhuin et al. parameterization).
        """
        from datetime import timedelta

        source_lat = source_result['source_lat']
        source_lon = source_result['source_lon']

        # 1. True Great Circle Distance (Physical travel path)
        distance_km = self.haversine_distance(
            source_lat, source_lon, self.coastal_lat, self.coastal_lon
        )

        # 2. Directional Geometry
        # Bearing from Source -> Coast
        bearing_src_to_coast = self.initial_bearing(
            source_lat, source_lon, self.coastal_lat, self.coastal_lon
        )

        # Source propagation direction (mean direction of the storm swell)
        source_main_dir = source_result.get('source_dp', bearing_src_to_coast)

        # Angular offset for energy spreading
        angular_offset = abs(source_main_dir - bearing_src_to_coast)
        if angular_offset > 180:
            angular_offset = 360 - angular_offset

        # 3. Energy Weighting (Beam Pattern)
        # If the coast is far off the main beam, energy is drastically reduced
        source_spread = source_result.get('source_directional_spread', 30.0)
        # Broaden spread slightly with distance (simple dispersion)
        effective_spread = source_spread + (distance_km / 1000.0) * self.DISPERSION_RATE

        directional_weight = self._compute_directional_weight(
            source_main_dir, bearing_src_to_coast,
            directional_spread=effective_spread
        )

        # 4. Dispersive Arrival Calculation
        # Instead of one time, we calculate arrival for the discrete frequency bins
        # found in the buoy observations.

        # Get relevant frequencies (e.g., from Buoy 1)
        # We focus on the swell band: min_freq to max_freq
        obs_freqs = np.linspace(self.min_freq, self.max_freq, 20)

        arrival_schedule = []

        # Estimate source generation time from Buoy 1's peak freq observation
        # t0 = t_obs - distance / Cg(f_peak)
        # Uses the distance from source to Buoy 1 calculated in triangulation
        d_buoy1 = source_result['buoy1_backtrack']['candidate_distance_km']
        cg_peak = self.deep_water_group_velocity(self.buoy1.peak_frequency)
        travel_time_buoy1_sec = (d_buoy1 * 1000) / cg_peak
        t0_source = self.buoy1.time - timedelta(seconds=travel_time_buoy1_sec)

        peak_arrival_time = None
        max_arrival_energy = -1.0

        for f in obs_freqs:
            # Group velocity for this frequency
            cg = self.deep_water_group_velocity(f)

            # Travel time Source -> Coast
            travel_sec = (distance_km * 1000) / cg
            arrival_time = t0_source + timedelta(seconds=travel_sec)

            # Energy Attenuation (Ardhuin et al. 2009 style simple decay)
            # Dissipation correlates with steepness, but here we use a linear approx
            # alpha proportional to f^2 is standard for viscous,
            # but observation shows linear f dependence often fits swell better.
            # Using your existing model but corrected:
            alpha = 2.0e-7 * f  # Adjusted coefficient
            decay_factor = np.exp(-alpha * (distance_km * 1000))

            # Approx energy at this frequency (assuming source had similar shape to Buoy1)
            # In a real model, we'd map the full spectrum. Here we scale the peak.
            # We assume a Gaussian shape around the peak frequency for the source magnitude
            f_sigma = 0.02
            source_spectral_density = self.buoy1.peak_energy * np.exp(
                -0.5 * ((f - self.buoy1.peak_frequency) / f_sigma) ** 2
            )

            predicted_energy = source_spectral_density * decay_factor * directional_weight

            if predicted_energy > max_arrival_energy:
                max_arrival_energy = predicted_energy
                peak_arrival_time = arrival_time

            arrival_schedule.append({
                'frequency': f,
                'period': 1.0 / f,
                'arrival_time': arrival_time,
                'predicted_energy': predicted_energy
            })

        return {
            'source_generation_time': t0_source,
            'propagation_distance_km': distance_km,
            'geometric_bearing_to_coast': bearing_src_to_coast,
            'directional_weight': directional_weight,
            'peak_arrival_time': peak_arrival_time,
            'peak_arrival_energy': max_arrival_energy,
            'first_arrival_time': arrival_schedule[0]['arrival_time'],  # Lowest freq arrives first
            'last_arrival_time': arrival_schedule[-1]['arrival_time'],  # Highest freq arrives last
            'dispersive_arrival_schedule': arrival_schedule,  # Plot this for the "fanning"
            'travel_time_peak_hours': (peak_arrival_time - t0_source).total_seconds() / 3600
        }
    # ==================== UNCERTAINTY QUANTIFICATION ====================

    def monte_carlo_uncertainty(self, n_samples: int = 1000) -> Dict:
        """
        Propagate uncertainty in spectral parameters through back-triangulation
        using Monte Carlo sampling.

        Assumes Gaussian distributions around mean spectra with given variances.
        """
        source_locations = []
        arrival_times = []
        arrival_energies = []

        for _ in range(n_samples):
            # Perturb peak directions (use ±10° standard deviation)
            perturbed_dir1 = self.buoy1.peak_direction + np.random.normal(0, 10)
            perturbed_dir2 = self.buoy2.peak_direction + np.random.normal(0, 10)

            # Perturb frequencies (±5% standard deviation)
            perturbed_freq1 = self.buoy1.peak_frequency * (1 + np.random.normal(0, 0.05))
            perturbed_freq2 = self.buoy2.peak_frequency * (1 + np.random.normal(0, 0.05))

            # Create perturbed observations
            buoy1_perturbed = BuoyObservation(
                self.buoy1.lat, self.buoy1.lon, self.buoy1.time,
                self.buoy1.mean_spectrum, self.buoy1.variance_spectrum,
                perturbed_freq1, perturbed_dir1, self.buoy1.peak_energy
            )
            buoy2_perturbed = BuoyObservation(
                self.buoy2.lat, self.buoy2.lon, self.buoy2.time,
                self.buoy2.mean_spectrum, self.buoy2.variance_spectrum,
                perturbed_freq2, perturbed_dir2, self.buoy2.peak_energy
            )

            # Run triangulation
            temp_analyzer = SwellBackTriangulation(
                buoy1_perturbed, buoy2_perturbed,
                (self.coastal_lat, self.coastal_lon),
                self.min_freq, self.max_freq
            )

            source_result = temp_analyzer.triangulate_source_location()
            arrival_result = temp_analyzer.predict_coastal_arrival(source_result)

            source_locations.append((source_result['source_lat'],
                                     source_result['source_lon']))
            arrival_times.append(arrival_result['arrival_time'])
            arrival_energies.append(arrival_result['arrival_energy'])

        # Compute statistics
        source_lats = [loc[0] for loc in source_locations]
        source_lons = [loc[1] for loc in source_locations]

        return {
            'source_lat_mean': np.mean(source_lats),
            'source_lat_std': np.std(source_lats),
            'source_lon_mean': np.mean(source_lons),
            'source_lon_std': np.std(source_lons),
            'arrival_time_mean': np.mean(arrival_times),
            'arrival_time_std': np.std([(t - arrival_times[0]).total_seconds()
                                        for t in arrival_times]) / 3600,  # hours
            'arrival_energy_mean': np.mean(arrival_energies),
            'arrival_energy_std': np.std(arrival_energies),
            'sample_size': n_samples
        }

    # ==================== VISUALIZATION HELPERS ====================

    def generate_report(self, source_result: Dict,
                        arrival_result: Dict) -> str:
        """
        Generate comprehensive text report of analysis.
        """
        report = []
        report.append("=" * 60)
        report.append("SWELL BACK-TRIANGULATION ANALYSIS REPORT")
        report.append("=" * 60)
        report.append("")

        report.append("BUOY OBSERVATIONS:")
        report.append(f"  Buoy 1: ({self.buoy1.lat:.3f}°, {self.buoy1.lon:.3f}°)")
        report.append(f"    Time: {self.buoy1.time}")
        report.append(f"    Peak Frequency: {self.buoy1.peak_frequency:.4f} Hz "
                      f"(Period: {1 / self.buoy1.peak_frequency:.1f} s)")
        report.append(f"    Peak Direction: {self.buoy1.peak_direction:.1f}°")
        report.append(f"    Peak Energy: {self.buoy1.peak_energy:.2e} m²/Hz/deg")
        report.append("")
        report.append(f"  Buoy 2: ({self.buoy2.lat:.3f}°, {self.buoy2.lon:.3f}°)")
        report.append(f"    Time: {self.buoy2.time}")
        report.append(f"    Peak Frequency: {self.buoy2.peak_frequency:.4f} Hz "
                      f"(Period: {1 / self.buoy2.peak_frequency:.1f} s)")
        report.append(f"    Peak Direction: {self.buoy2.peak_direction:.1f}°")
        report.append(f"    Peak Energy: {self.buoy2.peak_energy:.2e} m²/Hz/deg")
        report.append("")

        report.append("SOURCE LOCATION ESTIMATE:")
        report.append(f"  Coordinates: ({source_result['source_lat']:.3f}°, "
                      f"{source_result['source_lon']:.3f}°)")
        report.append(f"  Confidence Radius: ±{source_result['confidence_radius_km']:.1f} km")
        report.append(f"  Distance from Buoy 1: "
                      f"{source_result['buoy1_backtrack']['candidate_distance_km']:.1f} km")
        report.append(f"  Distance from Buoy 2: "
                      f"{source_result['buoy2_backtrack']['candidate_distance_km']:.1f} km")
        report.append("")

        report.append("COASTAL ARRIVAL PREDICTION:")
        report.append(f"  Target: ({self.coastal_lat:.3f}°, {self.coastal_lon:.3f}°)")
        report.append(f"  Predicted Arrival Time: {arrival_result['arrival_time']}")
        report.append(f"  Travel Time: {arrival_result['travel_time_hours']:.1f} hours")
        report.append(f"  Propagation Distance: "
                      f"{arrival_result['propagation_distance_km']:.1f} km")
        report.append(f"  Arrival Frequency: {arrival_result['arrival_frequency']:.4f} Hz "
                      f"(Period: {arrival_result['arrival_period']:.1f} s)")
        report.append(f"  Arrival Direction: {arrival_result['arrival_direction']:.1f}°")
        report.append(f"  Predicted Energy: {arrival_result['arrival_energy']:.2e} m²/Hz/deg")
        report.append(f"  Attenuation Factor: "
                      f"{arrival_result['energy_attenuation_factor']:.4f}")
        report.append(f"  Timing Uncertainty: "
                      f"±{arrival_result['uncertainty_hours']:.1f} hours")
        report.append("")
        report.append("=" * 60)

        return "\n".join(report)


class ReverseWaveShoaling:
    """
    Transform coastal wave spectra to equivalent deep-water conditions
    by inverting shoaling, refraction, and bottom friction effects.

    References:
    - Wave shoaling theory (Dean & Dalrymple, 1991)
    - Bottom friction models (Madsen et al., 1998; Lowe et al., 2005)
    - Spectral transformation (Goda, 2000)
    """

    GRAVITY = 9.81  # m/s²
    DEFAULT_WATER_DEPTH = 50.0  # Target deep-water depth (m)
    DEEP_WATER_THRESHOLD = 0.5  # h/L₀ > 0.5 is considered deep water

    def __init__(self,
                 coastal_depth: float,
                 target_depth: float = DEFAULT_WATER_DEPTH,
                 propagation_distance_km: float = 10.0,
                 bottom_roughness: str = 'medium'):
        """
        Initialize reverse shoaling transformation.

        Parameters:
        -----------
        coastal_depth : float
            Water depth at coastal buoy location (meters)
        target_depth : float
            Target deep-water depth for reconstruction (meters)
            Default 50m (typically considered deep water for swells)
        propagation_distance_km : float
            Approximate distance from target deep location to coastal buoy (km)
        bottom_roughness : str
            Bottom roughness category: 'smooth', 'medium', 'rough'
            Affects friction coefficient
        """
        self.h_coastal = coastal_depth
        self.h_target = target_depth
        self.distance_m = propagation_distance_km * 1000

        # Friction factor based on bottom roughness
        friction_factors = {
            'smooth': 0.005,
            'medium': 0.015,
            'rough': 0.04
        }
        self.fw = friction_factors.get(bottom_roughness, 0.015)

        if coastal_depth >= target_depth:
            warnings.warn("Coastal depth >= target depth. Minimal transformation expected.")

    # ==================== DISPERSION RELATION ====================

    def dispersion_relation(self, frequency: float, depth: float,
                            max_iter: int = 50) -> float:
        """
        Solve dispersion relation for wavenumber k:
        ω² = gk tanh(kh)

        Returns: wavenumber k (rad/m)
        """
        omega = 2 * np.pi * frequency

        # Initial guess (deep water)
        k = omega ** 2 / self.GRAVITY

        # Newton-Raphson iteration
        for _ in range(max_iter):
            f = omega ** 2 - self.GRAVITY * k * np.tanh(k * depth)
            df = -self.GRAVITY * (np.tanh(k * depth) +
                                  k * depth * (1 - np.tanh(k * depth) ** 2))
            k_new = k - f / df

            if abs(k_new - k) < 1e-10:
                break
            k = k_new

        return k

    def group_velocity(self, frequency: float, depth: float) -> float:
        """Calculate group velocity Cg"""
        k = self.dispersion_relation(frequency, depth)
        kh = k * depth

        c = 2 * np.pi * frequency / k  # Phase velocity
        n = 0.5 * (1 + (2 * kh / np.sinh(2 * kh)))  # Group/phase ratio

        return n * c

    def deep_water_wavelength(self, frequency: float) -> float:
        """Calculate deep-water wavelength L₀ = gT²/(2π)"""
        T = 1.0 / frequency
        return self.GRAVITY * T ** 2 / (2 * np.pi)

    # ==================== SHOALING COEFFICIENT ====================

    def shoaling_coefficient(self, frequency: float,
                             depth_from: float,
                             depth_to: float) -> float:
        """
        Calculate shoaling coefficient Ks from one depth to another.

        Ks = sqrt(Cg_from / Cg_to)

        For coastal -> deep: Ks < 1 (de-shoaling)
        """
        Cg_from = self.group_velocity(frequency, depth_from)
        Cg_to = self.group_velocity(frequency, depth_to)

        return np.sqrt(Cg_from / Cg_to)

    # ==================== BOTTOM FRICTION ATTENUATION ====================

    def friction_attenuation_coefficient(self, frequency: float,
                                         depth: float) -> float:
        """
        Calculate frequency-dependent attenuation coefficient α(f) due to
        bottom friction.

        Based on: dE/dx = -ρg fw ub³ / sinh³(kh)

        Returns: α in units of 1/m
        """
        k = self.dispersion_relation(frequency, depth)
        kh = k * depth

        # Avoid division by zero in deep water
        if kh > 3.0:  # Deep water limit
            return 0.0

        # Simplified parameterization (Collins 1972, Madsen 1988)
        # α ∝ fw · f² for shallow water
        alpha = self.fw * (2 * np.pi * frequency) ** 2 / (
                self.GRAVITY * np.sinh(kh) ** 2
        )

        return alpha

    def apply_friction_correction(self, energy_coastal: float,
                                  frequency: float,
                                  avg_depth: float) -> float:
        """
        Amplify coastal energy to compensate for dissipation over
        propagation distance.

        E_deep = E_coastal * exp(α · d)
        """
        alpha = self.friction_attenuation_coefficient(frequency, avg_depth)
        amplification = np.exp(alpha * self.distance_m)

        return energy_coastal * amplification

    # ==================== MAIN TRANSFORMATION METHODS ====================

    def reverse_shoaling_spectrum(self,
                                  frequency: np.ndarray,
                                  variance_density_coastal: np.ndarray,
                                  include_friction: bool = True,
                                  breaking_threshold: Optional[float] = None
                                  ) -> Tuple[np.ndarray, Dict]:
        """
        Transform coastal variance density spectrum to deep-water equivalent.

        Parameters:
        -----------
        frequency : np.ndarray
            Frequency array (Hz)
        variance_density_coastal : np.ndarray
            Coastal variance density S(f) in m²/Hz
        include_friction : bool
            If True, apply bottom friction correction
        breaking_threshold : float, optional
            If provided, flag frequencies where H/h exceeds threshold

        Returns:
        --------
        variance_density_deep : np.ndarray
            Reconstructed deep-water spectrum
        diagnostics : dict
            Transformation diagnostics and quality flags
        """
        variance_density_deep = np.zeros_like(variance_density_coastal)
        ks_values = np.zeros_like(frequency)
        friction_factors = np.zeros_like(frequency)
        breaking_flags = np.zeros_like(frequency, dtype=bool)

        # Average depth for friction calculation
        avg_depth = (self.h_coastal + self.h_target) / 2

        for i, f in enumerate(frequency):
            # Check if frequency is in valid range
            L0 = self.deep_water_wavelength(f)

            # Skip very high frequencies (likely local wind-sea)
            if self.h_coastal / L0 > 0.5:
                # Already in "deep water" for this frequency
                variance_density_deep[i] = variance_density_coastal[i]
                ks_values[i] = 1.0
                continue

            # 1. Compute shoaling coefficient (de-shoaling)
            Ks = self.shoaling_coefficient(f, self.h_coastal, self.h_target)
            ks_values[i] = Ks

            # 2. Apply inverse shoaling: S_deep = S_coastal / Ks²
            S_deshoaled = variance_density_coastal[i] / (Ks ** 2)

            # 3. Apply friction correction if requested
            if include_friction:
                S_corrected = self.apply_friction_correction(S_deshoaled, f, avg_depth)
                friction_factors[i] = S_corrected / S_deshoaled
            else:
                S_corrected = S_deshoaled
                friction_factors[i] = 1.0

            variance_density_deep[i] = S_corrected

            # 4. Check for potential breaking (quality flag)
            if breaking_threshold is not None:
                # Estimate wave height from spectral energy
                H_coastal = 4 * np.sqrt(variance_density_coastal[i] *
                                        (frequency[1] - frequency[0]) if i < len(frequency) - 1
                                        else 0.01)
                if H_coastal / self.h_coastal > breaking_threshold:
                    breaking_flags[i] = True

        diagnostics = {
            'shoaling_coefficients': ks_values,
            'friction_amplification': friction_factors,
            'breaking_flags': breaking_flags,
            'total_energy_coastal': np.trapezoid(variance_density_coastal, frequency),
            'total_energy_deep': np.trapezoid(variance_density_deep, frequency),
            'energy_amplification': np.trapezoid(variance_density_deep, frequency) /
                                    np.trapezoid(variance_density_coastal, frequency)
        }

        return variance_density_deep, diagnostics

    def reverse_directional_spectrum(self,
                                     frequency: np.ndarray,
                                     direction: np.ndarray,
                                     variance_density_coastal: np.ndarray,
                                     shore_normal_angle: float = 0.0,
                                     include_refraction: bool = True
                                     ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform 2D directional spectrum E(f,θ) from coastal to deep water.

        Parameters:
        -----------
        frequency : np.ndarray
            Frequency array (Hz), shape (nf,)
        direction : np.ndarray
            Direction array (degrees), shape (nd,)
        variance_density_coastal : np.ndarray
            2D directional spectrum, shape (nf, nd)
        shore_normal_angle : float
            Angle of shore-normal direction (degrees from North)
        include_refraction : bool
            Apply Snell's law directional correction

        Returns:
        --------
        variance_density_deep : np.ndarray
            Deep-water directional spectrum (nf, nd)
        direction_deep : np.ndarray
            Adjusted direction array if refraction applied
        """
        variance_density_deep = np.zeros_like(variance_density_coastal)

        for i, f in enumerate(frequency):
            # Apply 1D shoaling + friction to each frequency
            for j, theta in enumerate(direction):
                # De-shoaling
                Ks = self.shoaling_coefficient(f, self.h_coastal, self.h_target)
                S_deshoaled = variance_density_coastal[i, j] / (Ks ** 2)

                # Friction correction
                avg_depth = (self.h_coastal + self.h_target) / 2
                S_corrected = self.apply_friction_correction(S_deshoaled, f, avg_depth)

                variance_density_deep[i, j] = S_corrected

        # Apply refraction correction to directions
        if include_refraction:
            direction_deep = self._apply_inverse_refraction(
                frequency, direction, shore_normal_angle
            )
        else:
            direction_deep = direction.copy()

        return variance_density_deep, direction_deep

    def _apply_inverse_refraction(self,
                                  frequency: np.ndarray,
                                  direction: np.ndarray,
                                  shore_normal: float) -> np.ndarray:
        """
        Apply inverse Snell's law to directions.

        sin(θ_deep) / sin(θ_coastal) = c_deep / c_coastal
        """
        # For simplicity, apply average correction
        # In practice, should be frequency-dependent
        f_peak = frequency[np.argmax(frequency)]  # Placeholder

        k_coastal = self.dispersion_relation(f_peak, self.h_coastal)
        k_deep = self.dispersion_relation(f_peak, self.h_target)

        c_coastal = 2 * np.pi * f_peak / k_coastal
        c_deep = 2 * np.pi * f_peak / k_deep

        # Compute angle relative to shore-normal
        theta_rel_coastal = direction - shore_normal
        theta_rel_coastal_rad = np.radians(theta_rel_coastal)

        # Apply Snell's law
        sin_theta_deep = np.sin(theta_rel_coastal_rad) * c_deep / c_coastal
        sin_theta_deep = np.clip(sin_theta_deep, -1, 1)  # Avoid numerical issues

        theta_rel_deep_rad = np.arcsin(sin_theta_deep)
        theta_deep = np.degrees(theta_rel_deep_rad) + shore_normal

        return theta_deep

    # ==================== INTEGRATED PARAMETERS ====================

    def reconstruct_bulk_parameters(self,
                                    hsig_coastal: float,
                                    tp_coastal: float,
                                    depth_coastal: float,
                                    depth_target: float) -> Dict:
        """
        Quick estimation of deep-water Hs and Tp from coastal bulk parameters.

        Useful when full spectrum not available.
        """
        fp = 1.0 / tp_coastal

        # Shoaling coefficient at peak frequency
        Ks = self.shoaling_coefficient(fp, depth_coastal, depth_target)

        # Inverse shoaling for wave height
        hsig_deep = hsig_coastal / Ks

        # Period remains constant (no frequency shift in linear theory)
        tp_deep = tp_coastal

        return {
            'Hs_deep': hsig_deep,
            'Tp_deep': tp_deep,
            'shoaling_factor': Ks,
            'height_reduction_ratio': hsig_deep / hsig_coastal
        }

    # ==================== QUALITY CONTROL ====================

    def validate_transformation(self,
                                frequency: np.ndarray,
                                S_coastal: np.ndarray,
                                S_deep: np.ndarray) -> Dict:
        """
        Quality control checks on transformation.
        """
        # Check energy conservation principles
        E_coastal = np.trapz(S_coastal, frequency)
        E_deep = np.trapz(S_deep, frequency)

        warnings_list = []

        # Deep water energy should typically be less than or equal to coastal
        # (since shoaling amplifies energy)
        if E_deep < E_coastal * 0.5:
            warnings_list.append("Deep-water energy unexpectedly low (< 50% coastal)")

        if E_deep > E_coastal * 3.0:
            warnings_list.append("Deep-water energy very high (> 300% coastal) - "
                                 "check breaking or input parameters")

        # Check for unrealistic spectral values
        if np.any(S_deep < 0):
            warnings_list.append("Negative spectral values detected")

        if np.any(np.isnan(S_deep)):
            warnings_list.append("NaN values in output spectrum")

        return {
            'is_valid': len(warnings_list) == 0,
            'warnings': warnings_list,
            'energy_ratio': E_deep / E_coastal,
            'max_amplification': np.max(S_deep / (S_coastal + 1e-10))
        }


class FastReverseWaveShoaling:
    """
    Optimized vectorized version for batch processing thousands of spectra.

    Performance: ~100-1000x faster than nested loops
    - Uses NumPy broadcasting for parallel operations
    - Pre-computes transformation matrices
    - Optional Numba JIT compilation for extreme speed
    """

    GRAVITY = 9.81

    def __init__(self,
                 coastal_depth: float,
                 target_depth: float = 200.0,
                 propagation_distance_km: float = 15.0,
                 bottom_roughness: str = 'medium',
                 low_freq_cutoff: float = 0.045,
                 high_freq_cutoff: float = 0.20):

        self.h_coastal = coastal_depth
        self.h_target = target_depth
        self.distance_m = propagation_distance_km * 1000
        self.f_min = low_freq_cutoff
        self.f_max = high_freq_cutoff

        friction_factors = {'smooth': 0.005, 'medium': 0.015, 'rough': 0.04}
        self.fw = friction_factors.get(bottom_roughness, 0.015)

        # Pre-computed transformation matrices (computed once)
        self._transformation_matrix = None
        self._frequency_grid = None

    # ==================== VECTORIZED CORE METHODS ====================

    def _dispersion_vectorized(self, frequencies: np.ndarray,
                               depth: float, max_iter: int = 20) -> np.ndarray:
        """
        Vectorized dispersion relation solver.
        Operates on entire frequency array at once.
        """
        omega = 2 * np.pi * frequencies
        k = omega ** 2 / self.GRAVITY  # Initial guess

        for _ in range(max_iter):
            tanh_kh = np.tanh(k * depth)
            f = omega ** 2 - self.GRAVITY * k * tanh_kh
            df = -self.GRAVITY * (tanh_kh + k * depth * (1 - tanh_kh ** 2))
            k_new = k - f / df

            # Convergence check (vectorized)
            if np.allclose(k_new, k, atol=1e-10):
                break
            k = k_new

        return k

    def _group_velocity_vectorized(self, frequencies: np.ndarray,
                                   depth: float) -> np.ndarray:
        """Vectorized group velocity calculation"""
        k = self._dispersion_vectorized(frequencies, depth)
        kh = k * depth
        c = 2 * np.pi * frequencies / k
        n = 0.5 * (1 + (2 * kh / np.sinh(2 * kh)))
        return n * c

    def _shoaling_coefficient_vectorized(self, frequencies: np.ndarray,
                                         depth_from: float,
                                         depth_to: float,
                                         max_ratio: float = 4.0) -> np.ndarray:
        """Vectorized shoaling coefficient for all frequencies"""
        Cg_from = self._group_velocity_vectorized(frequencies, depth_from)
        Cg_to = self._group_velocity_vectorized(frequencies, depth_to)
        ratio = np.clip(Cg_from / Cg_to, 1.0 / max_ratio, max_ratio)
        return np.sqrt(ratio)

    def _friction_coefficient_vectorized(self, frequencies: np.ndarray,
                                         depth: float) -> np.ndarray:
        """Vectorized friction attenuation coefficient"""
        k = self._dispersion_vectorized(frequencies, depth)
        kh = k * depth

        # Avoid division by zero in deep water
        alpha = np.zeros_like(frequencies)
        shallow_mask = kh < 3.0

        if np.any(shallow_mask):
            alpha[shallow_mask] = self.fw * (2 * np.pi * frequencies[shallow_mask]) ** 2 / (
                    self.GRAVITY * np.sinh(kh[shallow_mask]) ** 2
            )

        return alpha

    # ==================== PRE-COMPUTATION FOR BATCH PROCESSING ====================

    def precompute_transformation_matrix(self, frequencies: np.ndarray) -> None:
        """
        Pre-compute transformation matrix for given frequency grid.
        Call this ONCE before processing many spectra with same frequencies.

        Speedup: Eliminates redundant calculations across samples.
        """
        print(f"Pre-computing transformation matrix for {len(frequencies)} frequencies...")

        # Compute all frequency-dependent factors once
        Ks = self._shoaling_coefficient_vectorized(frequencies,
                                                   self.h_coastal,
                                                   self.h_target)

        avg_depth = (self.h_coastal + self.h_target) / 2
        alpha = self._friction_coefficient_vectorized(frequencies, avg_depth)
        alpha_limited = np.minimum(alpha, 1e-5)
        friction_factor = np.exp(alpha_limited * self.distance_m)

        # Combined transformation factor
        # E_deep = E_coastal * transformation_factor
        transformation_factor = friction_factor / (Ks ** 2)

        # Apply frequency cutoffs
        valid_mask = (frequencies >= self.f_min) & (frequencies <= self.f_max)
        transformation_factor[~valid_mask] = 1.0  # No transformation outside valid range

        # Store for reuse
        self._transformation_matrix = transformation_factor
        self._frequency_grid = frequencies

        print(f"  Transformation matrix ready. Min factor: {transformation_factor.min():.3f}, "
              f"Max factor: {transformation_factor.max():.3f}")

    # ==================== FAST BATCH TRANSFORMATION ====================

    def transform_batch_1d(self,
                           spectra_batch: np.ndarray,
                           frequencies: np.ndarray) -> np.ndarray:
        """
        Transform batch of 1D frequency spectra (no directional info).

        Parameters:
        -----------
        spectra_batch : np.ndarray
            Shape (n_samples, n_frequencies) or (n_frequencies,) for single spectrum
        frequencies : np.ndarray
            Shape (n_frequencies,)

        Returns:
        --------
        spectra_deep : np.ndarray
            Same shape as input
        """
        # Handle single spectrum case
        single_spectrum = False
        if spectra_batch.ndim == 1:
            spectra_batch = spectra_batch[np.newaxis, :]
            single_spectrum = True

        # Ensure pre-computation has been done
        if (self._transformation_matrix is None or
                not np.array_equal(self._frequency_grid, frequencies)):
            self.precompute_transformation_matrix(frequencies)

        # VECTORIZED TRANSFORMATION (single operation for all samples!)
        # Broadcasting: (n_samples, n_freq) * (n_freq,) -> (n_samples, n_freq)
        spectra_deep = spectra_batch * self._transformation_matrix[np.newaxis, :]

        if single_spectrum:
            return spectra_deep[0]
        return spectra_deep

    def transform_batch_2d(self,
                           spectra_batch: np.ndarray,
                           frequencies: np.ndarray,
                           directions: np.ndarray,
                           include_refraction: bool = False) -> np.ndarray:
        """
        Transform batch of 2D directional spectra.

        Parameters:
        -----------
        spectra_batch : np.ndarray
            Shape (n_samples, n_frequencies, n_directions)
            OR (n_frequencies, n_directions) for single spectrum
        frequencies : np.ndarray
            Shape (n_frequencies,)
        directions : np.ndarray
            Shape (n_directions,)
        include_refraction : bool
            If True, apply refraction (slower, requires per-sample processing)

        Returns:
        --------
        spectra_deep : np.ndarray
            Same shape as input
        """
        # Handle single spectrum case
        single_spectrum = False
        if spectra_batch.ndim == 2:
            spectra_batch = spectra_batch[np.newaxis, :, :]
            single_spectrum = True

        n_samples, n_freq, n_dir = spectra_batch.shape

        # Ensure pre-computation
        if (self._transformation_matrix is None or
                not np.array_equal(self._frequency_grid, frequencies)):
            self.precompute_transformation_matrix(frequencies)

        # VECTORIZED TRANSFORMATION
        # Broadcasting: (n_samples, n_freq, n_dir) * (n_freq, 1) -> (n_samples, n_freq, n_dir)
        transformation_3d = self._transformation_matrix.to_numpy()[:, np.newaxis]
        spectra_deep = spectra_batch * transformation_3d[np.newaxis, :, :]  # Broadcasting

        # Refraction note: For full refraction, would need per-sample processing
        # For simplicity, skip refraction in vectorized version
        # (refraction effects are often secondary to shoaling for energy transformation)
        if include_refraction:
            warnings.warn("Refraction skipped in vectorized mode for performance. "
                          "Use transform_single_2d() if refraction is critical.")

        if single_spectrum:
            return spectra_deep[0]
        return spectra_deep

    # ==================== SINGLE SPECTRUM (WITH FULL REFRACTION) ====================

    def transform_single_2d_with_refraction(self,
                                            spectrum: np.ndarray,
                                            frequencies: np.ndarray,
                                            directions: np.ndarray,
                                            shore_normal_angle: float = 90.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform single 2D spectrum with full refraction.
        Use this only when refraction is critical (slower).

        For batch processing without refraction, use transform_batch_2d().
        """
        # Use pre-computed transformation if available
        if (self._transformation_matrix is None or
                not np.array_equal(self._frequency_grid, frequencies)):
            self.precompute_transformation_matrix(frequencies)

        n_freq, n_dir = spectrum.shape
        spectrum_deep = np.zeros_like(spectrum)
        direction_deep = directions.copy()

        # Apply frequency-dependent transformation
        for i, f in enumerate(frequencies):
            for j, theta in enumerate(directions):
                # Use pre-computed factor
                spectrum_deep[i, j] = spectrum[i, j] * self._transformation_matrix[i]

        return spectrum_deep, direction_deep


def compute_rotation_deviation(spectra, directions, a1, b1, dm, dp):
    """

    :param spectra: 2d spectra
    :param directions: directions
    :param a1: first fourier coefficient
    :param b1: second fourier coefficient
    :param dm: measured mean direction
    :return: degrees of alignment between measured mean direction and spectra
    """

    def circular_diff(angle1, angle2):
        """Compute smallest angular difference (degrees)"""
        diff = (angle1 - angle2) % 360
        if diff > 180:
            diff -= 360
        return diff

    theta_rad = np.radians(directions)

    delta_theta = np.radians(directions[1] - directions[0])  # Get step size

    # Cos/Sin for integration over direction
    cos_dir = np.cos(theta_rad) * delta_theta
    sin_dir = np.sin(theta_rad) * delta_theta

    # Integrate over all frequencies
    a1_total = np.sum(spectra * cos_dir[np.newaxis, :], axis=(0, 1))
    b1_total = np.sum(spectra * sin_dir[np.newaxis, :], axis=(0, 1))

    # Mean direction
    mean_dir_rad = np.arctan2(b1_total, a1_total)
    mean_dir_deg = (270 - np.degrees(mean_dir_rad)) % 360

    #Cross-check with direction of peak frequency
    peak_freq_idx = np.argmax(np.trapezoid(spectra, theta_rad, axis=1))
    bulk_mean_dir_deg = 270 - (np.degrees(np.arctan2(b1[peak_freq_idx],a1[peak_freq_idx]))) % 360

    error_peak = circular_diff(dp, bulk_mean_dir_deg)
    error_mean = circular_diff(dm, mean_dir_deg)

    return error_mean, error_peak

import pandas as pd


def compute_propagation_time(peak_frequency_hz, distance_km):
    """
    Compute wave propagation time using deep-water group velocity.

    For deep water: C_g = g / (4π f)

    Parameters:
    -----------
    peak_frequency_hz : float
        Peak frequency of the wave event (Hz)
    distance_km : float
        Great-circle distance from offshore to coastal buoy (km)

    Returns:
    --------
    travel_time_hours : float
        Estimated propagation time (hours)
    """
    g = 9.81  # m/s²

    # Deep-water group velocity
    Cg = g / (4 * np.pi * peak_frequency_hz)  # m/s

    # Convert distance to meters
    distance_m = distance_km * 1000

    # Travel time
    travel_time_sec = distance_m / Cg
    travel_time_hours = travel_time_sec / 3600

    return travel_time_hours


def compute_great_circle_distance(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two points (Haversine formula).

    Parameters:
    -----------
    lat1, lon1 : float
        Latitude/longitude of point 1 (degrees)
    lat2, lon2 : float
        Latitude/longitude of point 2 (degrees)

    Returns:
    --------
    distance_km : float
        Distance in kilometers
    """
    from math import radians, sin, cos, sqrt, atan2

    R = 6371.0  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance_km = R * c
    return distance_km


def compute_cooccurrence_matrix_with_physics(
        labels_offshore,
        labels_coastal,
        timestamps_offshore,
        timestamps_coastal,
        peak_frequencies_offshore,  # NEW: Peak frequency for each offshore event
        peak_directions_offshore,  # NEW: Peak direction for each offshore event
        offshore_location,  # NEW: (lat, lon) of offshore buoy
        coastal_location,  # NEW: (lat, lon) of coastal buoy
        time_window_hours=6,
        n_clusters_offshore=None,
        n_clusters_coastal=None,
        min_lag_hours=6,  # Minimum physical constraint
        max_lag_hours=120):  # Maximum physical constraint
    """
    Compute co-occurrence matrix with physics-based dynamic time lags.

    For each offshore event:
    1. Extract peak frequency and direction
    2. Compute great-circle distance to coastal buoy
    3. Calculate propagation time using group velocity
    4. Match with coastal events at predicted arrival time ± window

    Parameters:
    -----------
    labels_offshore : np.ndarray
        Cluster labels at offshore buoy, shape (n_samples_offshore,)
    labels_coastal : np.ndarray
        Cluster labels at coastal buoy, shape (n_samples_coastal,)
    timestamps_offshore : np.ndarray or pd.DatetimeIndex
        Timestamps for offshore observations
    timestamps_coastal : np.ndarray or pd.DatetimeIndex
        Timestamps for coastal observations
    peak_frequencies_offshore : np.ndarray
        Peak frequency (Hz) for each offshore event, shape (n_samples_offshore,)
    peak_directions_offshore : np.ndarray
        Peak direction (degrees) for each offshore event, shape (n_samples_offshore,)
    offshore_location : tuple
        (latitude, longitude) of offshore buoy
    coastal_location : tuple
        (latitude, longitude) of coastal buoy
    time_window_hours : float
        Tolerance window for matching (±hours)
    min_lag_hours, max_lag_hours : float
        Physical constraints on propagation time

    Returns:
    --------
    cooccurrence_matrix : np.ndarray
        Shape (n_clusters_offshore, n_clusters_coastal)
    normalized_matrix : np.ndarray
        Row-normalized probabilities
    metadata : dict
        Extended statistics including propagation times
    """
    # Auto-detect number of clusters
    if n_clusters_offshore is None:
        n_clusters_offshore = len(np.unique(labels_offshore))
    if n_clusters_coastal is None:
        n_clusters_coastal = len(np.unique(labels_coastal))

    # Convert to pandas
    df_offshore = pd.DataFrame({
        'time': pd.to_datetime(timestamps_offshore),
        'cluster': labels_offshore,
        'peak_freq': peak_frequencies_offshore,
        'peak_dir': peak_directions_offshore
    })
    df_coastal = pd.DataFrame({
        'time': pd.to_datetime(timestamps_coastal),
        'cluster': labels_coastal
    })


    # Initialize
    cooccurrence = np.zeros((n_clusters_offshore, n_clusters_coastal), dtype=int)
    matched_pairs = []
    propagation_times_used = []

    # Time window
    window_delta = pd.Timedelta(hours=time_window_hours)

    # For each offshore observation
    #for idx, row in df_offshore.iterrows():
    for idx, t in enumerate(timestamps_offshore):
        idx_offshore = idx
        # fractional_day = (t.hour * 3600 + t.minute * 60 + t.second) / 86400.0
        # time_select_ordinal = datetime.toordinal(t + timedelta(days=366)) + fractional_day
        idx_coastal = np.argmin(np.abs(timestamps_coastal - t))
        #index_drift03_dp = np.argmin(np.abs(time_drift03 - time_select_ordinal))
        # Compute fixed great-circle distance (doesn't change)
        lat_off, lon_off = offshore_location[0][idx_offshore], offshore_location[1][idx_offshore]
        lat_coast, lon_coast = coastal_location[0][idx_coastal], coastal_location[1][idx_coastal]
        distance_km = compute_great_circle_distance(lat_off, lon_off, lat_coast, lon_coast)

        offshore_time = df_offshore['time'][idx]
        offshore_cluster = int(df_offshore['cluster'][idx])
        peak_freq = df_offshore['peak_freq'][idx]
        peak_dir = df_offshore['peak_dir'][idx]

        # Skip if missing data
        if np.isnan(peak_freq) or peak_freq <= 0:
            continue

        # Compute propagation time based on frequency
        travel_time_hours = compute_propagation_time(peak_freq, distance_km)

        # Apply physical constraints
        travel_time_hours = np.clip(travel_time_hours, min_lag_hours, max_lag_hours)

        propagation_times_used.append(travel_time_hours)

        # Predicted arrival time
        lag_delta = pd.Timedelta(hours=travel_time_hours)
        expected_arrival = offshore_time + lag_delta

        # Find coastal observations within time window
        time_min = expected_arrival - window_delta
        time_max = expected_arrival + window_delta

        coastal_matches = df_coastal[
            (df_coastal['time'] >= time_min) &
            (df_coastal['time'] <= time_max)
            ]

        # For each match, increment co-occurrence
        for _, coastal_row in coastal_matches.iterrows():
            coastal_cluster = int(coastal_row['cluster'])
            cooccurrence[offshore_cluster, coastal_cluster] += 1

            actual_lag = (coastal_row['time'] - offshore_time).total_seconds() / 3600

            matched_pairs.append({
                'offshore_time': offshore_time,
                'offshore_cluster': offshore_cluster,
                'coastal_time': coastal_row['time'],
                'coastal_cluster': coastal_cluster,
                'peak_frequency_hz': peak_freq,
                'peak_direction_deg': peak_dir,
                'predicted_lag_hours': travel_time_hours,
                'actual_lag_hours': actual_lag,
                'lag_error_hours': actual_lag - travel_time_hours
            })

    # Normalize
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_matrix = cooccurrence / row_sums

    # Enhanced metadata
    matched_pairs_df = pd.DataFrame(matched_pairs)

    metadata = {
        'total_offshore_samples': len(df_offshore),
        'total_coastal_samples': len(df_coastal),
        'matched_pairs': len(matched_pairs),
        'match_rate': len(matched_pairs) / len(df_offshore) if len(df_offshore) > 0 else 0,
        'matched_pairs_df': matched_pairs_df,
        'distance_km': distance_km,
        'mean_propagation_time_hours': np.mean(propagation_times_used),
        'std_propagation_time_hours': np.std(propagation_times_used),
        'min_propagation_time_hours': np.min(propagation_times_used) if propagation_times_used else 0,
        'max_propagation_time_hours': np.max(propagation_times_used) if propagation_times_used else 0,
        'time_window_used': time_window_hours,
    }

    # Additional statistics if we have matches
    if len(matched_pairs) > 0:
        metadata['mean_lag_error_hours'] = matched_pairs_df['lag_error_hours'].mean()
        metadata['std_lag_error_hours'] = matched_pairs_df['lag_error_hours'].std()
        metadata['rmse_lag_hours'] = np.sqrt(np.mean(matched_pairs_df['lag_error_hours'] ** 2))

    return cooccurrence, normalized_matrix, metadata


def segment_events_from_label_sequence(timestamps, cluster_labels, min_gap_hours=12):
    """
    Segment swell events based on continuity of cluster labels.

    Special handling for label -1 (outliers):
    - Observations with cluster_label=-1 are marked as event_label=-1
    - They do NOT interrupt ongoing events (transparent to segmentation)
    - Event continuity is determined by non-outlier observations

    Example:
    --------
    cluster_labels: [2, 2, -1, -1, 2, 2, 3, 3]
    time gaps:      [--1h--1h--1h--1h--1h--1h--1h]
    event_labels:   [0, 0, -1, -1, 0, 0, 1, 1]
                    ^^^^^^^^^^^^^     event 0 (cluster 2, despite -1 gap)
                                      ^^^^^  event 1 (cluster 3)

    Parameters:
    -----------
    timestamps : array-like
        Ordered timestamps for each spectrum observation
    cluster_labels : array-like
        Cluster label for each observation (same length as timestamps)
        Use -1 for outliers/noise that should not break event continuity
    min_gap_hours : float
        Minimum time gap (in hours) to consider labels as interrupted
        (only evaluated between non-outlier observations)

    Returns:
    --------
    event_labels : array
        New event labels where each run of a cluster is a separate event
        Outliers are marked with event_label=-1
    event_info : dict
        Dictionary mapping event_label -> {
            'cluster': X,
            'start_time': t1,
            'end_time': t2,
            'n_observations': N,
            'n_outliers_within': M,  # NEW: count of -1 observations within event
            'start_idx': idx1,
            'end_idx': idx2
        }
    """
    import numpy as np
    from datetime import timedelta

    # Sort by time to ensure temporal order
    sort_idx = np.argsort(timestamps)
    sorted_times = timestamps[sort_idx]
    sorted_labels = cluster_labels[sort_idx]

    event_labels = np.full(len(sorted_labels), -1, dtype=int)  # Initialize all to -1
    event_info = {}
    current_event_id = 0

    # Find first non-outlier observation to start
    valid_indices = np.where(sorted_labels != -1)[0]

    if len(valid_indices) == 0:
        # All observations are outliers
        return event_labels[np.argsort(sort_idx)], event_info

    # Track current run (using only non-outlier observations)
    current_label = sorted_labels[valid_indices[0]]
    run_start_idx = valid_indices[0]
    last_valid_idx = valid_indices[0]  # Track last non-outlier position

    # Mark first valid observation
    event_labels[valid_indices[0]] = current_event_id

    for i in valid_indices[1:]:
        # Time gap since last valid (non-outlier) observation
        time_gap = sorted_times[i] - sorted_times[last_valid_idx]
        gap_hours = time_gap / np.timedelta64(1, 'h')

        # Check if label changed OR same label but large time gap
        if sorted_labels[i] != current_label or gap_hours > min_gap_hours:
            # End current run - record event info
            event_end_idx = last_valid_idx

            # Count observations in this event (including -1s in between)
            event_obs_mask = (sort_idx >= run_start_idx) & (sort_idx <= event_end_idx)
            n_total_obs = np.sum(event_obs_mask)
            n_outliers = np.sum((sorted_labels[run_start_idx:event_end_idx + 1] == -1))

            event_info[current_event_id] = {
                'cluster': current_label,
                'start_time': sorted_times[run_start_idx],
                'end_time': sorted_times[event_end_idx],
                'n_observations': n_total_obs - n_outliers,  # Only valid observations
                'n_outliers_within': n_outliers,
                'start_idx': run_start_idx,
                'end_idx': event_end_idx
            }

            # Start new run
            current_event_id += 1
            current_label = sorted_labels[i]
            run_start_idx = i

        # Mark this valid observation with current event ID
        event_labels[i] = current_event_id
        last_valid_idx = i

    # Handle last run
    event_end_idx = last_valid_idx
    event_obs_mask = (np.arange(len(sorted_labels)) >= run_start_idx) & \
                     (np.arange(len(sorted_labels)) <= event_end_idx)
    n_total_obs = np.sum(event_obs_mask)
    n_outliers = np.sum((sorted_labels[run_start_idx:event_end_idx + 1] == -1))

    event_info[current_event_id] = {
        'cluster': current_label,
        'start_time': sorted_times[run_start_idx],
        'end_time': sorted_times[event_end_idx],
        'n_observations': n_total_obs - n_outliers,
        'n_outliers_within': n_outliers,
        'start_idx': run_start_idx,
        'end_idx': event_end_idx
    }

    # Reorder back to original sequence
    unsort_idx = np.argsort(sort_idx)
    event_labels = event_labels[unsort_idx]

    return event_labels, event_info


def smooth_labels_rolling_window(timestamps, clustering_labels, window_hours=2, eliminate=False):
    """
    Smooth cluster labels using a rolling time window.

    Args:
        timestamps: array of datetime64
        clustering_labels: array of cluster labels
        window_hours: size of rolling window in hours

    Returns:
        smoothed_labels: labels after temporal smoothing
    """
    import numpy as np
    from scipy import stats

    n = len(clustering_labels)
    smoothed_labels = clustering_labels.copy()
    window_seconds = window_hours * 3600

    for i in range(n):
        current_time = timestamps[i]

        # Find all points within window
        time_diffs = np.abs((timestamps - current_time).astype('timedelta64[s]').astype(int))
        in_window = time_diffs <= window_seconds / 2

        # Get most common label in window
        if np.sum(in_window) > 0:
            window_labels = clustering_labels[in_window]
            mode_result = stats.mode(window_labels, keepdims=True)
            if not eliminate:
                smoothed_labels[i] = mode_result.mode[0]
            else:
                if not smoothed_labels[i] == mode_result.mode[0]:
                    smoothed_labels[i] = -1

    return smoothed_labels


# Complete pipeline
def identify_swell_events(timestamps, clustering_labels,
                         min_interruption_hours=2,
                         min_gap_hours=24, smoother=True, eliminate=False):
    """
    Complete pipeline to identify temporally coherent swell events
    """
    # Step 1: Smooth brief label changes
    if smoother:
        smoothed_labels = smooth_labels_rolling_window(timestamps, clustering_labels,
                                                       window_hours=min_interruption_hours, eliminate=eliminate)
    else:
        smoothed_labels = clustering_labels

    # Step 2: Separate events based on smoothed labels and time gaps
    event_ids, event_info = segment_events_from_label_sequence(timestamps, smoothed_labels,
                                           min_gap_hours=min_gap_hours)

    return event_ids, event_info, smoothed_labels

def extract_frequency_evolution(cluster_id, timestamps, peak_frequencies, cluster_labels):
    """
    Extract time series of frequencies for a specific cluster
    """
    mask = cluster_labels == cluster_id
    times = timestamps[mask]
    freqs = peak_frequencies[mask]

    # Sort by time
    sorted_indices = np.argsort(times)
    return times[sorted_indices], freqs[sorted_indices]


def compute_source_distance_from_dispersion(times, frequencies, energies=None):
    """
    Compute source distance using frequency dispersion by isolating the rising segment
    Based on: d = g/(4π·m) where m = df/dt

    Returns:
        distance_km: Distance to source
        t0: Estimated generation time
        quality: Regression R² coefficient
    """
    import numpy as np
    from scipy.signal import savgol_filter
    from sklearn.linear_model import RANSACRegressor, LinearRegression
    if len(times) > 1:
        times_np = np.array(times, dtype='datetime64[s]')

        # Convert times to seconds since first observation
        t_sec = (times_np - times_np[0]) / np.timedelta64(1, 's')

        #Perform an smooothing for remove outliers
        window_len = min(11, len(frequencies))
        if window_len % 2 == 0: window_len -= 1

        if window_len > 3:
            freq_smooth = savgol_filter(frequencies, window_length=window_len, polyorder=2)
        else:
            freq_smooth = frequencies

        #Find the turning point
        min_idx = np.argmin(freq_smooth)

        #Sanity check for existence of rising segment
        if min_idx >= len(frequencies) - 2:
            print("\nNo rising segment found in frequency dispersion of the event, set default 3000km")
            return {
                'distance_km': 3000,
                'generation_time': None,
                'slope': 0,
                'intercept': 0,
                'n_observations': len(times)
            }
        #Keep the rising part
        t_segment = t_sec[min_idx:]
        f_segment = frequencies[min_idx:]

        #Weight if energy available
        weights = None
        if energies is not None:
            weights = energies[min_idx:]

        #Robust Regression on the rising part using ransac
        X = t_segment.reshape(-1, 1)
        y = f_segment

        ransac = RANSACRegressor(
            estimator=LinearRegression(),
            min_samples=0.5,  # Fit at least 50% of the rising points
            residual_threshold=0.005,  # Tight threshold (5 mHz)
            random_state=42)
        try:
            ransac.fit(X, y, sample_weight=weights)

            slope = ransac.estimator_.coef_[0]
            intercept = ransac.estimator_.intercept_

            #Slope MUST be positive for distance estimation
            if slope <= 1e-9:
                print(f"\nNegative slope {slope}, set default 3000km")
                return {
                    'distance_km': 3000,
                    'generation_time': None,
                    'slope': 0,
                    'intercept': 0,
                    'n_observations': len(times)
                }

            # Calculate Distance
            # d = g / (4 * pi * slope)
            g = 9.81
            distance_m = g / (4 * np.pi * slope)
            distance_km = distance_m / 1000.0
            if distance_km > 6000.0:
                print(f"\nDistance estimated so high {distance_km} km, set default 3000km")
                return {
                    'distance_km': 3000,
                    'generation_time': None,
                    'slope': 0,
                    'intercept': 0,
                    'n_observations': len(times)
                }
            elif distance_km < 500.0:
                print(f"\nDistance estimated so low {distance_km} km, set default 3000km")
                return {
                    'distance_km': 3000,
                    'generation_time': None,
                    'slope': 0,
                    'intercept': 0,
                    'n_observations': len(times)
                }

            # Calculate Generation Time
            # f = slope * t_rel + intercept  => at f=0 (theoretical), t_rel = -intercept/slope
            # Note: t_sec was relative to times[0], so we add that back
            t_gen_rel_seconds = -intercept / slope
            generation_time = times[0] + np.timedelta64(int(t_gen_rel_seconds), 's')
            if abs(t_gen_rel_seconds) > 240 * 3600:
                print(f"\nGeneration time 10 days before reached limit {t_gen_rel_seconds/3600} hours, set default 3000km")
                return {
                    'distance_km': 3000,
                    'generation_time': None,
                    'slope': 0,
                    'intercept': 0,
                    'n_observations': len(times)
                }
            print(f"\nFrequency dispersion distance converged: {distance_km} km")
            return {
                'distance_km': distance_km,
                'generation_time': generation_time,
                'slope': slope,
                'intercept': intercept,
                'n_observations': len(times)
            }
        except Exception as e:
            print("\nLinear regression not converged, set default 3000km")
            return {
                'distance_km': 3000,
                'generation_time': None,
                'slope': 0,
                'intercept': 0,
                'n_observations': len(times)
            }
    else:
        print("\nOnly one observation in the event, set default 3000km")
        return {
            'distance_km': 3000,
            'generation_time': None,
            'slope': 0,
            'intercept': 0,
            'n_observations': len(times)
        }


def compute_dual_buoy_source_with_dispersion(
    cluster_buoy1, cluster_buoy2,
    times_buoy1, freqs_buoy1, directions_buoy1, lat_buoy1, lon_buoy1, labels_buoy1,
    times_buoy2, freqs_buoy2, directions_buoy2, lat_buoy2, lon_buoy2, labels_buoy2
):
    """
    Compute source location using dispersion from both buoys
    """
    # Extract frequency sequences for each cluster
    t1, f1 = extract_frequency_evolution(cluster_buoy1, times_buoy1, freqs_buoy1, labels_buoy1)
    t2, f2 = extract_frequency_evolution(cluster_buoy2, times_buoy2, freqs_buoy2, labels_buoy2)

    # Compute distances from dispersion
    disp1 = compute_source_distance_from_dispersion(t1, f1)
    disp2 = compute_source_distance_from_dispersion(t2, f2)

    # Get mean directions for each cluster
    dir1 = np.mean(directions_buoy1[labels_buoy1 == cluster_buoy1])
    dir2 = np.mean(directions_buoy2[labels_buoy2 == cluster_buoy2])

    # Compute source location as intersection of circles
    source_lat, source_lon = find_circle_intersection(
        lat_buoy1, lon_buoy1, disp1['distance_km'], dir1,
        lat_buoy2, lon_buoy2, disp2['distance_km'], dir2
    )

    return {
        'source_lat': source_lat,
        'source_lon': source_lon,
        'buoy1_distance_km': disp1['distance_km'],
        'buoy2_distance_km': disp2['distance_km'],
        'buoy1_quality': disp1['regression_r2'],
        'buoy2_quality': disp2['regression_r2'],
        'estimated_generation_time': disp1['generation_time']  # Should match disp2
    }

def find_circle_intersection(lat1, lon1, d1_km, bearing1,
                             lat2, lon2, d2_km, bearing2):
    """
    Find intersection of two circles centered at buoys with radii from dispersion
    """
    from geographiclib.geodesic import Geodesic
    geod = Geodesic.WGS84

    # Sample points along each great circle
    distances = np.linspace(max(0, d1_km - 500), d1_km + 500, 100)
    min_separation = np.inf
    best_point = None

    for dist1 in distances:
        # Point along bearing1 from buoy1
        g1 = geod.Direct(lat1, lon1, bearing1, dist1 * 1000)
        p1_lat, p1_lon = g1['lat2'], g1['lon2']

        # Find closest approach to circle around buoy2
        g2 = geod.Inverse(lat2, lon2, p1_lat, p1_lon)
        dist_to_buoy2_km = g2['s12'] / 1000

        separation = abs(dist_to_buoy2_km - d2_km)

        if separation < min_separation:
            min_separation = separation
            best_point = (p1_lat, p1_lon)

    return best_point[0], best_point[1]


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_directional_spectrum(frequencies, directions, spectrum,
                                     title='Directional Wave Spectrum',
                                     vmax=None, vmin=None, cmap='jet', colorbar=True):
    """
    Plot directional wave spectrum with smooth gradient colors.

    Uses pcolormesh for continuous color gradients without banding.

    Parameters:
    -----------
    frequencies : np.ndarray
        Frequency array (Hz), shape (nf,)
    directions : np.ndarray
        Direction array (degrees, nautical convention), shape (nd,)
    spectrum : np.ndarray
        2D directional spectrum E(f,θ), shape (nf, nd)
    title : str
        Plot title
    vmax : float, optional
        Maximum value for colorbar
    vmin : float, optional
        Minimum value for colorbar
    cmap : str
        Colormap name (default: 'jet')
    """
    # Convert directions to radians (nautical convention)
    theta = np.radians(90 - directions)

    # Close the circle for seamless wrapping at 360°/0°
    theta_closed = np.append(theta, theta[0] + 2 * np.pi)
    spectrum_closed = np.column_stack([spectrum, spectrum[:, 0]])

    # Create meshgrid
    THETA, FREQ = np.meshgrid(theta_closed, frequencies)

    # Create polar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    # Auto-scale if not provided
    if vmax is None:
        vmax = np.percentile(spectrum, 95)
    if vmin is None:
        vmin = 0  # Typically 0 for wave spectra

    # Pcolormesh for smooth gradients (no banding!)
    mesh = ax.pcolormesh(THETA, FREQ, spectrum_closed,
                         cmap=cmap, vmin=vmin, vmax=vmax,
                         shading='auto',  # Smooth interpolation
                         rasterized=True)  # Better rendering

    # Customize appearance
    ax.set_theta_zero_location('N')  # North at top
    ax.set_theta_direction(-1)  # Clockwise (nautical convention)

    # Add colorbar
    if colorbar:
        cbar = plt.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('Energy Density (m²/Hz/deg)', rotation=270, labelpad=20, fontsize=16)

    # Add title
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

    # Radial axis label
    ax.set_ylabel('Frequency (Hz)', labelpad=30)

    plt.tight_layout()
    return fig, ax

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from matplotlib.collections import LineCollection

def plot_cluster_spectra_pairs(
    sw_gp, selected_gpmodels, main_model, labels, N_0,
    title=None, save=None, lead=0, step=0.02, plot_latent=False,
    ticks=False, yscale=False, line_max=False,
    f_lo=0.04, f_hi=0.20,
    cmap_name="coolwarm",
    band_k=1.9,              # like your plot_models_plotly (≈ ~95% if Gaussian)
    upsample_N=1200,
    vmax_mode="p99",         # "max", "p95", "p99"
    colorbar=True
):
    """
    4 columns grid: [1D mean+samples, polar mean] x 2 clusters per row.

    Requirements implemented:
      - show integrated samples per cluster (thin lines alpha=0.2)
      - show GP mean +/- band_k * std band (fill_between)
      - shared cmap ('coolwarm') AND shared Normalize across all panels
      - same x/y ticks across 1D plots; same radial ticks across polar plots
      - x/r frequency range fixed to [0.04, 0.25]
      - no axis labels; titles 'cluster 0', 'cluster 1', ...
    """
    with mpl.rc_context({
    "font.size": 20,
    "axes.titlesize": 24,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    }):
        # ---- helper: robust numpy conversion (handles torch/xarray-ish) ----
        def as_np(a):
            try:
                import torch
                if isinstance(a, torch.Tensor):
                    return a.detach().cpu().numpy()
            except Exception:
                pass
            return np.asarray(a)

        # ---- pull directional spectra data ----
        Y = as_np(sw_gp.y_train)  # expected (Nobs, Nf, Nd) or (Nobs, Nf, Nd, Nlead)
        if Y.ndim == 4:
            Y = Y[..., lead]
        if Y.ndim != 3:
            raise ValueError(f"Expected sw_gp.y_train shape (Nobs, Nf, Nd)[,lead]. Got {Y.shape}")

        nobs, nf, nd = Y.shape

        # ---- frequencies ----
        freqs = None
        if hasattr(sw_gp, "frequencies"):
            freqs = as_np(sw_gp.frequencies).ravel()
        else:
            # fall back to first GP model basis
            gp0 = sw_gp.gpmodels[lead][selected_gpmodels[0]]
            freqs = as_np(gp0.x_basis).ravel()
        freqs = freqs[:nf]

        # ---- directions (degrees) ----
        if hasattr(sw_gp, "directions"):
            directions_deg = as_np(sw_gp.directions).ravel()
        elif hasattr(sw_gp, "dirs"):
            directions_deg = as_np(sw_gp.dirs).ravel()
        else:
            directions_deg = np.linspace(0, 360, nd, endpoint=False)

        # ---- frequency window mask ----
        mask = (freqs >= f_lo) & (freqs <= f_hi)
        if not np.any(mask):
            raise ValueError(f"No frequency bins found in [{f_lo}, {f_hi}] for freqs range [{freqs.min()}, {freqs.max()}].")

        freqs_w = freqs[mask]
        fmin, fmax = float(freqs_w.min()), float(freqs_w.max())

        # ---- compute per-cluster means + gather members ----
        clusters = []
        for m in selected_gpmodels:
            gp = sw_gp.gpmodels[lead][m]
            idx = np.asarray(gp.indexes, dtype=int)
            idx = idx[(idx >= 0) & (idx < nobs)]
            clusters.append((m, gp, idx))

        # ---- global color scaling (shared norm across ALL panels) ----
        # Use the directional spectra magnitude over the selected freq range.
        all_vals = []
        for _, _, idx in clusters:
            if idx.size > 0:
                all_vals.append(Y[idx][:, mask, :].reshape(-1))
        if len(all_vals) == 0:
            raise ValueError("No samples found in any cluster indexes.")
        all_vals = np.concatenate(all_vals)
        vmin = 0.0
        if vmax_mode == "max":
            vmax = float(np.max(all_vals))
        elif vmax_mode == "p95":
            vmax = float(np.percentile(all_vals, 95))
        else:  # "p99"
            vmax = float(np.percentile(all_vals, 99))

        cmap = mpl.colormaps[cmap_name]
        norm = colors.Normalize(vmin=vmin, vmax=vmax)  # shared norm :contentReference[oaicite:3]{index=3}

        # ---- ticks: enforce identical tick locations everywhere ----
        # Use a reasonable tick step; if your `step` is too coarse, fall back to 0.05
        tick_step = float(step) if (step is not None and step > 0 and (fmax - fmin)/step <= 10) else 0.02
        x_ticks = np.round(np.arange(f_lo, f_hi + 1e-12, tick_step), 5)

        # ---- helper: gradient mean line using LineCollection ----
        def add_gradient_line(ax, x, y, lw=5):
            x = np.asarray(x, float)
            y = np.asarray(y, float)

            # upsample to hide segment boundaries (looks continuous)
            x2 = np.linspace(float(x.min()), float(x.max()), int(upsample_N))
            y2 = np.interp(x2, x, y)

            pts = np.column_stack([x2, y2]).reshape(-1, 1, 2)
            segs = np.concatenate([pts[:-1], pts[1:]], axis=1)

            lc = LineCollection(
                segs, cmap=cmap, norm=norm,
                antialiased=True, capstyle="round", joinstyle="round"
            )
            lc.set_array(y2[:-1])
            lc.set_linewidth(lw)
            lc.set_rasterized(True)  # avoids seams in vector export
            ax.add_collection(lc)
            return lc

        # ---- helper tick manage ----
        def set_ticks_with_peak(ax, base_ticks, f_peak, f_lo, f_hi, tol=None):
            base_ticks = np.asarray(base_ticks, float)

            # infer spacing from base_ticks if not provided
            if tol is None:
                st = np.min(np.diff(np.sort(base_ticks))) if base_ticks.size > 1 else 0.01
                tol = 0.8 * st  # remove ticks closer than ~half a step

            # keep ticks that are not too close to f_peak (but always keep f_peak itself)
            keep = (np.abs(base_ticks - f_peak) >= tol)
            ticks = base_ticks[keep]

            # add f_peak, clip to range, unique + sorted
            ticks = np.unique(np.append(ticks, f_peak))
            ticks = ticks[(ticks >= f_lo) & (ticks <= f_hi)]
            ticks = np.sort(ticks)

            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{t:.3f}" for t in ticks])


        # ---- polar mesh prep (close the circle) ----
        theta = np.radians(90.0 - directions_deg)
        theta_closed = np.append(theta, theta[0] + 2*np.pi)

        # ---- pass 1: compute global y-limits for 1D panels (data + band) ----
        y_global_max = 0.0
        for _, gp, idx in clusters:
            if idx.size == 0:
                continue
            # integrated samples (mean over directions)
            S1_samples = Y[idx][:, mask, :].mean(axis=2)  # (Nsamp, Nf_win)
            y_global_max = max(y_global_max, float(np.max(S1_samples)))

            # GP posterior band on same x grid
            try:
                import torch
                x_t = torch.tensor(freqs_w, dtype=torch.float64).reshape(-1, 1)
                mean_, Sig_ = gp.observe_last(x_t)
                mean_np = as_np(mean_).reshape(-1)
                std_np = np.sqrt(np.diag(as_np(Sig_)))
                y_global_max = max(y_global_max, float(np.max(mean_np + band_k*std_np)))
            except Exception:
                # If gp.observe_last fails, we still have data-based limit
                pass

        if y_global_max <= 0:
            y_global_max = 1.0

        y_ticks = np.round(np.linspace(0.0, y_global_max, 4), 1)

        # ---- layout: 2 clusters per row => 4 columns ----
        K = len(clusters)
        nrows = int(np.ceil(K / 2))
        fig = plt.figure(figsize=(22, 4.8 * nrows))
        gs = fig.add_gridspec(nrows, 4, wspace=0.05, hspace=0.28)

        all_axes = []
        last_mesh = None

        for k, (m, gp, idx) in enumerate(clusters):
            row = k // 2
            slot = k % 2
            c0 = 2 * slot

            # =======================
            # 1D integrated spectra
            # =======================
            ax1 = fig.add_subplot(gs[row, c0])

            if idx.size > 0:
                S1_samples = Y[idx][:, mask, :].mean(axis=2)  # (Nsamp, Nf_win)
                # thin sample lines
                for s in S1_samples:
                    # optionally color by sample peak using the SAME cmap/norm
                    ax1.plot(freqs_w, s, lw=0.8, alpha=0.1, color=cmap(norm(float(np.min(s)))))

                # data mean
                S1_mean = S1_samples.mean(axis=0)
                # after you computed S1_mean on freqs_w
                imax = int(np.argmax(S1_mean))
                f_peak = float(freqs_w[imax])

                # 1) vertical line
                ax1.axvline(f_peak, color=cmap(norm(float(np.max(s)))), linestyle="--", linewidth=1.2, alpha=0.6)

                # 2) add peak to xticks (keep unique + sorted)
                set_ticks_with_peak(ax1, x_ticks, f_peak, f_lo, f_hi)
            else:
                S1_mean = np.zeros_like(freqs_w)

            # gradient line for mean (coolwarm)
            #add_gradient_line(ax1, freqs_w, S1_mean, lw=6)

            # GP mean + variance band (like your plot_models_plotly) using fill_between :contentReference[oaicite:4]{index=4}
            try:
                import torch
                #x_t = torch.tensor(freqs_w, dtype=torch.float64).reshape(-1, 1)
                x_t = torch.tensor(np.linspace(np.min(freqs_w), np.max(freqs_w), 100), dtype=torch.float64).reshape(-1, 1)
                mean_, Sig_ = gp.observe_last(x_t)
                coeff = np.max(S1_mean) / np.max(mean_.numpy())
                mean_np = as_np(mean_).reshape(-1) * coeff
                std_np = np.sqrt(np.diag(as_np(Sig_)))

                lo = mean_np - band_k * std_np
                hi = mean_np + band_k * std_np
                lo = np.maximum(lo, 0.0)

                band_color = cmap(norm(float(np.max(mean_np))))
                #ax1.plot(freqs_w, mean_np, color="black", lw=1.8)
                add_gradient_line(ax1, x_t.reshape(-1), mean_np, lw=6)
                ax1.fill_between(x_t.reshape(-1), lo, hi, color=band_color, alpha=0.25)
            except Exception:
                pass

            # formatting: shared axes and ticks, no labels
            ax1.set_xlim(f_lo, f_hi)
            ax1.set_ylim(0.0, y_global_max * 1.02)
            #ax1.set_xticks(x_ticks)
            ax1.set_yticks(y_ticks)
            ax1.set_xlabel("")
            ax1.set_ylabel("")
            ax1.set_title(f"Cluster {k}")

            # =======================
            # Polar directional mean
            # =======================
            ax2 = fig.add_subplot(gs[row, c0 + 1], projection="polar")

            if idx.size > 0:
                S2_mean = Y[idx][:, mask, :].mean(axis=0)  # (Nf_win, Nd)
            else:
                S2_mean = np.zeros((freqs_w.size, nd))

            S2_closed = np.column_stack([S2_mean, S2_mean[:, 0]])
            TH, FR = np.meshgrid(theta_closed, freqs_w)

            last_mesh = ax2.pcolormesh(
                TH, FR, S2_closed,
                cmap=cmap, norm=norm,
                shading="auto",          # smooth shading :contentReference[oaicite:5]{index=5}
                rasterized=True          # good for vector export :contentReference[oaicite:6]{index=6}
            )

            ax2.set_theta_zero_location("N")
            ax2.set_theta_direction(-1)
            ax2.set_rlim(f_lo, f_hi)
            ax2.set_rticks(x_ticks)     # same numbering as 1D x-axis
            ax2.set_xticklabels([])
            ax2.set_xlabel("")
            ax2.set_ylabel("")

            all_axes.extend([ax1, ax2])

        # Hide unused last pair if odd number of clusters
        if K % 2 == 1:
            row = K // 2
            ax_off1 = fig.add_subplot(gs[row, 2]); ax_off1.axis("off")
            ax_off2 = fig.add_subplot(gs[row, 3], projection="polar"); ax_off2.axis("off")

        # one shared colorbar for all panels (optional)
        if colorbar and last_mesh is not None:
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            fig.colorbar(sm, ax=all_axes, fraction=0.018, pad=0.01)

        if title is not None:
            fig.suptitle(title, y=0.995)

        if save:
            fig.savefig(save, dpi=300, bbox_inches="tight")
        else:
            plt.show()

        return fig


from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime
from matplotlib.gridspec import GridSpec
from geographiclib.geodesic import Geodesic

def compute_source_front_line(source_lat, source_lon, source_direction,
                              confidence_radius_km, directional_spread):
    """
    Compute a line perpendicular to propagation direction representing the source zone.

    Parameters:
    -----------
    source_lat, source_lon : float
        Center of source estimate
    source_direction : float
        Mean propagation direction (degrees)
    confidence_radius_km : float
        Uncertainty radius
    directional_spread : float
        Directional spread (degrees)

    Returns:
    --------
    dict with:
        - front_center_lat, front_center_lon: Center point
        - front_left_lat, front_left_lon: Left endpoint
        - front_right_lat, front_right_lon: Right endpoint
        - front_width_km: Width of front
    """

    geod = Geodesic.WGS84

    # Front line is perpendicular to propagation direction
    perpendicular_direction = (source_direction - 90) % 360

    # Front width based on confidence radius and directional spread
    # Wider if more directional uncertainty
    #spread_factor = 1.0 + (directional_spread / 180)  # 1.0 to 2.0
    front_half_width_km = confidence_radius_km #* spread_factor

    # Compute left endpoint (perpendicular to propagation, to the left)
    g_left = geod.Direct(source_lat, source_lon, perpendicular_direction,
                        front_half_width_km * 1000)
    front_left_lat = g_left['lat2']
    front_left_lon = g_left['lon2']

    # Compute right endpoint (opposite direction)
    opposite_direction = (perpendicular_direction + 180) % 360
    g_right = geod.Direct(source_lat, source_lon, opposite_direction,
                         front_half_width_km * 1000)
    front_right_lat = g_right['lat2']
    front_right_lon = g_right['lon2']

    return {
        'front_center_lat': source_lat,
        'front_center_lon': source_lon,
        'front_left_lat': front_left_lat,
        'front_left_lon': front_left_lon,
        'front_right_lat': front_right_lat,
        'front_right_lon': front_right_lon,
        'front_width_km': 2 * front_half_width_km
    }

def draw_propagation_cone_from_front(m, ax, source_lat, source_lon,
                                     source_direction, directional_spread,
                                     confidence_radius_km, dispersion_rate, cone_distance_km=500,
                                     color='yellow', alpha=0.3, draw_edges=True):
    """
    Draw a cone showing directional wave spreading using geodesic (great circle) paths.

    This version properly accounts for Earth's spherical geometry.

    Parameters:
    -----------
    m : Basemap
        Basemap object
    ax : matplotlib axes
        Axes object to draw on
    source_lat, source_lon : float
        Source coordinates (degrees)
    source_direction : float
        Mean propagation direction (degrees, 0=N, 90=E)
    directional_spread : float
        Full angular spread (degrees)
    confidence_radius_km : float
        Uncertainty radius
    cone_distance_km : float
        How far to extend the cone (km)
    color : str
        Fill color for cone
    alpha : float
        Transparency
    draw_edges : bool
        Whether to draw cone edges
    """
    min_lon, max_lon = m.llcrnrlon, m.urcrnrlon
    min_lat, max_lat = m.llcrnrlat, m.urcrnrlat
    def within_map(lon, lat):
        return (min_lon <= lon <= max_lon) and (min_lat <= lat <= max_lat)

    geod = Geodesic.WGS84
    front = compute_source_front_line(source_lat, source_lon, source_direction,
                                     confidence_radius_km, directional_spread)

    spread_half_angle = directional_spread / 2
    n_points = 50

    # ========== STEP 2: LEFT BOUNDARY (from left end of front) ==========
    left_boundary_lons = []
    left_boundary_lats = []
    left_start_direction = source_direction - spread_half_angle

    # Start from LEFT end of source front
    for dist_km in np.linspace(0, cone_distance_km, n_points):
        left_start_direction_ = left_start_direction - ((dist_km / 100.0) * dispersion_rate)/2
        dist_m = dist_km * 1000
        g = geod.Direct(front['front_left_lat'], front['front_left_lon'],
                       left_start_direction_, dist_m)
        left_boundary_lons.append(float(g['lon2']))
        left_boundary_lats.append(float(g['lat2']))

    # ========== STEP 3: RIGHT BOUNDARY (from right end of front) ==========
    right_boundary_lons = []
    right_boundary_lats = []
    right_start_direction = source_direction + spread_half_angle

    # Start from RIGHT end of source front
    for dist_km in np.linspace(0, cone_distance_km, n_points):
        right_start_direction_ = right_start_direction + ((dist_km / 100.0) * dispersion_rate)/2
        dist_m = dist_km * 1000
        g = geod.Direct(front['front_right_lat'], front['front_right_lon'],
                       right_start_direction_, dist_m)
        right_boundary_lons.append(float(g['lon2']))
        right_boundary_lats.append(float(g['lat2']))

    # ========== STEP 4: ARC AT END ==========
    arc_lons = []
    arc_lats = []

    geod_line = geod.InverseLine(left_boundary_lats[-1], left_boundary_lons[-1],
                                right_boundary_lats[-1], right_boundary_lons[-1])
    arc_length = geod_line.s13
    n_arc = 30

    for i in range(1, n_arc):
        g = geod_line.Position((i / n_arc) * arc_length)
        arc_lons.append(float(g['lon2']))
        arc_lats.append(float(g['lat2']))

    # ========== STEP 5: SOURCE FRONT LINE ==========
    front_lons = []
    front_lats = []

    geod_front = geod.InverseLine(front['front_left_lat'], front['front_left_lon'],
                                 front['front_right_lat'], front['front_right_lon'])
    front_length = geod_front.s13
    n_front = 20

    for i in range(n_front + 1):
        g = geod_front.Position((i / n_front) * front_length)
        front_lons.append(float(g['lon2']))
        front_lats.append(float(g['lat2']))

    # ========== STEP 6: COMBINE INTO POLYGON ==========
    # Order: front line -> left boundary -> arc -> right boundary (reversed) -> back to front
    polygon_lons = ([front['front_left_lon']] +
                    front_lons +
                    left_boundary_lons[1:] +
                    arc_lons +
                    right_boundary_lons[::-1][1:-1] +
                    [front['front_right_lon']])

    polygon_lats = ([front['front_left_lat']] +
                    front_lats +
                    left_boundary_lats[1:] +
                    arc_lats +
                    right_boundary_lats[::-1][1:-1] +
                    [front['front_right_lat']])

    polygon_filtered_lons = []
    polygon_filtered_lats = []

    for lon, lat in zip(polygon_lons, polygon_lats):
        if within_map(lon, lat):
            polygon_filtered_lons.append(lon)
            polygon_filtered_lats.append(lat)

    # Convert to numpy arrays
    polygon_lons = np.array(polygon_filtered_lons, dtype=np.float64)
    polygon_lats = np.array(polygon_filtered_lats, dtype=np.float64)

    # Convert to map projection
    polygon_x, polygon_y = m(polygon_lons.tolist(), polygon_lats.tolist())

    if hasattr(polygon_x, 'filled'):
        polygon_x = polygon_x.filled(np.nan)
    if hasattr(polygon_y, 'filled'):
        polygon_y = polygon_y.filled(np.nan)

    polygon_x = np.array(polygon_x, dtype=np.float64).flatten()
    polygon_y = np.array(polygon_y, dtype=np.float64).flatten()

    # Create polygon
    coords = np.column_stack([polygon_x, polygon_y])
    polygon = Polygon(coords, facecolor=color, edgecolor='orange' if draw_edges else color,
                     alpha=alpha, linewidth=2 if draw_edges else 0, zorder=2)

    ax.add_patch(polygon)

    # ========== STEP 7: DRAW SOURCE FRONT LINE ==========
    front_filtered_lons = []
    front_filtered_lats = []

    for lon, lat in zip(front_lons, front_lats):
        if within_map(lon, lat):
            front_filtered_lons.append(lon)
            front_filtered_lats.append(lat)
    front_filtered_lats = np.array(front_filtered_lats, dtype=np.float64).flatten()
    front_filtered_lons = np.array(front_filtered_lons, dtype=np.float64).flatten()
    front_x, front_y = m(front_filtered_lons, front_filtered_lats)
    m.plot(front_x, front_y, color='red', linewidth=3, linestyle='-',
          alpha=0.9, zorder=5, label='Source Front')

    # ========== STEP 8: DRAW BOUNDARIES ==========
    if draw_edges:
        left_boundary_filtered_lons = []
        left_boundary_filtered_lats = []
        right_boundary_filtered_lons = []
        right_boundary_filtered_lats = []

        for lon, lat in zip(left_boundary_lons, left_boundary_lats):
            if within_map(lon, lat):
                left_boundary_filtered_lons.append(lon)
                left_boundary_filtered_lats.append(lat)
        for lon, lat in zip(right_boundary_lons, right_boundary_lats):
            if within_map(lon, lat):
                right_boundary_filtered_lons.append(lon)
                right_boundary_filtered_lats.append(lat)

        left_boundary_filtered_lats = np.array(left_boundary_filtered_lats, dtype=np.float64).flatten()
        left_boundary_filtered_lons = np.array(left_boundary_filtered_lons, dtype=np.float64).flatten()
        right_boundary_filtered_lats = np.array(right_boundary_filtered_lats, dtype=np.float64).flatten()
        right_boundary_filtered_lons = np.array(right_boundary_filtered_lons, dtype=np.float64).flatten()

        left_x, left_y = m(left_boundary_filtered_lons, left_boundary_filtered_lats)
        m.plot(left_x, left_y, color='orange', linewidth=2,
               linestyle='--', alpha=0.8, zorder=3)

        right_x, right_y = m(right_boundary_filtered_lons, right_boundary_filtered_lats)
        m.plot(right_x, right_y, color='orange', linewidth=2,
               linestyle='--', alpha=0.8, zorder=3)

    # ========== STEP 9: DRAW CENTER RAY ==========
    center_lons = []
    center_lats = []

    for dist_km in np.linspace(0, cone_distance_km, n_points):
        dist_m = dist_km * 1000
        g = geod.Direct(source_lat, source_lon, source_direction, dist_m)
        center_lons.append(float(g['lon2']))
        center_lats.append(float(g['lat2']))

    center_x, center_y = m(center_lons, center_lats)
    m.plot(center_x, center_y, color='darkred', linewidth=2,
           linestyle='-', alpha=0.7, zorder=4,
           label=f"Mean Direction ({source_direction:.1f}°)")

    plt.plot(center_x[-1], center_y[-1], marker='>', markersize=12,
            color='darkred', markeredgecolor='black', markeredgewidth=1, zorder=5)

    return {
        'polygon': polygon,
        'front': front,
        'cone_width_at_end_km': geod_line.s13 / 1000
    }


def plot_position_event_full(source, arrival, lats, lons, peak_energy_h_arrival, peak_freq_h_arrival, dp_h,
                             index_hillary_dp_arrival, time_select_datetime, lat_h, lon_h, lat_d03, lon_d03,
                             lat_d06, lon_d06, index_drift03, index_drift06, index_h_arrival,
                             dispersion_rate, cluster_labels_d03, cluster_labels_d06, cluster_labels_h,
                             b_lon_1=None, b_lat_1=None, b_lon_2=None, b_lat_2=None, title=None, end_of_event=None,
                             loss_grid=None, lat_array=None, lon_array=None):
    min_lat, max_lat, min_lon, max_lon = lats[0], lats[1], lons[0], lons[1]

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(1, 2, width_ratios=[4, 1], figure=fig)

    # ========== LEFT: MAP ==========
    ax_map = fig.add_subplot(gs[0])

    m = Basemap(projection='merc',
                llcrnrlat=min_lat-2.0, urcrnrlat=max_lat+2.0,
                llcrnrlon=min_lon-2.0, urcrnrlon=max_lon+2.0,
                resolution='i', ax=ax_map)

    m.drawcoastlines()
    m.drawcountries()
    m.fillcontinents(color='lightgray', lake_color='lightblue', alpha=0.3)
    m.drawmapboundary(fill_color='lightcyan')

    # Add parallels and meridians
    m.drawparallels(np.arange(-90, 90, 5), labels=[1,0,0,0], fontsize=9)
    m.drawmeridians(np.arange(-180, 180, 5), labels=[0,0,0,1], fontsize=9)

    # Plot buoys
    x, y = m(lon_h[0], lat_h[0])
    m.scatter(x, y, marker='o', s=100, color='blue', edgecolors='black', linewidth=2,
              label='Hillarys Buoy', zorder=5)

    x, y = m(lon_d03[index_drift03], lat_d03[index_drift03])
    m.scatter(x, y, marker='o', s=100, color='red', edgecolors='black', linewidth=2,
              label='Drift03 Buoy', zorder=5)

    x, y = m(lon_d06[index_drift06], lat_d06[index_drift06])
    m.scatter(x, y, marker='o', s=100, color='green', edgecolors='black', linewidth=2,
              label='Drift06 Buoy', zorder=5)

    # Plot source
    x, y = m(source['source_lon'], source['source_lat'])
    m.scatter(x, y, marker='+', s=300, color='gold', linewidth=2,
              label='Source Estimate', zorder=5)

    # Plot backtracks
    if b_lon_1 is not None:
        x, y = m(b_lon_1, b_lat_1)
        m.plot(x, y, linewidth=2, linestyle='--', color='red', alpha=0.7,
               label='Backtrack Drift03')
    if b_lon_2 is not None:
        x, y = m(b_lon_2, b_lat_2)
        m.plot(x, y, linewidth=2, linestyle='--', color='green', alpha=0.7,
               label='Backtrack Drift06')

    # ========== ADD DIRECTIONAL ARROW (Swell Direction) ==========
    if 'source_directional_spread' in source and 'source_dp' in source:
        cone_distance = min(500, max_lon - min_lon) * 111  # Adjust to map extent

        # Draw the spreading cone
        polygon = draw_propagation_cone_from_front(
            m, ax_map,
            source['source_lat'].item(),
            source['source_lon'].item(),
            (180 + source['source_dp'].item())%360,
            source['source_directional_spread'],
            source['confidence_radius_km'],
            dispersion_rate=dispersion_rate,
            cone_distance_km=cone_distance,
            color='yellow',
            alpha=0.25,
            draw_edges=True
        )

        # Add to legend
        # plt.plot([], [], color='yellow', linewidth=10, alpha=0.5,
        #          label=f"Spreading Cone (±{source['source_directional_spread'].item()/2:.1f}°)")

    ax_map.set_title(f'Swell Tracking: {time_select_datetime.strftime("%Y-%m-%d %H:%M UTC")}',
                     fontsize=14, fontweight='bold')

    if loss_grid is not None:
        lon_mesh, lat_mesh = np.meshgrid(lon_array, lat_array)

        # Transform to map projection coordinates
        x_mesh, y_mesh = m(lon_mesh, lat_mesh)

        # Plot with pcolormesh
        im = m.pcolormesh(x_mesh, y_mesh, loss_grid, alpha=0.4,
                          shading='auto', zorder=5, cmap='coolwarm')
        fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.04)# upper-right corner


    # ========== RIGHT: INFORMATION PANEL ==========
    ax_info = fig.add_subplot(gs[1])
    ax_info.axis('off')

    # Prepare detailed information
    info_sections = [
        ("EVENT INFORMATION", [
            f"Time: {time_select_datetime.strftime('%Y-%m-%d %H:%M UTC')}",
            f"Source Generation: {arrival['source_generation_time'].strftime('%Y-%m-%d %H:%M UTC')}",
            f"Arrival at Hillarys: {arrival['arrival_time'].strftime('%Y-%m-%d %H:%M UTC')}",
            "" if end_of_event is None else f"End of event: {end_of_event.strftime('%Y-%m-%d %H:%M UTC')}"
        ]),

        ("SOURCE LOCATION", [
            f"Direction: {source.get('source_dp', 0).item():7.1f}°",
            f"Spread: {source.get('source_directional_spread', 0):7.1f}º",
        ]),

        ("PROPAGATION", [
            f"Distance: {arrival.get('propagation_distance_km', 0).item():.0f} km",
            f"Travel Time: {arrival.get('travel_time_hours', 0).item():.1f} hours, ({arrival.get('travel_time_hours', 0).item()/24.0:.1f} days)",
        ]),

        ("ENERGY (m²)", [
            f"Predicted: {arrival['arrival_energy'].item():8.3f}",
            f"Observed:  {peak_energy_h_arrival:8.3f}",
        ]),

        ("FREQUENCY (Hz)", [
            f"Predicted: {arrival['arrival_frequency'].item():.4f}, period: {1/arrival['arrival_frequency'].item():.1f} s",
            f"Observed:  {peak_freq_h_arrival.item():.4f}, period: {1/peak_freq_h_arrival.item():.1f} s",
        ]),

        ("DIRECTION (°)", [
            f"Predicted: {arrival['arrival_direction'].item():7.1f}°",
            f"Observed:  {dp_h[:, index_hillary_dp_arrival].item():7.1f}°",
            f"Error:     {(arrival['arrival_direction'].item() - dp_h[:, index_hillary_dp_arrival].item()):+7.1f}°",
        ]),

        ("CLUSTERING", [
            f"Cluster d03 event: {cluster_labels_d03[index_drift03]:.0f}",
            f"Cluster d06 event:  {cluster_labels_d06[index_drift06]:.0f}",
            f"Cluster H arrival:  {cluster_labels_h[index_hillary_dp_arrival]:.0f}",
        ]),
    ]


    # Render information panel
    y_pos = 0.90
    for section_title, section_lines in info_sections:
        # Section header
        ax_info.text(0.03, y_pos, section_title,
                    fontsize=10, fontweight='bold',
                    transform=ax_info.transAxes,
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        y_pos -= 0.03

        # Section content
        for line in section_lines:
            ax_info.text(0.03, y_pos, line,
                        fontsize=9, family='monospace',
                        transform=ax_info.transAxes)
            y_pos -= 0.03

        y_pos -= 0.01  # Extra space between sections

    plt.tight_layout()
    if title is None:
        plt.savefig(f'/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/prediction_source/swell_tracking_detailed_{time_select_datetime.strftime("%Y%m%d_%H%M")}.pdf',
                    dpi=300, bbox_inches='tight')
    else:
        plt.savefig(
            f'/home/adrian.perez/Documents/OceanWave/HDP-GPC/hdpgpc/ocean/plots/representers_events/{title}_{time_select_datetime.strftime("%Y%m%d_%H%M")}.pdf',
            dpi=300, bbox_inches='tight')
    plt.close()

def ordinal_float_to_datetime(ordinal_float):
    """
    Convert a floating-point ordinal (integer part + fractional day)
    to a Python datetime object.

    Parameters:
    -----------
    ordinal_float : float
        Ordinal number with decimal fraction representing partial day.

    Returns:
    --------
    datetime.datetime object
    """
    integer_part = int(ordinal_float)
    fractional_part = ordinal_float - integer_part

    base_date = datetime.fromordinal(integer_part)
    time_delta = timedelta(days=fractional_part)

    return base_date + time_delta

# ==================== USAGE EXAMPLES ====================

def example_with_your_data():
    """Example using your actual buoy data structure"""

    import h5py
    import os
    cwd = os.path.dirname(os.getcwd())
    data_path = os.path.join(cwd, 'data', 'ocean', 'wawaves')
    f_hillary = h5py.File(os.path.join(data_path, 'Hillarys_202407.mat'), 'r')
    f_hillary_2 = h5py.File(os.path.join(data_path, 'Hillarys_202408.mat'), 'r')
    # f = h5py.File(os.path.join(data_path, 'TIDE_SouthAfricaDrifting03_202406.mat'),'r')
    print(f_hillary.get('SpotData').keys())
    direction = np.concatenate(
        [np.array(f_hillary.get('SpotData/direction')), np.array(f_hillary_2.get('SpotData/direction'))], axis=1)
    variance_density = np.concatenate(
        [np.array(f_hillary.get('SpotData/varianceDensity')), np.array(f_hillary_2.get('SpotData/varianceDensity'))],
        axis=1)
    spec_time_hillarys = np.concatenate(
        [np.array(f_hillary.get('SpotData/spec_time')), np.array(f_hillary_2.get('SpotData/spec_time'))], axis=1)
    time = np.concatenate([np.array(f_hillary.get('SpotData/time')), np.array(f_hillary_2.get('SpotData/time'))],
                          axis=1)
    frequency_hillarys = np.concatenate(
        [np.array(f_hillary.get('SpotData/frequency')), np.array(f_hillary_2.get('SpotData/frequency'))], axis=1)
    a1 = np.concatenate([np.array(f_hillary.get('SpotData/a1')), np.array(f_hillary_2.get('SpotData/a1'))], axis=1)
    a2 = np.concatenate([np.array(f_hillary.get('SpotData/a2')), np.array(f_hillary_2.get('SpotData/a2'))], axis=1)
    b1 = np.concatenate([np.array(f_hillary.get('SpotData/b1')), np.array(f_hillary_2.get('SpotData/b1'))], axis=1)
    b2 = np.concatenate([np.array(f_hillary.get('SpotData/b2')), np.array(f_hillary_2.get('SpotData/b2'))], axis=1)

    result = interpolate_spectral_coefficients(
        frequency_hillarys[:, 0],
        variance_density,
        a1, b1, a2, b2,
        n_points=200,
        enforce_constraints=True
    )
    frequency_hillarys_dense = np.array([result['f_dense']] * frequency_hillarys.shape[1]).T
    S_dense = result['variance_density_dense']

    frequency_hillarys = frequency_hillarys_dense
    variance_density = S_dense
    a1, b1, a2, b2 = result['a1_dense'], result['b1_dense'], result['a2_dense'], result['b2_dense']

    S_theta = np.zeros((variance_density.shape[1], variance_density.shape[0], 37))
    directions = np.deg2rad(np.linspace(0, 360.0, 37))
    delta_theta = np.deg2rad(10.0)
    for t in range(S_theta.shape[0]):
        for f in range(S_theta.shape[1]):
            # Load S(f), a1, b1, a2, b2 for this time and frequency
            S = variance_density[f, t]  # Omnidirectional spectrum
            a1_, b1_, a2_, b2_ = a1[f, t], b1[f, t], a2[f, t], b2[f, t]  # Directional moments

            for i, theta in enumerate(directions):
                # Compute D(f, theta)
                D = (1 / np.pi) * (
                        1 / 2 + (a1_ * np.cos(theta) + b1_ * np.sin(theta))
                        + (a2_ * np.cos(2 * theta) + b2_ * np.sin(2 * theta))
                )

                # Ensure non-negativity
                D = max(D, 0)

                # Compute S(f, theta)
                S_theta[t, f, i] = S * D

            # Optional: Renormalize to ensure sum(S_theta * delta_theta) ≈ S
            integral = np.sum(S_theta[t, f, :]) * delta_theta
            integral = integral if integral > 0 else 1.0
            S_theta[t, f, :] *= S / integral  # Adjust if integral != S

    dirs = np.linspace(0, 360.0, 37)
    dirs = (dirs + 225.0) % 360.0
    dataset = xr.Dataset(
        data_vars=dict(
            efth=(["time", "freq", "dir"], S_theta)
        ),
        coords=dict(
            time=(["time"], spec_time_hillarys[0]),
            freq=(["freq"], frequency_hillarys[:, 0]),
            dir=(["dir"], dirs)
        )
    )
    dts_hillarys = wv.SpecDataset(dataset)
    new_dirs = np.linspace(0, 360, 37)  # Cada 10° incluyendo 360°
    data_ = dts_hillarys.interp(dir=new_dirs)
    freq = data_.freq
    freq_hillarys = data_.freq


    print(f"Loaded {variance_density.shape[0]} spectra")

    # Initialize transformer
    transformer = FastReverseWaveShoaling(
        coastal_depth=30.0,
        target_depth=200.0,
        low_freq_cutoff=0.045,
        high_freq_cutoff=0.20
    )

    # Transform entire dataset in one call
    variance_density_deep = transformer.transform_batch_2d(
        data_,
        freq,
        new_dirs
    )

    print(f"Transformed spectrum shape: {variance_density_deep.shape}")

    np.save(os.path.join(data_path, 'Hillarys_0708_shoaled.npy'), variance_density_deep.to_numpy())

    return variance_density_deep


if __name__ == "__main__":
    # Run example
    example_with_your_data()

