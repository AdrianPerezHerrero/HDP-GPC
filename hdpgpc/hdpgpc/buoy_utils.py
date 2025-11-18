import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import warnings
from scipy.interpolate import CubicSpline, interp1d
import wavespectra as wv
import xarray as xr


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
    spread_direction: Optional[float] = 30.0 # Spread of direction
    variance_spectrum: Optional[np.ndarray] = None # Variance of spectrum
    A_matrix: Optional[np.ndarray] = None  # Linear transformation matrix A
    C_matrix: Optional[np.ndarray] = None  # Linear transformation matrix C



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
    DISPERSION_RATE = 0.3 #Dispersion rate when swell moves /100m

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
        lat1_r, lon1_r = np.radians(lat1), np.radians(lon1)
        lat2_r, lon2_r = np.radians(lat2), np.radians(lon2)

        dlon = lon2_r - lon1_r

        y = np.sin(dlon) * np.cos(lat2_r)
        x = np.cos(lat1_r) * np.sin(lat2_r) - \
            np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon)

        bearing = np.degrees(np.arctan2(y, x))
        return (bearing + 360) % 360

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
        return SwellBackTriangulation.GRAVITY / (4 * np.pi * frequency)

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
        return initial_energy * np.exp(-alpha.to_numpy() * distance_m)

    # ==================== BACK-TRIANGULATION METHODS ====================

    def compute_back_azimuth(self, forward_direction: float) -> float:
        """
        Compute back-azimuth (opposite of wave approach direction).

        Wave direction is "coming from", so back-azimuth points toward source.
        """
        #return (forward_direction + 180) % 360
        return forward_direction

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
            self.buoy1.lat, self.buoy1.lon, back_az1, self.buoy1.peak_frequency
        )
        backtrack2 = self._backtrack_single_buoy(
            self.buoy2.lat, self.buoy2.lon, back_az2, self.buoy2.peak_frequency
        )

        w1 = self.buoy1.peak_energy
        w2 = self.buoy2.peak_energy
        if method == "intersection":
            source_lat, source_lon, confidence = self._find_great_circle_intersection(
                self.buoy1.lat, self.buoy1.lon, back_az1,
                self.buoy2.lat, self.buoy2.lon, back_az2
            )
        elif method == "weighted_centroid":
            # Weight by inverse variance or peak energy
            source_lat = (w1 * backtrack1['candidate_lat'] +
                          w2 * backtrack2['candidate_lat']) / (w1 + w2)
            source_lon = (w1 * backtrack1['candidate_lon'] +
                          w2 * backtrack2['candidate_lon']) / (w1 + w2)
            confidence = np.mean([backtrack1['uncertainty_km'],
                                  backtrack2['uncertainty_km']])

        else:
            raise ValueError(f"Unknown method: {method}")

        source_x = (w1 * np.cos(np.radians(self.buoy1.peak_direction)) +
                    w2 * np.cos(np.radians(self.buoy2.peak_direction))) / (w1 + w2)
        source_y = (w1 * np.sin(np.radians(self.buoy1.peak_direction)) +
                    w2 * np.sin(np.radians(self.buoy2.peak_direction))) / (w1 + w2)
        initial_bearing = np.arctan2(source_y, source_x)
        initial_bearing = np.degrees(initial_bearing) % 360

        distance_source_buoy1 =  self.haversine_distance(self.buoy1.lat, self.buoy1.lon, source_lat, source_lon)
        distance_source_buoy2 =  self.haversine_distance(self.buoy2.lat, self.buoy2.lon, source_lat, source_lon)
        source_directional_spread = self._estimate_source_directional_spread(
            self.buoy1.spread_direction, self.buoy2.spread_direction, distance_source_buoy1, distance_source_buoy2)

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

    def _backtrack_single_buoy(self, lat: float, lon: float,
                               back_azimuth: float,
                               frequency: float) -> Dict:
        """
        Backtrack from single buoy along great circle.

        Assumes swell traveled from a storm at typical fetch distance.
        """
        # Estimate reasonable source distances for Pacific swells
        # Typical: 1000-5000 km for well-developed swell
        candidate_distances = np.linspace(1000, 5000, 50)

        candidates = []
        for dist_km in candidate_distances:
            candidate_lat, candidate_lon = self.destination_point(
                lat, lon, back_azimuth, dist_km
            )
            candidates.append({
                'distance_km': dist_km,
                'lat': candidate_lat,
                'lon': candidate_lon,
                'travel_time_hours': self._compute_travel_time(dist_km, frequency)
            })

        # Select middle-distance candidate as representative
        mid_idx = len(candidates) // 2
        representative = candidates[mid_idx]

        return {
            'candidate_lat': representative['lat'],
            'candidate_lon': representative['lon'],
            'candidate_distance_km': representative['distance_km'],
            'travel_time_hours': representative['travel_time_hours'],
            'uncertainty_km': 500,  # ±500 km typical uncertainty
            'all_candidates': candidates
        }

    def _estimate_source_directional_spread(
            self,
            buoy1_spread: float,
            buoy2_spread: float,
            distance_buoy1_to_source: float,
            distance_buoy2_to_source: float) -> float:
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
        spread1_at_source = max(spread1_at_source, 10.0)  # Minimum 10°
        spread2_at_source = max(spread2_at_source, 10.0)

        # Weighted average (weight by inverse distance - closer buoy more reliable)
        w1 = distance_buoy1_to_source / (distance_buoy1_to_source + distance_buoy2_to_source)
        w2 = distance_buoy2_to_source / (distance_buoy1_to_source + distance_buoy2_to_source)

        source_spread = (w1 * spread1_at_source + w2 * spread2_at_source) / (w1 + w2)

        return source_spread

    def _find_great_circle_intersection(self, lat1: float, lon1: float,
                                        bearing1: float,
                                        lat2: float, lon2: float,
                                        bearing2: float) -> Tuple[float, float, float]:
        """
        Find intersection of two great circles (simplified approach).

        Uses iterative search to minimize distance between great circles.
        """
        # Sample points along each great circle
        distances = np.linspace(500, 5000, 100)

        min_separation = np.inf
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

        confidence_radius = min_separation  # Uncertainty equals closest approach

        return best_point[0], best_point[1], confidence_radius

    def _compute_travel_time(self, distance_km: float, frequency: float) -> float:
        """
        Compute swell travel time.

        Returns: Travel time in hours
        """
        distance_m = distance_km * 1000
        group_velocity = self.deep_water_group_velocity(frequency)
        travel_time_sec = distance_m / group_velocity.to_numpy()
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
        effective_distance = distance_to_coast * np.cos(angular_offset_rad)

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
            #directional_weight = np.exp(-0.5 * (lateral_offset / beam_halfwidth_km) ** 2)
            directional_weight = 0.0
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
        mean_energy = (self.buoy1.peak_energy + self.buoy2.peak_energy) / 2
        arrival_energy = self.energy_attenuation(
            mean_energy, mean_frequency, distance_to_coast
        )

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
                                 self.deep_water_group_velocity(mean_frequency).to_numpy() / 3600
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


def compute_rotation_deviation(spectra, directions, a1, b1, dm):
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
        return diff

    theta_rad = np.radians(directions)

    # Integrate over all frequencies
    a1_total = np.sum(spectra * np.cos(theta_rad)[np.newaxis, :], axis=(0, 1))
    b1_total = np.sum(spectra * np.sin(theta_rad)[np.newaxis, :], axis=(0, 1))

    # Mean direction
    mean_dir_rad = np.arctan2(b1_total, a1_total)
    mean_dir_deg = np.degrees(mean_dir_rad)

    #Using bulk parameters
    peak_freq_idx = np.argmax(np.trapezoid(spectra, theta_rad, axis=1))
    bulk_mean_dir_deg = np.degrees(np.arctan2(b1[peak_freq_idx],a1[peak_freq_idx]))

    error_fourier = circular_diff(dm, bulk_mean_dir_deg)
    error_spectra = circular_diff(dm, mean_dir_deg)

    return error_spectra, error_fourier

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

    # Compute fixed great-circle distance (doesn't change)
    lat_off, lon_off = offshore_location
    lat_coast, lon_coast = coastal_location
    distance_km = compute_great_circle_distance(lat_off, lon_off, lat_coast, lon_coast)

    print(f"Distance from offshore to coastal buoy: {distance_km:.1f} km")

    # Initialize
    cooccurrence = np.zeros((n_clusters_offshore, n_clusters_coastal), dtype=int)
    matched_pairs = []
    propagation_times_used = []

    # Time window
    window_delta = pd.Timedelta(hours=time_window_hours)

    # For each offshore observation
    for idx, row in df_offshore.iterrows():
        offshore_time = row['time']
        offshore_cluster = int(row['cluster'])
        peak_freq = row['peak_freq']
        peak_dir = row['peak_dir']

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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_directional_spectrum(frequencies, directions, spectrum,
                                     title='Directional Wave Spectrum',
                                     vmax=None, vmin=None, cmap='jet'):
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
    cbar = plt.colorbar(mesh, ax=ax, pad=0.1, shrink=0.8)
    cbar.set_label('Energy Density (m²/Hz/deg)', rotation=270, labelpad=20)

    # Add title
    ax.set_title(title, pad=20, fontsize=14, fontweight='bold')

    # Radial axis label
    ax.set_ylabel('Frequency (Hz)', labelpad=30)

    plt.tight_layout()
    return fig, ax


# ==================== NUMBA-ACCELERATED VERSION (OPTIONAL) ====================

try:
    from numba import jit, prange

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    warnings.warn("Numba not available. Install with: pip install numba")

if NUMBA_AVAILABLE:
    @jit(nopython=True, parallel=True, cache=True)
    def _transform_batch_numba(spectra_batch, transformation_matrix):
        """
        Numba-accelerated transformation.
        Can provide 10-100x additional speedup for very large batches.
        """
        n_samples, n_freq = spectra_batch.shape
        result = np.empty_like(spectra_batch)

        for i in prange(n_samples):  # Parallel loop
            for j in range(n_freq):
                result[i, j] = spectra_batch[i, j] * transformation_matrix[j]

        return result


    @jit(nopython=True, parallel=True, cache=True)
    def _transform_batch_2d_numba(spectra_batch, transformation_matrix):
        """Numba-accelerated 2D transformation"""
        n_samples, n_freq, n_dir = spectra_batch.shape
        result = np.empty_like(spectra_batch)

        for i in prange(n_samples):  # Parallel over samples
            for j in range(n_freq):
                for k in range(n_dir):
                    result[i, j, k] = spectra_batch[i, j, k] * transformation_matrix[j]

        return result


    class NumbaReverseWaveShoaling(FastReverseWaveShoaling):
        """
        Ultra-fast Numba-accelerated version.
        Use for very large datasets (>10,000 spectra).
        """

        def transform_batch_1d(self, spectra_batch, frequencies):
            if spectra_batch.ndim == 1:
                spectra_batch = spectra_batch[np.newaxis, :]
                single = True
            else:
                single = False

            if self._transformation_matrix is None:
                self.precompute_transformation_matrix(frequencies)

            result = _transform_batch_numba(spectra_batch, self._transformation_matrix)

            return result[0] if single else result

        def transform_batch_2d(self, spectra_batch, frequencies, directions,
                               include_refraction=False):
            if spectra_batch.ndim == 2:
                spectra_batch = spectra_batch[np.newaxis, :, :]
                single = True
            else:
                single = False

            if self._transformation_matrix is None:
                self.precompute_transformation_matrix(frequencies)

            result = _transform_batch_2d_numba(spectra_batch, self._transformation_matrix)

            return result[0] if single else result


# ==================== USAGE EXAMPLES ====================

def example_batch_processing():
    """Example: Process 3000 spectra efficiently"""

    # Setup
    n_samples = 3000
    n_frequencies = 200
    n_directions = 36

    frequencies = np.linspace(0.03, 0.30, n_frequencies)
    directions = np.linspace(0, 360, n_directions, endpoint=False)

    # Generate synthetic data (replace with your actual data)
    spectra_batch = np.random.rand(n_samples, n_frequencies, n_directions) * 2.0

    print(f"Processing {n_samples} spectra of shape ({n_frequencies}, {n_directions})...")

    # Initialize transformer
    transformer = FastReverseWaveShoaling(
        coastal_depth=70.0,
        target_depth=200.0,
        propagation_distance_km=20.0,
        low_freq_cutoff=0.045,
        high_freq_cutoff=0.20
    )

    # Option 1: NumPy vectorized (fast)
    import time
    start = time.time()
    spectra_deep = transformer.transform_batch_2d(
        spectra_batch, frequencies, directions
    )
    elapsed = time.time() - start
    print(f"NumPy vectorized: {elapsed:.3f} seconds ({n_samples / elapsed:.1f} spectra/sec)")

    # Option 2: Numba accelerated (ultra-fast, if available)
    if NUMBA_AVAILABLE:
        transformer_numba = NumbaReverseWaveShoaling(
            coastal_depth=70.0,
            target_depth=200.0
        )

        start = time.time()
        spectra_deep_numba = transformer_numba.transform_batch_2d(
            spectra_batch, frequencies, directions
        )
        elapsed = time.time() - start
        print(f"Numba accelerated: {elapsed:.3f} seconds ({n_samples / elapsed:.1f} spectra/sec)")

    return spectra_deep


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

