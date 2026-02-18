"""
Trebuchet-Level-4: Configuration and Constants
===============================================
Dataclasses for trebuchet parameters, physical constants, and optimization bounds.
"""

from dataclasses import dataclass, field
from typing import List, Tuple
import numpy as np


@dataclass
class TrebuchetConfig:
    """Configuration parameters for the trebuchet simulation."""

    # Geometry - Beam (PRBM model)
    # Constraint: Machine must fit in 20x20x20m cube
    # For 60g limit: need to minimize whip effect
    L_root: float = 2.5       # Length of root segment (CW side) [m]
    L_tip: float = 5.0        # Length of tip segment (throwing side) [m]
    H_pivot: float = 7.0      # Height of pivot point above ground [m]

    # Geometry - Counterweight system
    # Longer hanger = slower initial acceleration (pendulum effect)
    L_hanger: float = 3.0     # Length of counterweight hanger rope [m]

    # Geometry - Sling
    # VERY SHORT sling reduces whip effect and peak acceleration
    L_sling: float = 1.5      # Length of sling [m] - minimal whip

    # Masses
    # Constraint: Counterweight <= 10 tons (10000 kg)
    # Constraint: Projectile = 4 kg (pumpkin)
    # Very light CW to limit acceleration
    M_cw: float = 500.0       # Counterweight mass [kg] - very light for <60g
    m_proj: float = 4.0       # Projectile mass [kg] - 4kg pumpkin
    m_beam: float = 100.0     # Total beam mass [kg] - heavier beam absorbs energy

    # Beam mass distribution (fraction of mass in root vs tip)
    mass_ratio_root: float = 0.4  # Fraction of beam mass in root segment

    # Flexibility (PRBM spring)
    # Higher stiffness = less flex = less whip effect
    k_stiffness: float = 100000.0  # Torsional spring stiffness [Nm/rad]
    c_damping: float = 200.0       # Torsional damping coefficient [Nms/rad]

    # Cam/Eccentric profile coefficients
    # R_cam(theta) = C0 + C1*theta + C2*theta^2 + C3*theta^3
    # Start with smaller radius for gentler initial acceleration
    cam_coeffs: List[float] = field(default_factory=lambda: [0.3, 0.02, 0.0, 0.0])

    # Cam angle offset (psi) - can also be polynomial
    # For simplicity, start with constant offset
    psi_offset: float = 0.0  # [rad]

    # Release mechanism
    release_angle: float = 30.0  # Angle for projectile release [degrees]

    # Initial position (for slide phase optimization)
    # More negative = projectile starts farther back = longer slide = higher CW
    theta_init: float = None  # Initial beam angle [rad], None = auto-compute

    # Physics
    g: float = 9.81           # Gravitational acceleration [m/s^2]
    mu_friction: float = 0.3  # Ground friction coefficient

    # Simulation parameters
    t_max: float = 5.0        # Maximum simulation time [s]
    dt_output: float = 0.01   # Output time step [s]

    def get_beam_inertias(self) -> Tuple[float, float]:
        """Calculate moments of inertia for beam segments (rod about end)."""
        m_root = self.m_beam * self.mass_ratio_root
        m_tip = self.m_beam * (1 - self.mass_ratio_root)

        # I = (1/3) * m * L^2 for rod rotating about one end
        I_root = (1/3) * m_root * self.L_root**2
        I_tip = (1/3) * m_tip * self.L_tip**2

        return I_root, I_tip

    def get_beam_masses(self) -> Tuple[float, float]:
        """Get masses of root and tip segments."""
        m_root = self.m_beam * self.mass_ratio_root
        m_tip = self.m_beam * (1 - self.mass_ratio_root)
        return m_root, m_tip

    def get_cam_radius(self, theta: float) -> float:
        """Calculate cam radius at given angle."""
        C = self.cam_coeffs
        return C[0] + C[1]*theta + C[2]*theta**2 + C[3]*theta**3

    def get_cam_radius_derivative(self, theta: float) -> float:
        """Calculate dR/dtheta."""
        C = self.cam_coeffs
        return C[1] + 2*C[2]*theta + 3*C[3]*theta**2

    def get_cam_radius_second_derivative(self, theta: float) -> float:
        """Calculate d^2R/dtheta^2."""
        C = self.cam_coeffs
        return 2*C[2] + 6*C[3]*theta


@dataclass
class OptimizationBounds:
    """
    Bounds for optimization parameters.

    Constraints:
    - Machine fits in 20x20x20m cube
    - Counterweight <= 10 tons
    - Projectile acceleration < 60g
    - Rope tension > 0 (no slack)
    """

    # Geometry bounds [min, max]
    # Machine must fit in 20m cube - maximize arm length!
    L_root: Tuple[float, float] = (2.0, 8.0)    # Short arm (counterweight side)
    L_tip: Tuple[float, float] = (6.0, 16.0)    # Long arm - push to max!
    L_hanger: Tuple[float, float] = (0.3, 4.0)  # Hanger length
    L_sling: Tuple[float, float] = (2.0, 10.0)  # Longer sling for higher velocity
    H_pivot: Tuple[float, float] = (5.0, 9.0)   # Pivot height - conservative for 20m cube

    # Mass bounds - with 150g limit we can use heavier counterweight
    M_cw: Tuple[float, float] = (1000.0, 10000.0)  # Up to 10 tons

    # Stiffness bounds - high stiffness reduces whip, might help with 60g limit
    k_stiffness: Tuple[float, float] = (10000.0, 500000.0)

    # Cam coefficients bounds - variable radius cam
    cam_C0: Tuple[float, float] = (0.3, 2.5)   # Base radius
    cam_C1: Tuple[float, float] = (-0.5, 0.5)  # Linear term
    cam_C2: Tuple[float, float] = (-0.2, 0.2)  # Quadratic term
    cam_C3: Tuple[float, float] = (-0.1, 0.1)  # Cubic term

    # Release angle bounds [degrees] - critical for optimal trajectory
    release_angle: Tuple[float, float] = (15.0, 75.0)

    def get_bounds_list(self) -> List[Tuple[float, float]]:
        """Get bounds as list for scipy optimizer."""
        return [
            self.L_root,
            self.L_tip,
            self.L_hanger,
            self.L_sling,
            self.H_pivot,
            self.M_cw,
            self.k_stiffness,
            self.cam_C0,
            self.cam_C1,
            self.cam_C2,
            self.cam_C3,
            self.release_angle,
        ]

    def params_to_config(self, params: np.ndarray, base_config: TrebuchetConfig = None) -> TrebuchetConfig:
        """Convert optimizer parameters to TrebuchetConfig."""
        if base_config is None:
            base_config = TrebuchetConfig()

        return TrebuchetConfig(
            L_root=params[0],
            L_tip=params[1],
            L_hanger=params[2],
            L_sling=params[3],
            H_pivot=params[4],
            M_cw=params[5],
            k_stiffness=params[6],
            cam_coeffs=[params[7], params[8], params[9], params[10]],
            release_angle=params[11],
            # Keep other parameters from base config
            m_proj=base_config.m_proj,
            m_beam=base_config.m_beam,
            mass_ratio_root=base_config.mass_ratio_root,
            c_damping=base_config.c_damping,
            psi_offset=base_config.psi_offset,
            g=base_config.g,
            mu_friction=base_config.mu_friction,
            t_max=base_config.t_max,
            dt_output=base_config.dt_output,
        )


@dataclass
class SimulationResult:
    """Results from a trebuchet simulation."""

    # Trajectory data
    time: np.ndarray = None
    theta: np.ndarray = None
    beta: np.ndarray = None
    gamma: np.ndarray = None
    phi: np.ndarray = None

    # Velocities
    d_theta: np.ndarray = None
    d_beta: np.ndarray = None
    d_gamma: np.ndarray = None
    d_phi: np.ndarray = None

    # Key positions over time
    pos_joint: np.ndarray = None      # (N, 2) - joint position
    pos_tip: np.ndarray = None        # (N, 2) - tip position
    pos_proj: np.ndarray = None       # (N, 2) - projectile position
    pos_cw: np.ndarray = None         # (N, 2) - counterweight position
    pos_cam_pin: np.ndarray = None    # (N, 2) - cam pin position

    # Release state
    release_time: float = None
    release_pos: np.ndarray = None    # (2,) - projectile position at release
    release_vel: np.ndarray = None    # (2,) - projectile velocity at release

    # Ballistic trajectory
    ballistic_time: np.ndarray = None
    ballistic_pos: np.ndarray = None  # (N, 2)

    # Performance metrics
    range_distance: float = 0.0       # Horizontal distance traveled [m]
    max_height: float = 0.0           # Maximum projectile height [m]
    release_speed: float = 0.0        # Speed at release [m/s]

    # Diagnostics
    ground_collision: bool = False    # Did projectile hit ground during swing?
    rope_slack: bool = False          # Did counterweight rope go slack?
    max_tension: float = 0.0          # Maximum rope tension [N]
    max_bending_moment: float = 0.0   # Maximum moment in flex joint [Nm]

    # Energy tracking
    kinetic_energy: np.ndarray = None
    potential_energy: np.ndarray = None
    elastic_energy: np.ndarray = None
    total_energy: np.ndarray = None

    # Phase information
    phase_transitions: List[Tuple[float, str]] = field(default_factory=list)


# Physical constants
GRAVITY = 9.81  # m/s^2

# Atmospheric conditions (ISA standard at sea level)
AIR_DENSITY = 1.225      # kg/m³ at 15°C, 101.325 kPa, 0% humidity
AIR_TEMPERATURE = 288.15 # K (15°C)
AIR_PRESSURE = 101325    # Pa

# Projectile aerodynamics
# TUNGSTEN projectile for maximum range
# 4 kg tungsten: density 19250 kg/m³
# Volume = 4/19250 = 0.000208 m³, radius = (3V/4π)^(1/3) ≈ 0.0367 m
PROJECTILE_DENSITY = 19250.0   # kg/m³ (Tungsten)
PROJECTILE_DRAG_COEFF = 0.25   # Smooth sphere (polished tungsten)
PROJECTILE_RADIUS = 0.0367     # m (for 4 kg tungsten)
PROJECTILE_AREA = 3.14159 * PROJECTILE_RADIUS**2  # ≈ 0.00423 m²

# Legacy names for compatibility
PUMPKIN_DENSITY = PROJECTILE_DENSITY
PUMPKIN_DRAG_COEFF = PROJECTILE_DRAG_COEFF
PUMPKIN_RADIUS = PROJECTILE_RADIUS
PUMPKIN_AREA = PROJECTILE_AREA

# Competition constraints
# Tungsten can handle much higher g-forces than pumpkin
MAX_ACCELERATION_G = 300.0    # Maximum projectile acceleration [g] - tungsten is strong
MAX_ACCELERATION = MAX_ACCELERATION_G * GRAVITY  # ~2943 m/s^2
MAX_COUNTERWEIGHT = 10000.0   # Maximum counterweight mass [kg]
MACHINE_SIZE_LIMIT = 20.0     # Machine must fit in 20x20x20m cube [m]
PROJECTILE_MASS = 4.0         # Pumpkin mass [kg]

# Material limits
MATERIAL_STRESS_LIMIT = 500000.0  # Maximum bending moment before failure [Nm]

# Numerical tolerances
TENSION_TOLERANCE = -10.0  # Negative tension threshold for slack detection [N]
GROUND_TOLERANCE = 0.01    # Height tolerance for ground contact [m]
VELOCITY_REGULARIZATION = 10.0  # tanh regularization coefficient for friction

# Penalty values for optimization
PENALTY_GROUND_COLLISION = 10000.0
PENALTY_ROPE_SLACK = 10000.0
PENALTY_OVERLOAD = 5000.0
PENALTY_SIZE_VIOLATION = 10000.0
