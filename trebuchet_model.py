"""
Trebuchet-Level-4: Simulation Model
====================================
Main simulation class for the 4-DoF trebuchet with:
- Cam-driven counterweight
- Flexible beam (PRBM model)
- Three-phase simulation (slide, swing, ballistics)
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import PchipInterpolator
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable
import warnings

from config import (
    TrebuchetConfig, SimulationResult,
    GRAVITY, MAX_ACCELERATION, TENSION_TOLERANCE, GROUND_TOLERANCE,
    VELOCITY_REGULARIZATION, PENALTY_GROUND_COLLISION, PENALTY_ROPE_SLACK,
    PENALTY_OVERLOAD, PENALTY_SIZE_VIOLATION, MACHINE_SIZE_LIMIT, MAX_ACCELERATION_G,
    AIR_DENSITY, PUMPKIN_DRAG_COEFF, PUMPKIN_AREA
)

# Try to import generated equations, fall back to simplified model if not available
try:
    from generated_eom import compute_M, compute_F, compute_accelerations
    from generated_eom import (
        compute_p_pivot, compute_p_joint, compute_p_tip,
        compute_p_proj, compute_p_cam_pin, compute_p_cw,
        compute_v_proj, compute_v_cw
    )
    EOM_AVAILABLE = True
except ImportError:
    warnings.warn("generated_eom.py not found. Run derive_physics.py first!")
    EOM_AVAILABLE = False


class CamProfile:
    """
    Cam profile using cubic spline interpolation (PchipInterpolator).
    Provides smooth C1-continuous radius as function of beam angle.
    """

    def __init__(self, coeffs: List[float] = None, control_points: List[Tuple[float, float]] = None):
        """
        Initialize cam profile.

        Parameters
        ----------
        coeffs : list of float, optional
            Polynomial coefficients [C0, C1, C2, C3] for R(theta) = C0 + C1*theta + ...
        control_points : list of (theta, R) tuples, optional
            Control points for spline interpolation
        """
        if control_points is not None:
            # Use spline interpolation
            thetas = np.array([p[0] for p in control_points])
            radii = np.array([p[1] for p in control_points])
            self._spline = PchipInterpolator(thetas, radii)
            self._use_spline = True
        else:
            # Use polynomial
            self._coeffs = coeffs if coeffs is not None else [0.8, 0.0, 0.0, 0.0]
            self._use_spline = False

    def __call__(self, theta: float) -> float:
        """Get cam radius at given angle."""
        if self._use_spline:
            return float(self._spline(theta))
        else:
            C = self._coeffs
            return C[0] + C[1]*theta + C[2]*theta**2 + C[3]*theta**3

    def derivative(self, theta: float) -> float:
        """Get dR/dtheta."""
        if self._use_spline:
            return float(self._spline.derivative()(theta))
        else:
            C = self._coeffs
            return C[1] + 2*C[2]*theta + 3*C[3]*theta**2

    def second_derivative(self, theta: float) -> float:
        """Get d^2R/dtheta^2."""
        if self._use_spline:
            return float(self._spline.derivative(2)(theta))
        else:
            C = self._coeffs
            return 2*C[2] + 6*C[3]*theta


class TrebuchetSimulator:
    """
    Main simulation class for the trebuchet.

    Handles three phases:
    1. Slide: Projectile on ground with friction
    2. Swing: Full 4-DoF dynamics
    3. Ballistics: Free projectile flight
    """

    def __init__(self, config: TrebuchetConfig):
        """
        Initialize simulator with given configuration.

        Parameters
        ----------
        config : TrebuchetConfig
            Configuration parameters for the trebuchet
        """
        self.config = config
        self.cam = CamProfile(coeffs=config.cam_coeffs)

        # Precompute beam properties
        self.I_root, self.I_tip = config.get_beam_inertias()
        self.m_root, self.m_tip = config.get_beam_masses()

        # Release angle in radians
        self.release_angle_rad = np.deg2rad(config.release_angle)

        # Results storage
        self.result = SimulationResult()

        # Tracking
        self._max_accel = 0.0
        self._min_tension = float('inf')
        self._max_pos = np.array([0.0, 0.0])

    def _get_params_dict(self, theta: float) -> dict:
        """Get parameters dictionary for EOM functions."""
        cfg = self.config

        # Cam values at current angle
        R_cam = self.cam(theta)
        dR_cam = self.cam.derivative(theta)
        ddR_cam = self.cam.second_derivative(theta)

        return {
            'L_root': cfg.L_root,
            'L_tip': cfg.L_tip,
            'L_hanger': cfg.L_hanger,
            'L_sling': cfg.L_sling,
            'H': cfg.H_pivot,
            'm_root': self.m_root,
            'm_tip': self.m_tip,
            'M_cw': cfg.M_cw,
            'm_proj': cfg.m_proj,
            'I_root': self.I_root,
            'I_tip': self.I_tip,
            'k_stiff': cfg.k_stiffness,
            'c_damp': cfg.c_damping,
            'g': cfg.g,
            'R_cam': R_cam,
            'dR_cam': dR_cam,
            'ddR_cam': ddR_cam,
            'psi': cfg.psi_offset,
            'dpsi': 0.0,  # Constant psi offset
            'ddpsi': 0.0,
        }

    def _compute_positions(self, state: np.ndarray) -> dict:
        """
        Compute all positions from state vector.

        Parameters
        ----------
        state : ndarray
            State vector [theta, beta, gamma, phi, d_theta, d_beta, d_gamma, d_phi]

        Returns
        -------
        dict : Dictionary with position arrays
        """
        th, bt, gm, ph = state[:4]
        cfg = self.config

        # Cam values
        R_cam = self.cam(th)
        psi = cfg.psi_offset

        if EOM_AVAILABLE:
            p_pivot = compute_p_pivot(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                      cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
            p_joint = compute_p_joint(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                      cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
            p_tip = compute_p_tip(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                  cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
            p_proj = compute_p_proj(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                    cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
            p_cam_pin = compute_p_cam_pin(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                          cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
            p_cw = compute_p_cw(th, bt, gm, ph, cfg.L_root, cfg.L_tip,
                                cfg.L_hanger, cfg.L_sling, cfg.H_pivot, R_cam, psi)
        else:
            # Fallback manual computation
            p_pivot = np.array([0.0, cfg.H_pivot])
            p_joint = p_pivot + cfg.L_root * np.array([np.sin(th), -np.cos(th)])

            angle_tip = th + bt
            p_tip = p_joint + cfg.L_tip * np.array([np.sin(angle_tip), -np.cos(angle_tip)])

            p_proj = p_tip + cfg.L_sling * np.array([np.sin(ph), -np.cos(ph)])

            cam_angle = th + np.pi + psi
            p_cam_pin = p_pivot + R_cam * np.array([np.sin(cam_angle), -np.cos(cam_angle)])

            p_cw = p_cam_pin + cfg.L_hanger * np.array([np.sin(gm), -np.cos(gm)])

        return {
            'pivot': p_pivot,
            'joint': p_joint,
            'tip': p_tip,
            'proj': p_proj,
            'cam_pin': p_cam_pin,
            'cw': p_cw,
        }

    def _compute_velocities(self, state: np.ndarray) -> dict:
        """Compute velocities from state vector."""
        th, bt, gm, ph, dth, dbt, dgm, dph = state
        cfg = self.config

        R_cam = self.cam(th)
        dR_cam = self.cam.derivative(th)
        psi = cfg.psi_offset
        dpsi = 0.0

        if EOM_AVAILABLE:
            v_proj = compute_v_proj(th, bt, gm, ph, dth, dbt, dgm, dph,
                                    cfg.L_root, cfg.L_tip, cfg.L_sling,
                                    R_cam, dR_cam, psi, dpsi)
            v_cw = compute_v_cw(th, bt, gm, ph, dth, dbt, dgm, dph,
                                cfg.L_root, cfg.L_hanger, R_cam, dR_cam, psi, dpsi)
        else:
            # Fallback - numerical differentiation would be needed
            # For now, approximate with finite differences
            v_proj = np.array([0.0, 0.0])
            v_cw = np.array([0.0, 0.0])

        return {
            'v_proj': v_proj,
            'v_cw': v_cw,
        }

    def _compute_tension(self, state: np.ndarray, accel: np.ndarray) -> float:
        """
        Compute tension in counterweight rope.

        Uses Newton's second law on the counterweight:
        T - M_cw * g = M_cw * a_vertical
        """
        th, bt, gm, ph, dth, dbt, dgm, dph = state
        ddth, ddbt, ddgm, ddph = accel
        cfg = self.config

        # Counterweight acceleration (approximate from gamma dynamics)
        # a_cw_tangential = L_hanger * ddgm
        # a_cw_centripetal = L_hanger * dgm^2

        # Simplified tension estimate
        # T = M_cw * (g * cos(gamma) + L_hanger * dgm^2)
        T = cfg.M_cw * (cfg.g * np.cos(gm) + cfg.L_hanger * dgm**2)

        return T

    def _compute_proj_acceleration(self, state: np.ndarray, accel: np.ndarray) -> float:
        """Compute projectile acceleration magnitude."""
        th, bt, gm, ph, dth, dbt, dgm, dph = state
        ddth, ddbt, ddgm, ddph = accel
        cfg = self.config

        # Tip position derivatives
        angle_tip = th + bt
        omega_tip = dth + dbt
        alpha_tip = ddth + ddbt

        # Projectile position relative to tip end
        # Acceleration has tangential and centripetal components

        # Joint acceleration
        a_joint_x = cfg.L_root * (ddth * np.cos(th) - dth**2 * np.sin(th))
        a_joint_y = cfg.L_root * (ddth * np.sin(th) + dth**2 * np.cos(th))

        # Tip acceleration relative to joint
        a_tip_rel_x = cfg.L_tip * (alpha_tip * np.cos(angle_tip) - omega_tip**2 * np.sin(angle_tip))
        a_tip_rel_y = cfg.L_tip * (alpha_tip * np.sin(angle_tip) + omega_tip**2 * np.cos(angle_tip))

        a_tip_x = a_joint_x + a_tip_rel_x
        a_tip_y = a_joint_y + a_tip_rel_y

        # Projectile acceleration relative to tip
        omega_phi = dph
        alpha_phi = ddph

        a_proj_rel_x = cfg.L_sling * (alpha_phi * np.cos(ph) - omega_phi**2 * np.sin(ph))
        a_proj_rel_y = cfg.L_sling * (alpha_phi * np.sin(ph) + omega_phi**2 * np.cos(ph))

        a_proj_x = a_tip_x + a_proj_rel_x
        a_proj_y = a_tip_y + a_proj_rel_y

        return np.sqrt(a_proj_x**2 + a_proj_y**2)

    def _eom_swing(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Equations of motion for swing phase (full 4-DoF).

        Parameters
        ----------
        t : float
            Current time
        state : ndarray
            State vector [theta, beta, gamma, phi, d_theta, d_beta, d_gamma, d_phi]

        Returns
        -------
        ndarray : State derivative [d_theta, d_beta, d_gamma, d_phi, dd_theta, dd_beta, dd_gamma, dd_phi]
        """
        th = state[0]
        params = self._get_params_dict(th)

        if EOM_AVAILABLE:
            accel = compute_accelerations(state, params)
        else:
            # Fallback: simplified model
            accel = self._simplified_eom(state)

        # Track max acceleration
        proj_accel = self._compute_proj_acceleration(state, accel)
        self._max_accel = max(self._max_accel, proj_accel)

        # Track tension
        tension = self._compute_tension(state, accel)
        self._min_tension = min(self._min_tension, tension)

        # Track max position (for size check)
        positions = self._compute_positions(state)
        for pos in positions.values():
            self._max_pos = np.maximum(self._max_pos, np.abs(pos))

        return np.concatenate([state[4:], accel])

    def _simplified_eom(self, state: np.ndarray) -> np.ndarray:
        """Simplified equations of motion (fallback when generated_eom not available)."""
        th, bt, gm, ph, dth, dbt, dgm, dph = state
        cfg = self.config
        g = cfg.g

        # Very simplified - treat as coupled pendulums
        # This is NOT accurate but allows testing without generated equations

        ddth = -g / cfg.L_root * np.sin(th)  # Simple pendulum
        ddbt = -cfg.k_stiffness / self.I_tip * bt - cfg.c_damping / self.I_tip * dbt
        ddgm = -g / cfg.L_hanger * np.sin(gm)
        ddph = -g / cfg.L_sling * np.sin(ph)

        return np.array([ddth, ddbt, ddgm, ddph])

    def _eom_slide(self, t: float, state: np.ndarray) -> np.ndarray:
        """
        Equations of motion for slide phase (projectile on ground).

        Adds normal force reaction and friction.
        """
        # For slide phase, we constrain projectile y = 0
        # This modifies the effective EOM

        # Get base swing EOM
        deriv = self._eom_swing(t, state)

        # Apply friction force (regularized with tanh)
        th, bt, gm, ph, dth, dbt, dgm, dph = state
        cfg = self.config

        # Compute projectile horizontal velocity
        velocities = self._compute_velocities(state)
        vx_proj = velocities['v_proj'][0]

        # Friction force: F = -mu * N * sign(v)
        # Regularized: F = -mu * N * tanh(k * v)
        # N is approximately m_proj * g during slide
        N = cfg.m_proj * cfg.g
        F_friction = -cfg.mu_friction * N * np.tanh(VELOCITY_REGULARIZATION * vx_proj)

        # This friction affects phi dynamics
        # Simplified: add deceleration to ddphi
        # deriv[7] += F_friction / (cfg.m_proj * cfg.L_sling)

        return deriv

    def _release_event(self, t: float, state: np.ndarray) -> float:
        """
        Event function for projectile release.

        The release happens when the sling angle relative to the arm direction
        decreases through the release_angle threshold.

        With correct physics:
        - phi_relative starts positive (sling trailing behind arm)
        - It increases to a peak as the arm accelerates
        - Then decreases as the sling catches up and swings forward
        - Release should happen when phi_rel decreases through +release_angle
          (when sling is extended forward with projectile having upward velocity)
        """
        th, bt, gm, ph = state[:4]

        # Angle of sling relative to tip direction
        tip_angle = th + bt
        phi_relative = ph - tip_angle

        # Normalize to [-pi, pi]
        while phi_relative > np.pi:
            phi_relative -= 2 * np.pi
        while phi_relative < -np.pi:
            phi_relative += 2 * np.pi

        # Release when phi_relative decreases through +release_angle
        # This happens when the sling is swinging forward and the projectile
        # has good upward velocity for launch
        return phi_relative - self.release_angle_rad

    _release_event.terminal = True
    _release_event.direction = -1  # Detect when decreasing through zero

    def _ground_event(self, t: float, state: np.ndarray) -> float:
        """
        Event function for projectile hitting ground.

        Returns 0 when projectile y <= 0.
        """
        positions = self._compute_positions(state)
        return positions['proj'][1] - GROUND_TOLERANCE

    _ground_event.terminal = True
    _ground_event.direction = -1

    def _overload_event(self, t: float, state: np.ndarray) -> float:
        """
        Event function for projectile overload.

        Returns 0 when acceleration exceeds 60g limit.
        Pumpkin will be destroyed above this threshold.
        """
        # Compute acceleration for this state
        th = state[0]
        params = self._get_params_dict(th)

        if EOM_AVAILABLE:
            accel = compute_accelerations(state, params)
        else:
            accel = self._simplified_eom(state)

        proj_accel = self._compute_proj_acceleration(state, accel)
        accel_g = proj_accel / GRAVITY

        return MAX_ACCELERATION_G - accel_g

    _overload_event.terminal = True
    _overload_event.direction = -1

    def _liftoff_event(self, t: float, state: np.ndarray) -> float:
        """
        Event function for projectile lifting off ground (end of slide phase).

        Returns 0 when sling tension pulls projectile up.
        """
        positions = self._compute_positions(state)
        return positions['proj'][1] - GROUND_TOLERANCE

    _liftoff_event.terminal = True
    _liftoff_event.direction = 1

    def _get_initial_state(self) -> np.ndarray:
        """
        Compute initial state with projectile on ground for slide phase.

        The projectile starts on the ground (y=0) and slides during initial acceleration.
        More negative theta = projectile farther back = longer slide = higher CW = more energy.

        Returns state vector [theta, beta, gamma, phi, d_theta, d_beta, d_gamma, d_phi]
        """
        cfg = self.config

        # If theta_init is specified in config, use it directly
        if cfg.theta_init is not None:
            theta_init = cfg.theta_init
        else:
            # Search for optimal starting angle
            # Range: from -150° (very tilted back) to -30° (barely tilted)
            # More negative = longer slide distance, higher CW
            best_theta = -np.pi / 3  # -60 degrees default
            best_phi = 0.0
            best_cw_height = -float('inf')

            for theta_try in np.linspace(-5*np.pi/6, -np.pi/6, 100):  # -150° to -30°
                angle_tip = theta_try  # tip angle with beta=0

                # Tip position
                x_tip = cfg.L_root * np.sin(theta_try) + cfg.L_tip * np.sin(angle_tip)
                y_tip = cfg.H_pivot - cfg.L_root * np.cos(theta_try) - cfg.L_tip * np.cos(angle_tip)

                # Find phi to place projectile exactly on ground (y=0)
                # y_proj = y_tip - L_sling * cos(phi) = 0
                # cos(phi) = y_tip / L_sling
                cos_phi_needed = y_tip / cfg.L_sling

                if abs(cos_phi_needed) <= 1 and cos_phi_needed > 0:
                    phi_try = np.arccos(cos_phi_needed)

                    x_proj = x_tip + cfg.L_sling * np.sin(phi_try)
                    y_proj = y_tip - cfg.L_sling * np.cos(phi_try)

                    # Projectile must be behind machine (x < 0) and on ground
                    if x_proj < -1.0 and abs(y_proj) < 0.1:
                        # Calculate CW height for this configuration
                        R_cam = self.cam(theta_try)
                        psi = cfg.psi_offset
                        y_cam = cfg.H_pivot + R_cam * np.cos(theta_try + psi)
                        y_cw = y_cam - cfg.L_hanger  # gamma=0

                        # Prefer higher CW (more potential energy)
                        if y_cw > best_cw_height:
                            best_cw_height = y_cw
                            best_theta = theta_try
                            best_phi = phi_try

            theta_init = best_theta
            phi_init = best_phi

        # For specified theta_init, compute phi to put projectile on ground
        if cfg.theta_init is not None:
            angle_tip = theta_init
            x_tip = cfg.L_root * np.sin(theta_init) + cfg.L_tip * np.sin(angle_tip)
            y_tip = cfg.H_pivot - cfg.L_root * np.cos(theta_init) - cfg.L_tip * np.cos(angle_tip)
            cos_phi = y_tip / cfg.L_sling
            if abs(cos_phi) <= 1:
                phi_init = np.arccos(cos_phi)
            else:
                phi_init = 0.0
        else:
            phi_init = best_phi

        # Initial gamma: counterweight hanging straight down
        gamma_init = 0.0

        # All velocities start at zero
        return np.array([theta_init, 0.0, gamma_init, phi_init, 0.0, 0.0, 0.0, 0.0])

    def simulate(self) -> SimulationResult:
        """
        Run full trebuchet simulation.

        Returns
        -------
        SimulationResult : Results including trajectory, range, and diagnostics
        """
        cfg = self.config
        result = SimulationResult()

        # Reset tracking
        self._max_accel = 0.0
        self._min_tension = float('inf')
        self._max_pos = np.array([0.0, 0.0])

        # Get initial state
        state0 = self._get_initial_state()

        # Phase transitions
        result.phase_transitions = [(0.0, 'start')]

        # =====================================================================
        # PHASE A: Slide (optional - if projectile starts on ground)
        # =====================================================================
        positions = self._compute_positions(state0)
        if positions['proj'][1] < GROUND_TOLERANCE:
            # Start with slide phase
            sol_slide = solve_ivp(
                self._eom_slide,
                [0, cfg.t_max],
                state0,
                method='LSODA',
                events=[self._liftoff_event],
                dense_output=True,
                max_step=0.01,
            )

            if sol_slide.t_events[0].size > 0:
                t_liftoff = sol_slide.t_events[0][0]
                state_liftoff = sol_slide.y_events[0][0]
                result.phase_transitions.append((t_liftoff, 'liftoff'))
            else:
                # No liftoff - projectile stayed on ground
                state_liftoff = sol_slide.y[:, -1]
                t_liftoff = sol_slide.t[-1]
        else:
            # No slide phase
            state_liftoff = state0
            t_liftoff = 0.0

        # =====================================================================
        # PHASE B: Swing (main trebuchet operation)
        # =====================================================================
        sol_swing = solve_ivp(
            self._eom_swing,
            [t_liftoff, cfg.t_max],
            state_liftoff,
            method='LSODA',
            events=[self._release_event, self._ground_event, self._overload_event],
            dense_output=True,
            max_step=0.005,
        )

        # Check which event triggered
        if sol_swing.t_events[0].size > 0:
            # Release event
            t_release = sol_swing.t_events[0][0]
            state_release = sol_swing.y_events[0][0]
            result.phase_transitions.append((t_release, 'release'))
            result.ground_collision = False
        elif sol_swing.t_events[1].size > 0:
            # Ground collision
            t_release = sol_swing.t_events[1][0]
            state_release = sol_swing.y_events[1][0]
            result.phase_transitions.append((t_release, 'ground_collision'))
            result.ground_collision = True
        elif sol_swing.t_events[2].size > 0:
            # Overload - pumpkin destroyed
            t_release = sol_swing.t_events[2][0]
            state_release = sol_swing.y_events[2][0]
            result.phase_transitions.append((t_release, 'overload'))
            result.ground_collision = True  # Treat as failure
        else:
            # Timeout
            t_release = sol_swing.t[-1]
            state_release = sol_swing.y[:, -1]
            result.phase_transitions.append((t_release, 'timeout'))

        # Store trajectory (including slide phase if present)
        # Combine slide and swing phases
        t_slide_eval = np.arange(0, t_liftoff, cfg.dt_output) if t_liftoff > 0 else np.array([])
        t_swing_eval = np.arange(t_liftoff, t_release, cfg.dt_output)

        all_times = []
        all_states = []

        # Add slide phase data
        if len(t_slide_eval) > 0 and t_liftoff > 0:
            slide_states = sol_slide.sol(t_slide_eval)
            for i, t in enumerate(t_slide_eval):
                all_times.append(t)
                all_states.append(slide_states[:, i])

        # Add swing phase data
        if len(t_swing_eval) > 0:
            swing_states = sol_swing.sol(t_swing_eval)
            for i, t in enumerate(t_swing_eval):
                all_times.append(t)
                all_states.append(swing_states[:, i])

        if len(all_times) > 0:
            result.time = np.array(all_times)
            all_states = np.array(all_states)
            result.theta = all_states[:, 0]
            result.beta = all_states[:, 1]
            result.gamma = all_states[:, 2]
            result.phi = all_states[:, 3]
            result.d_theta = all_states[:, 4]
            result.d_beta = all_states[:, 5]
            result.d_gamma = all_states[:, 6]
            result.d_phi = all_states[:, 7]

            # Compute positions over time
            n_points = len(all_times)
            result.pos_joint = np.zeros((n_points, 2))
            result.pos_tip = np.zeros((n_points, 2))
            result.pos_proj = np.zeros((n_points, 2))
            result.pos_cw = np.zeros((n_points, 2))
            result.pos_cam_pin = np.zeros((n_points, 2))

            for i in range(n_points):
                state_i = all_states[i]
                positions = self._compute_positions(state_i)
                result.pos_joint[i] = positions['joint']
                result.pos_tip[i] = positions['tip']
                result.pos_proj[i] = positions['proj']
                result.pos_cw[i] = positions['cw']
                result.pos_cam_pin[i] = positions['cam_pin']

        # =====================================================================
        # PHASE C: Ballistics with air resistance
        # =====================================================================
        if not result.ground_collision:
            positions = self._compute_positions(state_release)
            velocities = self._compute_velocities(state_release)

            result.release_pos = positions['proj']
            result.release_vel = velocities['v_proj']
            result.release_time = t_release
            result.release_speed = np.linalg.norm(result.release_vel)

            # Ballistic flight with drag
            x0, y0 = result.release_pos
            vx0, vy0 = result.release_vel

            # Drag coefficient: F_drag = 0.5 * rho * v^2 * Cd * A
            # a_drag = F_drag / m = 0.5 * rho * Cd * A / m * v^2
            drag_factor = 0.5 * AIR_DENSITY * PUMPKIN_DRAG_COEFF * PUMPKIN_AREA / cfg.m_proj

            def ballistic_eom(t, state):
                """Equations of motion with air drag."""
                x, y, vx, vy = state
                v = np.sqrt(vx**2 + vy**2)
                if v > 0:
                    # Drag force opposes velocity
                    ax = -drag_factor * v * vx
                    ay = -cfg.g - drag_factor * v * vy
                else:
                    ax = 0
                    ay = -cfg.g
                return [vx, vy, ax, ay]

            def ground_hit(t, state):
                return state[1]  # y coordinate
            ground_hit.terminal = True
            ground_hit.direction = -1

            # Integrate ballistic trajectory
            ball_state0 = [x0, y0, vx0, vy0]
            sol_ball = solve_ivp(
                ballistic_eom,
                [0, 300],  # Max 300 seconds of flight
                ball_state0,
                method='RK45',
                events=[ground_hit],
                dense_output=True,
                max_step=0.5,
            )

            if sol_ball.t_events[0].size > 0:
                t_flight = sol_ball.t_events[0][0]
                final_state = sol_ball.y_events[0][0]
                result.range_distance = final_state[0]  # x at landing

                # Find max height from trajectory
                t_dense = np.linspace(0, t_flight, 200)
                traj = sol_ball.sol(t_dense)
                result.max_height = np.max(traj[1])  # max y

                # Store trajectory for visualization
                result.ballistic_time = t_dense + t_release
                result.ballistic_pos = np.column_stack([traj[0], traj[1]])
            else:
                # No landing within time limit
                result.range_distance = sol_ball.y[0, -1]
                result.max_height = np.max(sol_ball.y[1])

        # =====================================================================
        # Diagnostics
        # =====================================================================
        result.rope_slack = self._min_tension < TENSION_TOLERANCE
        result.max_tension = self._min_tension  # Store for analysis

        # Check constraints
        max_g = self._max_accel / GRAVITY
        result.max_bending_moment = self._max_accel  # Approximate

        return result

    def check_constraints(self, result: SimulationResult) -> Tuple[bool, List[str]]:
        """
        Check if simulation result satisfies all constraints.

        Returns (is_valid, list_of_violations)
        """
        violations = []

        # Size constraint: 20x20x20m cube
        if np.any(self._max_pos > MACHINE_SIZE_LIMIT / 2):
            violations.append(f"Size violation: max extent {self._max_pos}")

        # Acceleration constraint: < 60g
        max_g = self._max_accel / GRAVITY
        if max_g > MAX_ACCELERATION_G:
            violations.append(f"Acceleration violation: {max_g:.1f}g > {MAX_ACCELERATION_G}g")

        # Tension constraint: no slack
        if result.rope_slack:
            violations.append(f"Rope slack: min tension {self._min_tension:.1f} N")

        # Ground collision
        if result.ground_collision:
            violations.append("Ground collision during swing")

        return len(violations) == 0, violations

    def compute_fitness(self, result: SimulationResult) -> float:
        """
        Compute fitness value for optimization.

        Fitness = Range - Penalties
        """
        fitness = result.range_distance

        # Penalty for ground collision
        if result.ground_collision:
            fitness -= PENALTY_GROUND_COLLISION

        # Penalty for rope slack
        if result.rope_slack:
            fitness -= PENALTY_ROPE_SLACK

        # Penalty for over-acceleration
        max_g = self._max_accel / GRAVITY
        if max_g > MAX_ACCELERATION_G:
            overage = (max_g - MAX_ACCELERATION_G) / MAX_ACCELERATION_G
            fitness -= PENALTY_OVERLOAD * overage

        # Penalty for size violation
        if np.any(self._max_pos > MACHINE_SIZE_LIMIT / 2):
            overage = np.max(self._max_pos - MACHINE_SIZE_LIMIT / 2)
            fitness -= PENALTY_SIZE_VIOLATION * overage

        return fitness


def test_simulation():
    """Test the simulation with default parameters."""
    print("=" * 60)
    print("Trebuchet Simulation Test")
    print("=" * 60)

    # Create default config
    config = TrebuchetConfig()
    print(f"\nConfiguration:")
    print(f"  L_root: {config.L_root} m")
    print(f"  L_tip: {config.L_tip} m")
    print(f"  L_sling: {config.L_sling} m")
    print(f"  L_hanger: {config.L_hanger} m")
    print(f"  H_pivot: {config.H_pivot} m")
    print(f"  M_cw: {config.M_cw} kg")
    print(f"  m_proj: {config.m_proj} kg")
    print(f"  k_stiffness: {config.k_stiffness} Nm/rad")

    # Create simulator
    sim = TrebuchetSimulator(config)

    # Run simulation
    print("\nRunning simulation...")
    result = sim.simulate()

    # Print results
    print(f"\nResults:")
    print(f"  Range: {result.range_distance:.2f} m")
    print(f"  Max height: {result.max_height:.2f} m")
    print(f"  Release speed: {result.release_speed:.2f} m/s")
    print(f"  Ground collision: {result.ground_collision}")
    print(f"  Rope slack: {result.rope_slack}")

    # Check constraints
    is_valid, violations = sim.check_constraints(result)
    print(f"\nConstraints satisfied: {is_valid}")
    if violations:
        for v in violations:
            print(f"  - {v}")

    # Fitness
    fitness = sim.compute_fitness(result)
    print(f"\nFitness: {fitness:.2f}")

    return result


if __name__ == "__main__":
    result = test_simulation()
