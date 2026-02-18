"""
Trebuchet-Level-4: Visualization and Animation
===============================================
Interactive GUI for trebuchet simulation with:
- Real-time animation of the mechanism
- Energy and angle plots
- Parameter sliders
- Optimization controls
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Circle, FancyArrow
import matplotlib.gridspec as gridspec
from typing import Optional
import warnings

from config import TrebuchetConfig, SimulationResult, GRAVITY, MAX_ACCELERATION_G
from trebuchet_model import TrebuchetSimulator


class TrebuchetVisualizer:
    """
    Interactive visualization for trebuchet simulation.
    """

    def __init__(self, config: TrebuchetConfig = None):
        """
        Initialize visualizer.

        Parameters
        ----------
        config : TrebuchetConfig, optional
            Initial configuration
        """
        self.config = config if config is not None else TrebuchetConfig()
        self.simulator = None
        self.result = None
        self.animation = None

        # Figure and axes
        self.fig = None
        self.ax_main = None
        self.ax_energy = None
        self.ax_angles = None
        self.ax_tension = None

        # Plot elements
        self._plot_elements = {}

    def run_simulation(self):
        """Run simulation with current config."""
        self.simulator = TrebuchetSimulator(self.config)
        self.result = self.simulator.simulate()
        return self.result

    def create_figure(self):
        """Create the main figure with all subplots."""
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Trebuchet-Level-4 Simulator', fontsize=14, fontweight='bold')

        # Create grid layout
        gs = gridspec.GridSpec(3, 3, figure=self.fig, height_ratios=[2, 1, 1],
                               hspace=0.3, wspace=0.3)

        # Main animation plot (spans 2 columns)
        self.ax_main = self.fig.add_subplot(gs[0, :2])
        self.ax_main.set_title('Trebuchet Animation')
        self.ax_main.set_xlabel('X [m]')
        self.ax_main.set_ylabel('Y [m]')
        self.ax_main.set_aspect('equal')
        self.ax_main.grid(True, alpha=0.3)

        # Info panel (right side)
        self.ax_info = self.fig.add_subplot(gs[0, 2])
        self.ax_info.axis('off')

        # Energy plot
        self.ax_energy = self.fig.add_subplot(gs[1, 0])
        self.ax_energy.set_title('Energy')
        self.ax_energy.set_xlabel('Time [s]')
        self.ax_energy.set_ylabel('Energy [J]')
        self.ax_energy.grid(True, alpha=0.3)

        # Angles plot
        self.ax_angles = self.fig.add_subplot(gs[1, 1])
        self.ax_angles.set_title('Angles')
        self.ax_angles.set_xlabel('Time [s]')
        self.ax_angles.set_ylabel('Angle [deg]')
        self.ax_angles.grid(True, alpha=0.3)

        # Tension/Moment plot
        self.ax_tension = self.fig.add_subplot(gs[1, 2])
        self.ax_tension.set_title('Rope Tension')
        self.ax_tension.set_xlabel('Time [s]')
        self.ax_tension.set_ylabel('Tension [N]')
        self.ax_tension.grid(True, alpha=0.3)

        # Slider area
        self.ax_sliders = self.fig.add_subplot(gs[2, :])
        self.ax_sliders.axis('off')

        return self.fig

    def _init_plot_elements(self):
        """Initialize plot elements for animation."""
        ax = self.ax_main

        # Fixed 20x20 view (machine fits in 20m cube)
        ax.set_xlim(-10, 10)
        ax.set_ylim(-1, 21)

        # Ground
        self._plot_elements['ground'] = ax.axhline(y=0, color='brown', linewidth=2)

        # Pivot point
        self._plot_elements['pivot'] = Circle((0, self.config.H_pivot), 0.15,
                                               color='black', zorder=10)
        ax.add_patch(self._plot_elements['pivot'])

        # Beam segments
        self._plot_elements['beam_root'], = ax.plot([], [], 'b-', linewidth=4, label='Root')
        self._plot_elements['beam_tip'], = ax.plot([], [], 'b-', linewidth=3, label='Tip')

        # Sling
        self._plot_elements['sling'], = ax.plot([], [], 'g-', linewidth=2, label='Sling')

        # Counterweight system
        self._plot_elements['cam_arm'], = ax.plot([], [], 'r-', linewidth=3, label='Cam')
        self._plot_elements['hanger'], = ax.plot([], [], 'orange', linewidth=2, label='Hanger')
        self._plot_elements['cw'] = Circle((0, 0), 0.3, color='red', zorder=5)
        ax.add_patch(self._plot_elements['cw'])

        # Projectile
        self._plot_elements['proj'] = Circle((0, 0), 0.15, color='orange', zorder=5)
        ax.add_patch(self._plot_elements['proj'])

        # Trajectory trail
        self._plot_elements['trail'], = ax.plot([], [], 'orange', linewidth=1, alpha=0.5)

        # Ballistic trajectory
        self._plot_elements['ballistic'], = ax.plot([], [], 'orange', linewidth=1,
                                                     linestyle='--', alpha=0.7)

        # Time text
        self._plot_elements['time_text'] = ax.text(0.02, 0.98, '', transform=ax.transAxes,
                                                    verticalalignment='top', fontsize=10)

        ax.legend(loc='upper right', fontsize=8)

    def _update_main_plot(self, frame_idx: int):
        """Update main animation plot for given frame."""
        if self.result is None or self.result.time is None:
            return

        n_frames = len(self.result.time)
        if frame_idx >= n_frames:
            frame_idx = n_frames - 1

        # Get positions
        p_pivot = np.array([0, self.config.H_pivot])
        p_joint = self.result.pos_joint[frame_idx]
        p_tip = self.result.pos_tip[frame_idx]
        p_proj = self.result.pos_proj[frame_idx]
        p_cam_pin = self.result.pos_cam_pin[frame_idx]
        p_cw = self.result.pos_cw[frame_idx]

        # Update beam
        self._plot_elements['beam_root'].set_data(
            [p_pivot[0], p_joint[0]], [p_pivot[1], p_joint[1]])
        self._plot_elements['beam_tip'].set_data(
            [p_joint[0], p_tip[0]], [p_joint[1], p_tip[1]])

        # Update sling
        self._plot_elements['sling'].set_data(
            [p_tip[0], p_proj[0]], [p_tip[1], p_proj[1]])

        # Update counterweight system
        self._plot_elements['cam_arm'].set_data(
            [p_pivot[0], p_cam_pin[0]], [p_pivot[1], p_cam_pin[1]])
        self._plot_elements['hanger'].set_data(
            [p_cam_pin[0], p_cw[0]], [p_cam_pin[1], p_cw[1]])

        # Update counterweight position
        self._plot_elements['cw'].center = (p_cw[0], p_cw[1])

        # Update projectile position
        self._plot_elements['proj'].center = (p_proj[0], p_proj[1])

        # Update trail
        self._plot_elements['trail'].set_data(
            self.result.pos_proj[:frame_idx+1, 0],
            self.result.pos_proj[:frame_idx+1, 1])

        # Update time text
        t = self.result.time[frame_idx]
        self._plot_elements['time_text'].set_text(f't = {t:.3f} s')

        # Fixed 20x20 axis limits (centered on pivot, showing full machine envelope)
        # No dynamic rescaling - keeps consistent view throughout animation

    def _update_info_panel(self):
        """Update info panel with results."""
        self.ax_info.clear()
        self.ax_info.axis('off')

        if self.result is None:
            return

        lines = [
            "CONFIGURATION",
            f"L_root: {self.config.L_root:.2f} m",
            f"L_tip: {self.config.L_tip:.2f} m",
            f"L_sling: {self.config.L_sling:.2f} m",
            f"L_hanger: {self.config.L_hanger:.2f} m",
            f"H_pivot: {self.config.H_pivot:.2f} m",
            f"M_cw: {self.config.M_cw:.0f} kg",
            f"m_proj: {self.config.m_proj:.1f} kg",
            f"k_stiff: {self.config.k_stiffness:.0f} Nm/rad",
            "",
            "RESULTS",
            f"Range: {self.result.range_distance:.1f} m",
            f"Max height: {self.result.max_height:.1f} m",
            f"Release speed: {self.result.release_speed:.1f} m/s",
            f"Release time: {self.result.release_time:.3f} s" if self.result.release_time else "N/A",
            "",
            "CONSTRAINTS",
            f"Ground collision: {'YES' if self.result.ground_collision else 'No'}",
            f"Rope slack: {'YES' if self.result.rope_slack else 'No'}",
        ]

        text = '\n'.join(lines)
        self.ax_info.text(0.1, 0.95, text, transform=self.ax_info.transAxes,
                          verticalalignment='top', fontsize=9, family='monospace')

    def _plot_energy(self):
        """Plot energy vs time."""
        self.ax_energy.clear()
        self.ax_energy.set_title('Energy')
        self.ax_energy.set_xlabel('Time [s]')
        self.ax_energy.set_ylabel('Energy [kJ]')
        self.ax_energy.grid(True, alpha=0.3)

        if self.result is None or self.result.time is None:
            return

        t = self.result.time
        cfg = self.config
        g = cfg.g

        # Compute energies at each time step
        KE = []
        PE = []

        for i in range(len(t)):
            # Kinetic energy (approximate from velocities)
            dth = self.result.d_theta[i]
            dbt = self.result.d_beta[i]
            dgm = self.result.d_gamma[i]
            dph = self.result.d_phi[i]

            # Simplified KE calculation
            I_root, I_tip = cfg.get_beam_inertias()
            ke = 0.5 * I_root * dth**2
            ke += 0.5 * I_tip * (dth + dbt)**2
            ke += 0.5 * cfg.M_cw * (cfg.L_hanger * dgm)**2
            ke += 0.5 * cfg.m_proj * (cfg.L_sling * dph)**2
            KE.append(ke)

            # Potential energy
            y_cw = self.result.pos_cw[i, 1]
            y_proj = self.result.pos_proj[i, 1]
            pe = cfg.M_cw * g * y_cw + cfg.m_proj * g * y_proj
            PE.append(pe)

        KE = np.array(KE) / 1000  # Convert to kJ
        PE = np.array(PE) / 1000
        Total = KE + PE

        self.ax_energy.plot(t, KE, 'r-', label='Kinetic')
        self.ax_energy.plot(t, PE, 'b-', label='Potential')
        self.ax_energy.plot(t, Total, 'k--', label='Total')
        self.ax_energy.legend(fontsize=8)

    def _plot_angles(self):
        """Plot angles vs time."""
        self.ax_angles.clear()
        self.ax_angles.set_title('Angles')
        self.ax_angles.set_xlabel('Time [s]')
        self.ax_angles.set_ylabel('Angle [deg]')
        self.ax_angles.grid(True, alpha=0.3)

        if self.result is None or self.result.time is None:
            return

        t = self.result.time

        self.ax_angles.plot(t, np.rad2deg(self.result.theta), 'b-', label='θ (beam)')
        self.ax_angles.plot(t, np.rad2deg(self.result.beta), 'r-', label='β (flex)')
        self.ax_angles.plot(t, np.rad2deg(self.result.gamma), 'g-', label='γ (CW)')
        self.ax_angles.plot(t, np.rad2deg(self.result.phi), 'orange', label='φ (sling)')
        self.ax_angles.legend(fontsize=8)

    def _plot_tension(self):
        """Plot estimated tension vs time."""
        self.ax_tension.clear()
        self.ax_tension.set_title('Counterweight Rope Tension')
        self.ax_tension.set_xlabel('Time [s]')
        self.ax_tension.set_ylabel('Tension [kN]')
        self.ax_tension.grid(True, alpha=0.3)

        if self.result is None or self.result.time is None:
            return

        t = self.result.time
        cfg = self.config

        # Estimate tension from gamma dynamics
        tensions = []
        for i in range(len(t)):
            gm = self.result.gamma[i]
            dgm = self.result.d_gamma[i]

            # T = M_cw * (g * cos(gamma) + L_hanger * dgm^2)
            T = cfg.M_cw * (cfg.g * np.cos(gm) + cfg.L_hanger * dgm**2)
            tensions.append(T)

        tensions = np.array(tensions) / 1000  # Convert to kN

        self.ax_tension.plot(t, tensions, 'r-')
        self.ax_tension.axhline(y=0, color='k', linestyle='--', alpha=0.5)

        # Mark slack condition
        if np.any(tensions < 0):
            self.ax_tension.fill_between(t, 0, tensions, where=(tensions < 0),
                                          color='red', alpha=0.3, label='SLACK')
            self.ax_tension.legend(fontsize=8)

    def animate(self, interval: int = 50, repeat: bool = True):
        """
        Create and show animation.

        Parameters
        ----------
        interval : int
            Frame interval in milliseconds
        repeat : bool
            Whether to loop animation
        """
        if self.result is None:
            self.run_simulation()

        if self.fig is None:
            self.create_figure()

        self._init_plot_elements()

        # Plot static elements
        self._update_info_panel()
        self._plot_energy()
        self._plot_angles()
        self._plot_tension()

        # Plot ballistic trajectory
        if self.result.ballistic_pos is not None:
            self._plot_elements['ballistic'].set_data(
                self.result.ballistic_pos[:, 0],
                self.result.ballistic_pos[:, 1])

        n_frames = len(self.result.time) if self.result.time is not None else 1

        def init():
            return list(self._plot_elements.values())

        def update(frame):
            self._update_main_plot(frame)
            return list(self._plot_elements.values())

        self.animation = FuncAnimation(
            self.fig, update, frames=n_frames,
            init_func=init, interval=interval,
            blit=False, repeat=repeat
        )

        plt.tight_layout()
        plt.show()

    def plot_static(self):
        """Show static plots (no animation)."""
        if self.result is None:
            self.run_simulation()

        if self.fig is None:
            self.create_figure()

        self._init_plot_elements()

        # Plot final state
        if self.result.time is not None:
            final_frame = len(self.result.time) - 1
            self._update_main_plot(final_frame)

        # Plot static elements
        self._update_info_panel()
        self._plot_energy()
        self._plot_angles()
        self._plot_tension()

        # Plot ballistic trajectory
        if self.result.ballistic_pos is not None:
            self._plot_elements['ballistic'].set_data(
                self.result.ballistic_pos[:, 0],
                self.result.ballistic_pos[:, 1])

        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        """Plot just the projectile trajectory."""
        if self.result is None:
            self.run_simulation()

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Projectile Trajectory')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Ground
        ax.axhline(y=0, color='brown', linewidth=2)

        # Swing phase trajectory
        if self.result.pos_proj is not None:
            ax.plot(self.result.pos_proj[:, 0], self.result.pos_proj[:, 1],
                    'b-', linewidth=2, label='Swing phase')

        # Ballistic trajectory
        if self.result.ballistic_pos is not None:
            ax.plot(self.result.ballistic_pos[:, 0], self.result.ballistic_pos[:, 1],
                    'orange', linewidth=2, linestyle='--', label='Ballistic')

        # Release point
        if self.result.release_pos is not None:
            ax.plot(self.result.release_pos[0], self.result.release_pos[1],
                    'go', markersize=10, label='Release')

        # Landing point
        if self.result.range_distance > 0:
            ax.plot(self.result.range_distance, 0, 'rx', markersize=15,
                    markeredgewidth=3, label=f'Landing: {self.result.range_distance:.1f} m')

        ax.legend()
        plt.tight_layout()
        plt.show()


def get_optimal_config():
    """Return the best configuration found by optimization (tungsten, 300g limit, all constraints).

    Result: 2840m range with all constraints satisfied (6-hour DE optimization).
    """
    return TrebuchetConfig(
        L_root=5.047241802325359,
        L_tip=6.905045783887995,
        L_hanger=4.613214539369623,
        L_sling=9.479586613395131,
        H_pivot=5.219814758249189,
        M_cw=9995.004185098072,
        k_stiffness=33344.30801722248,
        cam_coeffs=[0.2975426902179485, -0.18514632192569783, 0.17366685335474752, 0.0],
        release_angle=70.11403663342277
    )


def interactive_demo(use_optimal=True):
    """Run interactive demo."""
    if use_optimal:
        print("Using OPTIMAL configuration (tungsten, 300g)...")
        config = get_optimal_config()
    else:
        print("Using DEFAULT configuration...")
        config = TrebuchetConfig()

    viz = TrebuchetVisualizer(config)

    print("Running simulation...")
    result = viz.run_simulation()

    print(f"\nResults:")
    print(f"  Range: {result.range_distance:.1f} m")
    print(f"  Release speed: {result.release_speed:.1f} m/s")
    print(f"  Ground collision: {result.ground_collision}")
    print(f"  Rope slack: {result.rope_slack}")

    print("\nStarting animation...")
    viz.animate(interval=30)


def compare_configs():
    """Compare different configurations."""
    configs = [
        ("Default", TrebuchetConfig()),
        ("Heavy CW", TrebuchetConfig(M_cw=8000)),
        ("Long arm", TrebuchetConfig(L_tip=10.0)),
        ("Stiff beam", TrebuchetConfig(k_stiffness=50000)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, (name, config) in zip(axes, configs):
        sim = TrebuchetSimulator(config)
        result = sim.simulate()

        ax.set_title(f'{name}\nRange: {result.range_distance:.1f} m')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)

        # Ground
        ax.axhline(y=0, color='brown', linewidth=2)

        # Trajectory
        if result.pos_proj is not None:
            ax.plot(result.pos_proj[:, 0], result.pos_proj[:, 1], 'b-', linewidth=1)

        if result.ballistic_pos is not None:
            ax.plot(result.ballistic_pos[:, 0], result.ballistic_pos[:, 1],
                    'orange', linewidth=1, linestyle='--')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--compare':
        compare_configs()
    elif len(sys.argv) > 1 and sys.argv[1] == '--trajectory':
        viz = TrebuchetVisualizer(get_optimal_config())
        viz.run_simulation()
        viz.plot_trajectory()
    elif len(sys.argv) > 1 and sys.argv[1] == '--static':
        viz = TrebuchetVisualizer(get_optimal_config())
        viz.run_simulation()
        viz.plot_static()
    elif len(sys.argv) > 1 and sys.argv[1] == '--default':
        interactive_demo(use_optimal=False)
    else:
        interactive_demo(use_optimal=True)
