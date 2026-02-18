"""
Trebuchet-Level-4: Genetic Algorithm Optimizer
===============================================
Multi-parameter optimization to maximize throwing range
while satisfying constraints.

Constraints:
- Machine fits in 20x20x20m cube
- Counterweight <= 10 tons
- Projectile acceleration < 60g
- Rope tension > 0 (no slack)
"""

import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
from typing import Tuple, List, Callable, Optional
import time
import warnings
from dataclasses import dataclass

from config import (
    TrebuchetConfig, OptimizationBounds, SimulationResult,
    PENALTY_GROUND_COLLISION, PENALTY_ROPE_SLACK, PENALTY_OVERLOAD,
    PENALTY_SIZE_VIOLATION, MAX_ACCELERATION_G, MACHINE_SIZE_LIMIT,
    MAX_COUNTERWEIGHT, GRAVITY
)
from trebuchet_model import TrebuchetSimulator


@dataclass
class OptimizationResult:
    """Results from optimization run."""
    best_params: np.ndarray
    best_fitness: float
    best_range: float
    best_config: TrebuchetConfig
    n_iterations: int
    n_evaluations: int
    convergence_history: List[float]
    constraint_violations: List[str]
    optimization_time: float


class TrebuchetOptimizer:
    """
    Genetic algorithm optimizer for trebuchet parameters.

    Uses scipy.optimize.differential_evolution for global optimization.
    """

    def __init__(
        self,
        bounds: OptimizationBounds = None,
        base_config: TrebuchetConfig = None,
        verbose: bool = True
    ):
        """
        Initialize optimizer.

        Parameters
        ----------
        bounds : OptimizationBounds, optional
            Parameter bounds for optimization
        base_config : TrebuchetConfig, optional
            Base configuration (non-optimized parameters)
        verbose : bool
            Print progress during optimization
        """
        self.bounds = bounds if bounds is not None else OptimizationBounds()
        self.base_config = base_config if base_config is not None else TrebuchetConfig()
        self.verbose = verbose

        # Tracking
        self.n_evaluations = 0
        self.convergence_history = []
        self.best_fitness = float('-inf')
        self.best_params = None

    def _params_to_config(self, params: np.ndarray) -> TrebuchetConfig:
        """Convert optimizer parameters to TrebuchetConfig."""
        return self.bounds.params_to_config(params, self.base_config)

    def _check_geometry_constraints(self, params: np.ndarray) -> Tuple[bool, float]:
        """
        Quick geometry check before running full simulation.

        Returns (is_valid, penalty)
        """
        L_root = params[0]
        L_tip = params[1]
        L_hanger = params[2]
        L_sling = params[3]
        H_pivot = params[4]
        M_cw = params[5]

        penalty = 0.0

        # CRITICAL: Beam must not hit the ground!
        # L_root + L_tip is the total beam length (both segments of the throwing arm)
        # When beam rotates down, it must not go below ground level
        beam_length = L_root + L_tip
        ground_clearance = 0.5  # 0.5m safety margin
        if beam_length > (H_pivot - ground_clearance):
            penalty += 5000 * (beam_length - (H_pivot - ground_clearance))

        # Height check: pivot + arm length + sling should fit in 20m cube
        max_height = H_pivot + L_tip + L_sling
        if max_height > MACHINE_SIZE_LIMIT:
            penalty += 1000 * (max_height - MACHINE_SIZE_LIMIT)

        # Counterweight mass check
        if M_cw > MAX_COUNTERWEIGHT:
            penalty += 1000 * (M_cw - MAX_COUNTERWEIGHT) / MAX_COUNTERWEIGHT

        return penalty == 0, penalty

    def _fitness_function(self, params: np.ndarray) -> float:
        """
        Fitness function for optimization.

        Returns negative fitness (for minimization).
        """
        self.n_evaluations += 1

        # Quick geometry check
        is_valid, geom_penalty = self._check_geometry_constraints(params)
        if geom_penalty > 0:
            return geom_penalty  # Return positive value (bad)

        try:
            # Create config and simulator
            config = self._params_to_config(params)
            simulator = TrebuchetSimulator(config)

            # Run simulation
            result = simulator.simulate()

            # Compute fitness
            fitness = simulator.compute_fitness(result)

            # Track best
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_params = params.copy()

                if self.verbose and self.n_evaluations % 10 == 0:
                    print(f"  Eval {self.n_evaluations}: New best fitness = {fitness:.2f} m")

        except Exception as e:
            if self.verbose:
                warnings.warn(f"Simulation failed: {e}")
            fitness = -100000  # Heavy penalty for failures

        # Return negative for minimization
        return -fitness

    def _callback(self, xk: np.ndarray, convergence: float = None):
        """Callback for optimization progress."""
        self.convergence_history.append(self.best_fitness)

        if self.verbose:
            print(f"  Generation: best = {self.best_fitness:.2f} m, "
                  f"evals = {self.n_evaluations}")

    def optimize(
        self,
        maxiter: int = 100,
        popsize: int = 20,
        mutation: Tuple[float, float] = (0.5, 1.0),
        recombination: float = 0.7,
        workers: int = 1,  # -1 for parallel
        seed: int = None,
        polish: bool = True
    ) -> OptimizationResult:
        """
        Run optimization.

        Parameters
        ----------
        maxiter : int
            Maximum number of generations
        popsize : int
            Population size multiplier (actual size = popsize * n_params)
        mutation : tuple
            Mutation constant range (dithering)
        recombination : float
            Recombination constant
        workers : int
            Number of parallel workers (-1 for all CPUs)
        seed : int, optional
            Random seed for reproducibility
        polish : bool
            Whether to polish best result with L-BFGS-B

        Returns
        -------
        OptimizationResult
        """
        # Reset tracking
        self.n_evaluations = 0
        self.convergence_history = []
        self.best_fitness = float('-inf')
        self.best_params = None

        bounds_list = self.bounds.get_bounds_list()

        if self.verbose:
            print("=" * 60)
            print("Trebuchet Optimization")
            print("=" * 60)
            print(f"Parameters: {len(bounds_list)}")
            print(f"Population: {popsize * len(bounds_list)}")
            print(f"Max iterations: {maxiter}")
            print(f"Workers: {workers}")
            print()

        start_time = time.time()

        # Run differential evolution
        result = differential_evolution(
            self._fitness_function,
            bounds_list,
            maxiter=maxiter,
            popsize=popsize,
            mutation=mutation,
            recombination=recombination,
            workers=workers,
            seed=seed,
            polish=polish,
            callback=self._callback,
            disp=self.verbose,
            updating='deferred' if workers != 1 else 'immediate',
        )

        elapsed = time.time() - start_time

        # Get best config
        best_params = self.best_params if self.best_params is not None else result.x
        best_config = self._params_to_config(best_params)

        # Run final simulation to get detailed results
        simulator = TrebuchetSimulator(best_config)
        final_result = simulator.simulate()
        _, violations = simulator.check_constraints(final_result)

        if self.verbose:
            print("\n" + "=" * 60)
            print("Optimization Complete")
            print("=" * 60)
            print(f"Time: {elapsed:.1f} s")
            print(f"Evaluations: {self.n_evaluations}")
            print(f"Best fitness: {self.best_fitness:.2f} m")
            print(f"Best range: {final_result.range_distance:.2f} m")
            print(f"\nOptimal parameters:")
            print(f"  L_root: {best_config.L_root:.3f} m")
            print(f"  L_tip: {best_config.L_tip:.3f} m")
            print(f"  L_hanger: {best_config.L_hanger:.3f} m")
            print(f"  L_sling: {best_config.L_sling:.3f} m")
            print(f"  H_pivot: {best_config.H_pivot:.3f} m")
            print(f"  M_cw: {best_config.M_cw:.1f} kg")
            print(f"  k_stiffness: {best_config.k_stiffness:.1f} Nm/rad")
            print(f"  cam_coeffs: {best_config.cam_coeffs}")
            print(f"  release_angle: {best_config.release_angle:.1f} deg")

            if violations:
                print(f"\nConstraint violations:")
                for v in violations:
                    print(f"  - {v}")

        return OptimizationResult(
            best_params=best_params,
            best_fitness=self.best_fitness,
            best_range=final_result.range_distance,
            best_config=best_config,
            n_iterations=result.nit,
            n_evaluations=self.n_evaluations,
            convergence_history=self.convergence_history,
            constraint_violations=violations,
            optimization_time=elapsed,
        )


def quick_test():
    """Quick test with few iterations."""
    print("Quick optimization test (5 iterations)...")

    optimizer = TrebuchetOptimizer(verbose=True)
    result = optimizer.optimize(maxiter=5, popsize=5, workers=1)

    return result


def full_optimization():
    """Full optimization run."""
    print("Full optimization run...")

    optimizer = TrebuchetOptimizer(verbose=True)
    result = optimizer.optimize(
        maxiter=100,
        popsize=20,
        workers=-1,  # Use all CPUs
        seed=42,
    )

    return result


def parameter_study():
    """
    Study individual parameter effects.

    Varies each parameter while keeping others at default.
    """
    print("=" * 60)
    print("Parameter Study")
    print("=" * 60)

    base_config = TrebuchetConfig()
    simulator = TrebuchetSimulator(base_config)
    base_result = simulator.simulate()
    base_range = base_result.range_distance

    print(f"Base range: {base_range:.2f} m\n")

    # Parameters to study
    studies = [
        ('L_root', np.linspace(1.0, 5.0, 10)),
        ('L_tip', np.linspace(3.0, 12.0, 10)),
        ('L_sling', np.linspace(2.0, 10.0, 10)),
        ('L_hanger', np.linspace(0.5, 5.0, 10)),
        ('M_cw', np.linspace(1000, 10000, 10)),
        ('k_stiffness', np.linspace(5000, 200000, 10)),
        ('release_angle', np.linspace(20, 70, 10)),
    ]

    results = {}

    for param_name, values in studies:
        print(f"\nStudying {param_name}...")
        ranges = []

        for val in values:
            config = TrebuchetConfig()
            setattr(config, param_name, val)

            try:
                sim = TrebuchetSimulator(config)
                result = sim.simulate()
                ranges.append(result.range_distance)
            except Exception:
                ranges.append(0.0)

        results[param_name] = (values, ranges)

        # Find best
        best_idx = np.argmax(ranges)
        print(f"  Best {param_name} = {values[best_idx]:.2f}, range = {ranges[best_idx]:.2f} m")

    return results


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == '--study':
        parameter_study()
    else:
        full_optimization()
