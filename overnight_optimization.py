"""
Overnight Optimization - Maximum Reliability
============================================
Runs for up to 6 hours to find the best possible trebuchet configuration.

Uses:
1. Differential Evolution with large population (robust global search)
2. Multiple random restarts
3. Final polishing with L-BFGS-B
4. Saves checkpoints every 10 iterations
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
from config import TrebuchetConfig, MACHINE_SIZE_LIMIT
from trebuchet_model import TrebuchetSimulator
import time
import json
from datetime import datetime
import os
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Output file for results
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "optimization_results.json")
CHECKPOINT_FILE = os.path.join(os.path.dirname(__file__), "optimization_checkpoint.json")

# Best known starting point
BEST_KNOWN = np.array([2.9656, 4.4926, 3.8398, 5.5292, 9.3779, 9929.1, 20076.8, 0.4072, 0.0244, 0.1315, 49.02])

# Optimization bounds - expanded for exploration
BOUNDS = [
    (1.5, 6.0),      # L_root - wider range
    (3.0, 10.0),     # L_tip - longer tip possible
    (0.5, 5.0),      # L_hanger
    (2.0, 10.0),     # L_sling - longer sling for more whip
    (5.0, 9.5),      # H_pivot
    (3000, 10000),   # M_cw - full range
    (5000, 200000),  # k_stiffness - wider range
    (0.1, 2.0),      # cam_C0
    (-0.5, 0.5),     # cam_C1
    (-0.3, 0.3),     # cam_C2
    (20.0, 80.0),    # release_angle - wider range
]

# Global tracking
best_ever = {"range": 0, "params": None, "time": 0}
eval_count = 0
start_time = None


def objective(params):
    """Objective function with constraint handling."""
    global eval_count, best_ever
    eval_count += 1

    try:
        config = TrebuchetConfig(
            L_root=params[0],
            L_tip=params[1],
            L_hanger=params[2],
            L_sling=params[3],
            H_pivot=params[4],
            M_cw=params[5],
            k_stiffness=params[6],
            cam_coeffs=[params[7], params[8], params[9], 0.0],
            release_angle=params[10]
        )

        sim = TrebuchetSimulator(config)
        result = sim.simulate()

        # Hard constraints - return large penalty
        if result.ground_collision:
            return 50000 + np.random.random()  # Add noise to help DE escape
        if result.rope_slack:
            return 40000 + np.random.random()

        # Size constraint
        if result.pos_proj is not None:
            max_h = np.max(result.pos_proj[:, 1])
            max_x = np.max(np.abs(result.pos_proj[:, 0]))
            if max_h > MACHINE_SIZE_LIMIT:
                return 30000 + (max_h - MACHINE_SIZE_LIMIT) * 100
            if max_x > MACHINE_SIZE_LIMIT / 2:
                return 30000 + (max_x - MACHINE_SIZE_LIMIT/2) * 100

        range_dist = result.range_distance

        # Track best ever
        if range_dist > best_ever["range"]:
            best_ever["range"] = range_dist
            best_ever["params"] = params.tolist()
            best_ever["time"] = time.time() - start_time

        return -range_dist

    except Exception as e:
        return 100000 + np.random.random()


def callback(xk, convergence=None):
    """Callback for progress reporting and checkpointing."""
    global best_ever, start_time

    elapsed = time.time() - start_time
    hours = elapsed / 3600

    # Print progress
    print(f"  [{hours:.2f}h] Best so far: {best_ever['range']:.1f} m, Evals: {eval_count}")

    # Save checkpoint
    checkpoint = {
        "best_range": best_ever["range"],
        "best_params": best_ever["params"],
        "elapsed_hours": hours,
        "eval_count": eval_count,
        "timestamp": datetime.now().isoformat()
    }
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

    # Stop after 6 hours
    if hours >= 6:
        print("Time limit reached (6 hours)")
        return True

    return False


def run_differential_evolution():
    """Run DE with optimal parameters for reliability."""
    global start_time, eval_count, best_ever

    print("=" * 70)
    print("PHASE 1: Differential Evolution (Global Search)")
    print("=" * 70)
    print(f"Population: 30 * 11 = 330 individuals")
    print(f"Max iterations: 500")
    print(f"Strategy: best1bin (exploitation) + rand1bin (exploration)")
    print()

    start_time = time.time()
    eval_count = 0

    # Run DE with large population
    result = differential_evolution(
        objective,
        BOUNDS,
        maxiter=500,
        popsize=30,           # Large population for diversity
        mutation=(0.5, 1.0),  # Adaptive mutation
        recombination=0.7,    # Good mixing
        strategy='best1bin',  # Exploit best solutions
        tol=1e-8,             # Very tight tolerance
        atol=0,
        polish=False,         # We'll polish manually
        disp=True,
        callback=callback,
        workers=1,            # Single worker for stability
        updating='immediate', # Update population immediately
        seed=42               # Reproducibility
    )

    print(f"\nDE finished: {-result.fun:.1f} m")
    return result.x, -result.fun


def run_basin_hopping(x0, n_iter=50):
    """Run Basin Hopping from a good starting point."""
    global start_time, eval_count

    print()
    print("=" * 70)
    print("PHASE 2: Basin Hopping (Escape Local Minima)")
    print("=" * 70)
    print(f"Starting from: {best_ever['range']:.1f} m")
    print(f"Iterations: {n_iter}")
    print()

    from scipy.optimize import basinhopping

    class BoundsEnforcer:
        def __init__(self, bounds):
            self.lb = np.array([b[0] for b in bounds])
            self.ub = np.array([b[1] for b in bounds])
        def __call__(self, **kwargs):
            x = kwargs["x_new"]
            return np.all(x >= self.lb) and np.all(x <= self.ub)

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': BOUNDS,
        'options': {'maxiter': 50}
    }

    result = basinhopping(
        objective,
        x0,
        minimizer_kwargs=minimizer_kwargs,
        niter=n_iter,
        stepsize=0.3,
        accept_test=BoundsEnforcer(BOUNDS),
        disp=True
    )

    print(f"\nBasin Hopping finished: {-result.fun:.1f} m")
    return result.x, -result.fun


def run_multistart_lbfgsb(n_starts=20):
    """Run L-BFGS-B from multiple random starting points."""
    global best_ever

    print()
    print("=" * 70)
    print("PHASE 3: Multi-Start L-BFGS-B (Local Refinement)")
    print("=" * 70)
    print(f"Random starts: {n_starts}")
    print()

    best_local = {"range": best_ever["range"], "params": best_ever["params"]}

    for i in range(n_starts):
        # Generate random starting point, biased toward best known
        if i == 0:
            x0 = np.array(best_ever["params"])
        elif i < 5:
            # Small perturbation of best
            x0 = np.array(best_ever["params"]) * (1 + 0.1 * np.random.randn(11))
        else:
            # Random point
            x0 = np.array([np.random.uniform(b[0], b[1]) for b in BOUNDS])

        # Clip to bounds
        x0 = np.clip(x0, [b[0] for b in BOUNDS], [b[1] for b in BOUNDS])

        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=BOUNDS,
                            options={'maxiter': 100})
            range_val = -result.fun

            if range_val > best_local["range"]:
                best_local["range"] = range_val
                best_local["params"] = result.x.tolist()
                print(f"  Start {i+1}/{n_starts}: {range_val:.1f} m (NEW BEST!)")
            else:
                print(f"  Start {i+1}/{n_starts}: {range_val:.1f} m")
        except:
            print(f"  Start {i+1}/{n_starts}: FAILED")

    return np.array(best_local["params"]), best_local["range"]


def final_polish(x0):
    """Final high-precision polish."""
    print()
    print("=" * 70)
    print("PHASE 4: Final Polish (High Precision)")
    print("=" * 70)

    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=BOUNDS,
        options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
    )

    print(f"Final result: {-result.fun:.1f} m")
    return result.x, -result.fun


def verify_result(params):
    """Verify and display final result."""
    print()
    print("=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    config = TrebuchetConfig(
        L_root=params[0],
        L_tip=params[1],
        L_hanger=params[2],
        L_sling=params[3],
        H_pivot=params[4],
        M_cw=params[5],
        k_stiffness=params[6],
        cam_coeffs=[params[7], params[8], params[9], 0.0],
        release_angle=params[10]
    )

    sim = TrebuchetSimulator(config)
    result = sim.simulate()

    print(f"Range: {result.range_distance:.2f} m")
    print(f"Release speed: {result.release_speed:.2f} m/s")
    print(f"Ground collision: {result.ground_collision}")
    print(f"Rope slack: {result.rope_slack}")

    if result.pos_proj is not None:
        print(f"Max height: {np.max(result.pos_proj[:, 1]):.2f} m")
        print(f"Max horizontal: {np.max(np.abs(result.pos_proj[:, 0])):.2f} m")

    return result


def main():
    global start_time, best_ever

    print("=" * 70)
    print("OVERNIGHT OPTIMIZATION - MAXIMUM RELIABILITY")
    print("=" * 70)
    print(f"Started: {datetime.now()}")
    print(f"Time limit: 6 hours")
    print(f"Initial best: {objective(BEST_KNOWN):.1f} m (negated)")
    print()

    start_time = time.time()
    best_ever = {"range": -objective(BEST_KNOWN), "params": BEST_KNOWN.tolist(), "time": 0}

    # Phase 1: Differential Evolution
    x1, r1 = run_differential_evolution()

    # Check time
    elapsed = (time.time() - start_time) / 3600
    if elapsed < 5:
        # Phase 2: Basin Hopping
        x2, r2 = run_basin_hopping(x1 if r1 > best_ever["range"] else np.array(best_ever["params"]), n_iter=100)

    # Check time
    elapsed = (time.time() - start_time) / 3600
    if elapsed < 5.5:
        # Phase 3: Multi-start L-BFGS-B
        x3, r3 = run_multistart_lbfgsb(n_starts=30)

    # Phase 4: Final polish
    best_params = np.array(best_ever["params"])
    x_final, r_final = final_polish(best_params)

    # Verify
    result = verify_result(x_final)

    # Save final results
    total_time = time.time() - start_time
    final_results = {
        "range_m": result.range_distance,
        "release_speed_ms": result.release_speed,
        "parameters": {
            "L_root": x_final[0],
            "L_tip": x_final[1],
            "L_hanger": x_final[2],
            "L_sling": x_final[3],
            "H_pivot": x_final[4],
            "M_cw": x_final[5],
            "k_stiffness": x_final[6],
            "cam_coeffs": [x_final[7], x_final[8], x_final[9], 0.0],
            "release_angle": x_final[10]
        },
        "constraints": {
            "ground_collision": result.ground_collision,
            "rope_slack": result.rope_slack
        },
        "optimization": {
            "total_time_hours": total_time / 3600,
            "total_evaluations": eval_count,
            "finished": datetime.now().isoformat()
        }
    }

    with open(RESULTS_FILE, 'w') as f:
        json.dump(final_results, f, indent=2)

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    print(f"Total time: {total_time/3600:.2f} hours")
    print(f"Total evaluations: {eval_count}")
    print(f"Best range: {result.range_distance:.2f} m")
    print(f"Results saved to: {RESULTS_FILE}")
    print()
    print("OPTIMAL PARAMETERS:")
    print(f"  L_root = {x_final[0]:.4f}")
    print(f"  L_tip = {x_final[1]:.4f}")
    print(f"  L_hanger = {x_final[2]:.4f}")
    print(f"  L_sling = {x_final[3]:.4f}")
    print(f"  H_pivot = {x_final[4]:.4f}")
    print(f"  M_cw = {x_final[5]:.1f}")
    print(f"  k_stiffness = {x_final[6]:.1f}")
    print(f"  cam_coeffs = [{x_final[7]:.4f}, {x_final[8]:.4f}, {x_final[9]:.4f}, 0.0]")
    print(f"  release_angle = {x_final[10]:.2f}")


if __name__ == "__main__":
    main()
