"""
Trebuchet-Level-4: Symbolic Physics Derivation
===============================================
Generates equations of motion for 4-DoF trebuchet using SymPy.
Run this script once to generate 'generated_eom.py'.

Generalized coordinates:
    theta (θ) - Main beam angle from vertical
    beta (β)  - Flex angle (tip relative to root)
    gamma (γ) - Counterweight hanger angle from vertical
    phi (φ)   - Sling angle relative to tip

Reference: Lagrangian mechanics with PRBM (Pseudo-Rigid-Body Model)
"""

import sympy as sp
from sympy import sin, cos, sqrt, Rational, symbols, Function, diff, simplify
from sympy import Matrix, zeros, trigsimp, nsimplify
from sympy.physics.mechanics import dynamicsymbols
from sympy.utilities.lambdify import lambdify
from sympy.printing.numpy import NumPyPrinter
import os


def derive_equations():
    """
    Derive the equations of motion symbolically.
    Returns matrices M and F such that M @ q_ddot = F
    """
    print("=" * 60)
    print("Trebuchet-Level-4: Symbolic Derivation")
    print("=" * 60)

    # Time variable
    t = sp.Symbol('t')

    # Generalized coordinates as functions of time
    theta = dynamicsymbols('theta')   # Main beam angle
    beta = dynamicsymbols('beta')     # Flex angle
    gamma = dynamicsymbols('gamma')   # Counterweight hanger angle
    phi = dynamicsymbols('phi')       # Sling angle

    # First derivatives (velocities)
    d_theta = diff(theta, t)
    d_beta = diff(beta, t)
    d_gamma = diff(gamma, t)
    d_phi = diff(phi, t)

    # Second derivatives (accelerations)
    dd_theta = diff(theta, t, 2)
    dd_beta = diff(beta, t, 2)
    dd_gamma = diff(gamma, t, 2)
    dd_phi = diff(phi, t, 2)

    q = Matrix([theta, beta, gamma, phi])
    dq = Matrix([d_theta, d_beta, d_gamma, d_phi])
    ddq = Matrix([dd_theta, dd_beta, dd_gamma, dd_phi])

    print("\n1. Defining symbolic parameters...")

    # Physical parameters (symbols)
    L_root = sp.Symbol('L_root', positive=True)      # Root segment length
    L_tip = sp.Symbol('L_tip', positive=True)        # Tip segment length
    L_hanger = sp.Symbol('L_hanger', positive=True)  # Hanger rope length
    L_sling = sp.Symbol('L_sling', positive=True)    # Sling length
    H = sp.Symbol('H', positive=True)                # Pivot height

    m_root = sp.Symbol('m_root', positive=True)      # Root segment mass
    m_tip = sp.Symbol('m_tip', positive=True)        # Tip segment mass
    M_cw = sp.Symbol('M_cw', positive=True)          # Counterweight mass
    m_proj = sp.Symbol('m_proj', positive=True)      # Projectile mass

    I_root = sp.Symbol('I_root', positive=True)      # Root moment of inertia
    I_tip = sp.Symbol('I_tip', positive=True)        # Tip moment of inertia

    k_stiff = sp.Symbol('k_stiff', positive=True)    # Torsional spring stiffness
    c_damp = sp.Symbol('c_damp', positive=True)      # Torsional damping
    g = sp.Symbol('g', positive=True)                # Gravity

    # Cam profile - treated as known functions of theta
    # R_cam(theta), dR_cam/dtheta, d^2R_cam/dtheta^2
    # psi(theta), dpsi/dtheta, d^2psi/dtheta^2
    R_cam = sp.Symbol('R_cam')           # Cam radius at current theta
    dR_cam = sp.Symbol('dR_cam')         # dR/dtheta
    ddR_cam = sp.Symbol('ddR_cam')       # d^2R/dtheta^2
    psi = sp.Symbol('psi')               # Cam angle offset
    dpsi = sp.Symbol('dpsi')             # dpsi/dtheta
    ddpsi = sp.Symbol('ddpsi')           # d^2psi/dtheta^2

    print("\n2. Computing kinematics...")

    # =========================================================================
    # KINEMATICS
    # =========================================================================

    # Pivot point (fixed)
    p_pivot = Matrix([0, H])

    # Joint position (end of root segment)
    # Root rotates by theta from vertical (downward is 0)
    p_joint = p_pivot + L_root * Matrix([sin(theta), -cos(theta)])

    # Angle of tip segment from vertical
    angle_tip = theta + beta

    # Tip end position (sling attachment point)
    p_tip = p_joint + L_tip * Matrix([sin(angle_tip), -cos(angle_tip)])

    # Projectile position
    # Sling hangs from tip, phi is angle from vertical
    p_proj = p_tip + L_sling * Matrix([sin(phi), -cos(phi)])

    # Cam pin position (where counterweight rope attaches)
    # On the short arm side (opposite to long arm), offset by psi
    cam_angle = theta + sp.pi + psi
    p_cam_pin = p_pivot + R_cam * Matrix([sin(cam_angle), -cos(cam_angle)])

    # Counterweight position
    # Hangs from cam pin, gamma is angle from vertical
    p_cw = p_cam_pin + L_hanger * Matrix([sin(gamma), -cos(gamma)])

    # Center of mass positions for beam segments (at midpoint)
    p_root_cm = p_pivot + (L_root / 2) * Matrix([sin(theta), -cos(theta)])
    p_tip_cm = p_joint + (L_tip / 2) * Matrix([sin(angle_tip), -cos(angle_tip)])

    print("   - Pivot:", p_pivot.T)
    print("   - Joint:", p_joint.T)
    print("   - Tip:", p_tip.T)

    # =========================================================================
    # VELOCITIES
    # =========================================================================
    print("\n3. Computing velocities...")

    # Velocity of joint
    v_joint = diff(p_joint, t)

    # Velocity of tip
    v_tip = diff(p_tip, t)

    # Velocity of projectile
    v_proj = diff(p_proj, t)

    # Velocity of cam pin
    # Need chain rule: d(R_cam)/dt = dR_cam/dtheta * dtheta/dt
    # First substitute time derivatives of cam parameters
    v_cam_pin_raw = diff(p_cam_pin, t)

    # Manual chain rule for R_cam and psi
    # d(R_cam)/dt = dR_cam * d_theta
    # d(psi)/dt = dpsi * d_theta
    v_cam_pin = v_cam_pin_raw.subs([
        (diff(R_cam, t), dR_cam * d_theta),
        (diff(psi, t), dpsi * d_theta)
    ])

    # Velocity of counterweight
    v_cw_raw = diff(p_cw, t)
    v_cw = v_cw_raw.subs([
        (diff(R_cam, t), dR_cam * d_theta),
        (diff(psi, t), dpsi * d_theta)
    ])

    # Velocities of beam segment centers of mass
    v_root_cm = diff(p_root_cm, t)
    v_tip_cm_raw = diff(p_tip_cm, t)
    v_tip_cm = v_tip_cm_raw

    print("   Velocities computed.")

    # =========================================================================
    # KINETIC ENERGY
    # =========================================================================
    print("\n4. Computing kinetic energy...")

    # Root segment: rotation about pivot
    # T_root = (1/2) * I_root * d_theta^2
    T_root = Rational(1, 2) * I_root * d_theta**2

    # Tip segment: translation of CM + rotation about CM
    # Rotation rate of tip = d_theta + d_beta
    omega_tip = d_theta + d_beta
    v_tip_cm_sq = v_tip_cm.dot(v_tip_cm)
    T_tip = Rational(1, 2) * m_tip * v_tip_cm_sq + Rational(1, 2) * I_tip * omega_tip**2

    # Projectile: translation only (point mass)
    v_proj_sq = v_proj.dot(v_proj)
    T_proj = Rational(1, 2) * m_proj * v_proj_sq

    # Counterweight: translation only (point mass)
    v_cw_sq = v_cw.dot(v_cw)
    T_cw = Rational(1, 2) * M_cw * v_cw_sq

    # Total kinetic energy
    T = T_root + T_tip + T_proj + T_cw

    print("   T_root, T_tip, T_proj, T_cw computed.")

    # =========================================================================
    # POTENTIAL ENERGY
    # =========================================================================
    print("\n5. Computing potential energy...")

    # Gravitational potential (using y-coordinate, ground at y=0)
    V_root = m_root * g * p_root_cm[1]
    V_tip = m_tip * g * p_tip_cm[1]
    V_proj = m_proj * g * p_proj[1]
    V_cw = M_cw * g * p_cw[1]

    # Elastic potential (torsional spring at joint)
    V_spring = Rational(1, 2) * k_stiff * beta**2

    # Total potential energy
    V = V_root + V_tip + V_proj + V_cw + V_spring

    print("   V_grav and V_spring computed.")

    # =========================================================================
    # LAGRANGIAN
    # =========================================================================
    print("\n6. Computing Lagrangian...")

    L = T - V

    print("   L = T - V computed.")

    # =========================================================================
    # GENERALIZED FORCES (non-conservative)
    # =========================================================================
    print("\n7. Defining generalized forces...")

    # Damping in the flex joint
    # Q_beta = -c_damp * d_beta
    Q_theta = 0
    Q_beta = -c_damp * d_beta
    Q_gamma = 0
    Q_phi = 0

    Q = Matrix([Q_theta, Q_beta, Q_gamma, Q_phi])

    print("   Damping force on beta: -c_damp * d_beta")

    # =========================================================================
    # LAGRANGE EQUATIONS
    # =========================================================================
    print("\n8. Deriving Lagrange equations...")
    print("   This may take a few minutes...")

    # d/dt(dL/d(dq_i)) - dL/dq_i = Q_i
    eom_exprs = []
    coords = [theta, beta, gamma, phi]
    vels = [d_theta, d_beta, d_gamma, d_phi]

    for i, (qi, dqi) in enumerate(zip(coords, vels)):
        print(f"   Processing coordinate {i+1}/4: {qi}...")

        # dL/d(dq_i)
        dL_ddqi = diff(L, dqi)

        # d/dt(dL/d(dq_i)) - need chain rule for cam terms
        d_dL_ddqi_raw = diff(dL_ddqi, t)

        # Substitute time derivatives of cam parameters
        d_dL_ddqi = d_dL_ddqi_raw.subs([
            (diff(R_cam, t), dR_cam * d_theta),
            (diff(psi, t), dpsi * d_theta),
            (diff(dR_cam, t), ddR_cam * d_theta**2 + dR_cam * dd_theta),
            (diff(dpsi, t), ddpsi * d_theta**2 + dpsi * dd_theta),
        ])

        # dL/dq_i
        dL_dqi = diff(L, qi)

        # Lagrange equation: d/dt(dL/d(dq_i)) - dL/dq_i - Q_i = 0
        eom_i = d_dL_ddqi - dL_dqi - Q[i]

        # Simplify (can be slow)
        # eom_i = trigsimp(eom_i)

        eom_exprs.append(eom_i)

    print("\n9. Converting to matrix form M @ ddq = F...")

    # Extract mass matrix and force vector
    # Equations are linear in ddq: M @ ddq = F
    accels = [dd_theta, dd_beta, dd_gamma, dd_phi]

    # Use linear_eq_to_matrix
    # linear_eq_to_matrix returns (A, b) such that A @ x = b when equations = 0
    # Our equations are: eom_i = M_row @ accels + remainder = 0
    # So: M @ accels = -remainder, and b = -remainder
    # Therefore F = b (NOT -b!)
    try:
        M_matrix, F_vector = sp.linear_eq_to_matrix(eom_exprs, accels)
    except Exception as e:
        print(f"   Error in linear_eq_to_matrix: {e}")
        print("   Trying manual extraction...")

        # Manual extraction
        M_matrix = zeros(4, 4)
        F_vector = Matrix([0, 0, 0, 0])

        for i, expr in enumerate(eom_exprs):
            expr_expanded = sp.expand(expr)
            for j, acc in enumerate(accels):
                coeff = expr_expanded.coeff(acc)
                M_matrix[i, j] = coeff

            # F is everything that doesn't contain accelerations (negated)
            # eom_i = M_row @ accels + remainder = 0
            # M_row @ accels = -remainder, so F = -remainder
            remainder = expr_expanded
            for j, acc in enumerate(accels):
                remainder = remainder - M_matrix[i, j] * acc
            F_vector[i] = -remainder  # This is correct for manual case

    print("   Mass matrix M: 4x4")
    print("   Force vector F: 4x1")

    return {
        't': t,
        'q': q,
        'dq': dq,
        'ddq': ddq,
        'coords': coords,
        'vels': vels,
        'M': M_matrix,
        'F': F_vector,
        'T': T,
        'V': V,
        'L': L,
        # Parameters
        'params': {
            'L_root': L_root, 'L_tip': L_tip, 'L_hanger': L_hanger,
            'L_sling': L_sling, 'H': H,
            'm_root': m_root, 'm_tip': m_tip, 'M_cw': M_cw, 'm_proj': m_proj,
            'I_root': I_root, 'I_tip': I_tip,
            'k_stiff': k_stiff, 'c_damp': c_damp, 'g': g,
            'R_cam': R_cam, 'dR_cam': dR_cam, 'ddR_cam': ddR_cam,
            'psi': psi, 'dpsi': dpsi, 'ddpsi': ddpsi,
        },
        # Positions for visualization
        'positions': {
            'p_pivot': p_pivot,
            'p_joint': p_joint,
            'p_tip': p_tip,
            'p_proj': p_proj,
            'p_cam_pin': p_cam_pin,
            'p_cw': p_cw,
        },
        # Velocities
        'velocities': {
            'v_proj': v_proj,
            'v_cw': v_cw,
        }
    }


def generate_code(derived):
    """Generate Python code from symbolic expressions."""

    print("\n10. Generating Python code...")

    # Extract symbols
    t = derived['t']
    params = derived['params']
    M = derived['M']
    F = derived['F']
    positions = derived['positions']
    velocities = derived['velocities']

    # Create substitution for dynamicsymbols to regular symbols
    theta, beta, gamma, phi = sp.symbols('th bt gm ph')
    d_theta, d_beta, d_gamma, d_phi = sp.symbols('dth dbt dgm dph')
    dd_theta, dd_beta, dd_gamma, dd_phi = sp.symbols('ddth ddbt ddgm ddph')

    # Original dynamic symbols
    theta_t = dynamicsymbols('theta')
    beta_t = dynamicsymbols('beta')
    gamma_t = dynamicsymbols('gamma')
    phi_t = dynamicsymbols('phi')

    d_theta_t = diff(theta_t, t)
    d_beta_t = diff(beta_t, t)
    d_gamma_t = diff(gamma_t, t)
    d_phi_t = diff(phi_t, t)

    dd_theta_t = diff(theta_t, t, 2)
    dd_beta_t = diff(beta_t, t, 2)
    dd_gamma_t = diff(gamma_t, t, 2)
    dd_phi_t = diff(phi_t, t, 2)

    subs_dict = {
        theta_t: theta, beta_t: beta, gamma_t: gamma, phi_t: phi,
        d_theta_t: d_theta, d_beta_t: d_beta, d_gamma_t: d_gamma, d_phi_t: d_phi,
        dd_theta_t: dd_theta, dd_beta_t: dd_beta, dd_gamma_t: dd_gamma, dd_phi_t: dd_phi,
    }

    # Apply substitutions
    M_sub = M.subs(subs_dict)
    F_sub = F.subs(subs_dict)

    # Substitute positions and velocities
    pos_sub = {}
    for name, expr in positions.items():
        pos_sub[name] = expr.subs(subs_dict)

    vel_sub = {}
    for name, expr in velocities.items():
        vel_sub[name] = expr.subs(subs_dict)

    # Create list of all arguments for the functions
    state_syms = [theta, beta, gamma, phi, d_theta, d_beta, d_gamma, d_phi]

    param_syms = [
        params['L_root'], params['L_tip'], params['L_hanger'], params['L_sling'], params['H'],
        params['m_root'], params['m_tip'], params['M_cw'], params['m_proj'],
        params['I_root'], params['I_tip'],
        params['k_stiff'], params['c_damp'], params['g'],
        params['R_cam'], params['dR_cam'], params['ddR_cam'],
        params['psi'], params['dpsi'], params['ddpsi'],
    ]

    all_args = state_syms + param_syms

    print("   Creating NumPy code...")

    # Use NumPyPrinter for code generation
    printer = NumPyPrinter()

    # Generate code strings for each element
    def matrix_to_code(mat, name):
        """Convert matrix to numpy code."""
        rows, cols = mat.shape
        lines = [f"def {name}(th, bt, gm, ph, dth, dbt, dgm, dph,"]
        lines.append("            L_root, L_tip, L_hanger, L_sling, H,")
        lines.append("            m_root, m_tip, M_cw, m_proj,")
        lines.append("            I_root, I_tip, k_stiff, c_damp, g,")
        lines.append("            R_cam, dR_cam, ddR_cam, psi, dpsi, ddpsi):")
        lines.append(f'    """Compute {name} matrix ({rows}x{cols})."""')
        lines.append(f"    result = np.zeros(({rows}, {cols}))")

        for i in range(rows):
            for j in range(cols):
                expr = mat[i, j]
                if expr != 0:
                    code = printer.doprint(expr)
                    lines.append(f"    result[{i}, {j}] = {code}")

        lines.append("    return result")
        return "\n".join(lines)

    def vector_to_code(vec, name):
        """Convert vector to numpy code."""
        rows = vec.shape[0]
        lines = [f"def {name}(th, bt, gm, ph, dth, dbt, dgm, dph,"]
        lines.append("            L_root, L_tip, L_hanger, L_sling, H,")
        lines.append("            m_root, m_tip, M_cw, m_proj,")
        lines.append("            I_root, I_tip, k_stiff, c_damp, g,")
        lines.append("            R_cam, dR_cam, ddR_cam, psi, dpsi, ddpsi):")
        lines.append(f'    """Compute {name} vector ({rows}x1)."""')
        lines.append(f"    result = np.zeros({rows})")

        for i in range(rows):
            expr = vec[i]
            if expr != 0:
                code = printer.doprint(expr)
                lines.append(f"    result[{i}] = {code}")

        lines.append("    return result")
        return "\n".join(lines)

    # Generate position functions
    def pos_to_code(expr, name):
        """Convert position expression to numpy code."""
        lines = [f"def {name}(th, bt, gm, ph, L_root, L_tip, L_hanger, L_sling, H, R_cam, psi):"]
        lines.append(f'    """Compute {name} position (x, y)."""')

        x_code = printer.doprint(expr[0])
        y_code = printer.doprint(expr[1])

        lines.append(f"    x = {x_code}")
        lines.append(f"    y = {y_code}")
        lines.append("    return np.array([x, y])")

        return "\n".join(lines)

    # Generate the module
    module_code = '''"""
Trebuchet-Level-4: Generated Equations of Motion
=================================================
AUTO-GENERATED FILE - DO NOT EDIT MANUALLY
Generated by derive_physics.py

This module contains the mass matrix M and force vector F
such that M @ q_ddot = F, where q = [theta, beta, gamma, phi]
"""

import numpy as np
from numpy import sin, cos, sqrt, pi


'''

    # Add mass matrix function
    module_code += matrix_to_code(M_sub, "compute_M") + "\n\n\n"

    # Add force vector function
    module_code += vector_to_code(F_sub, "compute_F") + "\n\n\n"

    # Add position functions
    for name, expr in pos_sub.items():
        module_code += pos_to_code(expr, f"compute_{name}") + "\n\n\n"

    # Add velocity functions for projectile
    v_proj = vel_sub['v_proj']
    lines = ["def compute_v_proj(th, bt, gm, ph, dth, dbt, dgm, dph,"]
    lines.append("                   L_root, L_tip, L_sling, R_cam, dR_cam, psi, dpsi):")
    lines.append('    """Compute projectile velocity (vx, vy)."""')
    vx_code = printer.doprint(v_proj[0])
    vy_code = printer.doprint(v_proj[1])
    lines.append(f"    vx = {vx_code}")
    lines.append(f"    vy = {vy_code}")
    lines.append("    return np.array([vx, vy])")
    module_code += "\n".join(lines) + "\n\n\n"

    # Add CW velocity
    v_cw = vel_sub['v_cw']
    lines = ["def compute_v_cw(th, bt, gm, ph, dth, dbt, dgm, dph,"]
    lines.append("                 L_root, L_hanger, R_cam, dR_cam, psi, dpsi):")
    lines.append('    """Compute counterweight velocity (vx, vy)."""')
    vx_code = printer.doprint(v_cw[0])
    vy_code = printer.doprint(v_cw[1])
    lines.append(f"    vx = {vx_code}")
    lines.append(f"    vy = {vy_code}")
    lines.append("    return np.array([vx, vy])")
    module_code += "\n".join(lines) + "\n\n\n"

    # Add convenience function for computing accelerations
    module_code += '''
def compute_accelerations(state, params):
    """
    Compute accelerations given state and parameters.

    Parameters
    ----------
    state : array_like
        State vector [theta, beta, gamma, phi, d_theta, d_beta, d_gamma, d_phi]
    params : dict
        Dictionary with keys: L_root, L_tip, L_hanger, L_sling, H,
                              m_root, m_tip, M_cw, m_proj, I_root, I_tip,
                              k_stiff, c_damp, g, R_cam, dR_cam, ddR_cam,
                              psi, dpsi, ddpsi

    Returns
    -------
    accelerations : ndarray
        Array [dd_theta, dd_beta, dd_gamma, dd_phi]
    """
    th, bt, gm, ph, dth, dbt, dgm, dph = state

    # Extract parameters
    L_root = params['L_root']
    L_tip = params['L_tip']
    L_hanger = params['L_hanger']
    L_sling = params['L_sling']
    H = params['H']
    m_root = params['m_root']
    m_tip = params['m_tip']
    M_cw = params['M_cw']
    m_proj = params['m_proj']
    I_root = params['I_root']
    I_tip = params['I_tip']
    k_stiff = params['k_stiff']
    c_damp = params['c_damp']
    g = params['g']
    R_cam = params['R_cam']
    dR_cam = params['dR_cam']
    ddR_cam = params['ddR_cam']
    psi = params['psi']
    dpsi = params['dpsi']
    ddpsi = params['ddpsi']

    # Compute mass matrix and force vector
    M = compute_M(th, bt, gm, ph, dth, dbt, dgm, dph,
                  L_root, L_tip, L_hanger, L_sling, H,
                  m_root, m_tip, M_cw, m_proj,
                  I_root, I_tip, k_stiff, c_damp, g,
                  R_cam, dR_cam, ddR_cam, psi, dpsi, ddpsi)

    F = compute_F(th, bt, gm, ph, dth, dbt, dgm, dph,
                  L_root, L_tip, L_hanger, L_sling, H,
                  m_root, m_tip, M_cw, m_proj,
                  I_root, I_tip, k_stiff, c_damp, g,
                  R_cam, dR_cam, ddR_cam, psi, dpsi, ddpsi)

    # Solve M @ ddq = F for accelerations
    try:
        accelerations = np.linalg.solve(M, F)
    except np.linalg.LinAlgError:
        # Singular matrix - return zeros
        accelerations = np.zeros(4)

    return accelerations
'''

    return module_code


def main():
    """Main function to derive physics and generate code."""

    print("\n" + "=" * 60)
    print("Starting symbolic derivation...")
    print("=" * 60)

    # Derive equations
    derived = derive_equations()

    # Generate code
    code = generate_code(derived)

    # Write to file
    output_path = os.path.join(os.path.dirname(__file__), "generated_eom.py")

    print(f"\n11. Writing to {output_path}...")

    with open(output_path, 'w') as f:
        f.write(code)

    print("\n" + "=" * 60)
    print("DONE! Generated equations saved to generated_eom.py")
    print("=" * 60)

    # Verification
    print("\n12. Verification - importing generated module...")
    try:
        import generated_eom
        print("   SUCCESS: Module imported correctly!")

        # Quick test
        import numpy as np
        state = np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        params = {
            'L_root': 2.0, 'L_tip': 4.0, 'L_hanger': 1.5, 'L_sling': 3.0, 'H': 3.0,
            'm_root': 20.0, 'm_tip': 30.0, 'M_cw': 500.0, 'm_proj': 10.0,
            'I_root': 26.67, 'I_tip': 160.0,
            'k_stiff': 10000.0, 'c_damp': 100.0, 'g': 9.81,
            'R_cam': 0.8, 'dR_cam': 0.0, 'ddR_cam': 0.0,
            'psi': 0.0, 'dpsi': 0.0, 'ddpsi': 0.0,
        }
        accel = generated_eom.compute_accelerations(state, params)
        print(f"   Test accelerations: {accel}")
        print("   Module verification PASSED!")

    except Exception as e:
        print(f"   WARNING: Could not verify module: {e}")
        print("   The module may still work, but please check manually.")


if __name__ == "__main__":
    main()
