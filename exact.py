import numpy as np
import matplotlib.pyplot as plt

# --- Problem Parameters ---
GAMMA = 1.4

# Initial conditions (Left state L, Right state R)
# (rho, u, p)
RHO_L, U_L, P_L = 1.0, 0.0, 1.0
RHO_R, U_R, P_R = 0.125, 0.0, 0.1

# Sound speeds in initial states
A_L = np.sqrt(GAMMA * P_L / RHO_L)
A_R = np.sqrt(GAMMA * P_R / RHO_R)

# --- Functions for Pressure Solver (Newton-Raphson) ---
def f_K(p_star, rho_K, p_K, a_K):
    """
    Function f_K(p_star) used in the pressure equation F(p_star)=0.
    K can be L (left) or R (right).
    """
    if p_star > p_K:  # Shock
        A_K_const = 2.0 / ((GAMMA + 1.0) * rho_K)
        B_K_const = (GAMMA - 1.0) / (GAMMA + 1.0) * p_K
        return (p_star - p_K) * np.sqrt(A_K_const / (p_star + B_K_const))
    else:  # Rarefaction
        return (2.0 * a_K / (GAMMA - 1.0)) * \
               ((p_star / p_K)**((GAMMA - 1.0) / (2.0 * GAMMA)) - 1.0)

def f_K_derivative(p_star, rho_K, p_K, a_K):
    """
    Derivative of f_K(p_star) w.r.t p_star.
    """
    if p_star > p_K:  # Shock derivative
        A_K_const = 2.0 / ((GAMMA + 1.0) * rho_K)
        B_K_const = (GAMMA - 1.0) / (GAMMA + 1.0) * p_K
        term1 = np.sqrt(A_K_const / (p_star + B_K_const))
        term2 = (p_star - p_K) / (2.0 * (p_star + B_K_const))
        return term1 * (1.0 - term2)
    else:  # Rarefaction derivative
        return (1.0 / (rho_K * a_K)) * (p_star / p_K)**(-(GAMMA + 1.0) / (2.0 * GAMMA))
        # Alternative form: (a_K / (GAMMA * p_K)) * (p_star / p_K)**(-(GAMMA + 1.0) / (2.0 * GAMMA))
        # The above is from Toro's f'_k directly related to his definition of f_k
        # Let's re-derive for the f_K defined above:
        # d/dp* [ (2aK/(g-1)) * ( (p*/pK)^alpha - 1 ) ] where alpha = (g-1)/(2g)
        # = (2aK/(g-1)) * alpha * (1/pK) * (p*/pK)^(alpha-1)
        # = (2aK/(g-1)) * ((g-1)/(2g)) * (1/pK) * (p*/pK)^(((g-1)/(2g)) - 1)
        # = (aK / (g*pK)) * (p*/pK)^(-(g+1)/(2g))
        # This one seems more consistent with formulas. Let's use it.
        # return (a_K / (GAMMA * p_K)) * (p_star / p_K)**(-(GAMMA + 1.0) / (2.0 * GAMMA))

def pressure_function(p_star):
    """
    F(p_star) = f_L(p_star) + f_R(p_star) + (U_R - U_L)
    We want to find p_star such that F(p_star) = 0.
    """
    val_L = f_K(p_star, RHO_L, P_L, A_L)
    val_R = f_K(p_star, RHO_R, P_R, A_R)
    return val_L + val_R + (U_R - U_L)

def pressure_function_derivative(p_star):
    """
    F'(p_star) = f'_L(p_star) + f'_R(p_star)
    """
    deriv_L = f_K_derivative(p_star, RHO_L, P_L, A_L)
    deriv_R = f_K_derivative(p_star, RHO_R, P_R, A_R)
    return deriv_L + deriv_R

# --- Solve for p_star using Newton-Raphson ---
TOLERANCE = 1e-6
MAX_ITER = 20
# Initial guess for p_star (e.g., average, or Two-Rarefaction Approx if du=0)
# p_star_guess = (P_L + P_R) / 2.0
# Toro's suggested guess (Eq. 4.46)
p_pvrs = 0.5*(P_L+P_R) - 0.125*(U_R-U_L)*(RHO_L+RHO_R)*(A_L+A_R)
p_star_guess = max(TOLERANCE, p_pvrs) # ensure positive

p_star = p_star_guess
for i in range(MAX_ITER):
    F_val = pressure_function(p_star)
    F_prime_val = pressure_function_derivative(p_star)
    
    if abs(F_prime_val) < 1e-10: # Avoid division by zero
        print(f"Derivative too small at iteration {i+1}, p_star = {p_star}")
        break
        
    p_star_new = p_star - F_val / F_prime_val
    
    if abs(p_star_new - p_star) < TOLERANCE:
        p_star = p_star_new
        # print(f"Converged at iteration {i+1} to p_star = {p_star}")
        break
    p_star = p_star_new
    if p_star < 0: # Pressure must be positive
        p_star = TOLERANCE # Reset to small positive if it goes negative
else:
    print(f"Newton-Raphson did not converge after {MAX_ITER} iterations. Final p_star = {p_star}")

print(f"Solved p_star = {p_star:.6f}")

# --- Calculate u_star (velocity in star region) ---
# u_star = U_L - f_K(p_star, RHO_L, P_L, A_L) # This is not directly u_star, f_K is related to u_star-u_K
# Correct formula for u_star using f_K as defined in Toro (Ch 4, Eq 4.9)
u_star = 0.5 * (U_L + U_R) + 0.5 * (f_K(p_star, RHO_R, P_R, A_R) - f_K(p_star, RHO_L, P_L, A_L))
print(f"Solved u_star = {u_star:.6f}")

# --- Calculate densities in star region (rho_star_L, rho_star_R) ---
# Left wave (expected Rarefaction for Sod problem, p_star < P_L)
if p_star <= P_L: # Rarefaction
    rho_star_L = RHO_L * (p_star / P_L)**(1.0 / GAMMA)
else: # Shock
    numerator = p_star / P_L + (GAMMA - 1.0) / (GAMMA + 1.0)
    denominator = ((GAMMA - 1.0) / (GAMMA + 1.0)) * (p_star / P_L) + 1.0
    rho_star_L = RHO_L * (numerator / denominator)

# Right wave (expected Shock for Sod problem, p_star > P_R)
if p_star >= P_R: # Shock
    numerator = p_star / P_R + (GAMMA - 1.0) / (GAMMA + 1.0)
    denominator = ((GAMMA - 1.0) / (GAMMA + 1.0)) * (p_star / P_R) + 1.0
    rho_star_R = RHO_R * (numerator / denominator)
else: # Rarefaction
    rho_star_R = RHO_R * (p_star / P_R)**(1.0 / GAMMA)
    
print(f"rho_star_L = {rho_star_L:.6f}, rho_star_R = {rho_star_R:.6f}")

# --- Calculate wave speeds ---
# Left wave
if p_star <= P_L: # Rarefaction
    a_star_L = np.sqrt(GAMMA * p_star / rho_star_L)
    S_HL = U_L - A_L  # Head of rarefaction
    S_TL = u_star - a_star_L # Tail of rarefaction
    print(f"Left wave: Rarefaction. Head speed S_HL = {S_HL:.4f}, Tail speed S_TL = {S_TL:.4f}")
else: # Shock
    S_L = U_L - A_L * np.sqrt(((GAMMA + 1.0) / (2.0 * GAMMA)) * (p_star / P_L) + (GAMMA - 1.0) / (2.0 * GAMMA))
    # Alternative S_L = (RHO_L * U_L - rho_star_L * u_star) / (RHO_L - rho_star_L)
    print(f"Left wave: Shock. Speed S_L = {S_L:.4f}")


# Contact discontinuity speed
S_C = u_star
print(f"Contact discontinuity speed S_C = {S_C:.4f}")

# Right wave
if p_star >= P_R: # Shock
    S_R = U_R + A_R * np.sqrt(((GAMMA + 1.0) / (2.0 * GAMMA)) * (p_star / P_R) + (GAMMA - 1.0) / (2.0 * GAMMA))
    # Alternative S_R = (RHO_R * U_R - rho_star_R * u_star) / (RHO_R - rho_star_R)
    print(f"Right wave: Shock. Speed S_R = {S_R:.4f}")
else: # Rarefaction
    a_star_R = np.sqrt(GAMMA * p_star / rho_star_R)
    S_HR = U_R + A_R  # Head of rarefaction
    S_TR = u_star + a_star_R # Tail of rarefaction
    print(f"Right wave: Rarefaction. Head speed S_HR = {S_HR:.4f}, Tail speed S_TR = {S_TR:.4f}")

# --- Sample solution at a given time t ---
def sample_solution(t_final, num_points=500, x_min=-0.5, x_max=0.5):
    x_coords = np.linspace(x_min, x_max, num_points)
    rho_sol = np.zeros(num_points)
    u_sol = np.zeros(num_points)
    p_sol = np.zeros(num_points)
    # e_sol = np.zeros(num_points) # Specific internal energy
    # E_sol = np.zeros(num_points) # Total energy

    for i, x in enumerate(x_coords):
        xi = x / t_final # Self-similar coordinate

        if p_star <= P_L: # Left Rarefaction
            if xi < S_HL: # Region 1 (Left initial state)
                rho_sol[i] = RHO_L
                u_sol[i] = U_L
                p_sol[i] = P_L
            elif xi < S_TL: # Region 2 (Left rarefaction fan)
                # u(xi) = (2/(g+1)) * (a_L + (g-1)/2*u_L + xi)
                # a(xi) = (2/(g+1)) * (a_L + (g-1)/2*u_L - (g-1)/2*xi)
                # rho = rho_L * (a(xi)/a_L)^(2/(g-1))
                # p = p_L * (a(xi)/a_L)^(2g/(g-1))
                u_sol[i] = (2.0 / (GAMMA + 1.0)) * (A_L + (GAMMA - 1.0) / 2.0 * U_L + xi)
                a_fan = (2.0 / (GAMMA + 1.0)) * (A_L + (GAMMA - 1.0) / 2.0 * U_L - (GAMMA - 1.0) / 2.0 * xi)
                rho_sol[i] = RHO_L * (a_fan / A_L)**(2.0 / (GAMMA - 1.0))
                p_sol[i] = P_L * (a_fan / A_L)**(2.0 * GAMMA / (GAMMA - 1.0))
            else: # Star region left of contact or further right
                # This part will be handled by contact and right wave logic
                pass # Will be overwritten if necessary
        else: # Left Shock (S_L is the shock speed)
            if xi < S_L: # Region 1
                rho_sol[i] = RHO_L
                u_sol[i] = U_L
                p_sol[i] = P_L
            else: # Star region or further right
                pass

        # Check star region and right wave (this logic needs to be careful about overlaps)
        # Assuming the standard Sod pattern: L-Rarefaction-StarL-Contact-StarR-Shock-R
        if xi >= S_TL and xi < S_C: # Region 3 (StarL: after left rarefaction, before contact)
            rho_sol[i] = rho_star_L
            u_sol[i] = u_star
            p_sol[i] = p_star
        elif xi >= S_C and xi < S_R: # Region 4 (StarR: after contact, before right shock)
            rho_sol[i] = rho_star_R
            u_sol[i] = u_star
            p_sol[i] = p_star
        elif xi >= S_R: # Region 5 (Right initial state, after right shock)
            rho_sol[i] = RHO_R
            u_sol[i] = U_R
            p_sol[i] = P_R
        # If left was a shock, need to adjust conditions for Region 3 and 4
        if p_star > P_L: # If left wave was a shock (S_L)
             if xi >= S_L and xi < S_C: # After left shock, before contact
                rho_sol[i] = rho_star_L
                u_sol[i] = u_star
                p_sol[i] = p_star


    # Calculate total energy E = p/(gamma-1) + 0.5 * rho * u^2
    E_sol = p_sol / (GAMMA - 1.0) + 0.5 * rho_sol * u_sol**2
    return x_coords, rho_sol, u_sol, p_sol, E_sol

# --- Plotting ---
t_final_plot = 10 # Standard time for Sod problem visualization
x_plot, rho_plot, u_plot, p_plot, E_plot = sample_solution(t_final_plot, x_min=-0.5, x_max=0.5)

fig, axes = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
plt.rcParams.update({'font.size': 12, 'font.family':'sans-serif'}) # Or specific like 'Arial'

axes[0].plot(x_plot, rho_plot, 'b-', linewidth=1.5)
axes[0].set_ylabel('Density ($\\rho$)')
axes[0].set_title(f'Sod Shock Tube Exact Solution at t = {t_final_plot}')
axes[0].grid(True, linestyle=':', alpha=0.7)

axes[1].plot(x_plot, u_plot, 'r-', linewidth=1.5)
axes[1].set_ylabel('Velocity ($u$)')
axes[1].grid(True, linestyle=':', alpha=0.7)

axes[2].plot(x_plot, p_plot, 'g-', linewidth=1.5)
axes[2].set_ylabel('Pressure ($p$)')
axes[2].grid(True, linestyle=':', alpha=0.7)

axes[3].plot(x_plot, E_plot, 'm-', linewidth=1.5)
axes[3].set_ylabel('Total Energy ($E$)')
axes[3].set_xlabel('Position ($x$)')
axes[3].grid(True, linestyle=':', alpha=0.7)
# axes[3].set_ylim(min(E_plot)*0.9, max(E_plot)*1.1)


plt.tight_layout()
plt.show()

# --- For comparison with numerical methods (placeholder) ---
print("\nExact solution generated. To compare with a numerical method:")
print("1. Implement a numerical scheme (e.g., Lax-Friedrichs, Godunov, HLLC) for the Euler equations.")
print("2. Run the numerical scheme with the same initial conditions and up to the same t_final.")
print("3. Plot the numerical solution on the same graphs as the exact solution for comparison.")
print("   For example, if you have numerical_rho, numerical_u, numerical_p at positions numerical_x:")
print("   axes[0].plot(numerical_x, numerical_rho, 'k--', label='Numerical')")
print("   axes[0].legend()")