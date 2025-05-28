

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve # For the exact Riemann solver
import time

# --- Problem Constants & Setup ---
# --- 问题常数与设置 ---
gamma = 1.4  # Ratio of specific heats / 比热比 (绝热指数)

# Initial conditions for Sod problem / Sod 问题的初始条件
rho_L_init, u_L_init, p_L_init = 1.0, 0.0, 1.0    # Left state / 左侧状态 (密度, 速度, 压力)
rho_R_init, u_R_init, p_R_init = 0.125, 0.0, 0.1 # Right state / 右侧状态 (密度, 速度, 压力)

# Computational domain / 计算区域
XMIN, XMAX = -0.5, 0.5  # x-coordinate range / x坐标范围
NX = 200  # Number of grid cells (can be increased, e.g., 400) / 网格单元数量
DX = (XMAX - XMIN) / NX # Grid spacing / 网格间距
X_CELL_CENTERS = np.linspace(XMIN + DX / 2, XMAX - DX / 2, NX) # Cell centers / 单元中心x坐标

# Time parameters / 时间参数
T_FINAL = 0.2     # Final simulation time / 最终模拟时间
CFL_CONST = 0.05  # CFL number / CFL数, 用于控制时间步长

# Number of ghost cells. WENO3 and MUSCL (as implemented) require N_ghost=2.
# For WENO5, N_ghost=3 would be needed.
# 虚拟单元数量。WENO3和MUSCL(当前实现)需要N_ghost=2。WENO5则需要N_ghost=3。
N_GHOST = 2
WENO_EPS = 1e-6 # Epsilon for WENO smoothness indicators to avoid division by zero
                # WENO光滑指示子的epsilon值, 防止除零

# --- Variable Conversion Functions ---
# --- 变量转换函数 ---
def primitive_to_conserved(rho, u, p, gamma_val=gamma):
    """Converts primitive variables (rho, u, p) to conserved variables (rho, rho*u, E)."""
    """将原始变量 (密度, 速度, 压力) 转换为守恒变量 (密度, 动量密度, 总能量密度)."""
    rho_u = rho * u
    E = p / (gamma_val - 1.0) + 0.5 * rho * u**2 # Total energy density / 总能量密度
    if np.isscalar(rho):
        return np.array([rho, rho_u, E])
    else:
        return np.array([rho, rho_u, E])

def conserved_to_primitive(U, gamma_val=gamma):
    """Converts conserved variables (rho, rho*u, E) to primitive variables (rho, u, p)."""
    """将守恒变量 (密度, 动量密度, 总能量密度) 转换为原始变量 (密度, 速度, 压力)."""
    if U.ndim == 1: # Single state / 单个状态
        rho = U[0]
        u = U[1] / rho
        E = U[2]
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2) # Pressure / 压力
        # Enforce positivity for safety, though ideally numerical scheme should maintain it
        # 为安全起见强制正值，尽管理想情况下数值格式应能维持
        rho = max(rho, 1e-9)
        p = max(p, 1e-9)
        return np.array([rho, u, p])
    else: # Array of states / 状态数组
        rho = U[0, :]
        u = U[1, :] / rho
        E = U[2, :]
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2)
        # Add positivity enforcement / 添加正值强制
        rho = np.maximum(rho, 1e-9) # Ensure density is positive / 确保密度为正
        p = np.maximum(p, 1e-9)   # Ensure pressure is positive / 确保压力为正
        return np.array([rho, u, p])

# --- Euler Flux Function (Physical Flux) ---
# --- 欧拉方程通量函数 (物理通量) ---
def euler_flux(rho, u, p):
    """Calculates physical flux F(U) from primitive variables."""
    """根据原始变量计算物理通量 F(U)."""
    rho_u = rho * u
    E = p / (gamma - 1.0) + 0.5 * rho * u**2 # Total energy density / 总能量密度
    F = np.zeros_like(rho_u, shape=(3,) + rho.shape) # Ensure correct shape for arrays / 确保数组形状正确
    F[0] = rho_u
    F[1] = rho_u * u + p  # Momentum flux + pressure term / 动量通量 + 压力项
    F[2] = u * (E + p)    # Energy flux / 能量通量
    return F

# --- TVD Limiters ---
# --- TVD 限制器 ---
def van_leer_limiter(r):
    """Van Leer flux limiter."""
    """Van Leer 通量限制器."""
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-12) # Add epsilon for stability / 添加epsilon以保证稳定性

def superbee_limiter(r):
    """Superbee flux limiter."""
    """Superbee 通量限制器."""
    return np.maximum(0, np.maximum(np.minimum(1, 2 * r), np.minimum(2, r)))

LIMITERS = {
    'VanLeer': van_leer_limiter,
    'Superbee': superbee_limiter
}

# --- Reconstruction Schemes ---
# --- 重构格式 ---
def muscl_reconstruction(P_gh, limiter_func, n_ghost, nx_domain):
    """
    MUSCL reconstruction for primitive variables P_gh (3, nx_total).
    Returns P_L_inter, P_R_inter (3, nx_domain + 1) which are states at cell interfaces.
    P_gh: primitive variables with ghost cells (rho, u, p) / 带虚拟单元的原始变量
    limiter_func: the chosen flux limiter function / 选择的通量限制器函数
    n_ghost: number of ghost cells on each side / 每侧虚拟单元数量
    nx_domain: number of internal computational cells / 内部计算单元数量
    """
    # P_L_inter: 左侧重构值在界面上 (q_{j+1/2}^L)
    # P_R_inter: 右侧重构值在界面上 (q_{j+1/2}^R)
    P_L_at_interfaces = np.zeros((3, nx_domain + 1)) # States on the left side of interfaces / 界面左侧的状态
    P_R_at_interfaces = np.zeros((3, nx_domain + 1)) # States on the right side of interfaces / 界面右侧的状态

    for k_var in range(3): # Iterate over rho, u, p / 遍历密度、速度、压力
        q = P_gh[k_var, :] # Current primitive variable array (with ghost cells) / 当前原始变量数组(带虚拟单元)
        
        for j_inter in range(nx_domain + 1): # Loop over nx_domain+1 interfaces / 遍历nx_domain+1个界面
            # Left state for interface j_inter (reconstructed from cell to its left)
            # 界面 j_inter 的左状态 (从其左侧单元重构得到)
            # Cell "providing" left state: P_gh[k_var, n_ghost + j_inter - 1] (this is cell i)
            # 提供左状态的单元索引: idx_cell_L = i
            idx_cell_L = n_ghost + j_inter - 1 # Index of cell i
            
            # Slopes for cell i (idx_cell_L)
            # dq_L_minus = q_i - q_{i-1}
            # dq_L_plus  = q_{i+1} - q_i
            dq_L_minus = q[idx_cell_L]     - q[idx_cell_L-1] # Slope "before" cell i / 单元i之前的斜率
            dq_L_plus  = q[idx_cell_L+1]   - q[idx_cell_L]   # Slope "after" cell i / 单元i之后的斜率
            
            # Ratio r for cell i
            r_L_num = dq_L_plus  # Numerator for r_i / r_i的分子
            r_L_den = dq_L_minus # Denominator for r_i / r_i的分母
            if np.abs(r_L_den) < 1e-9: # Avoid division by zero or near-zero / 避免除以零或接近零
                r_L = 2.0 if r_L_num * r_L_den >= 0 else -2.0 # Favor smooth or limit / 倾向于平滑或限制
            else:
                r_L = r_L_num / r_L_den
            phi_L = limiter_func(r_L) # Limiter function applied to r_i / 应用于r_i的限制器函数
            # Reconstructed state on the left of interface j_inter (i+1/2)
            # q_{i+1/2}^L = q_i + 0.5 * phi(r_i) * (q_i - q_{i-1})
            P_L_at_interfaces[k_var, j_inter] = q[idx_cell_L] + 0.5 * phi_L * dq_L_minus

            # Right state for interface j_inter (reconstructed from cell to its right)
            # 界面 j_inter 的右状态 (从其右侧单元重构得到)
            # Cell "providing" right state: P_gh[k_var, n_ghost + j_inter] (this is cell i+1 or j in some notations)
            # 提供右状态的单元索引: idx_cell_R = i+1
            idx_cell_R = n_ghost + j_inter # Index of cell i+1

            # Slopes for cell i+1 (idx_cell_R)
            # dq_R_minus = q_{i+1} - q_i
            # dq_R_plus  = q_{i+2} - q_{i+1}
            dq_R_minus = q[idx_cell_R]     - q[idx_cell_R-1] # slope "before" cell i+1 / 单元i+1之前的斜率
            dq_R_plus  = q[idx_cell_R+1]   - q[idx_cell_R]   # slope "after" cell i+1 / 单元i+1之后的斜率
            
            # Ratio r for cell i+1
            r_R_num = dq_R_plus 
            r_R_den = dq_R_minus
            if np.abs(r_R_den) < 1e-9:
                r_R = 2.0 if r_R_num * r_R_den >= 0 else -2.0
            else:
                r_R = r_R_num / r_R_den
            phi_R = limiter_func(r_R) # Limiter applied to r_{i+1} / 应用于r_{i+1}的限制器函数
            # Reconstructed state on the right of interface j_inter (i+1/2)
            # q_{i+1/2}^R = q_{i+1} - 0.5 * phi(r_{i+1}) * (q_{i+1} - q_i)
            P_R_at_interfaces[k_var, j_inter] = q[idx_cell_R] - 0.5 * phi_R * dq_R_minus
            
    return P_L_at_interfaces, P_R_at_interfaces

def weno3_reconstruction(P_gh, n_ghost, nx_domain):
    """
    Component-wise WENO3 reconstruction for primitive variables P_gh (3, nx_total).
    Returns P_L_inter, P_R_inter (3, nx_domain + 1)
    (Based on Shu, C.-W. (1998). Essentially non-oscillatory..., section 2.3, r=2 case)
    P_gh: primitive variables with ghost cells / 带虚拟单元的原始变量
    n_ghost: number of ghost cells / 虚拟单元数量
    nx_domain: number of internal cells / 内部单元数量
    """
    P_L_at_interfaces = np.zeros((3, nx_domain + 1)) # 界面左侧的状态 q_{j+1/2}^L
    P_R_at_interfaces = np.zeros((3, nx_domain + 1)) # 界面右侧的状态 q_{j+1/2}^R
    
    # Linear weights for 3rd order (r=2 stencil) / 3阶(r=2模板)的线性权重
    # For P_L (q_{i+1/2}^-), using stencils S0={i-1,i}, S1={i,i+1} relative to cell i
    d0_L, d1_L = 2./3., 1./3. 
    # For P_R (q_{j-1/2}^+), which is q_{i+1/2}^R if j=i+1.
    # Stencils relative to cell j (idx_R_cell) are S0={j-1,j}, S1={j,j+1}
    # For q_{j-1/2}^+ (right value at interface j-1/2, or left interface of cell j)
    d0_R, d1_R = 1./3., 2./3. 

    for k_var in range(3): # Iterate over rho, u, p / 遍历密度、速度、压力
        q = P_gh[k_var, :] # Current primitive variable / 当前原始变量
        for j_inter in range(nx_domain + 1): # Interface index j_inter (corresponds to i+1/2) / 界面索引 (对应 i+1/2)
            # --- P_L at interface j_inter (q_{i+1/2}^-) ---
            # This value is reconstructed using cell i (idx_L_cell) as the central cell for the stencil.
            # Stencil uses cells: idx_L_cell-1, idx_L_cell, idx_L_cell+1
            # 界面 j_inter 左侧的值 (q_{i+1/2}^-), 使用单元 i (idx_L_cell) 作为模板中心
            idx_L_cell = n_ghost + j_inter - 1 # Cell i, immediately to the left of interface j_inter
            
            q_m1 = q[idx_L_cell - 1] # q_{i-1}
            q_0  = q[idx_L_cell]     # q_i
            q_p1 = q[idx_L_cell + 1] # q_{i+1}

            # Candidate polynomials evaluated at interface (q_{i+1/2}) / 在界面处计算的候选多项式值
            p0_L = -0.5 * q_m1 + 1.5 * q_0  # Stencil S0 = {q_{i-1}, q_i}
            p1_L =  0.5 * q_0  + 0.5 * q_p1  # Stencil S1 = {q_i, q_{i+1}}

            # Smoothness indicators (beta factors) / 光滑度指示子 (beta因子)
            IS0_L = (q_0 - q_m1)**2 # beta_0 for P_L
            IS1_L = (q_p1 - q_0)**2 # beta_1 for P_L

            # Nonlinear weights (alpha, then omega) / 非线性权重 (alpha, 然后 omega)
            alpha0_L = d0_L / (WENO_EPS + IS0_L)**2
            alpha1_L = d1_L / (WENO_EPS + IS1_L)**2
            
            w0_L = alpha0_L / (alpha0_L + alpha1_L)
            w1_L = alpha1_L / (alpha0_L + alpha1_L)

            P_L_at_interfaces[k_var, j_inter] = w0_L * p0_L + w1_L * p1_L

            # --- P_R at interface j_inter (q_{i+1/2}^+) ---
            # This value is reconstructed using cell i+1 (idx_R_cell) as the central cell.
            # It corresponds to q_{j-1/2}^+ if we let j = idx_R_cell.
            # Stencil uses cells: idx_R_cell-1, idx_R_cell, idx_R_cell+1
            # 界面 j_inter 右侧的值 (q_{i+1/2}^+), 使用单元 i+1 (idx_R_cell) 作为模板中心 (记为单元j)
            idx_R_cell = n_ghost + j_inter # Cell i+1 (or j), immediately to the right of interface j_inter
            
            q_m1 = q[idx_R_cell - 1] # q_{j-1} (relative to cell j=idx_R_cell)
            q_0  = q[idx_R_cell]     # q_j
            q_p1 = q[idx_R_cell + 1] # q_{j+1}

            # Candidate polynomials evaluated at interface (q_{j-1/2}) / 在界面处计算的候选多项式值
            p0_R =  0.5 * q_m1 + 0.5 * q_0   # Stencil S0 = {q_{j-1}, q_j}
            p1_R =  1.5 * q_0  - 0.5 * q_p1   # Stencil S1 = {q_j, q_{j+1}} (Note: -0.5*q_p1 is correct for q_{j-1/2})

            # Smoothness indicators (beta factors) / 光滑度指示子
            IS0_R = (q_0 - q_m1)**2 # beta_0 for P_R (relative to cell j)
            IS1_R = (q_p1 - q_0)**2 # beta_1 for P_R (relative to cell j)

            # Nonlinear weights / 非线性权重
            alpha0_R = d0_R / (WENO_EPS + IS0_R)**2 # Note: d0_R, d1_R are for right-biased stencil
            alpha1_R = d1_R / (WENO_EPS + IS1_R)**2
            
            w0_R = alpha0_R / (alpha0_R + alpha1_R)
            w1_R = alpha1_R / (alpha0_R + alpha1_R)

            P_R_at_interfaces[k_var, j_inter] = w0_R * p0_R + w1_R * p1_R
            
    return P_L_at_interfaces, P_R_at_interfaces


# --- Numerical Flux Schemes ---
# --- 数值通量格式 ---
def van_leer_fvs_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    """
    Van Leer Flux Vector Splitting.
    P_L_inter, P_R_inter are (3, nx+1) arrays of primitive variables at interfaces.
    Returns numerical flux F_num (3, nx+1).
    Van Leer 通量矢量分裂。
    P_L_inter, P_R_inter 是界面处原始变量的(3, nx+1)数组。
    返回数值通量 F_num (3, nx+1)。
    """
    rho_L, u_L, p_L = P_L_inter[0], P_L_inter[1], P_L_inter[2] # Left states at interfaces / 界面左状态
    rho_R, u_R, p_R = P_R_inter[0], P_R_inter[1], P_R_inter[2] # Right states at interfaces / 界面右状态

    # F_num = F^+(P_L) + F^-(P_R)
    F_plus_L, _ = van_leer_flux_split_vectorized(rho_L, u_L, p_L, gamma_val) # F^+ from left state / 左状态的F^+
    _, F_minus_R = van_leer_flux_split_vectorized(rho_R, u_R, p_R, gamma_val)# F^- from right state / 右状态的F^-
    
    return F_plus_L + F_minus_R

def van_leer_flux_split_vectorized(rho_vec, u_vec, p_vec, gamma_val=gamma):
    """Helper for van_leer_fvs_flux. Computes F+ and F- components."""
    """van_leer_fvs_flux 的辅助函数。计算 F+ 和 F- 分量。"""
    a_vec = np.sqrt(gamma_val * p_vec / rho_vec) # Sound speed / 声速
    M_vec = u_vec / a_vec                        # Mach number / 马赫数

    rho_u_vec = rho_vec * u_vec
    E_vec = p_vec / (gamma_val - 1.0) + 0.5 * rho_vec * u_vec**2

    F_plus_vec = np.zeros((3, len(rho_vec)))
    F_minus_vec = np.zeros((3, len(rho_vec)))

    # Indices for different Mach number regimes / 不同马赫数区域的索引
    idx_M_ge_1 = M_vec >= 1.0      # Supersonic, positive direction / 超音速, 正向
    idx_M_le_m1 = M_vec <= -1.0     # Supersonic, negative direction / 超音速, 负向
    idx_M_abs_lt_1 = np.abs(M_vec) < 1.0 # Subsonic / 亚音速

    # Full physical flux components / 完整物理通量分量
    F_full_0 = rho_u_vec
    F_full_1 = rho_u_vec**2 / rho_vec + p_vec # rho*u^2 + p
    F_full_2 = u_vec * (E_vec + p_vec)

    # For M >= 1, F+ = F_physical, F- = 0
    F_plus_vec[0, idx_M_ge_1] = F_full_0[idx_M_ge_1]
    F_plus_vec[1, idx_M_ge_1] = F_full_1[idx_M_ge_1]
    F_plus_vec[2, idx_M_ge_1] = F_full_2[idx_M_ge_1]
    # F_minus_vec remains zero for these indices

    # For M <= -1, F+ = 0, F- = F_physical
    F_minus_vec[0, idx_M_le_m1] = F_full_0[idx_M_le_m1]
    F_minus_vec[1, idx_M_le_m1] = F_full_1[idx_M_le_m1]
    F_minus_vec[2, idx_M_le_m1] = F_full_2[idx_M_le_m1]
    # F_plus_vec remains zero for these indices
    
    # Split flux for |M| < 1 (subsonic) / 亚音速情况下的分裂通量
    if np.any(idx_M_abs_lt_1): # Process only if there are subsonic points /仅当存在亚音速点时处理
        u_sub = u_vec[idx_M_abs_lt_1]
        a_sub = a_vec[idx_M_abs_lt_1]
        rho_sub = rho_vec[idx_M_abs_lt_1]
        M_sub = M_vec[idx_M_abs_lt_1]

        # F+ components for subsonic flow / 亚音速流的 F+ 分量
        f_mass_plus_sub = rho_sub * a_sub * 0.25 * (M_sub + 1.0)**2
        term_plus = ( (gamma_val - 1.0) * u_sub + 2.0 * a_sub )
        F_plus_vec[0, idx_M_abs_lt_1] = f_mass_plus_sub
        F_plus_vec[1, idx_M_abs_lt_1] = f_mass_plus_sub * ( term_plus / gamma_val )
        F_plus_vec[2, idx_M_abs_lt_1] = f_mass_plus_sub * ( term_plus**2 / (2.0 * (gamma_val**2 - 1.0)) )
        
        # F- components for subsonic flow / 亚音速流的 F- 分量
        f_mass_minus_sub = -rho_sub * a_sub * 0.25 * (M_sub - 1.0)**2
        term_minus = ( (gamma_val - 1.0) * u_sub - 2.0 * a_sub )
        F_minus_vec[0, idx_M_abs_lt_1] = f_mass_minus_sub
        F_minus_vec[1, idx_M_abs_lt_1] = f_mass_minus_sub * ( term_minus / gamma_val )
        F_minus_vec[2, idx_M_abs_lt_1] = f_mass_minus_sub * ( term_minus**2 / (2.0 * (gamma_val**2 - 1.0)) )
            
    return F_plus_vec, F_minus_vec

def roe_fds_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    """
    Roe Flux Difference Splitting.
    P_L_inter, P_R_inter are (3, nx+1) arrays of primitive variables at interfaces.
    Returns numerical flux F_num (3, nx+1).
    Roe 通量差分分裂。
    P_L_inter, P_R_inter 是界面处原始变量的(3, nx+1)数组。
    返回数值通量 F_num (3, nx+1)。
    """
    rho_L, u_L, p_L = P_L_inter[0], P_L_inter[1], P_L_inter[2] # Left states / 界面左状态
    rho_R, u_R, p_R = P_R_inter[0], P_R_inter[1], P_R_inter[2] # Right states / 界面右状态

    # Convert to conserved variables for Roe formulation / 转换为守恒变量以用于Roe格式
    U_L_inter = primitive_to_conserved(rho_L, u_L, p_L, gamma_val)
    U_R_inter = primitive_to_conserved(rho_R, u_R, p_R, gamma_val)

    # Physical fluxes F(U_L) and F(U_R) / 物理通量 F(U_L) 和 F(U_R)
    F_L = euler_flux(rho_L, u_L, p_L)
    F_R = euler_flux(rho_R, u_R, p_R)

    # Roe averages / Roe平均值
    sqrt_rho_L = np.sqrt(rho_L)
    sqrt_rho_R = np.sqrt(rho_R)
    
    rho_hat = sqrt_rho_L * sqrt_rho_R # Roe averaged density (geometric mean) / Roe平均密度 (几何平均)
    u_hat = (sqrt_rho_L * u_L + sqrt_rho_R * u_R) / (sqrt_rho_L + sqrt_rho_R) # Roe averaged velocity / Roe平均速度
    
    # Enthalpy H = (E+p)/rho = (gamma*p/((gamma-1)*rho)) + 0.5*u^2 / 焓
    E_L = U_L_inter[2]
    E_R = U_R_inter[2]
    H_L = (E_L + p_L) / rho_L # Left enthalpy / 左焓
    H_R = (E_R + p_R) / rho_R # Right enthalpy / 右焓
    H_hat = (sqrt_rho_L * H_L + sqrt_rho_R * H_R) / (sqrt_rho_L + sqrt_rho_R) # Roe averaged enthalpy / Roe平均焓
    
    a_hat_sq = (gamma_val - 1.0) * (H_hat - 0.5 * u_hat**2) # Square of Roe averaged sound speed / Roe平均声速的平方
    a_hat_sq = np.maximum(a_hat_sq, 1e-9) # Ensure positivity for sound speed / 确保声速为正
    a_hat = np.sqrt(a_hat_sq)             # Roe averaged sound speed / Roe平均声速

    # Differences in conserved variables / 守恒变量的差
    dU = U_R_inter - U_L_inter # dU = [d_rho, d_rho_u, d_E]

    # Eigenvalues of Roe matrix (u_hat-a_hat, u_hat, u_hat+a_hat) / Roe矩阵的特征值
    lambda_hat = np.array([u_hat - a_hat, u_hat, u_hat + a_hat])

    # Wave strengths (alpha_tilde_k from Anderson CFD book, based on projecting dU)
    # 波强度 (alpha_tilde_k, 参考 Anderson CFD 教材, 基于dU的投影)
    # These are coefficients when dU is expanded in terms of right eigenvectors of Roe matrix
    # (dU = sum_k alpha_tilde_k * R_hat_k, where R_hat_k are specific forms of eigenvectors)
    alpha_2_tilde = (gamma_val-1)/a_hat**2 * \
                    ( (H_hat - u_hat**2)*dU[0] + u_hat*dU[1] - dU[2] )
    alpha_1_tilde = ( dU[0]*(u_hat+a_hat) - dU[1] - a_hat*alpha_2_tilde ) / (2*a_hat)
    alpha_3_tilde = dU[0] - alpha_1_tilde - alpha_2_tilde

    # Contributions of each wave to dU (alpha_tilde_k * R_hat_k)
    # R_hat_k (eigenvectors) for [d_rho, d_rhou, d_E] are:
    # R_hat_1_col = [1, u_hat - a_hat, H_hat - u_hat * a_hat]
    # R_hat_2_col = [1, u_hat,         0.5 * u_hat**2     ]
    # R_hat_3_col = [1, u_hat + a_hat, H_hat + u_hat * a_hat]
    # 各波对dU的贡献 (alpha_tilde_k * R_hat_k)
    
    term1 = np.zeros_like(dU) # Contribution from (u-a) wave / (u-a)波的贡献
    term2 = np.zeros_like(dU) # Contribution from (u) wave / (u)波的贡献
    term3 = np.zeros_like(dU) # Contribution from (u+a) wave / (u+a)波的贡献

    # Wave 1 (u-a)
    term1[0,:] = alpha_1_tilde
    term1[1,:] = alpha_1_tilde * (u_hat - a_hat)
    term1[2,:] = alpha_1_tilde * (H_hat - u_hat * a_hat)
    # Wave 2 (u)
    term2[0,:] = alpha_2_tilde
    term2[1,:] = alpha_2_tilde * u_hat
    term2[2,:] = alpha_2_tilde * 0.5 * u_hat**2
    # Wave 3 (u+a)
    term3[0,:] = alpha_3_tilde
    term3[1,:] = alpha_3_tilde * (u_hat + a_hat)
    term3[2,:] = alpha_3_tilde * (H_hat + u_hat * a_hat)

    # Entropy fix (Harten-Hyman type for |lambda_k|) / 熵修正 (Harten-Hyman类型, 作用于|lambda_k|)
    # This prevents non-physical expansion shocks / 防止非物理的膨胀激波
    epsilon_roe = 0.1 # Parameter for entropy fix strength / 熵修正强度参数
    delta_k = epsilon_roe * a_hat # Threshold for applying entropy fix / 应用熵修正的阈值
    
    abs_lambda_fixed = np.abs(lambda_hat) # |lambda_k|
    for k_wave in range(3): # For each eigenvalue / 对每个特征值
        idx_fix = abs_lambda_fixed[k_wave,:] < delta_k # Check if eigenvalue is too small / 检查特征值是否过小
        # Apply Harten's entropy fix where needed / 在需要的地方应用Harten熵修正
        abs_lambda_fixed[k_wave, idx_fix] = (lambda_hat[k_wave, idx_fix]**2 + delta_k[idx_fix]**2) / (2 * delta_k[idx_fix])
    
    # Numerical dissipation term: sum_k |lambda_k|_fixed * (alpha_tilde_k * R_hat_k)
    # 数值耗散项: sum_k |lambda_k|_fixed * (alpha_tilde_k * R_hat_k)
    dissipation = abs_lambda_fixed[0] * term1 + \
                  abs_lambda_fixed[1] * term2 + \
                  abs_lambda_fixed[2] * term3
                  
    # Roe numerical flux: F_Roe = 0.5 * (F_L + F_R) - 0.5 * dissipation
    # Roe 数值通量
    F_numerical = 0.5 * (F_L + F_R) - 0.5 * dissipation
    return F_numerical

# --- Boundary Conditions ---
# --- 边界条件 ---
def apply_boundary_conditions(U_internal, num_ghost, nx_domain):
    """Applies outflow (zeroth-order extrapolation) boundary conditions."""
    """应用出流边界条件 (零阶外插)。"""
    U_with_ghost = np.zeros((3, nx_domain + 2 * num_ghost)) # Array with ghost cells / 带虚拟单元的数组
    U_with_ghost[:, num_ghost : num_ghost + nx_domain] = U_internal # Fill internal cells / 填充内部单元

    # Outflow: copy values from the nearest internal cell to ghost cells
    # 出流: 将最近的内部单元值复制到虚拟单元
    for i in range(num_ghost):
        U_with_ghost[:, i] = U_internal[:, 0]           # Left boundary / 左边界
        U_with_ghost[:, -(i + 1)] = U_internal[:, -1]   # Right boundary / 右边界
    return U_with_ghost

# --- RHS Calculation ---
# --- 右端项 (dU/dt) 计算 ---
def calculate_rhs(U_current_internal, dx_val, reconstruction_method_name, flux_method_name,
                  limiter_name=None, n_ghost_cells=N_GHOST, nx_val=NX, gamma_val=gamma):
    """
    Calculates the right-hand side of dU/dt = - (F_{j+1/2} - F_{j-1/2}) / dx.
    计算 dU/dt = - (F_{j+1/2} - F_{j-1/2}) / dx 的右端项。
    """
    
    # 1. Apply BCs to get U_gh (conserved variables with ghost cells)
    #    应用边界条件得到 U_gh (带虚拟单元的守恒变量)
    U_gh = apply_boundary_conditions(U_current_internal, n_ghost_cells, nx_val)
    
    # 2. Convert all cells (including ghost) to primitive variables P_gh
    #    将所有单元(包括虚拟单元)转换为原始变量 P_gh
    P_gh = conserved_to_primitive(U_gh, gamma_val)
    
    # 3. Reconstruction to get P_L_inter, P_R_inter (primitive states at interfaces)
    #    重构得到界面处的原始状态 P_L_inter, P_R_inter
    if reconstruction_method_name == 'MUSCL':
        if limiter_name is None or limiter_name not in LIMITERS:
            raise ValueError(f"Invalid or missing limiter for MUSCL: {limiter_name}")
        limiter_func = LIMITERS[limiter_name]
        P_L_inter, P_R_inter = muscl_reconstruction(P_gh, limiter_func, n_ghost_cells, nx_val)
    elif reconstruction_method_name == 'MUSCL_CHAR': # New option with characteristic limiting / 特征变量限制的新选项
        if limiter_name is None or limiter_name not in LIMITERS:
            raise ValueError(f"Invalid or missing limiter for MUSCL_CHAR: {limiter_name}")
        limiter_func = LIMITERS[limiter_name]
        P_L_inter, P_R_inter = muscl_char_reconstruction(P_gh, limiter_func, n_ghost_cells, nx_val, gamma_val)
    elif reconstruction_method_name == 'WENO3':
        P_L_inter, P_R_inter = weno3_reconstruction(P_gh, n_ghost_cells, nx_val)
    else:
        raise ValueError(f"Unknown reconstruction method: {reconstruction_method_name}")

    # Enforce positivity for reconstructed primitive states at interfaces
    # 对界面处重构的原始状态强制正值
    P_L_inter[0,:] = np.maximum(P_L_inter[0,:], 1e-9) # rho_L
    P_L_inter[2,:] = np.maximum(P_L_inter[2,:], 1e-9) # p_L
    P_R_inter[0,:] = np.maximum(P_R_inter[0,:], 1e-9) # rho_R
    P_R_inter[2,:] = np.maximum(P_R_inter[2,:], 1e-9) # p_R

    # 4. Numerical Flux Calculation at interfaces F_numerical_at_interfaces
    #    计算界面处的数值通量 F_numerical_at_interfaces
    if flux_method_name == 'FVS_VanLeer':
        F_numerical_at_interfaces = van_leer_fvs_flux(P_L_inter, P_R_inter, gamma_val)
    elif flux_method_name == 'FDS_Roe':
        F_numerical_at_interfaces = roe_fds_flux(P_L_inter, P_R_inter, gamma_val)
    else:
        raise ValueError(f"Unknown flux method: {flux_method_name}")
        
    # 5. Compute dU/dt for internal cells using finite difference of fluxes
    #    使用通量的有限差分计算内部单元的 dU/dt
    #    dU/dt = - (F_{j+1/2} - F_{j-1/2}) / dx
    rhs_U = -(F_numerical_at_interfaces[:, 1:] - F_numerical_at_interfaces[:, :-1]) / dx_val
    return rhs_U

# --- Exact Sod Solver (Based on Toro's book, Chapter 4) ---
# --- Sod 问题精确解 (基于 Toro 教材第四章) ---
def exact_sod_solution(x_points, t, gamma_val, rho_L, u_L, p_L, rho_R, u_R, p_R):
    """
    Calculates the exact solution of the 1D Euler Riemann problem.
    (Sod problem is a specific instance of a Riemann problem).

    Args:
        x_points (np.ndarray): Array of x-coordinates where the solution is sought. / 求解位置的x坐标数组
        t (float): Time at which the solution is sought. / 求解时间
        gamma_val (float): Adiabatic index. / 绝热指数
        rho_L, u_L, p_L: Initial left state (density, velocity, pressure). / 初始左状态
        rho_R, u_R, p_R: Initial right state. / 初始右状态

    Returns:
        tuple: (rho_sol, u_sol, p_sol, E_sol) / (密度解, 速度解, 压力解, 总能量解)
    """

    # Initial sound speeds in left and right states / 左右状态的初始声速
    a_L = np.sqrt(gamma_val * p_L / rho_L)
    a_R = np.sqrt(gamma_val * p_R / rho_R)

    # Handle t=0 case (initial discontinuity) / 处理 t=0 的情况 (初始间断)
    if t == 0:
        rho_sol = np.where(x_points < 0, rho_L, rho_R) # Diaphragm at x=0 / 膜片在 x=0
        u_sol = np.where(x_points < 0, u_L, u_R)
        p_sol = np.where(x_points < 0, p_L, p_R)
        E_sol = p_sol / (gamma_val - 1.0) + 0.5 * rho_sol * u_sol**2
        return rho_sol, u_sol, p_sol, E_sol

    # Function f_K(p_star, p_K, rho_K, a_K) from Toro's book (eq. 4.5, 4.8)
    # This function relates p_star to u_star via either shock or rarefaction relations.
    # Toro 书中 (eq. 4.5, 4.8) 的函数 f_K(p_star, p_K, rho_K, a_K)
    # 该函数通过激波或稀疏波关系将 p_star 与 u_star 联系起来。
    def shock_tube_relations_func(p_star_val, p_K, rho_K, a_K): # K is L or R state / K代表左或右状态
        if p_star_val > p_K:  # Shock wave case / 激波情况 (Toro eq. 4.7)
            # Returns |u_star - u_K| for a shock
            A_K = 2.0 / ((gamma_val + 1.0) * rho_K)
            B_K = (gamma_val - 1.0) / (gamma_val + 1.0) * p_K
            val = (p_star_val - p_K) * np.sqrt(A_K / (p_star_val + B_K))
        else:  # Rarefaction wave case / 稀疏波情况 (Toro eq. 4.8)
            # Returns u_star - u_K for left rarefaction, or -(u_star - u_K) for right rarefaction
            # if f_K is defined appropriately. Toro's f_L and f_R definition (p.128) makes this work.
            val = (2.0 * a_K / (gamma_val - 1.0)) * \
                  ( (p_star_val / p_K)**((gamma_val - 1.0) / (2.0 * gamma_val)) - 1.0 )
        return val

    # Pressure function for root finding: f(p_star) = 0 (Toro eq. 4.6)
    # f(p_star) = f_L(p_star) + f_R(p_star) + (u_R - u_L)
    # 用于求根的压力函数: f(p_star) = 0
    def pressure_func_root(p_star_guess):
        f_L_val = shock_tube_relations_func(p_star_guess, p_L, rho_L, a_L) # Contribution from left wave
        f_R_val = shock_tube_relations_func(p_star_guess, p_R, rho_R, a_R) # Contribution from right wave
        return f_L_val + f_R_val + (u_R - u_L)

    # Initial guess for p_star (pressure in the star region) / p_star (星区压力) 的初始猜测值
    p_star_guess = 0.5 * (p_L + p_R) 
    if p_star_guess <= 0: # Ensure guess is positive / 确保猜测值为正
        p_star_guess = 1e-6

    # Solve for p_star using a numerical root finder (fsolve) / 使用数值求根器 (fsolve) 求解 p_star
    try:
        p_star = fsolve(pressure_func_root, p_star_guess, xtol=1e-12)[0]
        if p_star < 0: # Check for non-physical pressure / 检查非物理压力
            raise ValueError("Solver returned negative pressure for p_star.")
    except Exception as e: # If fsolve fails, try a more robust guess / 如果 fsolve 失败，尝试更稳健的猜测值
        print(f"Warning: fsolve failed with initial p_star guess. Error: {e}. Trying alternative guess.")
        # Alternative guess (e.g., two-shock approximation or PVRS from Toro)
        # For Sod, simpler guesses usually work, but this adds robustness.
        p_pvrs_num = (a_L + a_R - 0.5*(gamma_val-1.0)*(u_R-u_L))
        p_pvrs_den = ( a_L/(p_L**((gamma_val-1.0)/(2.0*gamma_val))) + \
                       a_R/(p_R**((gamma_val-1.0)/(2.0*gamma_val))) )
        p_star_guess_alt = (p_pvrs_num / p_pvrs_den)**( (2.0*gamma_val)/(gamma_val-1.0) ) # Toro eq. 4.47 (PVRS)
        p_star_guess_alt = max(1e-6, p_star_guess_alt) # Ensure positive
        try:
            p_star = fsolve(pressure_func_root, p_star_guess_alt, xtol=1e-12)[0]
            if p_star < 0:
                 raise ValueError("Solver returned negative pressure for p_star even with alternative guess.")
        except Exception as e_alt:
             print(f"Critical: fsolve failed with alternative guess for p_star. Error: {e_alt}")
             raise ValueError("Could not determine p_star for exact solution.") from e_alt

    # Calculate u_star (velocity in the star region) using p_star (Toro eq. 4.9)
    # 使用 p_star 计算 u_star (星区速度)
    f_L_at_p_star = shock_tube_relations_func(p_star, p_L, rho_L, a_L)
    f_R_at_p_star = shock_tube_relations_func(p_star, p_R, rho_R, a_R)
    u_star = 0.5 * (u_L + u_R) + 0.5 * (f_R_at_p_star - f_L_at_p_star)

    # Densities in the star region (rho_star_L and rho_star_R) / 星区密度
    if p_star > p_L:  # Left shock (Toro eq. 4.50) / 左激波
        rho_star_L = rho_L * ( (p_star / p_L + (gamma_val - 1.0) / (gamma_val + 1.0)) / \
                               ( (gamma_val - 1.0) / (gamma_val + 1.0) * (p_star / p_L) + 1.0) )
    else:  # Left rarefaction (Toro eq. 4.51) / 左稀疏波 (等熵关系)
        rho_star_L = rho_L * (p_star / p_L)**(1.0 / gamma_val)
        
    if p_star > p_R:  # Right shock (Toro eq. 4.57) / 右激波
        rho_star_R = rho_R * ( (p_star / p_R + (gamma_val - 1.0) / (gamma_val + 1.0)) / \
                               ( (gamma_val - 1.0) / (gamma_val + 1.0) * (p_star / p_R) + 1.0) )
    else:  # Right rarefaction (Toro eq. 4.58) / 右稀疏波 (等熵关系)
        rho_star_R = rho_R * (p_star / p_R)**(1.0 / gamma_val)

    # Wave speeds / 波速
    S_C = u_star  # Contact discontinuity speed / 接触间断速度 (Toro eq. 4.32)

    # Determine speeds for left-traveling wave / 确定左行波的速度
    if p_star > p_L:  # Left wave is a shock / 左波是激波
        S_L_shock = u_L - a_L * np.sqrt( ((gamma_val + 1.0) / (2.0 * gamma_val)) * (p_star / p_L) + \
                                         ((gamma_val - 1.0) / (2.0 * gamma_val)) ) # Toro eq. 4.52
    else:  # Left wave is a rarefaction / 左波是稀疏波
        a_star_L = a_L * (p_star / p_L)**((gamma_val - 1.0) / (2.0 * gamma_val)) # Sound speed at star region edge / 星区边缘声速 (Toro eq. 4.53)
        S_HL_raref = u_L - a_L           # Head of left rarefaction / 左稀疏波头部速度 (Toro eq. 4.54)
        S_TL_raref = u_star - a_star_L   # Tail of left rarefaction / 左稀疏波尾部速度 (Toro eq. 4.55)

    # Determine speeds for right-traveling wave / 确定右行波的速度
    if p_star > p_R:  # Right wave is a shock / 右波是激波
        S_R_shock = u_R + a_R * np.sqrt( ((gamma_val + 1.0) / (2.0 * gamma_val)) * (p_star / p_R) + \
                                         ((gamma_val - 1.0) / (2.0 * gamma_val)) ) # Toro eq. 4.59
    else:  # Right wave is a rarefaction / 右波是稀疏波
        a_star_R = a_R * (p_star / p_R)**((gamma_val - 1.0) / (2.0 * gamma_val)) # Sound speed at star region edge / 星区边缘声速 (Toro eq. 4.60)
        S_HR_raref = u_R + a_R           # Head of right rarefaction / 右稀疏波头部速度 (Toro eq. 4.61)
        S_TR_raref = u_star + a_star_R   # Tail of right rarefaction / 右稀疏波尾部速度 (Toro eq. 4.62)
        
    # Solution arrays to be filled / 待填充的解数组
    rho_sol = np.zeros_like(x_points, dtype=float)
    u_sol = np.zeros_like(x_points, dtype=float)
    p_sol = np.zeros_like(x_points, dtype=float)

    # Sample solution at each x_point based on s = x/t / 根据 s = x/t 在每个 x_point 处采样解
    # The structure of the solution depends on the wave pattern (shock/rarefaction on left/right)
    # 解的结构取决于波型 (左/右是激波还是稀疏波)
    for i, x_val in enumerate(x_points):
        s_query = x_val / t  # Query speed relative to diaphragm origin / 查询点相对于膜片原点的速度

        if s_query <= S_C:  # Point is to the left of or at the contact discontinuity / 点在接触间断左侧或接触间断上
            if p_star > p_L:  # Left wave is a SHOCK / 左波是激波
                if s_query <= S_L_shock:  # Region 1 (Undisturbed Left State) / 区域1 (未扰动左状态)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_L, u_L, p_L
                else:  # Region 3 (Star Region Left - between shock and contact) / 区域3 (左星区 - 激波与接触间断之间)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_star_L, u_star, p_star
            else:  # Left wave is a RAREFACTION / 左波是稀疏波
                if s_query <= S_HL_raref:  # Region 1 (Undisturbed Left State) / 区域1 (未扰动左状态)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_L, u_L, p_L
                elif s_query <= S_TL_raref:  # Region 2 (Inside Left Rarefaction Fan, Toro eq. 4.56) / 区域2 (左稀疏扇内部)
                    # u_fan = (2/(g+1)) * (a_L + (g-1)/2 * u_L + s_query)
                    u_sol[i] = (2.0 / (gamma_val + 1.0)) * (a_L + (gamma_val - 1.0) / 2.0 * u_L + s_query)
                    # rho_fan = rho_L * [ (2/(g+1)) + ((g-1)/((g+1)*a_L)) * (u_L - s_query) ]^(2/(g-1))
                    # p_fan = p_L * (rho_fan/rho_L)^g
                    common_factor = (2.0 / (gamma_val + 1.0)) + \
                                    ((gamma_val - 1.0) / ((gamma_val + 1.0) * a_L)) * (u_L - s_query)
                    rho_sol[i] = rho_L * common_factor**(2.0 / (gamma_val - 1.0))
                    p_sol[i] = p_L * common_factor**(2.0 * gamma_val / (gamma_val - 1.0))
                else:  # Region 3 (Star Region Left - between rarefaction tail and contact) / 区域3 (左星区 - 稀疏波尾与接触间断之间)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_star_L, u_star, p_star
        else:  # Point is to the right of the contact discontinuity (s_query > S_C) / 点在接触间断右侧
            if p_star > p_R:  # Right wave is a SHOCK / 右波是激波
                if s_query >= S_R_shock:  # Region 5 (Undisturbed Right State) / 区域5 (未扰动右状态)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_R, u_R, p_R
                else:  # Region 4 (Star Region Right - between contact and shock) / 区域4 (右星区 - 接触间断与激波之间)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_star_R, u_star, p_star
            else:  # Right wave is a RAREFACTION / 右波是稀疏波
                if s_query >= S_HR_raref:  # Region 5 (Undisturbed Right State) / 区域5 (未扰动右状态)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_R, u_R, p_R
                elif s_query >= S_TR_raref:  # Region 4b (Inside Right Rarefaction Fan, Toro eq. 4.63) / 区域4b (右稀疏扇内部)
                    # u_fan = (2/(g+1)) * (-a_R + (g-1)/2 * u_R + s_query)
                    u_sol[i] = (2.0 / (gamma_val + 1.0)) * (-a_R + (gamma_val - 1.0) / 2.0 * u_R + s_query)
                    # rho_fan = rho_R * [ (2/(g+1)) - ((g-1)/((g+1)*a_R)) * (u_R - s_query) ]^(2/(g-1))
                    # p_fan = p_R * (rho_fan/rho_R)^g
                    common_factor = (2.0 / (gamma_val + 1.0)) - \
                                    ((gamma_val - 1.0) / ((gamma_val + 1.0) * a_R)) * (u_R - s_query)
                    rho_sol[i] = rho_R * common_factor**(2.0 / (gamma_val - 1.0))
                    p_sol[i] = p_R * common_factor**(2.0 * gamma_val / (gamma_val - 1.0))
                else:  # Region 4a (Star Region Right - between contact and rarefaction tail) / 区域4a (右星区 - 接触间断与稀疏波尾之间)
                    rho_sol[i], u_sol[i], p_sol[i] = rho_star_R, u_star, p_star
                    
    # Calculate total energy per unit volume E = p/(gamma-1) + 0.5*rho*u^2 / 计算单位体积总能量
    E_sol = p_sol / (gamma_val - 1.0) + 0.5 * rho_sol * u_sol**2
    
    return rho_sol, u_sol, p_sol, E_sol

# --- Main Simulation Runner for a Single Configuration ---
# --- 单个配置的主模拟运行器 ---
def run_simulation_single(U_init_sim, t_final_sim, dx_sim, cfl_val, nx_val,
                          reconstruction_method, flux_method, limiter=None, gamma_val=gamma):
    """Runs the simulation for one combination of schemes."""
    """为一种格式组合运行模拟。"""
    U = np.copy(U_init_sim) # Current conserved variables / 当前守恒变量
    t_curr = 0.0            # Current time / 当前时间
    iter_count = 0          # Iteration counter / 迭代计数器

    while t_curr < t_final_sim:
        # Calculate dt based on CFL condition / 根据CFL条件计算dt
        rho_curr_iter, u_curr_iter, p_curr_iter = conserved_to_primitive(U, gamma_val)
        # Sound speed (abs for safety, though p,rho should be positive)
        # 声速 (为安全使用abs，尽管p,rho应为正)
        a_curr_iter = np.sqrt(gamma_val * np.abs(p_curr_iter) / np.abs(rho_curr_iter)) 
        max_speed = np.max(np.abs(u_curr_iter) + a_curr_iter) # Max characteristic speed / 最大特征速度
        
        dt_val = cfl_val * dx_sim / max_speed if max_speed > 1e-9 else cfl_val * dx_sim # Time step / 时间步长
        
        # Adjust dt if it overshoots t_final_sim / 如果dt超过t_final_sim，则调整dt
        if t_curr + dt_val > t_final_sim:
            dt_val = t_final_sim - t_curr
        
        if dt_val <= 1e-12: # Avoid infinitely small dt if t_curr is already t_final_sim
                            # 如果t_curr已达到t_final_sim，避免无限小dt
            break

        # 3rd Order Runge-Kutta (SSP-RK3) time integration / 3阶龙格-库塔 (SSP-RK3) 时间积分
        # U* = U^n + dt * RHS(U^n)
        rhs1 = calculate_rhs(U, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U_1 = U + dt_val * rhs1
        
        # U** = (3/4)U^n + (1/4)U* + (1/4)dt * RHS(U*)
        rhs2 = calculate_rhs(U_1, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U_2 = 0.75 * U + 0.25 * U_1 + 0.25 * dt_val * rhs2
        
        # U^{n+1} = (1/3)U^n + (2/3)U** + (2/3)dt * RHS(U**)
        rhs3 = calculate_rhs(U_2, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U = (1.0/3.0) * U + (2.0/3.0) * U_2 + (2.0/3.0) * dt_val * rhs3
        
        t_curr += dt_val
        iter_count += 1
        # Optional: Print progress / 可选: 打印进度
        # if iter_count % 200 == 0: 
        #     print(f"  Iter: {iter_count}, Time: {t_curr:.4f}/{t_final_sim:.4f}, dt: {dt_val:.2e}")
    return U

def get_eigenvectors_primitive(rho, u, p, gamma_val=gamma):
    """
    Calculates Left (L) and Right (R) eigenvector matrices for the 1D Euler
    equations in primitive variables Q = [rho, u, p].
    L transforms dQ to dW (characteristic variable changes). dW = L @ dQ
    R transforms dW to dQ. dQ = R @ dW
    L @ R = I (identity matrix)
    Eigenvalues: lambda_1 = u-a, lambda_2 = u, lambda_3 = u+a
    
    计算一维欧拉方程原始变量 Q = [rho, u, p] 的左(L)和右(R)特征向量矩阵。
    L 将 dQ 转换为 dW (特征变量变化)。
    R 将 dW 转换为 dQ。
    L @ R = I (单位矩阵)
    特征值: lambda_1 = u-a, lambda_2 = u, lambda_3 = u+a
    """
    if rho < 1e-9 or p < 1e-9: # Fallback for unphysical states / 非物理状态的备用方案
        a = 1e-3 # Small positive sound speed / 小正声速
    else:
        a = np.sqrt(gamma_val * p / rho) # Sound speed / 声速
    
    # L matrix (rows are left eigenvectors) / L矩阵 (行为左特征向量)
    # dW[0] (from u-a wave): (dp - rho*a*du) / (2*a^2)
    # dW[1] (from u wave):   drho - dp/a^2
    # dW[2] (from u+a wave): (dp + rho*a*du) / (2*a^2)
    # Normalization can vary; this one matches common forms where L R = I.
    L = np.array([
        [0,          -rho / (2 * a),   1 / (2 * a**2)], # Corresponds to u-a eigenvalue
        [1,           0,              -1 / a**2     ], # Corresponds to u eigenvalue
        [0,           rho / (2 * a),   1 / (2 * a**2)]  # Corresponds to u+a eigenvalue
    ])

    # R matrix (columns are right eigenvectors) / R矩阵 (列为右特征向量)
    # Column 1 (for u-a): [1, -a/rho, a^2]^T
    # Column 2 (for u):   [1,  0,     0  ]^T (Careful, some sources use [0,1,0] if basis is different)
    #                     This form for R matches the L above for L@R=I
    # Column 3 (for u+a): [1,  a/rho, a^2]^T
    R = np.array([
        [1,           1,          1          ], # drho component
        [-a / rho,    0,          a / rho    ], # du component
        [a**2,        0,          a**2       ]  # dp component
    ])
    return L, R

# --- MUSCL Reconstruction with Characteristic Limiting ---
# --- 带特征变量限制的 MUSCL 重构 ---
def muscl_char_reconstruction(P_gh, limiter_func, n_ghost, nx_domain, gamma_val=gamma):
    """
    MUSCL reconstruction using characteristic variable limiting.
    Slopes are computed and limited in characteristic space, then projected back.
    P_gh: primitive variables (rho, u, p) with ghost cells (3, nx_total) / 带虚拟单元的原始变量
    limiter_func: the chosen TVD flux limiter function / 选择的TVD通量限制器函数
    Returns P_L_inter, P_R_inter (3, nx_domain + 1) / 返回界面左右状态
    """
    P_L_at_interfaces = np.zeros((3, nx_domain + 1)) # 界面左侧状态
    P_R_at_interfaces = np.zeros((3, nx_domain + 1)) # 界面右侧状态
    epsilon_slope = 1e-12 # For r_char denominator, to avoid division by zero / r_char分母的epsilon, 防零除

    for j_inter in range(nx_domain + 1): # Loop over nx_domain+1 interfaces / 遍历界面
        # --- Left state for interface j_inter (reconstructed from cell idx_cell_L) ---
        # --- 界面 j_inter 的左状态 (从单元 idx_cell_L 重构) ---
        idx_cell_L = n_ghost + j_inter - 1 # Cell i, to the left of interface i+1/2
        
        # Primitive variables for the stencil around cell L (cell i)
        # 单元 L (即单元 i) 周围模板的原始变量
        Q_L_m1 = P_gh[:, idx_cell_L - 1] # Q_{i-1}
        Q_L_0  = P_gh[:, idx_cell_L]     # Q_i (state used for averaging eigenvectors) / 用于平均特征向量的状态
        Q_L_p1 = P_gh[:, idx_cell_L + 1] # Q_{i+1}

        # Get eigenvectors based on state Q_L_0 (cell i) / 基于状态 Q_L_0 (单元 i) 获取特征向量
        L_avg_L, R_avg_L = get_eigenvectors_primitive(Q_L_0[0], Q_L_0[1], Q_L_0[2], gamma_val)

        # Differences in primitive variables for cell i / 单元 i 的原始变量差分
        dQ_minus_prim_L = Q_L_0 - Q_L_m1 # q_i - q_{i-1} (slope "before" cell i)
        dQ_plus_prim_L  = Q_L_p1 - Q_L_0  # q_{i+1} - q_i (slope "after" cell i)
        
        # Project to characteristic differences: dW = L @ dQ / 投影到特征差分
        dW_minus_L = L_avg_L @ dQ_minus_prim_L # Characteristic slope "before"
        dW_plus_L  = L_avg_L @ dQ_plus_prim_L  # Characteristic slope "after"

        # Apply limiter to each characteristic field / 对每个特征场应用限制器
        limited_char_slope_L = np.zeros(3) # This will be 0.5 * phi(r_char) * dW_minus_L
        for k_char in range(3): # For each characteristic variable / 对每个特征变量
            r_char_L_num = dW_plus_L[k_char] # Numerator for r_char for cell i
            r_char_L_den = dW_minus_L[k_char] # Denominator for r_char for cell i
            
            if np.abs(r_char_L_den) < epsilon_slope: # Avoid division by zero
                r_char_L = 2.0 if r_char_L_num * r_char_L_den >=0 else -2.0 # Default for zero denominator
            else:
                r_char_L = r_char_L_num / r_char_L_den
            
            phi_char_L = limiter_func(r_char_L) # Limiter function value
            # Limited characteristic slope component for extrapolation from cell i
            limited_char_slope_L[k_char] = 0.5 * phi_char_L * dW_minus_L[k_char] 
            
        # Project limited characteristic slope back to primitive space: dQ_limited = R @ (0.5 * phi * dW_minus)
        # 将限制后的特征斜率投影回原始空间
        dQ_limited_slope_L = R_avg_L @ limited_char_slope_L
        # P_L = Q_i + dQ_limited_slope (extrapolation to i+1/2 interface)
        P_L_at_interfaces[:, j_inter] = Q_L_0 + dQ_limited_slope_L

        # --- Right state for interface j_inter (reconstructed from cell idx_cell_R) ---
        # --- 界面 j_inter 的右状态 (从单元 idx_cell_R 重构) ---
        idx_cell_R = n_ghost + j_inter # Cell i+1 (or j), to the right of interface i+1/2

        # Primitive variables for stencil around cell R (cell i+1 or j)
        # 单元 R (即单元 i+1 或 j) 周围模板的原始变量
        Q_R_m1 = P_gh[:, idx_cell_R - 1] # Q_{j-1} (or Q_i)
        Q_R_0  = P_gh[:, idx_cell_R]     # Q_j (or Q_{i+1}) (state used for averaging eigenvectors)
        Q_R_p1 = P_gh[:, idx_cell_R + 1] # Q_{j+1} (or Q_{i+2})

        # Eigenvectors based on state Q_R_0 (cell j or i+1)
        L_avg_R, R_avg_R = get_eigenvectors_primitive(Q_R_0[0], Q_R_0[1], Q_R_0[2], gamma_val)
        
        # Differences in primitive variables for cell j (or i+1)
        dQ_minus_prim_R = Q_R_0 - Q_R_m1 # q_j - q_{j-1} (slope "before" cell j)
        dQ_plus_prim_R  = Q_R_p1 - Q_R_0  # q_{j+1} - q_j (slope "after" cell j)
        
        # Project to characteristic differences
        dW_minus_R = L_avg_R @ dQ_minus_prim_R # Characteristic slope "before" cell j
        dW_plus_R  = L_avg_R @ dQ_plus_prim_R  # Characteristic slope "after" cell j
        
        limited_char_slope_R = np.zeros(3) # This will be 0.5 * phi(r_char) * dW_minus_R
        for k_char in range(3):
            r_char_R_num = dW_plus_R[k_char] # Numerator for r_char for cell j
            r_char_R_den = dW_minus_R[k_char] # Denominator for r_char for cell j
            
            if np.abs(r_char_R_den) < epsilon_slope:
                r_char_R = 2.0 if r_char_R_num * r_char_R_den >= 0 else -2.0
            else:
                r_char_R = r_char_R_num / r_char_R_den

            phi_char_R = limiter_func(r_char_R) # Limiter function value
            # Limited characteristic slope component for extrapolation from cell j
            # The slope dW_minus_R is used for extrapolation from cell Q_R_0 (cell j)
            limited_char_slope_R[k_char] = 0.5 * phi_char_R * dW_minus_R[k_char]
            
        # Project limited characteristic slope back to primitive space
        dQ_limited_slope_R = R_avg_R @ limited_char_slope_R
        # P_R = Q_j - dQ_limited_slope (extrapolation to j-1/2 interface, which is i+1/2)
        P_R_at_interfaces[:, j_inter] = Q_R_0 - dQ_limited_slope_R 
        
    return P_L_at_interfaces, P_R_at_interfaces


# --- Modified calculate_rhs (to include MUSCL_CHAR) ---
# --- 修改后的 calculate_rhs (以包含 MUSCL_CHAR) ---
def calculate_rhs(U_current_internal, dx_val, reconstruction_method_name, flux_method_name,
                  limiter_name=None, n_ghost_cells=N_GHOST, nx_val=NX, gamma_val=gamma):
    
    U_gh = apply_boundary_conditions(U_current_internal, n_ghost_cells, nx_val)
    P_gh = conserved_to_primitive(U_gh, gamma_val)
    
    if reconstruction_method_name == 'MUSCL': # Component-wise MUSCL / 分量MUSCL
        if limiter_name is None or limiter_name not in LIMITERS:
            raise ValueError(f"Invalid or missing limiter for MUSCL: {limiter_name}")
        limiter_func = LIMITERS[limiter_name]
        P_L_inter, P_R_inter = muscl_reconstruction(P_gh, limiter_func, n_ghost_cells, nx_val)
    elif reconstruction_method_name == 'MUSCL_CHAR': # MUSCL with characteristic limiting / 带特征限制的MUSCL
        if limiter_name is None or limiter_name not in LIMITERS:
            raise ValueError(f"Invalid or missing limiter for MUSCL_CHAR: {limiter_name}")
        limiter_func = LIMITERS[limiter_name]
        P_L_inter, P_R_inter = muscl_char_reconstruction(P_gh, limiter_func, n_ghost_cells, nx_val, gamma_val)
    elif reconstruction_method_name == 'WENO3':
        P_L_inter, P_R_inter = weno3_reconstruction(P_gh, n_ghost_cells, nx_val)
    else:
        raise ValueError(f"Unknown reconstruction method: {reconstruction_method_name}")

    # Ensure positivity of reconstructed states at interfaces / 确保界面重构状态的正性
    P_L_inter[0,:] = np.maximum(P_L_inter[0,:], 1e-9); P_L_inter[2,:] = np.maximum(P_L_inter[2,:], 1e-9)
    P_R_inter[0,:] = np.maximum(P_R_inter[0,:], 1e-9); P_R_inter[2,:] = np.maximum(P_R_inter[2,:], 1e-9)

    # Calculate numerical flux at interfaces / 计算界面数值通量
    if flux_method_name == 'FVS_VanLeer':
        F_numerical_at_interfaces = van_leer_fvs_flux(P_L_inter, P_R_inter, gamma_val)
    elif flux_method_name == 'FDS_Roe':
        F_numerical_at_interfaces = roe_fds_flux(P_L_inter, P_R_inter, gamma_val)
    else:
        raise ValueError(f"Unknown flux method: {flux_method_name}")
        
    # Compute RHS: dU/dt = - (F_j+1/2 - F_j-1/2) / dx
    rhs_U = -(F_numerical_at_interfaces[:, 1:] - F_numerical_at_interfaces[:, :-1]) / dx_val
    return rhs_U

# --- Main Execution (add new scheme combinations) ---
# --- 主执行程序 (添加新的格式组合) ---
if __name__ == "__main__":
    # Initial condition setup / 初始条件设置
    U_initial = np.zeros((3, NX)) # Array for initial conserved variables / 初始守恒变量数组
    U_L_cons_init = primitive_to_conserved(rho_L_init, u_L_init, p_L_init, gamma) # Left conserved state
    U_R_cons_init = primitive_to_conserved(rho_R_init, u_R_init, p_R_init, gamma) # Right conserved state
    # Populate initial condition based on diaphragm at x=0 / 根据x=0处的膜片填充初始条件
    for i in range(NX):
        if X_CELL_CENTERS[i] < 0:
            U_initial[:, i] = U_L_cons_init
        else:
            U_initial[:, i] = U_R_cons_init

    # Get exact solution for plotting comparison / 获取精确解用于绘图比较
    x_exact_plot = np.linspace(XMIN, XMAX, 500) # Points for plotting exact solution / 精确解绘图点
    rho_ex, u_ex, p_ex, e_ex_cons = exact_sod_solution(x_exact_plot, T_FINAL, gamma,
                                                  rho_L_init, u_L_init, p_L_init,
                                                  rho_R_init, u_R_init, p_R_init)
    # For plotting, often specific internal energy e = E_total_per_mass - 0.5*u^2 = p/((gamma-1)*rho) is used
    # 绘图时常用比内能 e
    e_ex_specific = p_ex / ((gamma - 1.0) * rho_ex) # Specific internal energy from exact solution

    # Define schemes to test / 定义要测试的格式
    schemes_to_test = [
        # Previous schemes / 先前的格式
        {'recon': 'MUSCL', 'flux': 'FVS_VanLeer', 'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FVS(VL)'},
        {'recon': 'MUSCL', 'flux': 'FVS_VanLeer', 'limiter': 'Superbee', 'label': 'MUSCL(SB)-FVS(VL)'},
        {'recon': 'WENO3', 'flux': 'FVS_VanLeer', 'limiter': None,       'label': 'WENO3-FVS(VL)'},
        {'recon': 'MUSCL', 'flux': 'FDS_Roe',     'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FDS(Roe)'},
        {'recon': 'MUSCL', 'flux': 'FDS_Roe',     'limiter': 'Superbee', 'label': 'MUSCL(SB)-FDS(Roe)'},
        {'recon': 'WENO3', 'flux': 'FDS_Roe',     'limiter': None,       'label': 'WENO3-FDS(Roe)'},
        # New schemes with Characteristic Reconstruction for FVS / 使用特征重构的FVS新格式
        {'recon': 'MUSCL_CHAR', 'flux': 'FVS_VanLeer', 'limiter': 'VanLeer',  'label': 'MUSCL_Char(VL)-FVS(VL)'},
        {'recon': 'MUSCL_CHAR', 'flux': 'FVS_VanLeer', 'limiter': 'Superbee', 'label': 'MUSCL_Char(SB)-FVS(VL)'},
        # Optional: Characteristic reconstruction with FDS_Roe (this is very common and high-performing)
        # 可选: 特征重构与FDS_Roe (非常常见且高性能的组合)
        # {'recon': 'MUSCL_CHAR', 'flux': 'FDS_Roe', 'limiter': 'VanLeer',  'label': 'MUSCL_Char(VL)-FDS(Roe)'},
        # {'recon': 'MUSCL_CHAR', 'flux': 'FDS_Roe', 'limiter': 'Superbee', 'label': 'MUSCL_Char(SB)-FDS(Roe)'},
    ]

    results_all = {} # Dictionary to store results for all schemes / 存储所有格式结果的字典
    # Extend colors if needed, or use a colormap / 如果需要，扩展颜色或使用颜色映射
    plot_colors = plt.cm.get_cmap('tab10', len(schemes_to_test)).colors


    print(f"Starting simulations for {len(schemes_to_test)} scheme(s) with NX={NX}, T_final={T_FINAL}, CFL={CFL_CONST}\n")

    # Run simulations for each scheme configuration / 为每个格式配置运行模拟
    for i, config in enumerate(schemes_to_test):
        start_time_sim = time.time() # Start timer for simulation / 模拟开始计时
        print(f"Running: {config['label']}...")
        U_final_run = run_simulation_single(U_initial, T_FINAL, DX, CFL_CONST, NX,
                                        config['recon'], config['flux'], config['limiter'], gamma)
        end_time_sim = time.time() # End timer / 模拟结束计时
        rho_num, u_num, p_num = conserved_to_primitive(U_final_run, gamma) # Convert to primitive for plotting / 转换为原始变量用于绘图
        # Calculate specific internal energy e = p / ( (gamma-1)*rho ) for numerical solution
        # 计算数值解的比内能
        e_num_specific = p_num / ((gamma - 1.0) * rho_num) 
        results_all[config['label']] = {'rho': rho_num, 'u': u_num, 'p': p_num, 'e': e_num_specific, 'color': plot_colors[i % len(plot_colors)]}
        print(f"Finished: {config['label']} in {end_time_sim - start_time_sim:.2f} seconds.\n")

    # Plotting the results / 绘制结果
    plot_vars = [
        ('Density (ρ)', 'rho', rho_ex),
        ('Velocity (u)', 'u', u_ex),
        ('Pressure (p)', 'p', p_ex),
        ('Specific Internal Energy (e)', 'e', e_ex_specific) # Plot specific internal energy / 绘制比内能
    ]

    plt.figure(figsize=(18, 14)) # Adjusted size for more legends / 调整大小以容纳更多图例
    for i_var, (var_title, key, exact_sol) in enumerate(plot_vars):
        plt.subplot(2, 2, i_var + 1)
        plt.plot(x_exact_plot, exact_sol, 'k-', linewidth=2.5, label='Exact (精确解)', zorder=100) # Exact on top / 精确解在最上层
        for config_label, data in results_all.items():
            plt.plot(X_CELL_CENTERS, data[key], marker='.' if NX <= 100 else 'None', markersize=4, linestyle='-', color=data['color'], label=config_label, alpha=0.8)
        
        plt.title(f'{var_title} at t={T_FINAL:.2f}')
        plt.xlabel('x')
        plt.ylabel(var_title.split(' ')[0])
        plt.legend(fontsize='x-small', loc='best') # Smaller legend, auto-placement / 较小图例, 自动放置
        plt.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle(f'Sod Shock Tube Comparison (incl. Char. Recon. for FVS) - NX={NX}, CFL={CFL_CONST}', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96]) 
    plt.savefig(f"sod_tube_comparison_char_NX{NX}.png")
    plt.show()

    # --- some normal Discussion ---
    print("\n--- Discussion (Including Characteristic Reconstruction for FVS) ---")
    print(f"1. Computational Domain & Grid / 计算域与网格: Domain [{XMIN}, {XMAX}] with {NX} cells (DX={DX:.2e}).")
    print("2. Shock Capturing Formats & Observations / 激波捕捉格式与观察:")
    print("   - TVD-MUSCL (Component-wise Primitive Limiting / 分量原始变量限制): As discussed before.")
    print("   - TVD-MUSCL_CHAR (Characteristic Limiting / 特征变量限制): This method reconstructs states by limiting slopes in characteristic space.")
    print("     - Motivation / 动机: Aligning the limiting process with the direction of wave propagation should theoretically lead to sharper resolution of discontinuities and fewer oscillations compared to component-wise limiting.")
    print("       (将限制过程与波传播方向对齐，理论上应比分量限制导致更清晰的间断分辨率和更少的振荡。)")
    print("   - WENO3: As discussed before.")
    print("3. Flux Treatment Methods & Observations / 通量处理方法与观察:")
    print("   - FVS (Van Leer): As discussed before. Inherently dissipative for certain wave types.")
    print("     (对于某些波类型具有固有的耗散性。)")
    print("   - FDS (Roe): As discussed before. Generally superior for resolving all wave types.")
    print("     (通常在解析所有波类型方面表现更优。)")
    print("4. Combinations with Characteristic Reconstruction for FVS / 特征重构与FVS的组合:")
    print("   - MUSCL_Char + FVS_VanLeer vs. MUSCL (primitive) + FVS_VanLeer:")
    print("     - Expectation / 期望: MUSCL_Char should provide 'better' (sharper, less oscillatory) P_L and P_R states to the FVS flux.")
    print("       (MUSCL_Char 应为FVS通量提供“更好”(更清晰，更少振荡)的 P_L 和 P_R 状态。)")
    print("     - Impact / 影响: The improvement might be noticeable, leading to somewhat sharper discontinuities compared to component-wise MUSCL with FVS. However, the inherent dissipation of the Van Leer FVS itself might still be the dominant factor in smearing, especially for contact discontinuities.")
    print("       (与使用FVS的分量MUSCL相比，改进可能很明显，导致间断更清晰。然而，Van Leer FVS本身的固有耗散可能仍然是涂抹的主要因素，特别是对于接触间断。)")
    print("     - The Van Leer FVS does not explicitly use the characteristic structure of the Euler equations to compute the flux (it splits based on Mach number). So, while better input states help, the FVS mechanism doesn't fully leverage the characteristic information in the same way an FDS scheme does.")
    print("       (Van Leer FVS 在计算通量时不显式使用欧拉方程的特征结构(它基于马赫数进行分裂)。因此，尽管更好的输入状态有帮助，FVS机制不像FDS格式那样充分利用特征信息。)")
    print("     - We might see reduced numerical noise or oscillations near shocks, and perhaps slightly steeper shock profiles.")
    print("       (我们可能会看到激波附近的数值噪声或振荡减少，以及可能略微更陡峭的激波剖面。)")
    print("5. General Observations / 一般观察:")
    print("   - Cost / 计算成本: Characteristic reconstruction is computationally more expensive than component-wise reconstruction due to the matrix-vector multiplications (projections to/from characteristic space) required for each cell/interface involved in the slope calculations.")
    print("     (由于斜率计算中涉及的每个单元/界面都需要矩阵-向量乘法(到/从特征空间的投影)，特征重构在计算上比分量重构更昂贵。)")
    print("   - Robustness / 稳健性: Characteristic methods, if not carefully implemented (e.g., handling near-zero sound speed or ensuring physical averages), can sometimes be slightly less robust, though TVD limiters generally maintain stability.")
    print("     (如果实施不当(例如，处理接近零的声速或确保物理平均值)，特征方法有时可能稳健性稍差，尽管TVD限制器通常能保持稳定性。)")
    print("   - Ideal Pairing / 理想配对: Characteristic reconstruction is most effective when paired with FDS schemes (like Roe's), as FDS schemes are designed to resolve individual characteristic waves. Using MUSCL_Char with FDS-Roe is a very common and high-performing combination.")
    print("     (特征重构与FDS格式(如Roe格式)配对时最有效，因为FDS格式旨在解析单个特征波。将MUSCL_Char与FDS-Roe结合使用是一种非常常见且高性能的组合。)")
    print("   - FVS limitation / FVS的局限性: The primary benefit of FVS is its robustness and simplicity. While characteristic reconstruction improves the input, FVS might not be the best partner to fully exploit the sharpened characteristic information. Nevertheless, any improvement in the reconstructed states is generally beneficial.")
    print("     (FVS的主要优点是其稳健性和简单性。虽然特征重构改善了输入，但FVS可能不是充分利用锐化特征信息的最佳伙伴。然而，重构状态的任何改进通常都是有益的。)")
    print("6. Conclusion on Characteristic Reconstruction with FVS / 关于特征重构与FVS的结论:")
    print("   - It's a valid attempt to improve the quality of states fed into the FVS. Some improvement in resolution or reduction of oscillations is expected.")
    print("     (这是改进输入到FVS的状态质量的有效尝试。预计在分辨率或振荡减少方面会有一些改进。)")
    print("   - The degree of improvement will likely be less dramatic than when characteristic reconstruction is paired with an FDS flux, due to the nature of FVS.")
    print("     (由于FVS的性质，改进程度可能不如特征重构与FDS通量配对时那么显著。)")
    print("   - The results will show if the added complexity and computational cost of characteristic reconstruction yield a significant enough benefit when the final flux calculation is still FVS-based.")
    print("     (结果将显示，当最终通量计算仍基于FVS时，特征重构增加的复杂性和计算成本是否能带来足够显著的好处。)")

