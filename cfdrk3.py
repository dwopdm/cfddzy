import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve # 导入用于精确 Riemann 解算器的 fsolve
import time
import matplotlib.animation as animation # 导入用于动画制作的模块

# --- Problem Constants & Setup ---
# --- 问题常数与设置 ---
gamma = 1.4  # 比热比 (绝热指数)

# Initial conditions for Sod problem / Sod 问题的初始条件
rho_L_init, u_L_init, p_L_init = 1.0, 0.0, 1.0    # Left state / 左侧状态 (密度, 速度, 压力)
rho_R_init, u_R_init, p_R_init = 0.125, 0.0, 0.1 # Right state / 右侧状态 (密度, 速度, 压力)

# Computational domain / 计算区域
XMIN, XMAX = -0.5, 0.5  # x-coordinate range / x坐标范围
NX = 200  # Number of grid cells (can be increased, e.g., 400) / 网格单元数量 (物理网格点)
DX = (XMAX - XMIN) / NX # Grid spacing / 网格间距
X_CELL_CENTERS = np.linspace(XMIN + DX / 2, XMAX - DX / 2, NX) # Cell centers / 单元中心x坐标

# Time parameters / 时间参数
T_FINAL_STATIC_PLOT = 0.14 # Final simulation time FOR STATIC PLOTS / 静态图的最终模拟时间
CFL_CONST = 0.5  # CFL number / CFL数, 用于控制时间步长

# Number of ghost cells. WENO3 and MUSCL (as implemented) require N_ghost=2.
# For WENO5, N_ghost=3 would be needed.
# 虚拟单元数量。WENO3和MUSCL(当前实现)需要N_ghost=2。WENO5则需要N_ghost=3。
N_GHOST = 2
WENO_EPS = 1e-6 # Epsilon for WENO smoothness indicators to avoid division by zero
                # WENO光滑指示子的epsilon值, 防止除零

# --- Variable Conversion Functions ---
# --- 变量转换函数 ---
def primitive_to_conserved(rho, u, p, gamma_val=gamma):
    """
    Converts primitive variables (rho, u, p) to conserved variables (rho, rho*u, E).
    将原始变量 (密度 rho, 速度 u, 压力 p) 转换为守恒变量 (密度 rho, 动量密度 rho*u, 总能量密度 E).
    """
    rho_u = rho * u  # 动量密度
    E = p / (gamma_val - 1.0) + 0.5 * rho * u**2 # 总能量密度 E = p/(γ-1) + 0.5*ρ*u^2
    if np.isscalar(rho): # 如果输入是标量
        return np.array([rho, rho_u, E])
    else: # 如果输入是数组
        return np.array([rho, rho_u, E])

def conserved_to_primitive(U, gamma_val=gamma):
    """
    Converts conserved variables (rho, rho*u, E) to primitive variables (rho, u, p).
    将守恒变量 (密度, 动量密度, 总能量密度) 转换为原始变量 (密度, 速度, 压力).
    U 是一个包含守恒变量的数组 (U[0]=rho, U[1]=rho*u, U[2]=E).
    """
    if U.ndim == 1: # Single state / 单个状态 (一个网格点)
        rho = U[0]
        # 为避免除以零 (或非常小的rho), 使用 np.maximum
        u = U[1] / np.maximum(rho, 1e-12) 
        E = U[2]
        # 从总能量密度E中恢复压力p: p = (γ-1)*(E - 0.5*ρ*u^2)
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2) 
        # 确保密度和压力为正，增加数值稳定性
        rho = max(rho, 1e-9)
        p = max(p, 1e-9)
        return np.array([rho, u, p])
    else: # Array of states / 状态数组 (多个网格点)
        rho = U[0, :]
        u = U[1, :] / np.maximum(rho, 1e-12) 
        E = U[2, :]
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2)
        rho = np.maximum(rho, 1e-9)
        p = np.maximum(p, 1e-9)
        return np.array([rho, u, p])

# --- Euler Flux Function (Physical Flux) ---
# --- 欧拉方程通量函数 (物理通量) ---
def euler_flux(rho, u, p):
    """
    Calculates physical flux F(U) from primitive variables for 1D Euler equations.
    根据原始变量 (rho, u, p) 计算一维欧拉方程的物理通量 F(U).
    F(U) = [rho*u, rho*u^2 + p, u*(E+p)]^T
    """
    rho_u = rho * u # 动量密度
    # 总能量密度 E = p/(γ-1) + 0.5*ρ*u^2
    E = p / (gamma - 1.0) + 0.5 * rho * u**2 
    # 初始化通量向量 F
    F = np.zeros_like(rho_u, shape=(3,) + rho.shape) #确保数组形状正确
    F[0] = rho_u                # 质量通量
    F[1] = rho_u * u + p        # 动量通量 + 压力项
    F[2] = u * (E + p)          # 能量通量
    return F

# --- TVD Limiters ---
# --- TVD 限制器 ---
# TVD (Total Variation Diminishing) 限制器用于在高阶格式中抑制数值震荡，尤其在间断附近。
# r 是连续梯度的比值，r_j = (u_j - u_{j-1}) / (u_{j+1} - u_j) (或类似形式)

def van_leer_limiter(r):
    """Van Leer flux limiter."""
    """Van Leer 通量限制器. phi(r) = (r + |r|) / (1 + |r|)"""
    # 添加一个小的 epsilon (1e-12) 到分母以防止除零，增强稳定性
    return (r + np.abs(r)) / (1.0 + np.abs(r) + 1e-12) 

def superbee_limiter(r):
    """Superbee flux limiter."""
    """Superbee 通量限制器. phi(r) = max(0, min(1, 2r), min(2, r))"""
    return np.maximum(0, np.maximum(np.minimum(1, 2 * r), np.minimum(2, r)))

LIMITERS = {
    'VanLeer': van_leer_limiter,
    'Superbee': superbee_limiter
}

# --- Reconstruction Schemes ---
# --- 重构格式 ---
# 重构是从单元平均值恢复单元界面处点值的过程。这是高阶格式的关键步骤。

def muscl_reconstruction(P_gh, limiter_func, n_ghost, nx_domain):
    """
    MUSCL (Monotone Upstream-centered Schemes for Conservation Laws) reconstruction with a TVD limiter.
    使用TVD限制器的MUSCL重构。
    P_gh: 带虚拟单元的原始变量数组 (rho, u, p)。
    limiter_func: 选择的通量限制器函数。
    n_ghost: 每侧的虚拟单元数量。
    nx_domain: 计算域内的网格单元数量。
    返回: P_L_at_interfaces (界面左侧重构值), P_R_at_interfaces (界面右侧重构值)
    """
    P_L_at_interfaces = np.zeros((3, nx_domain + 1)) # 存储在每个界面 i+1/2 处的左侧值 q_{L, i+1/2}
    P_R_at_interfaces = np.zeros((3, nx_domain + 1)) # 存储在每个界面 i+1/2 处的右侧值 q_{R, i+1/2}

    # 对每个原始变量 (rho, u, p) 进行重构
    for k_var in range(3): # 遍历 rho, u, p
        q = P_gh[k_var, :] # 当前处理的变量 (例如密度场 q = P_gh[0,:])
        # 遍历所有内部界面 (从 j_inter=0 到 nx_domain)
        # j_inter = 0 对应 x_{1/2} 界面, j_inter = nx_domain 对应 x_{NX+1/2} 界面
        for j_inter in range(nx_domain + 1):
            # --- 左侧状态重构 q_{L, j+1/2} (在单元 i 的右界面) ---
            # 界面 j_inter 左侧的单元索引是 (n_ghost + j_inter - 1)
            # 例如，对于第一个物理界面 j_inter=0 (即 x_{1/2}), 左侧单元是 n_ghost-1 (最后一个左虚拟单元)
            # 对于内部界面 j_inter (对应 x_{i+1/2}), 左侧单元是 i
            idx_cell_L = n_ghost + j_inter - 1 # 这是用于重构左侧值的中心单元 i
            
            # 计算梯度： dq_L_minus 是 q_i - q_{i-1}, dq_L_plus 是 q_{i+1} - q_i
            dq_L_minus = q[idx_cell_L]     - q[idx_cell_L-1] # (q_i - q_{i-1})
            dq_L_plus  = q[idx_cell_L+1]   - q[idx_cell_L]   # (q_{i+1} - q_i)
            
            # 计算 r_L = (q_{i+1} - q_i) / (q_i - q_{i-1}) (顺风梯度比)
            r_L_den = dq_L_minus 
            if np.abs(r_L_den) < 1e-9: # 避免除零；如果分母为零，说明是平台或极值
                r_L = 2.0 if dq_L_plus * r_L_den >= 0 else -2.0 # 给予一个较大的r值，限制器会处理
            else:
                r_L = dq_L_plus / r_L_den
            
            phi_L = limiter_func(r_L) # 计算限制器函数值
            # MUSCL 重构公式: q_{L, i+1/2} = q_i + 0.5 * phi(r_i) * (q_i - q_{i-1})
            P_L_at_interfaces[k_var, j_inter] = q[idx_cell_L] + 0.5 * phi_L * dq_L_minus

            # --- 右侧状态重构 q_{R, j+1/2} (在单元 i+1 的左界面) ---
            # 界面 j_inter 右侧的单元索引是 (n_ghost + j_inter)
            # 例如，对于第一个物理界面 j_inter=0 (即 x_{1/2}), 右侧单元是 n_ghost (第一个物理单元)
            # 对于内部界面 j_inter (对应 x_{i+1/2}), 右侧单元是 i+1
            idx_cell_R = n_ghost + j_inter # 这是用于重构右侧值的中心单元 i+1

            # 计算梯度： dq_R_minus 是 q_{i+1} - q_i, dq_R_plus 是 q_{i+2} - q_{i+1}
            # 注意这里是相对于单元 idx_cell_R (即 i+1) 的梯度
            dq_R_minus = q[idx_cell_R]     - q[idx_cell_R-1] # (q_{i+1} - q_i)
            dq_R_plus  = q[idx_cell_R+1]   - q[idx_cell_R]   # (q_{i+2} - q_{i+1})

            # 计算 r_R = (q_{i+2} - q_{i+1}) / (q_{i+1} - q_i) (顺风梯度比，但从单元 i+1 的角度看)
            # 为了使用相同的限制器 phi(r), 通常使用 r_{i+1} = (q_{i+1} - q_i) / (q_{i+2} - q_{i+1}) (逆风梯度比)
            # 或者使用 r_i^{R} = (q_i - q_{i+1}) / (q_{i-1} - q_i) 
            # 这里实现的是对称形式，对单元 i+1 (idx_cell_R) 使用 r_{i+1} = (q_{i+2}-q_{i+1})/(q_{i+1}-q_i)
            r_R_den = dq_R_minus 
            if np.abs(r_R_den) < 1e-9:
                r_R = 2.0 if dq_R_plus * r_R_den >= 0 else -2.0
            else:
                r_R = dq_R_plus / r_R_den
            
            phi_R = limiter_func(r_R)
            # MUSCL 重构公式: q_{R, i+1/2} = q_{i+1} - 0.5 * phi(r_{i+1}) * (q_{i+1} - q_i)
            # (这里的 phi(r_{i+1}) 是基于单元 i+1 的梯度计算的)
            P_R_at_interfaces[k_var, j_inter] = q[idx_cell_R] - 0.5 * phi_R * dq_R_minus
            
    return P_L_at_interfaces, P_R_at_interfaces

def GVC2_reconstruction(P_gh, n_ghost, nx_domain):
    """
    GVC2 (Generalized Viscosity Capturing 2nd order) reconstruction.
    这是一种混合格式，根据局部流场的光滑度在不同的重构模板间切换。
    P_gh: 带虚拟单元的原始变量数组。
    n_ghost: 虚拟单元数。
    nx_domain: 物理网格单元数。
    返回: 界面左右两侧的重构值。
    """
    P_L_at_interfaces = np.zeros((3, nx_domain + 1))
    P_R_at_interfaces = np.zeros((3, nx_domain + 1))

    for k_var in range(3): # 对 rho, u, p 分别进行
        q = P_gh[k_var, :]
        for j_inter in range(nx_domain + 1): # 遍历所有内部界面 x_{j+1/2}
            # --- 左侧状态 q_{L, j+1/2} 重构 (基于单元 i = idx_L_cell) ---
            idx_L_cell = n_ghost + j_inter - 1 # 界面左侧的单元索引 i
            q_im1L = q[idx_L_cell-1] # q_{i-1}
            q_iL   = q[idx_L_cell]   # q_i
            q_ip1L = q[idx_L_cell+1] # q_{i+1}
            
            # 比较单元i左右两侧的梯度绝对值
            # K_iL = |q_i - q_{i-1}|, K_ip1L = |q_{i+1} - q_i|
            is_smoother_on_left = np.abs(q_iL - q_im1L) < np.abs(q_ip1L - q_iL)
    
            if is_smoother_on_left: # 如果左侧更光滑 (梯度更小)，使用二阶迎风格式 (基于 i 和 i-1)
                # q_L = q_i + 0.5 * (q_i - q_{i-1}) = (3*q_i - q_{i-1}) / 2
                q_interfacel = (3.0 * q_iL - q_im1L) / 2.0
            else: # 如果右侧更光滑或梯度相近，使用中心格式 (基于 i 和 i+1)
                # q_L = (q_i + q_{i+1}) / 2
                q_interfacel = (q_ip1L + q_iL) / 2.0
            P_L_at_interfaces[k_var, j_inter] = q_interfacel
            
            # --- 右侧状态 q_{R, j+1/2} 重构 (基于单元 i+1 = idx_R_cell) ---
            idx_R_cell = n_ghost + j_inter # 界面右侧的单元索引 i+1
            q_iR   = q[idx_R_cell-1] # q_i (单元 (i+1) 左侧的邻居)
            q_ip1R = q[idx_R_cell]   # q_{i+1} (单元 (i+1) 自身)
            q_ip2R = q[idx_R_cell+1] # q_{i+2} (单元 (i+1) 右侧的邻居)
            
            # 比较单元 (i+1) 左右两侧的梯度绝对值
            # K_ip1R = |q_{i+1} - q_i|, K_ip2R = |q_{i+2} - q_{i+1}|
            is_smoother_on_right = np.abs(q_ip2R - q_ip1R) < np.abs(q_ip1R - q_iR)   
            
            if is_smoother_on_right: # 如果右侧更光滑 (梯度更小)，使用二阶迎风格式 (基于 i+1 和 i+2)
                # q_R = q_{i+1} - 0.5 * (q_{i+2} - q_{i+1}) = (3*q_{i+1} - q_{i+2}) / 2
                q_interfacer = (3.0 * q_ip1R - q_ip2R) / 2.0
            else: # 如果左侧更光滑或梯度相近，使用中心格式 (基于 i 和 i+1)
                # q_R = (q_i + q_{i+1}) / 2
                q_interfacer = (q_iR + q_ip1R) / 2.0 
            P_R_at_interfaces[k_var, j_inter] = q_interfacer
            
    return P_L_at_interfaces, P_R_at_interfaces

def weno3_reconstruction(P_gh, n_ghost, nx_domain):
    """
    WENO3 (Weighted Essentially Non-Oscillatory, 3rd order) reconstruction.
    P_gh: 带虚拟单元的原始变量数组。
    n_ghost: 虚拟单元数。
    nx_domain: 物理网格单元数。
    返回: 界面左右两侧的重构值。
    """
    P_L_at_interfaces = np.zeros((3, nx_domain + 1))
    P_R_at_interfaces = np.zeros((3, nx_domain + 1))
    # WENO3 的线性权重 (optimal weights for smooth solutions)
    d0_L, d1_L = 2./3., 1./3. # 左侧重构的线性权重 (对应模板 S0, S1)
    d0_R, d1_R = 1./3., 2./3. # 右侧重构的线性权重 (对应模板 S0, S1)

    for k_var in range(3): # 对 rho, u, p 分别进行
        q = P_gh[k_var, :]
        for j_inter in range(nx_domain + 1): # 遍历所有内部界面 x_{j+1/2}
            # --- 左侧状态 q_{L, j+1/2} 重构 (基于单元 i = idx_L_cell) ---
            # 模板包含单元 i-1, i, i+1 (q_m1L, q_0L, q_p1L)
            idx_L_cell = n_ghost + j_inter - 1 # 中心单元 i
            q_m1L, q_0L, q_p1L = q[idx_L_cell-1], q[idx_L_cell], q[idx_L_cell+1]
            
            # 候选模板 (stencils) 的重构值
            # S0: 使用 q_{i-1}, q_i  (二阶精度) -> p0_L = -0.5*q_{i-1} + 1.5*q_i
            p0_L = -0.5*q_m1L + 1.5*q_0L 
            # S1: 使用 q_i, q_{i+1} (二阶精度) -> p1_L =  0.5*q_i + 0.5*q_{i+1}
            p1_L =  0.5*q_0L + 0.5*q_p1L
            
            # 光滑度指示子 (Smoothness Indicators)
            IS0_L = (q_0L - q_m1L)**2  # IS_k = sum_{l=1}^{r-1} int (d^l p_k(x) / dx^l)^2 dx
            IS1_L = (q_p1L - q_0L)**2  # 对于三阶WENO，简化为 (差分)^2
            
            # 非线性权重 alpha_k = d_k / (epsilon + IS_k)^2
            alpha0_L = d0_L / (WENO_EPS + IS0_L)**2
            alpha1_L = d1_L / (WENO_EPS + IS1_L)**2
            
            # 归一化权重 omega_k = alpha_k / sum(alpha_j)
            # 加权组合得到最终的重构值
            P_L_at_interfaces[k_var,j_inter]=(alpha0_L*p0_L + alpha1_L*p1_L) / (alpha0_L + alpha1_L)

            # --- 右侧状态 q_{R, j+1/2} 重构 (基于单元 i+1 = idx_R_cell) ---
            idx_R_cell = n_ghost + j_inter # 这是单元 i+1
            # q_i, q_{i+1}, q_{i+2}
            q_m1R, q_0R, q_p1R = q[idx_R_cell-1], q[idx_R_cell], q[idx_R_cell+1] 
            
            # 候选模板 (stencils) 的重构值 for q_R at i+1/2
            # S0 (from q_i, q_{i+1}): 0.5*q_i + 0.5*q_{i+1}
            p0_R =  0.5*q_m1R + 0.5*q_0R 
            # S1 (from q_{i+1}, q_{i+2}): 1.5*q_{i+1} - 0.5*q_{i+2} 
            p1_R = 1.5*q_0R - 0.5*q_p1R 
            
            IS0_R = (q_0R - q_m1R)**2 # Based on q_i, q_{i+1}
            IS1_R = (q_p1R - q_0R)**2 # Based on q_{i+1}, q_{i+2}
            
            alpha0_R = d0_R / (WENO_EPS + IS0_R)**2 
            alpha1_R = d1_R / (WENO_EPS + IS1_R)**2 
            
            P_R_at_interfaces[k_var,j_inter] = (alpha0_R*p0_R + alpha1_R*p1_R) / (alpha0_R + alpha1_R)
    return P_L_at_interfaces, P_R_at_interfaces


# --- Numerical Flux Schemes ---
# --- 数值通量格式 ---
# 数值通量用于计算通过单元界面的物理量的交换。

def van_leer_fvs_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    """
    Van Leer Flux Vector Splitting (FVS).
    基于界面左右两侧的重构原始变量 P_L_inter, P_R_inter 计算数值通量。
    通量 F = F^+(P_L) + F^-(P_R)
    """
    rho_L, u_L, p_L = P_L_inter[0], P_L_inter[1], P_L_inter[2]
    rho_R, u_R, p_R = P_R_inter[0], P_R_inter[1], P_R_inter[2]
    
    # 从左侧状态计算正通量分裂 F^+
    F_plus_L, _ = van_leer_flux_split_vectorized(rho_L, u_L, p_L, gamma_val)
    # 从右侧状态计算负通量分裂 F^-
    _, F_minus_R = van_leer_flux_split_vectorized(rho_R, u_R, p_R, gamma_val)
    
    return F_plus_L + F_minus_R # 数值通量

def van_leer_flux_split_vectorized(rho_vec, u_vec, p_vec, gamma_val=gamma):
    """
    Vectorized Van Leer flux splitting.
    根据原始变量计算分裂后的正通量 F^+ 和负通量 F^-.
    """
    # 计算声速 a 和马赫数 M
    a_vec = np.sqrt(gamma_val * np.maximum(p_vec, 1e-9) / np.maximum(rho_vec, 1e-9))
    M_vec = u_vec / np.maximum(a_vec, 1e-9) # 避免除零
    
    rho_u_vec = rho_vec * u_vec
    E_vec = p_vec / (gamma_val - 1.0) + 0.5 * rho_vec * u_vec**2
    
    F_plus_vec = np.zeros((3, len(rho_vec)))
    F_minus_vec = np.zeros((3, len(rho_vec)))
    
    # 条件索引
    idx_M_ge_1 = M_vec >= 1.0     # M >= 1 (超声速，正向)
    idx_M_le_m1 = M_vec <= -1.0   # M <= -1 (超声速，负向)
    idx_M_abs_lt_1 = np.abs(M_vec) < 1.0 # |M| < 1 (亚声速)
    
    # 物理通量 (用于 M >= 1 或 M <= -1 的情况)
    F_full_0 = rho_u_vec
    F_full_1 = rho_u_vec * u_vec + p_vec
    F_full_2 = u_vec * (E_vec + p_vec)
    
    # 当 M >= 1: F^+ = F_physical, F^- = 0
    F_plus_vec[:, idx_M_ge_1] = np.array([F_full_0[idx_M_ge_1], F_full_1[idx_M_ge_1], F_full_2[idx_M_ge_1]])
    # F_minus_vec[:, idx_M_ge_1] 保持为0
    
    # 当 M <= -1: F^+ = 0, F^- = F_physical
    # F_plus_vec[:, idx_M_le_m1] 保持为0
    F_minus_vec[:, idx_M_le_m1] = np.array([F_full_0[idx_M_le_m1], F_full_1[idx_M_le_m1], F_full_2[idx_M_le_m1]])
    
    # 当 |M| < 1 (亚声速):
    if np.any(idx_M_abs_lt_1):
        u_s, a_s, rho_s, M_s = u_vec[idx_M_abs_lt_1], a_vec[idx_M_abs_lt_1], rho_vec[idx_M_abs_lt_1], M_vec[idx_M_abs_lt_1]
        
        # F^+ (Van Leer 1982, AIAA Journal)
        f_m_p = rho_s * a_s * 0.25 * (M_s + 1.0)**2
        u_p_vl = ((gamma_val - 1.0) * u_s + 2.0 * a_s) / gamma_val
        t_p_sq_di = 1.0 / (2.0 * (gamma_val**2 - 1.0))
        
        F_plus_vec[0, idx_M_abs_lt_1] = f_m_p
        F_plus_vec[1, idx_M_abs_lt_1] = f_m_p * u_p_vl
        F_plus_vec[2, idx_M_abs_lt_1] = f_m_p * (((gamma_val - 1.0) * u_s + 2.0 * a_s)**2 * t_p_sq_di)

        # F^-
        f_m_m = -rho_s * a_s * 0.25 * (M_s - 1.0)**2
        u_m_vl = ((gamma_val - 1.0) * u_s - 2.0 * a_s) / gamma_val
        F_minus_vec[0, idx_M_abs_lt_1] = f_m_m
        F_minus_vec[1, idx_M_abs_lt_1] = f_m_m * u_m_vl
        F_minus_vec[2, idx_M_abs_lt_1] = f_m_m * (((gamma_val - 1.0) * u_s - 2.0 * a_s)**2 * t_p_sq_di)
        
    return F_plus_vec, F_minus_vec

def roe_fds_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    """
    Roe Flux Difference Splitting (FDS).
    基于界面左右两侧的重构原始变量 P_L_inter, P_R_inter 计算数值通量。
    F_Roe = 0.5 * (F(P_L) + F(P_R)) - 0.5 * |A_Roe(P_L,P_R)| * (U(P_R) - U(P_L))
    """
    rho_L, u_L, p_L = P_L_inter[0], P_L_inter[1], P_L_inter[2]
    rho_R, u_R, p_R = P_R_inter[0], P_R_inter[1], P_R_inter[2]

    # 转换为守恒变量
    U_L = primitive_to_conserved(rho_L, u_L, p_L, gamma_val)
    U_R = primitive_to_conserved(rho_R, u_R, p_R, gamma_val)
    
    # 计算物理通量
    F_L = euler_flux(rho_L, u_L, p_L)
    F_R = euler_flux(rho_R, u_R, p_R)
    
    # Roe 平均值计算
    srho_L = np.sqrt(rho_L); srho_R = np.sqrt(rho_R) # sqrt(rho)
    # u_hat = (sqrt(rho_L)*u_L + sqrt(rho_R)*u_R) / (sqrt(rho_L) + sqrt(rho_R))
    u_hat = (srho_L * u_L + srho_R * u_R) / (srho_L + srho_R)
    
    # H = (E+p)/rho (比焓)
    H_L = (U_L[2] + p_L) / np.maximum(rho_L, 1e-9) 
    H_R = (U_R[2] + p_R) / np.maximum(rho_R, 1e-9)
    # H_hat = (sqrt(rho_L)*H_L + sqrt(rho_R)*H_R) / (sqrt(rho_L) + sqrt(rho_R))
    H_hat = (srho_L * H_L + srho_R * H_R) / (srho_L + srho_R)
    
    # a_hat^2 = (gamma-1)*(H_hat - 0.5*u_hat^2)
    a_h_sq = (gamma_val - 1.0) * (H_hat - 0.5 * u_hat**2)
    a_h_sq = np.maximum(a_h_sq, 1e-9) # 确保声速平方为正
    a_hat = np.sqrt(a_h_sq) # Roe 平均声速
    
    dU = U_R - U_L # 守恒变量的差 Delta U
    
    # Roe 平均矩阵的特征值: lambda_1 = u_hat - a_hat, lambda_2 = u_hat, lambda_3 = u_hat + a_hat
    l_hat = np.array([u_hat - a_hat, u_hat, u_hat + a_hat]) # (3, num_interfaces)
    
    # 波强度 alpha_k (dU 在特征向量方向上的投影)
    # alpha_2 (对应 lambda_2 = u_hat):
    a2t = ((gamma_val - 1.0) / np.maximum(a_hat**2, 1e-9)) * \
          (dU[0] * (H_hat - u_hat**2) + u_hat * dU[1] - dU[2])
    # alpha_1 (对应 lambda_1 = u_hat - a_hat):
    a1t = (dU[0] * (u_hat + a_hat) - dU[1] - a_hat * a2t) / np.maximum(2.0 * a_hat, 1e-9)
    # alpha_3 (对应 lambda_3 = u_hat + a_hat):
    a3t = dU[0] - a1t - a2t
    
    # Roe 矩阵的右特征向量 R_hat (这里直接用 alpha_k * R_k_hat)
    t1,t2,t3 = np.zeros_like(dU),np.zeros_like(dU),np.zeros_like(dU)
    # R_1_hat = [1, u_hat-a_hat, H_hat-u_hat*a_hat]^T
    t1[0,:] = a1t; t1[1,:] = a1t*(u_hat-a_hat); t1[2,:] = a1t*(H_hat-u_hat*a_hat)
    # R_2_hat = [1, u_hat, 0.5*u_hat^2]^T (注意这里的能量项)
    t2[0,:] = a2t; t2[1,:] = a2t*u_hat;       t2[2,:] = a2t*(0.5*u_hat**2)
    # R_3_hat = [1, u_hat+a_hat, H_hat+u_hat*a_hat]^T
    t3[0,:] = a3t; t3[1,:] = a3t*(u_hat+a_hat); t3[2,:] = a3t*(H_hat+u_hat*a_hat)
    
    # Harten-Hyman 熵修正 (或其他熵修正, 如 epsilon-Roe)
    # 防止在声速点 (u=a 或 u=-a) 或滞止点 (u=0 附近) 产生非物理扩张激波
    # |lambda_k|_fixed = |lambda_k| if |lambda_k| >= delta_k else (lambda_k^2 + delta_k^2)/(2*delta_k)
    eps_r = 0.1 # 熵修正参数 (可调)
    d_k_r = eps_r * a_hat # delta_k
    abs_l_f = np.abs(l_hat)
    
    for k_w in range(3): # 对每个特征值
        idx_f = abs_l_f[k_w,:] < d_k_r # 找到需要修正的特征值
        abs_l_f[k_w,idx_f] = (l_hat[k_w,idx_f]**2 + d_k_r[idx_f]**2) / np.maximum(2.0 * d_k_r[idx_f], 1e-12)
        
    # 数值耗散项 0.5 * sum( |lambda_k|_fixed * alpha_k * R_k_hat )
    diss = abs_l_f[0]*t1 + abs_l_f[1]*t2 + abs_l_f[2]*t3
    
    # Roe 数值通量
    return 0.5 * (F_L + F_R) - 0.5 * diss

# --- Boundary Conditions ---
# --- 边界条件 ---
def apply_boundary_conditions(U_internal, num_ghost, nx_domain):
    """
    Applies simple zero-gradient (extrapolation) boundary conditions.
    应用简单的零梯度 (外推) 边界条件。
    U_internal: 内部网格的守恒变量数组。
    num_ghost: 每侧的虚拟单元数量。
    nx_domain: 物理网格单元数量。
    返回: 带虚拟单元的守恒变量数组。
    """
    # 创建一个包含虚拟单元的总数组
    U_with_ghost = np.zeros((3, nx_domain + 2 * num_ghost))
    # 将内部解复制到中心部分
    U_with_ghost[:, num_ghost : num_ghost + nx_domain] = U_internal
    # 应用左边界条件 (零梯度: 虚拟单元的值等于最近的物理单元的值)
    for i in range(num_ghost):
        U_with_ghost[:, i] = U_internal[:, 0]       # 左侧虚拟单元
        U_with_ghost[:, -(i + 1)] = U_internal[:, -1] # 右侧虚拟单元
    return U_with_ghost

# --- Eigenvector calculations for Characteristic Limiting ---
# --- 用于特征限制的特征向量计算 (此处未在MUSCL中使用，但可用于特征变量重构) ---
def get_eigenvectors_primitive(rho, u, p, gamma_val=gamma):
    """
    Calculates left and right eigenvectors for the Euler equations in primitive variables.
    计算原始变量形式的欧拉方程的左右特征向量。 (主要用于特征变量WENO或特征变量限制)
    """
    rho_s = max(rho,1e-9); p_s = max(p,1e-9) # 保证正值
    a = np.sqrt(gamma_val*p_s/rho_s)
    a_s = max(a,1e-9) # 保证声速正值

    # 左特征向量矩阵 L (行向量是左特征向量)
    # L * dP = dW (特征变量的微分)
    L = np.array([
        [0,          -rho_s*0.5/a_s,  0.5/(a_s**2)],  # 对应 u-a 特征值
        [1,          0,              -1.0/(a_s**2)],  # 对应 u   特征值
        [0,           rho_s*0.5/a_s,  0.5/(a_s**2)]   # 对应 u+a 特征值
    ])
    # 右特征向量矩阵 R (列向量是右特征向量)
    # R * dW = dP
    R = np.array([
        [1,               1,            1],
        [-a_s/rho_s,      0,            a_s/rho_s],
        [a_s**2,          0,            a_s**2]
    ])
    return L, R


# --- RHS Calculation ---
# --- 右端项 (空间离散) 计算 ---
def calculate_rhs(U_curr_int, dx_v, recon_m, flux_m, lim_n=None, n_gc=N_GHOST, nx_v=NX, gam_v=gamma):
    """
    Calculates the right-hand side (RHS) of the semi-discretized Euler equations: dU/dt = - (F_{j+1/2} - F_{j-1/2}) / dx.
    计算半离散欧拉方程的右端项 dU/dt = - (F_{j+1/2} - F_{j-1/2}) / dx。
    U_curr_int: 当前时刻的内部网格守恒变量。
    dx_v: 网格间距。
    recon_m: 重构方法 ('MUSCL', 'GVC2', 'WENO3')。
    flux_m: 数值通量方法 ('FVS_VanLeer', 'FDS_Roe')。
    lim_n: MUSCL使用的限制器名称 (如果 recon_m == 'MUSCL')。
    n_gc, nx_v, gam_v: 虚拟单元数, 物理网格数, 比热比。
    返回: RHS 向量。
    """
    # 1. 应用边界条件并转换为原始变量
    U_g = apply_boundary_conditions(U_curr_int, n_gc, nx_v) # 添加虚拟单元
    P_g = conserved_to_primitive(U_g, gam_v)               # 转换为原始变量，用于重构

    # 2. 界面值重构
    if recon_m == 'MUSCL':
        if lim_n is None or lim_n not in LIMITERS: 
            raise ValueError(f"Invalid limiter for MUSCL: {lim_n}")
        lim_f = LIMITERS[lim_n] # 获取限制器函数
        P_L, P_R = muscl_reconstruction(P_g, lim_f, n_gc, nx_v)
    elif recon_m == 'GVC2':
         P_L,P_R = GVC2_reconstruction(P_g, n_gc, nx_v)
    elif recon_m == 'WENO3':
        P_L,P_R = weno3_reconstruction(P_g, n_gc, nx_v)
    else:
        raise ValueError(f"Unknown reconstruction method: {recon_m}")
    
    # 确保重构后的界面原始变量物理上合理 (密度和压力为正)
    P_L[0,:] = np.maximum(P_L[0,:], 1e-9) # rho_L >= 0
    P_L[2,:] = np.maximum(P_L[2,:], 1e-9) # p_L >= 0
    P_R[0,:] = np.maximum(P_R[0,:], 1e-9) # rho_R >= 0
    P_R[2,:] = np.maximum(P_R[2,:], 1e-9) # p_R >= 0

    # 3. 计算数值通量
    if flux_m == 'FVS_VanLeer':
        F_num = van_leer_fvs_flux(P_L, P_R, gam_v)
    elif flux_m == 'FDS_Roe':
        F_num = roe_fds_flux(P_L, P_R, gam_v)
    else:
        raise ValueError(f"Unknown numerical flux method: {flux_m}")
        
    # 4. 计算右端项 (空间导数的负值)
    # RHS = - (F_{j+1/2} - F_{j-1/2}) / dx
    # F_num 维度是 (3, nx_domain + 1)， F_num[:,j] 是界面 x_{j+1/2} 处的通量
    # F_num[:,1:] 是 F_{j+1/2} for j=1..nx_domain (即 F_{3/2} 到 F_{NX+1/2})
    # F_num[:,:-1] 是 F_{j-1/2} for j=1..nx_domain (即 F_{1/2} 到 F_{NX-1/2})
    return -(F_num[:, 1:] - F_num[:, :-1]) / dx_v

# --- Exact Sod Solver (Based on Toro's book, Chapter 4) ---
# --- 精确 Sod 问题解算器 (基于 Toro 的书, 第4章) ---
def exact_sod_solution(x_pts, t, g, rhoL, uL, pL, rhoR, uR, pR):
    """
    Computes the exact solution of the Sod Riemann problem.
    (代码来自Toro书的Fortran程序或其他类似实现，已向量化和适配Python)
    x_pts: 查询解的空间点。
    t: 查询解的时间。
    g: 比热比 gamma。
    rhoL, uL, pL: 左侧初始状态。
    rhoR, uR, pR: 右侧初始状态。
    返回: rho, u, p, E (在x_pts处的精确解)
    """
    # 计算初始左右声速
    aL = np.sqrt(g * max(pL, 1e-9) / max(rhoL, 1e-9))
    aR = np.sqrt(g * max(pR, 1e-9) / max(rhoR, 1e-9))

    # 处理 t=0 的情况 (初始条件)
    if abs(t) < 1e-12:
        r_s = np.where(x_pts < 0, rhoL, rhoR)
        u_s = np.where(x_pts < 0, uL, uR)
        p_s = np.where(x_pts < 0, pL, pR)
        E_s = p_s / (g - 1.) + 0.5 * r_s * u_s**2
        return r_s, u_s, p_s, E_s

    # 压力函数 f(p_star, pK, rhoK, aK) (Toro Eq. 4.6)
    def s_t_r_f(p_s_v, pK, rhoK, aK): # p_s_v is p_star (contact region pressure)
        pK_s = max(pK, 1e-9); rhoK_s = max(rhoK, 1e-9); aK_s = max(aK, 1e-9)
        if p_s_v > pK_s: # Shock wave
            AK = 2. / ((g + 1.) * rhoK_s)
            BK = (g - 1.) / (g + 1.) * pK_s
            return (p_s_v - pK_s) * np.sqrt(AK / max(p_s_v + BK, 1e-9))
        else: # Rarefaction wave
            return (2. * aK_s / (g - 1.)) * ((max(p_s_v, 0) / pK_s)**((g - 1.) / (2. * g)) - 1.)

    # 方程 F(p_star) = f(p_star, pL, rhoL, aL) + f(p_star, pR, rhoR, aR) + (uR - uL) = 0 (Toro Eq. 4.5)
    def p_f_r(p_s_g_arr): # p_s_g_arr is an array containing guess for p_star
        p_s_g = max(p_s_g_arr[0], 1e-9) # Ensure p_star guess is positive
        return s_t_r_f(p_s_g, pL, rhoL, aL) + s_t_r_f(p_s_g, pR, rhoR, aR) + (uR - uL)

    # 求解 p_star (接触区压力)
    p_s_g_i = 0.5 * (pL + pR) # 初始猜测值
    p_s_g_i = max(p_s_g_i, 1e-6) # 确保猜测值为正
    try:
        p_s = fsolve(p_f_r, [p_s_g_i], xtol=1e-12)[0] # 使用fsolve数值求解
    except: # 如果fsolve失败，尝试Toro书中建议的两激波或两稀疏波近似作为初值
        # PVRS (Primitive Variable Riemann Solver) 近似初值 (Toro Eq. 9.32 for two-rarefaction)
        p_pv_n = (aL + aR - 0.5 * (g - 1.) * (uR - uL))
        p_L_pt = aL / (max(pL, 1e-9)**((g - 1.) / (2. * g)))
        p_R_pt = aR / (max(pR, 1e-9)**((g - 1.) / (2. * g)))
        p_pv_d = max(p_L_pt + p_R_pt, 1e-9)
        p_s_g_a = (p_pv_n / p_pv_d)**((2. * g) / (g - 1.))
        p_s_g_a = max(1e-6, p_s_g_a if np.isfinite(p_s_g_a) else 1e-6)
        p_s = fsolve(p_f_r, [p_s_g_a], xtol=1e-12)[0]
    p_s = max(p_s, 1e-9) # 确保求解的 p_star 为正

    # 计算 u_star (接触区速度) (Toro Eq. 4.9)
    u_s = 0.5 * (uL + uR) + 0.5 * (s_t_r_f(p_s, pR, rhoR, aR) - s_t_r_f(p_s, pL, rhoL, aL))

    # 根据 p_star 和左右状态确定波的类型并计算相关区域的密度
    pL_s, rhoL_s = max(pL, 1e-9), max(rhoL, 1e-9)
    pR_s, rhoR_s = max(pR, 1e-9), max(rhoR, 1e-9)

    # 左侧波后区域密度 rho_star_L
    if p_s > pL_s: # Left shock
        rho_sL = rhoL_s * ((p_s / pL_s + (g - 1.) / (g + 1.)) / \
                           ((g - 1.) / (g + 1.) * (p_s / pL_s) + 1.)) # Toro Eq. 4.12
    else: # Left rarefaction
        rho_sL = rhoL_s * (p_s / pL_s)**(1. / g) # Toro Eq. 4.16

    # 右侧波后区域密度 rho_star_R
    if p_s > pR_s: # Right shock
        rho_sR = rhoR_s * ((p_s / pR_s + (g - 1.) / (g + 1.)) / \
                           ((g - 1.) / (g + 1.) * (p_s / pR_s) + 1.)) # Toro Eq. 4.13
    else: # Right rarefaction
        rho_sR = rhoR_s * (p_s / pR_s)**(1. / g) # Toro Eq. 4.17
    
    rho_sL = max(rho_sL, 1e-9); rho_sR = max(rho_sR, 1e-9)

    # 计算波速
    S_C = u_s # 接触间断速度
    
    if p_s > pL_s: # Left shock
        S_L_sh = uL - aL * np.sqrt(((g + 1.) / (2. * g)) * (p_s / pL_s) + ((g - 1.) / (2. * g))) # Toro Eq. 4.54
    else: # Left rarefaction
        a_sL = max(aL * (p_s / pL_s)**((g - 1.) / (2. * g)), 1e-9) # 声速在稀疏波尾部 (Toro Eq. 4.58)
        S_HL_r = uL - aL     # 稀疏波头部速度 (Toro Eq. 4.23)
        S_TL_r = u_s - a_sL  # 稀疏波尾部速度 (Toro Eq. 4.24)

    if p_s > pR_s: # Right shock
        S_R_sh = uR + aR * np.sqrt(((g + 1.) / (2. * g)) * (p_s / pR_s) + ((g - 1.) / (2. * g))) # Toro Eq. 4.59
    else: # Right rarefaction
        a_sR = max(aR * (p_s / pR_s)**((g - 1.) / (2. * g)), 1e-9) # 声速在稀疏波尾部
        S_HR_r = uR + aR     # 稀疏波头部速度 (Toro Eq. 4.27)
        S_TR_r = u_s + a_sR  # 稀疏波尾部速度 (Toro Eq. 4.28)

    # 根据 x/t 的值确定每个采样点所处的区域并赋值
    rs, us, ps = np.zeros_like(x_pts), np.zeros_like(x_pts), np.zeros_like(x_pts)
    for i, x_v in enumerate(x_pts):
        s_q = x_v / t # 采样射线的速度
        
        if s_q <= S_C: # 点在接触间断左侧或接触间断上
            if p_s > pL_s: # Left shock
                if s_q <= S_L_sh: # 在激波左侧 (未扰动区L)
                    rs[i], us[i], ps[i] = rhoL, uL, pL
                else: # 在激波右侧 (星区L)
                    rs[i], us[i], ps[i] = rho_sL, u_s, p_s
            else: # Left rarefaction
                if s_q <= S_HL_r: # 在稀疏波头部左侧 (未扰动区L)
                    rs[i], us[i], ps[i] = rhoL, uL, pL
                elif s_q <= S_TL_r: # 在稀疏波内部 (星区L)
                    aL_s = max(aL, 1e-9)
                    us[i] = (2. / (g + 1.)) * (aL_s + (g - 1.) / 2. * uL + s_q)
                    cf = max((2./(g+1.)) + ((g-1.)/((g+1.)*aL_s))*(uL-s_q), 0) 
                    rs[i] = rhoL_s * cf**(2./(g-1.))
                    ps[i] = pL_s * cf**(2.*g/(g-1.))
                else: # 在稀疏波尾部右侧 (星区L，但已是均匀状态)
                    rs[i], us[i], ps[i] = rho_sL, u_s, p_s
        else: # 点在接触间断右侧
            if p_s > pR_s: # Right shock
                if s_q >= S_R_sh: # 在激波右侧 (未扰动区R)
                    rs[i], us[i], ps[i] = rhoR, uR, pR
                else: # 在激波左侧 (星区R)
                    rs[i], us[i], ps[i] = rho_sR, u_s, p_s
            else: # Right rarefaction
                if s_q >= S_HR_r: # 在稀疏波头部右侧 (未扰动区R)
                    rs[i], us[i], ps[i] = rhoR, uR, pR
                elif s_q >= S_TR_r: # 在稀疏波内部 (星区R)
                    aR_s = max(aR, 1e-9)
                    us[i] = (2. / (g + 1.)) * (-aR_s + (g - 1.) / 2. * uR + s_q)
                    cf = max((2./(g+1.)) - ((g-1.)/((g+1.)*aR_s))*(uR-s_q), 0) 
                    rs[i] = rhoR_s * cf**(2./(g-1.))
                    ps[i] = pR_s * cf**(2.*g/(g-1.))
                else: # 在稀疏波尾部左侧 (星区R，但已是均匀状态)
                    rs[i], us[i], ps[i] = rho_sR, u_s, p_s
                    
    rs = np.maximum(rs, 1e-9); ps = np.maximum(ps, 1e-9) # 确保正值
    Es = ps / (g - 1.) + 0.5 * rs * us**2
    return rs, us, ps, Es

# --- Main Simulation Runner for a Single Configuration ---
# --- 单个配置的主模拟运行程序 ---
def run_simulation_single(U_init,t_fin,dx_s,cfl,nx_s,recon,flux,lim=None,gam=gamma,verb=True):
    """
    Runs a single simulation configuration up to a final time t_fin.
    使用指定的参数运行一次模拟直到最终时间 t_fin。
    时间推进格式: 三阶经典Runge-Kutta (Classical RK3)
    """
    U = np.copy(U_init) # 复制初始条件以避免修改原始数组
    t_c = 0.0           # 当前时间
    n_iter = 0          # 迭代次数
    t_start = time.time() # 记录开始时间

    if verb: print(f"  Running: {recon}/{lim if lim else ''}-{flux} to T={t_fin:.2f}")

    while t_c < t_fin:
        # 1. 计算时间步长 dt (根据 CFL 条件)
        rho, u, p = conserved_to_primitive(U, gam) # 获取当前原始变量
        a = np.sqrt(gam * np.maximum(p, 1e-9) / np.maximum(rho, 1e-9)) # 计算声速
        max_wave_speed = np.max(np.abs(u) + a) # 最大波速 (特征速度)
        dt = cfl * dx_s / max_wave_speed if max_wave_speed > 1e-9 else cfl * dx_s # CFL 条件 dt = CFL * dx / max_speed
        
        # 确保不会超出最终时间
        if t_c + dt > t_fin:
            dt = t_fin - t_c
        
        if dt <= 1e-12: # 如果时间步长过小，则停止
            break

        # =====================================================================
        # --- 三阶经典Runge-Kutta (Classical RK3) 时间推进格式 ---
        # k1 = dt * RHS(U^n)
        # k2 = dt * RHS(U^n + k1/2)
        # k3 = dt * RHS(U^n - k1 + 2*k2)
        # U^(n+1) = U^n + (1/6)*(k1 + 4*k2 + k3)
        # ---------------------------------------------------------------------

        # Stage 1
        rhs_n = calculate_rhs(U, dx_s, recon, flux, lim, N_GHOST, nx_s, gam)
        k1 = dt * rhs_n
        
        # Stage 2
        U_temp1 = U + k1 / 2.0
        rhs_temp1 = calculate_rhs(U_temp1, dx_s, recon, flux, lim, N_GHOST, nx_s, gam)
        k2 = dt * rhs_temp1
        
        # Stage 3
        U_temp2 = U - k1 + 2.0 * k2
        rhs_temp2 = calculate_rhs(U_temp2, dx_s, recon, flux, lim, N_GHOST, nx_s, gam)
        k3 = dt * rhs_temp2
        
        # Final update
        U = U + (1.0/6.0) * (k1 + 4.0 * k2 + k3)
        # =====================================================================
        
        t_c += dt      # 更新当前时间
        n_iter += 1    # 更新迭代次数
        
        if verb and n_iter % 200 == 0: # 每200次迭代打印一次进度
            print(f"    Iter:{n_iter}, T:{t_c:.3f}/{t_fin:.3f}")
            
    if verb: print(f"  Finished in {time.time()-t_start:.2f}s. Iters: {n_iter}")
    return U


# --- Function to run simulation and collect frames for animation ---
# --- 运行模拟并收集动画帧的函数 ---
def run_simulation_and_collect_frames(U_init_sim, t_final_anim, dx_sim, cfl_val, nx_val,
                                     config_dict, gamma_val, anim_frame_dt_target):
    """
    Runs a simulation and collects data frames at specified time intervals for animation.
    运行模拟并在指定时间间隔收集数据帧以制作动画。
    时间推进格式: 三阶经典Runge-Kutta (Classical RK3)
    """
    U = np.copy(U_init_sim)
    t_curr = 0.0
    iter_count = 0
    animation_frames = [] # 存储动画帧数据的列表
    
    reconstruction_method = config_dict['recon']
    flux_method = config_dict['flux']
    limiter = config_dict.get('limiter', None) 
    scheme_label = config_dict['label']

    # 存储初始状态 (t=0)
    rho_iter, u_iter, p_iter = conserved_to_primitive(U, gamma_val)
    # 计算比内能 e = p / (rho * (gamma - 1))
    e_iter = p_iter / ((gamma_val - 1.0) * np.maximum(rho_iter, 1e-9)) 
    animation_frames.append({
        't': 0.0,
        'rho': np.copy(rho_iter), 'u': np.copy(u_iter),
        'p': np.copy(p_iter), 'e': np.copy(e_iter)
    })

    next_frame_capture_time = anim_frame_dt_target # 下一个捕获帧的时间点
    # 打印进度的间隔时间
    progress_interval = max(0.01, t_final_anim / 10.0) 
    next_progress_print_time = progress_interval if t_final_anim > 0 else float('inf')


    print(f"  Sim for Anim Frame Collection: {scheme_label} to T={t_final_anim:.2f}")
    sim_start_time = time.time()

    while t_curr < t_final_anim:
        rho_iter, u_iter, p_iter = conserved_to_primitive(U, gamma_val)
        a_iter = np.sqrt(gamma_val * np.maximum(p_iter, 1e-9) / np.maximum(rho_iter, 1e-9))
        max_speed = np.max(np.abs(u_iter) + a_iter)
        dt_val = cfl_val * dx_sim / max_speed if max_speed > 1e-9 else cfl_val * dx_sim

        if t_curr + dt_val > t_final_anim: dt_val = t_final_anim - t_curr
        if dt_val <= 1e-12: break 

        # =====================================================================
        # --- 三阶经典Runge-Kutta (Classical RK3) 时间推进格式 ---
        # k1 = dt * RHS(U^n)
        # k2 = dt * RHS(U^n + k1/2)
        # k3 = dt * RHS(U^n - k1 + 2*k2)
        # U^(n+1) = U^n + (1/6)*(k1 + 4*k2 + k3)
        # ---------------------------------------------------------------------

        # Stage 1
        rhs_n = calculate_rhs(U, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        k1 = dt_val * rhs_n
        
        # Stage 2
        U_temp1 = U + k1 / 2.0
        rhs_temp1 = calculate_rhs(U_temp1, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        k2 = dt_val * rhs_temp1
        
        # Stage 3
        U_temp2 = U - k1 + 2.0 * k2
        rhs_temp2 = calculate_rhs(U_temp2, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        k3 = dt_val * rhs_temp2
        
        # Final update
        U_new = U + (1.0/6.0) * (k1 + 4.0 * k2 + k3)
        # =====================================================================

        t_new = t_curr + dt_val # 更新后的时间
        
        # 帧捕获逻辑: 如果当前模拟时间已越过目标帧时间，则存储帧
        while next_frame_capture_time <= t_new + 1e-9 and next_frame_capture_time <= t_final_anim + 1e-9:
            rho_f, u_f, p_f = conserved_to_primitive(U_new, gamma_val) # 使用 U_new 的状态
            e_f = p_f / ((gamma_val - 1.0) * np.maximum(rho_f, 1e-9))
            animation_frames.append({
                't': next_frame_capture_time, # 标记帧的目标时间
                'rho': np.copy(rho_f), 'u': np.copy(u_f),
                'p': np.copy(p_f), 'e': np.copy(e_f)
            })
            next_frame_capture_time += anim_frame_dt_target # 更新下一个捕获时间
        
        U = U_new      # 更新守恒变量
        t_curr = t_new # 更新当前时间
        iter_count += 1

        # 打印动画模拟进度
        if t_curr >= next_progress_print_time and t_final_anim > 0:
            print(f"    {scheme_label[:20]:<20s} Anim Sim: T {t_curr:.3f}/{t_final_anim:.3f} ({t_curr/t_final_anim*100:.1f}%), Frames: {len(animation_frames)}")
            while next_progress_print_time <= t_curr and next_progress_print_time <= t_final_anim :
                 next_progress_print_time += progress_interval
    
    # 确保在 t_final_anim 处有一帧 (如果尚未捕获且需要)
    if t_final_anim > 0 and \
       (not animation_frames or abs(animation_frames[-1]['t'] - t_final_anim) > anim_frame_dt_target * 0.5) and \
       t_curr >= t_final_anim - 1e-9 : 
        rho_f, u_f, p_f = conserved_to_primitive(U, gamma_val) # 使用最终的 U
        e_f = p_f / ((gamma_val - 1.0) * np.maximum(rho_f, 1e-9))
        animation_frames.append({
            't': t_final_anim, 
            'rho': np.copy(rho_f), 'u': np.copy(u_f),
            'p': np.copy(p_f), 'e': np.copy(e_f)
        })

    print(f"  Finished Anim Frame Collection for {scheme_label} in {time.time()-sim_start_time:.2f}s. Frames: {len(animation_frames)}")
    return animation_frames


# --- Main Execution ---
# --- 主执行程序 ---
if __name__ == "__main__":
    # --- 初始化守恒变量场 U_initial ---
    U_initial = np.zeros((3, NX)) # (3 variables, NX cells)
    # 将左右初始状态从原始变量转换为守恒变量
    U_L_cons_init = primitive_to_conserved(rho_L_init, u_L_init, p_L_init, gamma)
    U_R_cons_init = primitive_to_conserved(rho_R_init, u_R_init, p_R_init, gamma)
    # 根据间断位置 x=0 设置初始条件
    for i in range(NX):
        if X_CELL_CENTERS[i] < 0: # 间断左侧
            U_initial[:, i] = U_L_cons_init
        else: # 间断右侧
            U_initial[:, i] = U_R_cons_init

    # 用于绘制精确解的 x 坐标点
    x_exact_plot = np.linspace(XMIN, XMAX, 500)

    # 定义要测试的数值格式组合
    schemes_to_test = [
        {'recon': 'MUSCL', 'flux': 'FVS_VanLeer', 'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FVS(VL)'},
        {'recon': 'GVC2',  'flux': 'FVS_VanLeer', 'limiter': None,       'label': 'GVC2-FVS(VL)'},
        {'recon': 'WENO3', 'flux': 'FVS_VanLeer', 'limiter': None,       'label': 'WENO3-FVS(VL)'},
        {'recon': 'MUSCL', 'flux': 'FDS_Roe',     'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FDS(Roe)'},
        {'recon': 'GVC2',  'flux': 'FDS_Roe',     'limiter': None,       'label': 'GVC2-FDS(Roe)'},
        {'recon': 'WENO3', 'flux': 'FDS_Roe',     'limiter': None,       'label': 'WENO3-FDS(Roe)'},
    ]

    # 为每个格式分配颜色，以便在绘图中保持一致性
    num_schemes = len(schemes_to_test)
    if hasattr(plt, 'colormaps'):
        cmap_func = plt.colormaps.get_cmap
    else: 
        cmap_func = plt.cm.get_cmap

    if num_schemes <= 10: cmap = cmap_func('tab10')
    elif num_schemes <= 20: cmap = cmap_func('tab20')
    else: cmap = cmap_func('viridis', num_schemes) 
    
    for i, config in enumerate(schemes_to_test):
        if num_schemes <= 20 and num_schemes > 1 : 
             config['color'] = cmap(i)
        elif num_schemes == 1:
             config['color'] = cmap(0.5) 
        else: 
             config['color'] = cmap(i / (num_schemes -1 if num_schemes > 1 else 1.0))


    # --- Part 1: 运行静态比较图 ---
    RUN_STATIC_COMPARISON = True
    if RUN_STATIC_COMPARISON:
        print(f"--- Running Static Scheme Comparison up to T={T_FINAL_STATIC_PLOT} ---")
        # 计算静态图时刻的精确解
        rho_ex_stat, u_ex_stat, p_ex_stat, _ = exact_sod_solution(x_exact_plot, T_FINAL_STATIC_PLOT, gamma,
                                                      rho_L_init, u_L_init, p_L_init,
                                                      rho_R_init, u_R_init, p_R_init)
        e_ex_specific_stat = p_ex_stat / ((gamma - 1.0) * np.maximum(rho_ex_stat, 1e-9)) # 精确解的比内能
        
        results_all_static = {} # 存储所有格式的静态结果
        print(f"Starting static simulations for {len(schemes_to_test)} schemes...\n")
        for config in schemes_to_test:
            # 运行单个模拟
            U_final_run = run_simulation_single(U_initial, T_FINAL_STATIC_PLOT, DX, CFL_CONST, NX,
                                            config['recon'], config['flux'], config.get('limiter'), gamma)
            # 转换回原始变量以便绘图
            rho_n, u_n, p_n = conserved_to_primitive(U_final_run, gamma)
            e_n_spec = p_n / ((gamma - 1.0) * np.maximum(rho_n,1e-9)) # 数值解的比内能
            results_all_static[config['label']] = {'rho':rho_n,'u':u_n,'p':p_n,'e':e_n_spec}

        # 绘制静态比较图
        plot_vars_static = [('Density (ρ)','rho',rho_ex_stat), 
                            ('Velocity (u)','u',u_ex_stat),
                            ('Pressure (p)','p',p_ex_stat), 
                            ('Specific Internal Energy (e)','e',e_ex_specific_stat)]
        
        plt.figure(figsize=(18, 14))
        for i_v, (title, key, ex_sol) in enumerate(plot_vars_static):
            plt.subplot(2,2,i_v+1)
            plt.plot(x_exact_plot,ex_sol,'k-',lw=2.5,label='Exact',zorder=100) # 绘制精确解
            for config in schemes_to_test: # 绘制每个数值格式的结果
                plt.plot(X_CELL_CENTERS, results_all_static[config['label']][key], 
                         marker='.' if NX<=100 else 'None', # 网格少时用点标记
                         markersize=4, linestyle='-', color=config['color'], 
                         label=config['label'], alpha=0.8)
            plt.title(f'{title} at t={T_FINAL_STATIC_PLOT:.2f}')
            plt.xlabel('x'); plt.ylabel(title.split(' ')[0])
            plt.legend(fontsize='x-small',loc='best'); plt.grid(True,ls=':',alpha=0.6)
        
        plt.suptitle(f'Sod Shock Tube Static Comparison - NX={NX}, T={T_FINAL_STATIC_PLOT}', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0,0,1,0.96]) # 调整布局防止标题重叠
        plt.savefig(f"sod_tube_static_NX{NX}_T{T_FINAL_STATIC_PLOT}.png")
        print(f"\nStatic plot saved. Showing now...")
        plt.show()

    # --- Part 2: 生成多格式组合动画 ---
    CREATE_ANIMATION = True           # 是否创建动画
    ANIM_T_FINAL = 0.25               # 动画的最终模拟时间 
    ANIM_FRAME_DT_TARGET = 0.0025     # 目标动画帧之间的时间间隔 
    ANIM_FPS = 30                     # 动画的帧率 

    if CREATE_ANIMATION:
        print(f"\n--- Generating Multi-Scheme Animation up to T={ANIM_T_FINAL} ---")
        all_animation_frames_data = {} # 存储所有格式的动画帧数据
        min_frames_collected = float('inf') # 记录所有格式中最少的帧数，以确保动画长度一致

        # 为每个格式运行模拟并收集帧
        for config in schemes_to_test:
            frames_list = run_simulation_and_collect_frames(
                U_initial, ANIM_T_FINAL, DX, CFL_CONST, NX,
                config, gamma, ANIM_FRAME_DT_TARGET
            )
            all_animation_frames_data[config['label']] = frames_list
            if frames_list: 
                min_frames_collected = min(min_frames_collected, len(frames_list))
            else: 
                min_frames_collected = 0 
                print(f"Warning: Scheme {config['label']} failed to produce animation frames.")

        if not all_animation_frames_data or min_frames_collected < 2:
            print("Not enough frames collected across all schemes for animation. Exiting multi-scheme animation generation.")
        else:
            num_render_frames = min_frames_collected # 动画将渲染的帧数
            print(f"Collected data for all schemes. Will render {num_render_frames} frames in multi-scheme animation.")

            # 创建动画图形和子图
            fig_anim, axes_anim_flat = plt.subplots(2, 2, figsize=(16, 11)) 
            axes_anim = {'rho': axes_anim_flat[0,0], 'u': axes_anim_flat[0,1],
                         'p': axes_anim_flat[1,0], 'e': axes_anim_flat[1,1]}
            plot_vars_keys = ['rho', 'u', 'p', 'e'] # 要绘制的变量键名
            plot_vars_titles = ['Density (ρ)', 'Velocity (u)', 'Pressure (p)', 'Specific Internal Energy (e)']
            
            lines_numerical_all_schemes = {cfg['label']: {} for cfg in schemes_to_test} # 存储数值解的线对象
            lines_exact_anim = {} # 存储精确解的线对象

            # 设置精确解的线
            for key_idx, key_var in enumerate(plot_vars_keys):
                ax = axes_anim[key_var]
                line_e, = ax.plot([], [], 'k-', lw=2.5, label='Exact', zorder=200) # 精确解置于顶层
                lines_exact_anim[key_var] = line_e
            
            # 设置数值解的线
            for config in schemes_to_test:
                scheme_label = config['label']
                color = config['color']
                for key_idx, key_var in enumerate(plot_vars_keys):
                    ax = axes_anim[key_var]
                    line_n, = ax.plot([], [], lw=1.5, label=scheme_label, color=color, alpha=0.7)
                    lines_numerical_all_schemes[scheme_label][key_var] = line_n

            # 配置坐标轴属性 (标题, 标签, 网格, 图例)
            for key_idx, key_var in enumerate(plot_vars_keys):
                ax = axes_anim[key_var]
                ax.set_title(plot_vars_titles[key_idx].split(' ')[0]) # 子图标题
                ax.set_xlabel('x'); ax.set_ylabel(plot_vars_titles[key_idx].split(' ')[0]) # 轴标签
                ax.grid(True, linestyle=':', alpha=0.6) # 网格
            
            # 稳健地设置y轴限制
            min_max_for_ylim = {key: [np.inf, -np.inf] for key in plot_vars_keys}
            # 通过采样精确解和数值解的首尾帧来确定y轴范围
            anim_times_for_ylim = np.linspace(0, ANIM_T_FINAL, max(20, num_render_frames // 5 + 2)) 
            for t_sample in anim_times_for_ylim:
                r_ex_s,u_ex_s,p_ex_s,_ = exact_sod_solution(x_exact_plot,t_sample,gamma,rho_L_init,u_L_init,p_L_init,rho_R_init,u_R_init,p_R_init)
                e_ex_s = p_ex_s / ((gamma-1.0)*np.maximum(r_ex_s,1e-9))
                ex_s_data = {'rho':r_ex_s, 'u':u_ex_s, 'p':p_ex_s, 'e':e_ex_s}
                for key in plot_vars_keys:
                    valid_data = ex_s_data[key][np.isfinite(ex_s_data[key])]
                    if valid_data.size > 0:
                        min_max_for_ylim[key][0]=min(min_max_for_ylim[key][0], np.min(valid_data))
                        min_max_for_ylim[key][1]=max(min_max_for_ylim[key][1], np.max(valid_data))
            
            for config in schemes_to_test: 
                frames = all_animation_frames_data[config['label']]
                if len(frames) >= num_render_frames and num_render_frames > 0: 
                    # 检查一些帧 (首, 中, 尾)
                    indices_to_check = np.unique(np.linspace(0, num_render_frames-1, min(10,num_render_frames), dtype=int))
                    for frame_idx_check in indices_to_check:
                        f_data = frames[frame_idx_check]
                        for key in plot_vars_keys:
                            valid_data = f_data[key][np.isfinite(f_data[key])]
                            if valid_data.size > 0:
                                min_max_for_ylim[key][0]=min(min_max_for_ylim[key][0],np.min(valid_data))
                                min_max_for_ylim[key][1]=max(min_max_for_ylim[key][1],np.max(valid_data))
            
            for key in plot_vars_keys: # 应用计算出的y轴限制
                ax = axes_anim[key]
                min_v, max_v = min_max_for_ylim[key]
                span = max_v - min_v
                if not np.isfinite(span) or span < 1e-6*abs(max_v) or span < 1e-6: span = abs(max_v)*0.2 if abs(max_v)>1e-6 else 0.2
                if not np.isfinite(span) or span < 1e-6: span = 0.2
                low_lim=min_v-0.1*span; upp_lim=max_v+0.1*span
                if key in ['rho','p','e']: low_lim=max(0,low_lim if np.isfinite(low_lim) else 0)
                if not np.isfinite(low_lim): low_lim = 0.0
                if not np.isfinite(upp_lim): upp_lim = low_lim + 1.0 
                ax.set_ylim(low_lim, upp_lim); ax.set_xlim(XMIN, XMAX)

            # 将图例添加到其中一个子图 (例如左上角)
            axes_anim['rho'].legend(fontsize='xx-small', loc='upper right', ncol=2)
            fig_anim.suptitle('', fontsize=16) # 初始为空，由update函数更新
            fig_anim.tight_layout(rect=[0, 0.03, 1, 0.95]) # 调整布局，为总标题留出空间
            
            # 确定用于时间参考的格式 (例如，第一个有足够帧数的格式)
            ref_scheme_label_for_time = None
            for cfg_item in schemes_to_test:
                if len(all_animation_frames_data[cfg_item['label']]) >= num_render_frames:
                    ref_scheme_label_for_time = cfg_item['label']
                    break
            if ref_scheme_label_for_time is None and schemes_to_test: 
                print("Warning: Could not find a reference scheme with enough frames. Using first scheme as fallback.")
                ref_scheme_label_for_time = schemes_to_test[0]['label']
            elif not schemes_to_test:
                print("Error: No schemes to test for animation timing.")
                # Handle error, perhaps by not creating animation
                
            # 动画更新函数
            def update_anim_plot_multi(frame_idx):
                if ref_scheme_label_for_time is None or frame_idx >= len(all_animation_frames_data[ref_scheme_label_for_time]):
                    return [] # 如果参考格式数据不足，则不更新

                t_current_frame = all_animation_frames_data[ref_scheme_label_for_time][frame_idx]['t']
                
                artists_changed = [] # 存储已更改的艺术家对象，用于blit=True
                # 更新精确解
                r_ex_f,u_ex_f,p_ex_f,_=exact_sod_solution(x_exact_plot,t_current_frame,gamma,rho_L_init,u_L_init,p_L_init,rho_R_init,u_R_init,p_R_init)
                e_ex_f=p_ex_f/((gamma-1.0)*np.maximum(r_ex_f,1e-9))
                exact_f_data={'rho':r_ex_f,'u':u_ex_f,'p':p_ex_f,'e':e_ex_f}
                for key_var in plot_vars_keys:
                    lines_exact_anim[key_var].set_data(x_exact_plot, exact_f_data[key_var])
                    artists_changed.append(lines_exact_anim[key_var])

                # 更新所有格式的数值解
                for config_item in schemes_to_test:
                    scheme_label = config_item['label']
                    # 确保该格式有足够的帧
                    if len(all_animation_frames_data[scheme_label]) > frame_idx:
                        frame_data_scheme = all_animation_frames_data[scheme_label][frame_idx]
                        for key_var in plot_vars_keys:
                            lines_numerical_all_schemes[scheme_label][key_var].set_data(X_CELL_CENTERS, frame_data_scheme[key_var])
                            artists_changed.append(lines_numerical_all_schemes[scheme_label][key_var])
                
                # 更新总标题
                fig_anim.suptitle(f'Sod Shock Tube Multi-Scheme Comparison at t={t_current_frame:.4f}s (NX={NX})')
                artists_changed.append(fig_anim.texts[0]) # suptitle 是 fig.texts[0]
                return artists_changed

            # 创建动画对象
            ani = animation.FuncAnimation(fig_anim, update_anim_plot_multi, frames=num_render_frames,
                                interval=1000/ANIM_FPS, blit=True, repeat=False)
            
            anim_filename = f"sod_tube_anim_T{ANIM_T_FINAL:.2f}_MULTI_NX{NX}.gif"
            print(f"Saving multi-scheme animation to {anim_filename} ({num_render_frames} frames, {ANIM_FPS} FPS)...")
            try:
                ani.save(anim_filename, writer='pillow', fps=ANIM_FPS) # 使用 'pillow' 作为 GIF 编码器
                print(f"Multi-scheme animation saved: {anim_filename}")
            except Exception as e:
                print(f"Error saving multi-scheme animation: {e}")
                print("Ensure 'pillow' is installed (pip install pillow).")
            plt.close(fig_anim) # 关闭图形以释放内存


        # --- Part 3: 生成单个格式的独立动画 ---
        CREATE_INDIVIDUAL_ANIMATIONS = True 
        if CREATE_INDIVIDUAL_ANIMATIONS and all_animation_frames_data:
            print(f"\n--- Generating Individual Scheme Animations up to T={ANIM_T_FINAL} ---")
            
            for config_individual in schemes_to_test: # 遍历每个要测试的格式
                scheme_label_indiv = config_individual['label']
                scheme_color_indiv = config_individual['color']
                
                # 检查该格式是否有收集到的帧数据
                if scheme_label_indiv not in all_animation_frames_data:
                    print(f"Skipping individual animation for {scheme_label_indiv}: No frame data found.")
                    continue
                
                frames_data_indiv = all_animation_frames_data[scheme_label_indiv]

                if not frames_data_indiv or len(frames_data_indiv) < 2: # 至少需要2帧才能制作动画
                    print(f"Skipping individual animation for {scheme_label_indiv} due to insufficient frames ({len(frames_data_indiv)}).")
                    continue
                
                num_render_frames_indiv = len(frames_data_indiv) # 该格式动画的帧数
                print(f"  Preparing individual animation for: {scheme_label_indiv} ({num_render_frames_indiv} frames)")

                # 为单个格式动画创建新的图形和子图
                fig_anim_indiv, axes_anim_flat_indiv = plt.subplots(2, 2, figsize=(16, 11))
                axes_anim_indiv = {'rho': axes_anim_flat_indiv[0,0], 'u': axes_anim_flat_indiv[0,1],
                                   'p': axes_anim_flat_indiv[1,0], 'e': axes_anim_flat_indiv[1,1]}
                
                lines_numerical_indiv = {} # 存储该格式数值解的线对象
                lines_exact_indiv = {}   # 存储精确解的线对象

                # 设置线对象
                for key_idx, key_var in enumerate(plot_vars_keys):
                    ax = axes_anim_indiv[key_var]
                    line_e, = ax.plot([], [], 'k-', lw=2.5, label='Exact', zorder=200)
                    lines_exact_indiv[key_var] = line_e
                    
                    line_n, = ax.plot([], [], lw=1.5, label=scheme_label_indiv, color=scheme_color_indiv, alpha=0.7)
                    lines_numerical_indiv[key_var] = line_n

                # 配置坐标轴 (使用之前为组合动画计算的y轴范围，以保持一致性)
                for key_idx, key_var in enumerate(plot_vars_keys):
                    ax = axes_anim_indiv[key_var]
                    ax.set_title(plot_vars_titles[key_idx].split(' ')[0])
                    ax.set_xlabel('x'); ax.set_ylabel(plot_vars_titles[key_idx].split(' ')[0])
                    ax.grid(True, linestyle=':', alpha=0.6)
                    
                    # 使用从多格式动画中确定的y轴限制
                    min_v_glob, max_v_glob = min_max_for_ylim[key_var] 
                    span_glob = max_v_glob - min_v_glob
                    if not np.isfinite(span_glob) or span_glob < 1e-6*abs(max_v_glob) or span_glob < 1e-6: span_glob = abs(max_v_glob)*0.2 if abs(max_v_glob)>1e-6 else 0.2
                    if not np.isfinite(span_glob) or span_glob < 1e-6: span_glob = 0.2
                    low_lim_glob=min_v_glob-0.1*span_glob; upp_lim_glob=max_v_glob+0.1*span_glob
                    if key_var in ['rho','p','e']: low_lim_glob=max(0,low_lim_glob if np.isfinite(low_lim_glob) else 0)
                    if not np.isfinite(low_lim_glob): low_lim_glob = 0.0
                    if not np.isfinite(upp_lim_glob): upp_lim_glob = low_lim_glob + 1.0
                    ax.set_ylim(low_lim_glob, upp_lim_glob); ax.set_xlim(XMIN, XMAX)
                
                axes_anim_indiv['rho'].legend(fontsize='small', loc='upper right') # 添加图例
                fig_anim_indiv.suptitle('', fontsize=16) # 初始空标题
                fig_anim_indiv.tight_layout(rect=[0, 0.03, 1, 0.95])

                # 单个动画的更新函数
                def update_anim_plot_individual(frame_idx_indiv):
                    if frame_idx_indiv >= len(frames_data_indiv): return [] # 帧索引超出范围

                    t_current_frame_indiv = frames_data_indiv[frame_idx_indiv]['t']
                    
                    artists_changed_indiv = []
                    # 更新精确解
                    r_ex_f,u_ex_f,p_ex_f,_ = exact_sod_solution(x_exact_plot,t_current_frame_indiv,gamma,rho_L_init,u_L_init,p_L_init,rho_R_init,u_R_init,p_R_init)
                    e_ex_f = p_ex_f/((gamma-1.0)*np.maximum(r_ex_f,1e-9))
                    exact_f_data = {'rho':r_ex_f,'u':u_ex_f,'p':p_ex_f,'e':e_ex_f}
                    for key_var in plot_vars_keys:
                        lines_exact_indiv[key_var].set_data(x_exact_plot, exact_f_data[key_var])
                        artists_changed_indiv.append(lines_exact_indiv[key_var])

                    # 更新该格式的数值解
                    frame_data_scheme_indiv = frames_data_indiv[frame_idx_indiv]
                    for key_var in plot_vars_keys:
                        lines_numerical_indiv[key_var].set_data(X_CELL_CENTERS, frame_data_scheme_indiv[key_var])
                        artists_changed_indiv.append(lines_numerical_indiv[key_var])
                    
                    # 更新总标题
                    fig_anim_indiv.suptitle(f'Sod: {scheme_label_indiv} at t={t_current_frame_indiv:.4f}s (NX={NX})')
                    artists_changed_indiv.append(fig_anim_indiv.texts[0])
                    return artists_changed_indiv

                # 创建并保存单个动画
                ani_indiv = animation.FuncAnimation(fig_anim_indiv, update_anim_plot_individual, frames=num_render_frames_indiv,
                                            interval=1000/ANIM_FPS, blit=True, repeat=False)
                
                # 创建安全的文件名 (替换特殊字符)
                safe_scheme_label = "".join(c if c.isalnum() else "_" for c in scheme_label_indiv)
                anim_filename_indiv = f"sod_tube_anim_T{ANIM_T_FINAL:.2f}_{safe_scheme_label}_NX{NX}.gif"
                print(f"  Saving individual animation to {anim_filename_indiv} ({num_render_frames_indiv} frames, {ANIM_FPS} FPS)...")
                try:
                    ani_indiv.save(anim_filename_indiv, writer='pillow', fps=ANIM_FPS)
                    print(f"  Individual animation saved: {anim_filename_indiv}")
                except Exception as e:
                    print(f"  Error saving individual animation for {scheme_label_indiv}: {e}")
                plt.close(fig_anim_indiv) # 关闭图形