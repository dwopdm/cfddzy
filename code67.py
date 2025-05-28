import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve # For the exact Riemann solver
import time
import matplotlib.animation as animation # Added for animation

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
T_FINAL_STATIC_PLOT = 20 # Final simulation time FOR STATIC PLOTS
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
        u = U[1] / np.maximum(rho, 1e-12) # Avoid division by zero
        E = U[2]
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2) # Pressure / 压力
        rho = max(rho, 1e-9)
        p = max(p, 1e-9)
        return np.array([rho, u, p])
    else: # Array of states / 状态数组
        rho = U[0, :]
        u = U[1, :] / np.maximum(rho, 1e-12) # Avoid division by zero
        E = U[2, :]
        p = (gamma_val - 1.0) * (E - 0.5 * rho * u**2)
        rho = np.maximum(rho, 1e-9)
        p = np.maximum(p, 1e-9)
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
    P_L_at_interfaces = np.zeros((3, nx_domain + 1))
    P_R_at_interfaces = np.zeros((3, nx_domain + 1))
    for k_var in range(3):
        q = P_gh[k_var, :]
        for j_inter in range(nx_domain + 1):
            idx_cell_L = n_ghost + j_inter - 1
            dq_L_minus = q[idx_cell_L] - q[idx_cell_L-1]
            dq_L_plus  = q[idx_cell_L+1] - q[idx_cell_L]
            r_L_den = dq_L_minus
            if np.abs(r_L_den) < 1e-9: r_L = 2.0 if dq_L_plus * r_L_den >= 0 else -2.0
            else: r_L = dq_L_plus / r_L_den
            phi_L = limiter_func(r_L)
            P_L_at_interfaces[k_var, j_inter] = q[idx_cell_L] + 0.5 * phi_L * dq_L_minus
            idx_cell_R = n_ghost + j_inter
            dq_R_minus = q[idx_cell_R] - q[idx_cell_R-1]
            dq_R_plus  = q[idx_cell_R+1] - q[idx_cell_R]
            r_R_den = dq_R_minus
            if np.abs(r_R_den) < 1e-9: r_R = 2.0 if dq_R_plus * r_R_den >= 0 else -2.0
            else: r_R = dq_R_plus / r_R_den
            phi_R = limiter_func(r_R)
            P_R_at_interfaces[k_var, j_inter] = q[idx_cell_R] - 0.5 * phi_R * dq_R_minus
    return P_L_at_interfaces, P_R_at_interfaces

def weno3_reconstruction(P_gh, n_ghost, nx_domain):
    P_L_at_interfaces = np.zeros((3, nx_domain + 1))
    P_R_at_interfaces = np.zeros((3, nx_domain + 1))
    d0_L, d1_L = 2./3., 1./3.; d0_R, d1_R = 1./3., 2./3.
    for k_var in range(3):
        q = P_gh[k_var, :]
        for j_inter in range(nx_domain + 1):
            idx_L_cell = n_ghost + j_inter - 1
            q_m1L, q_0L, q_p1L = q[idx_L_cell-1], q[idx_L_cell], q[idx_L_cell+1]
            p0_L = -0.5*q_m1L + 1.5*q_0L; p1_L = 0.5*q_0L + 0.5*q_p1L
            IS0_L=(q_0L-q_m1L)**2; IS1_L=(q_p1L-q_0L)**2
            alpha0_L=d0_L/(WENO_EPS+IS0_L)**2; alpha1_L=d1_L/(WENO_EPS+IS1_L)**2
            P_L_at_interfaces[k_var,j_inter]=(alpha0_L*p0_L+alpha1_L*p1_L)/(alpha0_L+alpha1_L)
            idx_R_cell = n_ghost + j_inter
            q_m1R, q_0R, q_p1R = q[idx_R_cell-1], q[idx_R_cell], q[idx_R_cell+1]
            p0_R = 0.5*q_m1R + 0.5*q_0R; p1_R = 1.5*q_0R - 0.5*q_p1R
            IS0_R=(q_0R-q_m1R)**2; IS1_R=(q_p1R-q_0R)**2
            alpha0_R=d0_R/(WENO_EPS+IS0_R)**2; alpha1_R=d1_R/(WENO_EPS+IS1_R)**2
            P_R_at_interfaces[k_var,j_inter]=(alpha0_R*p0_R+alpha1_R*p1_R)/(alpha0_R+alpha1_R)
    return P_L_at_interfaces, P_R_at_interfaces


# --- Numerical Flux Schemes ---
def van_leer_fvs_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    rho_L, u_L, p_L = P_L_inter[0], P_L_inter[1], P_L_inter[2]
    rho_R, u_R, p_R = P_R_inter[0], P_R_inter[1], P_R_inter[2]
    F_plus_L, _ = van_leer_flux_split_vectorized(rho_L, u_L, p_L, gamma_val)
    _, F_minus_R = van_leer_flux_split_vectorized(rho_R, u_R, p_R, gamma_val)
    return F_plus_L + F_minus_R

def van_leer_flux_split_vectorized(rho_vec, u_vec, p_vec, gamma_val=gamma):
    a_vec = np.sqrt(gamma_val * np.maximum(p_vec, 1e-9) / np.maximum(rho_vec, 1e-9))
    M_vec = u_vec / np.maximum(a_vec, 1e-9)
    rho_u_vec = rho_vec * u_vec
    E_vec = p_vec / (gamma_val - 1.0) + 0.5 * rho_vec * u_vec**2
    F_plus_vec = np.zeros((3, len(rho_vec))); F_minus_vec = np.zeros((3, len(rho_vec)))
    idx_M_ge_1 = M_vec >= 1.0; idx_M_le_m1 = M_vec <= -1.0; idx_M_abs_lt_1 = np.abs(M_vec) < 1.0
    F_full_0 = rho_u_vec; F_full_1 = rho_u_vec * u_vec + p_vec; F_full_2 = u_vec * (E_vec + p_vec)
    F_plus_vec[:, idx_M_ge_1] = np.array([F_full_0[idx_M_ge_1], F_full_1[idx_M_ge_1], F_full_2[idx_M_ge_1]])
    F_minus_vec[:, idx_M_le_m1] = np.array([F_full_0[idx_M_le_m1], F_full_1[idx_M_le_m1], F_full_2[idx_M_le_m1]])
    if np.any(idx_M_abs_lt_1):
        u_s,a_s,rho_s,M_s = u_vec[idx_M_abs_lt_1],a_vec[idx_M_abs_lt_1],rho_vec[idx_M_abs_lt_1],M_vec[idx_M_abs_lt_1]
        f_m_p = rho_s*a_s*0.25*(M_s+1.0)**2; u_p_vl=((gamma_val-1.0)*u_s+2.0*a_s)/gamma_val
        t_p_sq_di=1.0/(2.0*(gamma_val**2-1.0))
        F_plus_vec[0,idx_M_abs_lt_1]=f_m_p; F_plus_vec[1,idx_M_abs_lt_1]=f_m_p*u_p_vl
        F_plus_vec[2,idx_M_abs_lt_1]=f_m_p*(((gamma_val-1.0)*u_s+2.0*a_s)**2*t_p_sq_di)
        f_m_m = -rho_s*a_s*0.25*(M_s-1.0)**2; u_m_vl=((gamma_val-1.0)*u_s-2.0*a_s)/gamma_val
        F_minus_vec[0,idx_M_abs_lt_1]=f_m_m; F_minus_vec[1,idx_M_abs_lt_1]=f_m_m*u_m_vl
        F_minus_vec[2,idx_M_abs_lt_1]=f_m_m*(((gamma_val-1.0)*u_s-2.0*a_s)**2*t_p_sq_di)
    return F_plus_vec, F_minus_vec

def roe_fds_flux(P_L_inter, P_R_inter, gamma_val=gamma):
    rho_L,u_L,p_L=P_L_inter[0],P_L_inter[1],P_L_inter[2]
    rho_R,u_R,p_R=P_R_inter[0],P_R_inter[1],P_R_inter[2]
    U_L=primitive_to_conserved(rho_L,u_L,p_L,gamma_val); U_R=primitive_to_conserved(rho_R,u_R,p_R,gamma_val)
    F_L=euler_flux(rho_L,u_L,p_L); F_R=euler_flux(rho_R,u_R,p_R)
    srho_L=np.sqrt(rho_L); srho_R=np.sqrt(rho_R)
    u_hat=(srho_L*u_L+srho_R*u_R)/(srho_L+srho_R)
    H_L=(U_L[2]+p_L)/np.maximum(rho_L,1e-9); H_R=(U_R[2]+p_R)/np.maximum(rho_R,1e-9)
    H_hat=(srho_L*H_L+srho_R*H_R)/(srho_L+srho_R)
    a_h_sq=(gamma_val-1.0)*(H_hat-0.5*u_hat**2); a_h_sq=np.maximum(a_h_sq,1e-9); a_hat=np.sqrt(a_h_sq)
    dU=U_R-U_L; l_hat=np.array([u_hat-a_hat,u_hat,u_hat+a_hat])
    a2t=((gamma_val-1.0)/np.maximum(a_hat**2,1e-9))*(dU[0]*(H_hat-u_hat**2)+u_hat*dU[1]-dU[2])
    a1t=(dU[0]*(u_hat+a_hat)-dU[1]-a_hat*a2t)/np.maximum(2.0*a_hat,1e-9)
    a3t=dU[0]-a1t-a2t
    t1,t2,t3 = np.zeros_like(dU),np.zeros_like(dU),np.zeros_like(dU)
    t1[0,:]=a1t; t1[1,:]=a1t*(u_hat-a_hat); t1[2,:]=a1t*(H_hat-u_hat*a_hat)
    t2[0,:]=a2t; t2[1,:]=a2t*u_hat; t2[2,:]=a2t*(0.5*u_hat**2)
    t3[0,:]=a3t; t3[1,:]=a3t*(u_hat+a_hat); t3[2,:]=a3t*(H_hat+u_hat*a_hat)
    eps_r=0.1; d_k_r=eps_r*a_hat; abs_l_f=np.abs(l_hat)
    for k_w in range(3):
        idx_f=abs_l_f[k_w,:]<d_k_r
        abs_l_f[k_w,idx_f]=(l_hat[k_w,idx_f]**2+d_k_r[idx_f]**2)/np.maximum(2.0*d_k_r[idx_f],1e-12)
    diss=abs_l_f[0]*t1+abs_l_f[1]*t2+abs_l_f[2]*t3
    return 0.5*(F_L+F_R)-0.5*diss

# --- Boundary Conditions ---
def apply_boundary_conditions(U_internal, num_ghost, nx_domain):
    U_with_ghost = np.zeros((3, nx_domain + 2 * num_ghost))
    U_with_ghost[:, num_ghost : num_ghost + nx_domain] = U_internal
    for i in range(num_ghost):
        U_with_ghost[:, i] = U_internal[:, 0]
        U_with_ghost[:, -(i + 1)] = U_internal[:, -1]
    return U_with_ghost

# --- Eigenvector calculations for Characteristic Limiting ---
def get_eigenvectors_primitive(rho, u, p, gamma_val=gamma):
    rho_s=max(rho,1e-9); p_s=max(p,1e-9); a=np.sqrt(gamma_val*p_s/rho_s); a_s=max(a,1e-9)
    L=np.array([[0,-rho_s*0.5/a_s,0.5/(a_s**2)],[1,0,-1.0/(a_s**2)],[0,rho_s*0.5/a_s,0.5/(a_s**2)]])
    R=np.array([[1,1,1],[-a_s/rho_s,0,a_s/rho_s],[a_s**2,0,a_s**2]])
    return L, R

# --- MUSCL Reconstruction with Characteristic Limiting ---
def muscl_char_reconstruction(P_gh, limiter_func, n_ghost, nx_domain, gamma_val=gamma):
    P_L_inter = np.zeros((3, nx_domain+1)); P_R_inter = np.zeros((3, nx_domain+1)); eps_s = 1e-12
    for j_inter in range(nx_domain + 1):
        idx_L = n_ghost+j_inter-1
        Q_Lm1,Q_L0,Q_Lp1 = P_gh[:,idx_L-1],P_gh[:,idx_L],P_gh[:,idx_L+1]
        L_L,R_L = get_eigenvectors_primitive(Q_L0[0],Q_L0[1],Q_L0[2],gamma_val)
        dQm_p_L=Q_L0-Q_Lm1; dQp_p_L=Q_Lp1-Q_L0
        dWm_L=L_L@dQm_p_L; dWp_L=L_L@dQp_p_L
        lim_cs_L=np.zeros(3)
        for k_c in range(3):
            r_den=dWm_L[k_c]
            if np.abs(r_den)<eps_s: r_c_L=2. if dWp_L[k_c]*r_den>=0 else -2.
            else: r_c_L=dWp_L[k_c]/r_den
            phi_L=limiter_func(r_c_L); lim_cs_L[k_c]=0.5*phi_L*dWm_L[k_c]
        P_L_inter[:,j_inter]=Q_L0+R_L@lim_cs_L
        idx_R = n_ghost+j_inter
        Q_Rm1,Q_R0,Q_Rp1 = P_gh[:,idx_R-1],P_gh[:,idx_R],P_gh[:,idx_R+1]
        L_R,R_R = get_eigenvectors_primitive(Q_R0[0],Q_R0[1],Q_R0[2],gamma_val)
        dQm_p_R=Q_R0-Q_Rm1; dQp_p_R=Q_Rp1-Q_R0
        dWm_R=L_R@dQm_p_R; dWp_R=L_R@dQp_p_R
        lim_cs_R=np.zeros(3)
        for k_c in range(3):
            r_den=dWm_R[k_c]
            if np.abs(r_den)<eps_s: r_c_R=2. if dWp_R[k_c]*r_den>=0 else -2.
            else: r_c_R=dWp_R[k_c]/r_den
            phi_R=limiter_func(r_c_R); lim_cs_R[k_c]=0.5*phi_R*dWm_R[k_c]
        P_R_inter[:,j_inter]=Q_R0-R_R@lim_cs_R
    return P_L_inter, P_R_inter

# --- RHS Calculation ---
def calculate_rhs(U_curr_int, dx_v, recon_m, flux_m, lim_n=None, n_gc=N_GHOST, nx_v=NX, gam_v=gamma):
    U_g = apply_boundary_conditions(U_curr_int,n_gc,nx_v); P_g = conserved_to_primitive(U_g,gam_v)
    if recon_m=='MUSCL':
        if lim_n is None or lim_n not in LIMITERS: raise ValueError(f"Invalid limiter: {lim_n}")
        lim_f = LIMITERS[lim_n]; P_L,P_R = muscl_reconstruction(P_g,lim_f,n_gc,nx_v)
    elif recon_m=='MUSCL_CHAR':
        if lim_n is None or lim_n not in LIMITERS: raise ValueError(f"Invalid limiter: {lim_n}")
        lim_f = LIMITERS[lim_n]; P_L,P_R = muscl_char_reconstruction(P_g,lim_f,n_gc,nx_v,gam_v)
    elif recon_m=='WENO3': P_L,P_R = weno3_reconstruction(P_g,n_gc,nx_v)
    else: raise ValueError(f"Unknown recon method: {recon_m}")
    P_L[0,:]=np.maximum(P_L[0,:],1e-9); P_L[2,:]=np.maximum(P_L[2,:],1e-9)
    P_R[0,:]=np.maximum(P_R[0,:],1e-9); P_R[2,:]=np.maximum(P_R[2,:],1e-9)
    if flux_m=='FVS_VanLeer': F_num=van_leer_fvs_flux(P_L,P_R,gam_v)
    elif flux_m=='FDS_Roe': F_num=roe_fds_flux(P_L,P_R,gam_v)
    else: raise ValueError(f"Unknown flux method: {flux_m}")
    return -(F_num[:,1:]-F_num[:,:-1])/dx_v

# --- Exact Sod Solver (Based on Toro's book, Chapter 4) ---
def exact_sod_solution(x_pts, t, g, rhoL, uL, pL, rhoR, uR, pR):
    aL=np.sqrt(g*max(pL,1e-9)/max(rhoL,1e-9)); aR=np.sqrt(g*max(pR,1e-9)/max(rhoR,1e-9))
    if abs(t)<1e-12:
        r_s=np.where(x_pts<0,rhoL,rhoR); u_s=np.where(x_pts<0,uL,uR); p_s=np.where(x_pts<0,pL,pR)
        return r_s,u_s,p_s,p_s/(g-1.)+0.5*r_s*u_s**2
    def s_t_r_f(p_s_v,pK,rhoK,aK):
        pK_s=max(pK,1e-9); rhoK_s=max(rhoK,1e-9); aK_s=max(aK,1e-9)
        if p_s_v>pK_s:
            AK=2./((g+1.)*rhoK_s); BK=(g-1.)/(g+1.)*pK_s
            return (p_s_v-pK_s)*np.sqrt(AK/max(p_s_v+BK,1e-9))
        return (2.*aK_s/(g-1.))*( (max(p_s_v,0)/pK_s)**((g-1.)/(2.*g)) -1.)
    def p_f_r(p_s_g_arr):
        p_s_g=max(p_s_g_arr[0],1e-9)
        return s_t_r_f(p_s_g,pL,rhoL,aL)+s_t_r_f(p_s_g,pR,rhoR,aR)+(uR-uL)
    p_s_g_i=0.5*(pL+pR); p_s_g_i=max(p_s_g_i,1e-6)
    try: p_s=fsolve(p_f_r,[p_s_g_i],xtol=1e-12)[0]
    except:
        p_pv_n=(aL+aR-0.5*(g-1.)*(uR-uL))
        p_L_pt=aL/(max(pL,1e-9)**((g-1.)/(2.*g))); p_R_pt=aR/(max(pR,1e-9)**((g-1.)/(2.*g)))
        p_pv_d=max(p_L_pt+p_R_pt,1e-9)
        p_s_g_a=(p_pv_n/p_pv_d)**((2.*g)/(g-1.)); p_s_g_a=max(1e-6,p_s_g_a if np.isfinite(p_s_g_a) else 1e-6)
        p_s=fsolve(p_f_r,[p_s_g_a],xtol=1e-12)[0]
    p_s=max(p_s,1e-9)
    u_s=0.5*(uL+uR)+0.5*(s_t_r_f(p_s,pR,rhoR,aR)-s_t_r_f(p_s,pL,rhoL,aL))
    pL_s,rhoL_s=max(pL,1e-9),max(rhoL,1e-9); pR_s,rhoR_s=max(pR,1e-9),max(rhoR,1e-9)
    if p_s>pL_s: rho_sL=rhoL_s*((p_s/pL_s+(g-1.)/(g+1.))/((g-1.)/(g+1.)*(p_s/pL_s)+1.))
    else: rho_sL=rhoL_s*(p_s/pL_s)**(1./g)
    if p_s>pR_s: rho_sR=rhoR_s*((p_s/pR_s+(g-1.)/(g+1.))/((g-1.)/(g+1.)*(p_s/pR_s)+1.))
    else: rho_sR=rhoR_s*(p_s/pR_s)**(1./g)
    rho_sL=max(rho_sL,1e-9); rho_sR=max(rho_sR,1e-9); S_C=u_s
    if p_s>pL_s: S_L_sh=uL-aL*np.sqrt(((g+1.)/(2.*g))*(p_s/pL_s)+((g-1.)/(2.*g)))
    else: a_sL=max(aL*(p_s/pL_s)**((g-1.)/(2.*g)),1e-9); S_HL_r=uL-aL; S_TL_r=u_s-a_sL
    if p_s>pR_s: S_R_sh=uR+aR*np.sqrt(((g+1.)/(2.*g))*(p_s/pR_s)+((g-1.)/(2.*g)))
    else: a_sR=max(aR*(p_s/pR_s)**((g-1.)/(2.*g)),1e-9); S_HR_r=uR+aR; S_TR_r=u_s+a_sR
    rs,us,ps=np.zeros_like(x_pts),np.zeros_like(x_pts),np.zeros_like(x_pts)
    for i,x_v in enumerate(x_pts):
        s_q=x_v/t
        if s_q<=S_C:
            if p_s>pL_s:
                if s_q<=S_L_sh: rs[i],us[i],ps[i]=rhoL,uL,pL
                else: rs[i],us[i],ps[i]=rho_sL,u_s,p_s
            else:
                if s_q<=S_HL_r: rs[i],us[i],ps[i]=rhoL,uL,pL
                elif s_q<=S_TL_r:
                    aL_s=max(aL,1e-9)
                    us[i]=(2./(g+1.))*(aL_s+(g-1.)/2.*uL+s_q)
                    cf=max((2./(g+1.))+((g-1.)/((g+1.)*aL_s))*(uL-s_q),0)
                    rs[i]=rhoL_s*cf**(2./(g-1.)); ps[i]=pL_s*cf**(2.*g/(g-1.))
                else: rs[i],us[i],ps[i]=rho_sL,u_s,p_s
        else:
            if p_s>pR_s:
                if s_q>=S_R_sh: rs[i],us[i],ps[i]=rhoR,uR,pR
                else: rs[i],us[i],ps[i]=rho_sR,u_s,p_s
            else:
                if s_q>=S_HR_r: rs[i],us[i],ps[i]=rhoR,uR,pR
                elif s_q>=S_TR_r:
                    aR_s=max(aR,1e-9)
                    us[i]=(2./(g+1.))*(-aR_s+(g-1.)/2.*uR+s_q)
                    cf=max((2./(g+1.))-((g-1.)/((g+1.)*aR_s))*(uR-s_q),0)
                    rs[i]=rhoR_s*cf**(2./(g-1.)); ps[i]=pR_s*cf**(2.*g/(g-1.))
                else: rs[i],us[i],ps[i]=rho_sR,u_s,p_s
    rs=np.maximum(rs,1e-9); ps=np.maximum(ps,1e-9)
    return rs,us,ps,ps/(g-1.)+0.5*rs*us**2

# --- Main Simulation Runner for a Single Configuration ---
def run_simulation_single(U_init,t_fin,dx_s,cfl,nx_s,recon,flux,lim=None,gam=gamma,verb=True):
    U=np.copy(U_init); t_c=0.0; n_iter=0; t_start=time.time()
    if verb: print(f"  Running: {recon}/{lim if lim else ''}-{flux} to T={t_fin:.2f}")
    while t_c<t_fin:
        rho,u,p=conserved_to_primitive(U,gam)
        a=np.sqrt(gam*np.maximum(p,1e-9)/np.maximum(rho,1e-9))
        dt=cfl*dx_s/np.max(np.abs(u)+a) if np.max(np.abs(u)+a)>1e-9 else cfl*dx_s
        if t_c+dt>t_fin: dt=t_fin-t_c
        if dt<=1e-12: break
        r1=calculate_rhs(U,dx_s,recon,flux,lim,N_GHOST,nx_s,gam)
        U1=U+dt*r1
        r2=calculate_rhs(U1,dx_s,recon,flux,lim,N_GHOST,nx_s,gam)
        U2=0.75*U+0.25*U1+0.25*dt*r2
        r3=calculate_rhs(U2,dx_s,recon,flux,lim,N_GHOST,nx_s,gam)
        U=(1./3.)*U+(2./3.)*U2+(2./3.)*dt*r3
        t_c+=dt; n_iter+=1
        if verb and n_iter%200==0: print(f"    Iter:{n_iter}, T:{t_c:.3f}/{t_fin:.3f}")
    if verb: print(f"  Finished in {time.time()-t_start:.2f}s. Iters: {n_iter}")
    return U


# --- Function to run simulation and collect frames for animation ---
def run_simulation_and_collect_frames(U_init_sim, t_final_anim, dx_sim, cfl_val, nx_val,
                                     config_dict, gamma_val, anim_frame_dt_target): # Pass full config
    U = np.copy(U_init_sim)
    t_curr = 0.0
    iter_count = 0
    animation_frames = []
    
    reconstruction_method = config_dict['recon']
    flux_method = config_dict['flux']
    limiter = config_dict.get('limiter', None) # Use .get for optional keys
    scheme_label = config_dict['label']


    # Store initial state (t=0)
    rho_iter, u_iter, p_iter = conserved_to_primitive(U, gamma_val)
    e_iter = p_iter / ((gamma_val - 1.0) * np.maximum(rho_iter, 1e-9))
    animation_frames.append({
        't': 0.0,
        'rho': np.copy(rho_iter), 'u': np.copy(u_iter),
        'p': np.copy(p_iter), 'e': np.copy(e_iter)
    })

    next_frame_capture_time = anim_frame_dt_target
    progress_interval = max(0.01, t_final_anim / 10.0) # Ensure progress interval is reasonable
    next_progress_print_time = progress_interval

    print(f"  Sim for Anim Frame Collection: {scheme_label} to T={t_final_anim:.2f}")
    sim_start_time = time.time()

    while t_curr < t_final_anim:
        rho_iter, u_iter, p_iter = conserved_to_primitive(U, gamma_val)
        a_iter = np.sqrt(gamma_val * np.maximum(p_iter, 1e-9) / np.maximum(rho_iter, 1e-9))
        max_speed = np.max(np.abs(u_iter) + a_iter)
        dt_val = cfl_val * dx_sim / max_speed if max_speed > 1e-9 else cfl_val * dx_sim

        if t_curr + dt_val > t_final_anim: dt_val = t_final_anim - t_curr
        if dt_val <= 1e-12: break

        rhs1 = calculate_rhs(U, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U_1 = U + dt_val * rhs1
        rhs2 = calculate_rhs(U_1, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U_2 = 0.75 * U + 0.25 * U_1 + 0.25 * dt_val * rhs2
        rhs3 = calculate_rhs(U_2, dx_sim, reconstruction_method, flux_method, limiter, N_GHOST, nx_val, gamma_val)
        U_new = (1.0/3.0) * U + (2.0/3.0) * U_2 + (2.0/3.0) * dt_val * rhs3

        t_new = t_curr + dt_val
        U = U_new

        if t_new >= next_frame_capture_time - 1e-9:
            while next_frame_capture_time <= t_new + 1e-9 and next_frame_capture_time <= t_final_anim + 1e-9 :
                rho_f, u_f, p_f = conserved_to_primitive(U, gamma_val)
                e_f = p_f / ((gamma_val - 1.0) * np.maximum(rho_f, 1e-9))
                animation_frames.append({
                    't': t_new, 'rho': np.copy(rho_f), 'u': np.copy(u_f),
                    'p': np.copy(p_f), 'e': np.copy(e_f)
                })
                next_frame_capture_time += anim_frame_dt_target
        
        t_curr = t_new
        iter_count += 1

        if t_curr >= next_progress_print_time and t_final_anim > 0:
            print(f"    {scheme_label[:20]:<20s} Anim Sim: T {t_curr:.3f}/{t_final_anim:.3f} ({t_curr/t_final_anim*100:.1f}%), Frames: {len(animation_frames)}")
            while next_progress_print_time <= t_curr and next_progress_print_time <= t_final_anim :
                 next_progress_print_time += progress_interval
    
    print(f"  Finished Anim Frame Collection for {scheme_label} in {time.time()-sim_start_time:.2f}s. Frames: {len(animation_frames)}")
    return animation_frames


# --- Main Execution ---
if __name__ == "__main__":
    U_initial = np.zeros((3, NX))
    U_L_cons_init = primitive_to_conserved(rho_L_init, u_L_init, p_L_init, gamma)
    U_R_cons_init = primitive_to_conserved(rho_R_init, u_R_init, p_R_init, gamma)
    for i in range(NX):
        if X_CELL_CENTERS[i] < 0: U_initial[:, i] = U_L_cons_init
        else: U_initial[:, i] = U_R_cons_init

    x_exact_plot = np.linspace(XMIN, XMAX, 500)

    schemes_to_test = [
        {'recon': 'MUSCL', 'flux': 'FVS_VanLeer', 'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FVS(VL)'},
        {'recon': 'MUSCL', 'flux': 'FVS_VanLeer', 'limiter': 'Superbee', 'label': 'MUSCL(SB)-FVS(VL)'},
        {'recon': 'WENO3', 'flux': 'FVS_VanLeer', 'limiter': None,       'label': 'WENO3-FVS(VL)'},
        {'recon': 'MUSCL', 'flux': 'FDS_Roe',     'limiter': 'VanLeer',  'label': 'MUSCL(VL)-FDS(Roe)'},
        {'recon': 'MUSCL', 'flux': 'FDS_Roe',     'limiter': 'Superbee', 'label': 'MUSCL(SB)-FDS(Roe)'},
        {'recon': 'WENO3', 'flux': 'FDS_Roe',     'limiter': None,       'label': 'WENO3-FDS(Roe)'},
        {'recon': 'MUSCL_CHAR', 'flux': 'FVS_VanLeer', 'limiter': 'VanLeer',  'label': 'MUSCL_Char(VL)-FVS(VL)'},
        {'recon': 'MUSCL_CHAR', 'flux': 'FVS_VanLeer', 'limiter': 'Superbee', 'label': 'MUSCL_Char(SB)-FVS(VL)'},
        {'recon': 'MUSCL_CHAR', 'flux': 'FDS_Roe', 'limiter': 'Superbee', 'label': 'MUSCL_Char(SB)-FDS(Roe)'},
    ]

    # Assign colors to schemes_to_test for consistent plotting
    num_schemes = len(schemes_to_test)
    if num_schemes <= 10: cmap = plt.cm.get_cmap('tab10', num_schemes)
    elif num_schemes <= 20: cmap = plt.cm.get_cmap('tab20', num_schemes)
    else: cmap = plt.cm.get_cmap('viridis', num_schemes)
    
    for i, config in enumerate(schemes_to_test):
        config['color'] = cmap(i)


    RUN_STATIC_COMPARISON = True
    if RUN_STATIC_COMPARISON:
        print(f"--- Running Static Scheme Comparison up to T={T_FINAL_STATIC_PLOT} ---")
        rho_ex_stat, u_ex_stat, p_ex_stat, _ = exact_sod_solution(x_exact_plot, T_FINAL_STATIC_PLOT, gamma,
                                                      rho_L_init, u_L_init, p_L_init,
                                                      rho_R_init, u_R_init, p_R_init)
        e_ex_specific_stat = p_ex_stat / ((gamma - 1.0) * np.maximum(rho_ex_stat, 1e-9))
        results_all_static = {}
        print(f"Starting static simulations for {len(schemes_to_test)} schemes...\n")
        for config in schemes_to_test:
            U_final_run = run_simulation_single(U_initial, T_FINAL_STATIC_PLOT, DX, CFL_CONST, NX,
                                            config['recon'], config['flux'], config.get('limiter'), gamma)
            rho_n, u_n, p_n = conserved_to_primitive(U_final_run, gamma)
            e_n_spec = p_n / ((gamma - 1.0) * np.maximum(rho_n,1e-9))
            results_all_static[config['label']] = {'rho':rho_n,'u':u_n,'p':p_n,'e':e_n_spec}

        plot_vars_static = [('Density (ρ)','rho',rho_ex_stat), ('Velocity (u)','u',u_ex_stat),
                            ('Pressure (p)','p',p_ex_stat), ('Specific Internal Energy (e)','e',e_ex_specific_stat)]
        plt.figure(figsize=(18, 14))
        for i_v, (title, key, ex_sol) in enumerate(plot_vars_static):
            plt.subplot(2,2,i_v+1); plt.plot(x_exact_plot,ex_sol,'k-',lw=2.5,label='Exact',zorder=100)
            for config in schemes_to_test:
                plt.plot(X_CELL_CENTERS, results_all_static[config['label']][key], marker='.' if NX<=100 else 'None',
                         markersize=4, linestyle='-', color=config['color'], label=config['label'], alpha=0.8)
            plt.title(f'{title} at t={T_FINAL_STATIC_PLOT:.2f}'); plt.xlabel('x'); plt.ylabel(title.split(' ')[0])
            plt.legend(fontsize='x-small',loc='best'); plt.grid(True,ls=':',alpha=0.6)
        plt.suptitle(f'Sod Shock Tube Static Comparison - NX={NX}, T={T_FINAL_STATIC_PLOT}', fontsize=16, y=0.99)
        plt.tight_layout(rect=[0,0,1,0.96]); plt.savefig(f"sod_tube_static_NX{NX}_T{T_FINAL_STATIC_PLOT}.png")
        print(f"\nStatic plot saved. Showing now...")
        plt.show()

    # --- Part 2: Generate multi-scheme animation ---
    CREATE_ANIMATION = True
    ANIM_T_FINAL = 5
    ANIM_FRAME_DT_TARGET = 0.01
    ANIM_FPS = 60 # Adjusted FPS

    if CREATE_ANIMATION:
        print(f"\n--- Generating Multi-Scheme Animation up to T={ANIM_T_FINAL} ---")
        all_animation_frames_data = {}
        min_frames_collected = float('inf')

        for config in schemes_to_test:
            frames_list = run_simulation_and_collect_frames(
                U_initial, ANIM_T_FINAL, DX, CFL_CONST, NX,
                config, gamma, ANIM_FRAME_DT_TARGET
            )
            all_animation_frames_data[config['label']] = frames_list
            if frames_list: min_frames_collected = min(min_frames_collected, len(frames_list))
            else: min_frames_collected = 0 # If any scheme fails to produce frames

        if not all_animation_frames_data or min_frames_collected < 2:
            print("Not enough frames collected across all schemes for animation. Exiting animation generation.")
        else:
            num_render_frames = min_frames_collected
            print(f"Collected data for all schemes. Will render {num_render_frames} frames in animation.")

            fig_anim, axes_anim_flat = plt.subplots(2, 2, figsize=(16, 11)) # Slightly adjusted size
            axes_anim = {'rho': axes_anim_flat[0,0], 'u': axes_anim_flat[0,1],
                         'p': axes_anim_flat[1,0], 'e': axes_anim_flat[1,1]}
            plot_vars_keys = ['rho', 'u', 'p', 'e']
            plot_vars_titles = ['Density (ρ)', 'Velocity (u)', 'Pressure (p)', 'Specific Internal Energy (e)']
            
            lines_numerical_all_schemes = {cfg['label']: {} for cfg in schemes_to_test}
            lines_exact_anim = {}

            # Setup lines for exact solution
            for key_idx, key_var in enumerate(plot_vars_keys):
                ax = axes_anim[key_var]
                line_e, = ax.plot([], [], 'k-', lw=2.5, label='Exact', zorder=200) # Exact on top
                lines_exact_anim[key_var] = line_e
            
            # Setup lines for numerical solutions
            for config in schemes_to_test:
                scheme_label = config['label']
                color = config['color']
                for key_idx, key_var in enumerate(plot_vars_keys):
                    ax = axes_anim[key_var]
                    line_n, = ax.plot([], [], lw=1.5, label=scheme_label, color=color, alpha=0.7)
                    lines_numerical_all_schemes[scheme_label][key_var] = line_n

            # Configure axes properties (titles, labels, grid, legend)
            for key_idx, key_var in enumerate(plot_vars_keys):
                ax = axes_anim[key_var]
                ax.set_title(plot_vars_titles[key_idx].split(' ')[0])
                ax.set_xlabel('x'); ax.set_ylabel(plot_vars_titles[key_idx].split(' ')[0])
                ax.grid(True, linestyle=':', alpha=0.6)
            
            # Set y-axis limits robustly
            min_max_for_ylim = {key: [np.inf, -np.inf] for key in plot_vars_keys}
            # Exact solution over time
            anim_times_for_ylim = np.linspace(0, ANIM_T_FINAL, 20)
            for t_sample in anim_times_for_ylim:
                r_ex_s,u_ex_s,p_ex_s,_ = exact_sod_solution(x_exact_plot,t_sample,gamma,rho_L_init,u_L_init,p_L_init,rho_R_init,u_R_init,p_R_init)
                e_ex_s = p_ex_s / ((gamma-1.0)*np.maximum(r_ex_s,1e-9))
                ex_s_data = {'rho':r_ex_s, 'u':u_ex_s, 'p':p_ex_s, 'e':e_ex_s}
                for key in plot_vars_keys:
                    min_max_for_ylim[key][0]=min(min_max_for_ylim[key][0], np.min(ex_s_data[key]))
                    min_max_for_ylim[key][1]=max(min_max_for_ylim[key][1], np.max(ex_s_data[key]))
            # Numerical solutions (first and last rendered frame)
            for config in schemes_to_test:
                frames = all_animation_frames_data[config['label']]
                if len(frames) >= num_render_frames: # Ensure this scheme has enough frames
                    f_init = frames[0]; f_final = frames[num_render_frames-1]
                    for key in plot_vars_keys:
                        min_max_for_ylim[key][0]=min(min_max_for_ylim[key][0],np.min(f_init[key]),np.min(f_final[key]))
                        min_max_for_ylim[key][1]=max(min_max_for_ylim[key][1],np.max(f_init[key]),np.max(f_final[key]))
            
            for key in plot_vars_keys:
                ax = axes_anim[key]
                min_v, max_v = min_max_for_ylim[key]
                span = max_v - min_v
                if span < 1e-6*abs(max_v) or span < 1e-6: span = abs(max_v)*0.2 if abs(max_v)>1e-6 else 0.2
                if span < 1e-6: span = 0.2
                low_lim=min_v-0.1*span; upp_lim=max_v+0.1*span
                if key in ['rho','p','e']: low_lim=max(0,low_lim)
                ax.set_ylim(low_lim, upp_lim); ax.set_xlim(XMIN, XMAX)

            # Add legend to one of the subplots (e.g., top-left)
            axes_anim['rho'].legend(fontsize='xx-small', loc='upper right', ncol=2)


            fig_anim.suptitle('', fontsize=16)
            fig_anim.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            # Determine the reference scheme for time (e.g., the first one that has enough frames)
            ref_scheme_label_for_time = None
            for config in schemes_to_test:
                if len(all_animation_frames_data[config['label']]) >= num_render_frames:
                    ref_scheme_label_for_time = config['label']
                    break
            if ref_scheme_label_for_time is None: # Should not happen if num_render_frames > 0
                print("Error: Could not find a reference scheme for animation timing.")


            def update_anim_plot_multi(frame_idx):
                # Time from reference scheme, or calculate based on frame_idx and target dt
                # For simplicity, use time from the first scheme's frame data
                t_current_frame = all_animation_frames_data[ref_scheme_label_for_time][frame_idx]['t']
                
                artists_changed = []
                # Exact solution
                r_ex_f,u_ex_f,p_ex_f,_=exact_sod_solution(x_exact_plot,t_current_frame,gamma,rho_L_init,u_L_init,p_L_init,rho_R_init,u_R_init,p_R_init)
                e_ex_f=p_ex_f/((gamma-1.0)*np.maximum(r_ex_f,1e-9))
                exact_f_data={'rho':r_ex_f,'u':u_ex_f,'p':p_ex_f,'e':e_ex_f}
                for key in plot_vars_keys:
                    lines_exact_anim[key].set_data(x_exact_plot, exact_f_data[key])
                    artists_changed.append(lines_exact_anim[key])

                # Numerical solutions for all schemes
                for config in schemes_to_test:
                    scheme_label = config['label']
                    # Ensure the scheme has enough frames before accessing
                    if len(all_animation_frames_data[scheme_label]) > frame_idx:
                        frame_data_scheme = all_animation_frames_data[scheme_label][frame_idx]
                        for key in plot_vars_keys:
                            lines_numerical_all_schemes[scheme_label][key].set_data(X_CELL_CENTERS, frame_data_scheme[key])
                            artists_changed.append(lines_numerical_all_schemes[scheme_label][key])
                
                title_obj = fig_anim.texts[0] if fig_anim.texts else fig_anim.suptitle('')
                title_obj.set_text(f'Sod Shock Tube Multi-Scheme Comparison at t={t_current_frame:.3f}s (NX={NX})')
                artists_changed.append(title_obj)
                return artists_changed

            ani = animation.FuncAnimation(fig_anim, update_anim_plot_multi, frames=num_render_frames,
                                interval=1000/ANIM_FPS, blit=True, repeat=False)
            
            anim_filename = f"sod_tube_anim_T{ANIM_T_FINAL}_MULTI_NX{NX}.gif"
            print(f"Saving multi-scheme animation to {anim_filename} ({num_render_frames} frames, {ANIM_FPS} FPS)...")
            try:
                ani.save(anim_filename, writer='pillow', fps=ANIM_FPS)
                print(f"Multi-scheme animation saved: {anim_filename}")
            except Exception as e:
                print(f"Error saving multi-scheme animation: {e}")
                print("Ensure 'pillow' is installed (pip install pillow).")
            plt.close(fig_anim)


  