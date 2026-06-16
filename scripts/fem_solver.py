#!/usr/bin/env python3
"""
结构动力学 Q3 — Euler-Bernoulli 梁 FEM 求解器 (Python 实现)
算法与 fem_beam.f90 完全一致:
  直接刚度法组装 → 一致质量矩阵 → 广义特征值问题 K·φ = ω²·M·φ
  → scipy.linalg.eigh (底层 LAPACK DSYGV, 与 Fortran 版算法相同)
输出 6 个 CSV 文件供后处理使用。
"""

import numpy as np
from scipy.linalg import eigh, cholesky
from scipy.integrate import simpson
import csv, os

# ============================================================
# 0. 物理参数 (与 fem_beam.f90 完全一致)
# ============================================================
ell     = 1.0               # 特征长度 [m]
E_mod   = 2.10e11           # 弹性模量 [Pa]
rho     = 7800.0            # 密度 [kg/m³]
S_area  = 0.01              # 截面积 [m²]
J_inert = 8.33e-6           # 截面惯性矩 [m⁴]
EI      = E_mod * J_inert   # 弯曲刚度 [N·m²]
mu      = rho * S_area      # 线密度 [kg/m]
m_mass  = mu * ell * 0.8    # 集中质量 [kg]
k_spr   = EI / ell**3 * 30.0  # 弹簧刚度 [N/m]
v0      = 0.1               # C 点初始速度 [m/s]

# ============================================================
# 1. FEM 离散化
# ============================================================
n_seg      = 4
ne_per_seg = 10
ne         = n_seg * ne_per_seg          # 总单元数 = 40
nn         = ne + 1                      # 总节点数 = 41
ndof_total = 2 * nn                     # 总自由度数 = 82
le         = ell / ne_per_seg            # 单元长度

# 节点坐标
x_nodes = np.array([i * le for i in range(nn)])

# 关键节点 (0-based Python)
node_B = ne_per_seg            # = 10, x = ell
node_C = 2 * ne_per_seg        # = 20, x = 2*ell
node_D = 3 * ne_per_seg        # = 30, x = 3*ell

# DOF 编号 (0-based): w_i = 2*i, θ_i = 2*i+1
dof_B_w = 2 * node_B           # = 20
dof_C_w = 2 * node_C           # = 40
dof_D_w = 2 * node_D           # = 60

# ============================================================
# 2. 单元矩阵 (Hermite 梁单元)
# ============================================================
def elem_stiffness(le_len):
    """Euler-Bernoulli 梁单元刚度矩阵 [12, 6L, -12, 6L; ...] * EI/L³"""
    L = le_len
    k = np.array([
        [12.0,   6.0*L,  -12.0,   6.0*L],
        [6.0*L,  4.0*L*L, -6.0*L, 2.0*L*L],
        [-12.0, -6.0*L,   12.0,  -6.0*L],
        [6.0*L,  2.0*L*L, -6.0*L, 4.0*L*L]
    ])
    return k * (EI / L**3)

def elem_mass(le_len):
    """Euler-Bernoulli 梁单元一致质量矩阵 * ρS·L/420"""
    L = le_len
    m = np.array([
        [156.0,   22.0*L,   54.0,  -13.0*L],
        [22.0*L,   4.0*L*L, 13.0*L, -3.0*L*L],
        [54.0,    13.0*L,  156.0,  -22.0*L],
        [-13.0*L, -3.0*L*L, -22.0*L, 4.0*L*L]
    ])
    return m * (mu * L / 420.0)

# ============================================================
# 3. 组装全局矩阵
# ============================================================
K_global = np.zeros((ndof_total, ndof_total))
M_global = np.zeros((ndof_total, ndof_total))

for e in range(ne):
    n1 = e        # 左节点
    n2 = e + 1    # 右节点
    dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1]
    Ke = elem_stiffness(le)
    Me = elem_mass(le)
    for i_local in range(4):
        for j_local in range(4):
            gi, gj = dofs[i_local], dofs[j_local]
            K_global[gi, gj] += Ke[i_local, j_local]
            M_global[gi, gj] += Me[i_local, j_local]

# ---- 添加弹簧与集中质量 ----
K_global[dof_D_w, dof_D_w] += k_spr
M_global[dof_C_w, dof_C_w] += m_mass

# ============================================================
# 4. 消去铰支约束 (dof_B_w)
# ============================================================
constrained_dof = dof_B_w
active_dofs = [i for i in range(ndof_total) if i != constrained_dof]
n_active = len(active_dofs)

K_red = K_global[np.ix_(active_dofs, active_dofs)]
M_red = M_global[np.ix_(active_dofs, active_dofs)]

# ============================================================
# 5. 求解广义特征值问题 K·φ = ω²·M·φ
#    scipy.linalg.eigh → LAPACK DSYGV (与 Fortran 版算法一致)
# ============================================================
eig_vals, eig_vecs_red = eigh(K_red, M_red)

# 按特征值升序排列 (eigh 已排序)
omegas = np.sqrt(np.maximum(eig_vals, 0.0))
betas = (eig_vals * mu / EI)**0.25 * ell
freqs_hz = omegas / (2.0 * np.pi)

# ============================================================
# 6. 重构完整振型 (含被约束 DOF = 0)
# ============================================================
n_modes = min(n_active, 12)
mode_shapes_full = np.zeros((ndof_total, n_modes))
for j in range(n_modes):
    for i in range(n_active):
        mode_shapes_full[active_dofs[i], j] = eig_vecs_red[i, j]

# ============================================================
# 7. 振型归一化: M_n = φ_nᵀ · M · φ_n = 1
# ============================================================
modal_masses = np.zeros(n_modes)
mass_dist = np.zeros(n_modes)    # 分布质量贡献
mass_conc = np.zeros(n_modes)    # 集中质量贡献

for n in range(n_modes):
    phi = mode_shapes_full[:, n]
    M_n_raw = phi @ M_global @ phi
    scale = 1.0 / np.sqrt(M_n_raw) if M_n_raw > 1e-30 else 1.0
    mode_shapes_full[:, n] *= scale
    # 重算归一化后的分量
    phi = mode_shapes_full[:, n]
    modal_masses[n] = phi @ M_global @ phi  # 应为 1.0
    mass_dist[n] = phi @ (M_global - np.diag(np.diag(M_global) * 0)) @ phi
    # 重新准确计算
    M_no_conc = M_global.copy()
    M_no_conc[dof_C_w, dof_C_w] -= m_mass
    mass_dist[n] = phi @ M_no_conc @ phi
    mass_conc[n] = m_mass * phi[dof_C_w]**2

# ============================================================
# 8. 振型插值到密集网格
# ============================================================
def interpolate_mode_shape(mode_idx, n_pts=1001):
    """用 Hermite 插值在密集网格上计算振型"""
    xi = np.linspace(0, 4 * ell, n_pts)
    Y = np.zeros(n_pts)
    phi = mode_shapes_full[:, mode_idx]

    for i_pt, x in enumerate(xi):
        # 找到 x 所在的单元
        elem_idx = min(int(x / le), ne - 1)
        if elem_idx < 0:
            elem_idx = 0
        if elem_idx >= ne:
            elem_idx = ne - 1

        n1, n2 = elem_idx, elem_idx + 1
        x1 = n1 * le
        xi_local = (x - x1) / le  # [0, 1]

        # Hermite 形函数
        H1 = 1.0 - 3.0*xi_local**2 + 2.0*xi_local**3
        H2 = le * (xi_local - 2.0*xi_local**2 + xi_local**3)
        H3 = 3.0*xi_local**2 - 2.0*xi_local**3
        H4 = le * (-xi_local**2 + xi_local**3)

        w1, th1 = phi[2*n1], phi[2*n1+1]
        w2, th2 = phi[2*n2], phi[2*n2+1]
        Y[i_pt] = H1*w1 + H2*th1 + H3*w2 + H4*th2

    return xi, Y

mode_shapes_interp = []
for n in range(n_modes):
    mode_shapes_interp.append(interpolate_mode_shape(n, 1001))

# ============================================================
# 9. 动力学响应 — C 点
# ============================================================
dt = 5e-5
t_end = 0.15
n_t = int(t_end / dt) + 1
t = np.linspace(0, t_end, n_t)

# C 点各阶振型值
phi_C = np.array([mode_shapes_full[dof_C_w, n] for n in range(n_modes)])

w_C = np.zeros(n_t)
v_C = np.zeros(n_t)
a_C = np.zeros(n_t)

for n in range(n_modes):
    if omegas[n] < 1e-10:
        continue
    coeff = m_mass * v0 * phi_C[n] / (modal_masses[n] * omegas[n])
    w_C += phi_C[n] * coeff * np.sin(omegas[n] * t)
    v_C += phi_C[n] * coeff * omegas[n] * np.cos(omegas[n] * t)
    a_C += phi_C[n] * (-coeff * omegas[n]**2) * np.sin(omegas[n] * t)

# ============================================================
# 10. 输出 CSV 文件
# ============================================================
os.makedirs('output', exist_ok=True)

# --- frequencies.csv ---
with open('output/frequencies.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode', 'beta', 'omega_rad_s', 'freq_Hz', 'period_s'])
    for n in range(n_modes):
        w.writerow([n+1, f'{betas[n]:.8f}', f'{omegas[n]:.8f}',
                     f'{freqs_hz[n]:.8f}', f'{1.0/freqs_hz[n]:.8f}' if freqs_hz[n] > 0 else 'inf'])

# --- mode_shapes.csv ---
with open('output/mode_shapes.csv', 'w', newline='') as f:
    w = csv.writer(f)
    header = ['x_m']
    for n in range(n_modes):
        header.append(f'phi_{n+1}')
    w.writerow(header)
    xi, Y0 = mode_shapes_interp[0]
    for i in range(len(xi)):
        row = [f'{xi[i]:.8f}']
        for n in range(n_modes):
            row.append(f'{mode_shapes_interp[n][1][i]:.12f}')
        w.writerow(row)

# --- response_C.csv ---
with open('output/response_C.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['t_s', 'displacement_m', 'velocity_m_s', 'acceleration_m_s2'])
    for i in range(n_t):
        w.writerow([f'{t[i]:.8f}', f'{w_C[i]:.12e}', f'{v_C[i]:.12e}', f'{a_C[i]:.12e}'])

# --- beam_params.csv ---
with open('output/beam_params.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['parameter', 'value', 'unit'])
    params = [
        ('l', ell, 'm'), ('E', E_mod, 'Pa'), ('rho', rho, 'kg/m³'),
        ('S', S_area, 'm²'), ('J', J_inert, 'm⁴'), ('EI', EI, 'N·m²'),
        ('mu', mu, 'kg/m'), ('m', m_mass, 'kg'), ('k', k_spr, 'N/m'),
        ('v0', v0, 'm/s'), ('alpha', m_mass/(mu*ell), '-'),
        ('kappa', k_spr*ell**3/EI, '-'),
        ('ne_per_seg', ne_per_seg, '-'), ('ne', ne, '-'),
        ('nn', nn, '-'), ('ndof_total', ndof_total, '-'),
        ('n_active', n_active, '-')
    ]
    for row in params:
        w.writerow(row)

# --- modal_masses.csv ---
with open('output/modal_masses.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode', 'M_n', 'M_distributed', 'M_concentrated', 'conc_ratio'])
    for n in range(n_modes):
        ratio = mass_conc[n] / modal_masses[n] if modal_masses[n] > 0 else 0
        w.writerow([n+1, f'{modal_masses[n]:.8f}', f'{mass_dist[n]:.8f}',
                     f'{mass_conc[n]:.8f}', f'{ratio:.6f}'])

# --- mode_shapes_nodes.csv (节点值) ---
with open('output/mode_shapes_nodes.csv', 'w', newline='') as f:
    w = csv.writer(f)
    header = ['x_m', 'w', 'theta']
    for n in range(n_modes):
        header.append(f'w_mode_{n+1}')
    w.writerow(header)
    for i_node in range(nn):
        x = x_nodes[i_node]
        w_val = mode_shapes_full[2*i_node, 0]  # mode 1 placeholder for w column
        th_val = mode_shapes_full[2*i_node+1, 0]
        row = [f'{x:.8f}', f'{w_val:.12f}', f'{th_val:.12f}']
        for n in range(n_modes):
            row.append(f'{mode_shapes_full[2*i_node, n]:.12f}')
        w.writerow(row)

# ============================================================
# 11. 正交性验证
# ============================================================
max_offdiag = 0.0
for i in range(n_modes):
    for j in range(i+1, n_modes):
        val = abs(mode_shapes_full[:, i] @ M_global @ mode_shapes_full[:, j])
        max_offdiag = max(max_offdiag, val)

print("=" * 65)
print("  FEM Beam Solver — Python (algorithm ≡ fem_beam.f90)")
print("=" * 65)
print(f"  Nodes: {nn}, Elements: {ne}, DOFs: {ndof_total}")
print(f"  Active DOFs (after hinge constraint): {n_active}")
print(f"  Element length: {le:.6f} m")
print()
print(f"  Eigenvalue solver: scipy.linalg.eigh → LAPACK DSYGV")
print(f"  Same algorithm as Fortran Cholesky + standard eigenproblem")
print()
print("  Natural Frequencies:")
print(f"  {'n':>3s}  {'ω_n [rad/s]':>14s}  {'f_n [Hz]':>10s}  {'β_n':>10s}")
print("  " + "-" * 50)
for n in range(min(n_modes, 8)):
    print(f"  {n+1:3d}  {omegas[n]:14.4f}  {freqs_hz[n]:10.4f}  {betas[n]:10.5f}")

print()
print(f"  Orthogonality check: max off-diagonal = {max_offdiag:.2e}")
print(f"  All CSV files exported to ./output/")
print()
for n in range(min(n_modes, 6)):
    print(f"  Mode {n+1}: M_n={modal_masses[n]:.4f}, "
          f"dist={mass_dist[n]:.4f}, conc={mass_conc[n]:.4f}, "
          f"φ_C={mode_shapes_full[dof_C_w, n]:.5f}")
