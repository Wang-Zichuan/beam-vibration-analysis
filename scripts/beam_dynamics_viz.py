#!/usr/bin/env python3
"""
结构动力学 Q3 — Euler-Bernoulli 梁自由振动解析解可视化
传递矩阵法 + 频率方程求解 + 振型与动力学响应绘图

梁布局 (总长 4l):
  A(free) -- B(hinge) -- C(mass m) -- D(spring k) -- E(free)
   x=0        x=l         x=2l         x=3l           x=4l
"""

import numpy as np
from scipy.optimize import bisect
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import scienceplots  # noqa: F401 — 注册样式
import os, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 0. 参数设置 (SI 单位)
# ============================================================
# 梁几何与材料
l      = 1.0              # 特征长度 [m], 总长 L = 4l = 4 m
E_mod  = 2.1e11           # 弹性模量 [Pa] (钢)
rho    = 7800.0           # 密度 [kg/m³]
S_area = 0.01             # 截面积 [m²]
J      = 8.33e-6          # 截面惯性矩 [m⁴]  (矩形 ~0.1×0.1 m → b h³/12)

mu     = rho * S_area      # 线密度 [kg/m]
EJ     = E_mod * J          # 弯曲刚度 [N·m²]

# 集中质量与弹簧
m_mass = mu * l * 0.8     # 集中质量 [kg] （约等于一个梁段质量的 80%）
k_spring = EJ / l**3 * 30.0  # 弹簧刚度 [N/m] (无量纲 κ≈30)

# 初始条件
v0 = 0.1                  # C 点初始速度 [m/s]

# 无量纲参数
alpha = m_mass / (mu * l)   # 集中质量比
kappa = k_spring * l**3 / EJ  # 弹簧刚度比

print("=" * 65)
print("参数摘要")
print("=" * 65)
print(f"  l      = {l:.3f} m          (总长 4l = {4*l:.3f} m)")
print(f"  E      = {E_mod:.3e} Pa")
print(f"  ρ      = {rho:.1f} kg/m³")
print(f"  S      = {S_area:.4f} m²")
print(f"  J      = {J:.3e} m⁴")
print(f"  EJ     = {EJ:.3e} N·m²")
print(f"  μ      = {mu:.3f} kg/m")
print(f"  m      = {m_mass:.3f} kg")
print(f"  k      = {k_spring:.3e} N/m")
print(f"  α      = {alpha:.4f}      (m / μ l)")
print(f"  κ      = {kappa:.4f}      (k l³ / EJ)")
print(f"  v₀     = {v0:.3f} m/s")
print()


# ============================================================
# 1. 传递矩阵 P(s, β)
# ============================================================
def transfer_matrix(s, beta):
    """
    Euler-Bernoulli 梁段传递矩阵 P(s)。
    s  : 无量纲长度
    beta: 频率参数 β = λ l
    返回 4×4 矩阵，满足 z(ξ+s) = P(s) · z(ξ)
    状态向量 z = [Y, Y', Y'', Y''']^T
    """
    if beta < 1e-12:
        # 刚体模态极限
        P = np.eye(4)
        P[0, 1] = s
        P[0, 2] = s**2 / 2
        P[0, 3] = s**3 / 6
        P[1, 2] = s
        P[1, 3] = s**2 / 2
        P[2, 3] = s
        return P

    b = beta
    bs = b * s
    Ch, Sh = np.cosh(bs), np.sinh(bs)
    C, S   = np.cos(bs), np.sin(bs)

    P = np.zeros((4, 4))
    # 第 0 行: Y
    P[0, 0] = (Ch + C) / 2
    P[0, 1] = (Sh + S) / (2 * b)
    P[0, 2] = (Ch - C) / (2 * b**2)
    P[0, 3] = (Sh - S) / (2 * b**3)

    # 第 1 行: Y'
    P[1, 0] = b * (Sh - S) / 2
    P[1, 1] = (Ch + C) / 2
    P[1, 2] = (Sh + S) / (2 * b)
    P[1, 3] = (Ch - C) / (2 * b**2)

    # 第 2 行: Y''
    P[2, 0] = b**2 * (Ch - C) / 2
    P[2, 1] = b * (Sh - S) / 2
    P[2, 2] = (Ch + C) / 2
    P[2, 3] = (Sh + S) / (2 * b)

    # 第 3 行: Y'''
    P[3, 0] = b**3 * (Sh + S) / 2
    P[3, 1] = b**2 * (Ch - C) / 2
    P[3, 2] = b * (Sh - S) / 2
    P[3, 3] = (Ch + C) / 2

    return P


# ============================================================
# 2. 跳跃矩阵
# ============================================================
def jump_matrix_mass(beta):
    """集中质量 m 的跳跃矩阵 J_m (在 ξ=2)"""
    J = np.eye(4)
    J[3, 0] = alpha * beta**4
    return J

def jump_matrix_spring():
    """弹簧 k 的跳跃矩阵 J_k (在 ξ=3)"""
    J = np.eye(4)
    J[3, 0] = -kappa
    return J


# ============================================================
# 3. 构建特征矩阵 A(β) 并求解频率方程
# ============================================================
e1 = np.array([1.0, 0.0, 0.0, 0.0])
e3 = np.array([0.0, 0.0, 1.0, 0.0])
e4 = np.array([0.0, 0.0, 0.0, 1.0])
g1 = np.array([1.0, 0.0, 0.0, 0.0])
g2 = np.array([0.0, 1.0, 0.0, 0.0])


def build_A(beta):
    """
    构建 3×3 特征矩阵 A(β)。
    未知向量 c = [a, b, R]^T, 满足 A(β) c = 0。
    """
    P1   = transfer_matrix(1.0, beta)
    Jm   = jump_matrix_mass(beta)
    Jk   = jump_matrix_spring()
    # T(β) = P(1) · J_k · P(1) · J_m · P(1)
    T = P1 @ Jk @ P1 @ Jm @ P1

    A = np.zeros((3, 3))
    # Row 0: Y(1) = 0 → e1^T · P1 · [a·g1 + b·g2] = 0
    P1g1 = P1 @ g1
    P1g2 = P1 @ g2
    A[0, 0] = e1 @ P1g1
    A[0, 1] = e1 @ P1g2
    A[0, 2] = 0.0                 # R 不影响 Y(1⁻)

    # T·P1·g1, T·P1·g2, T·e4
    TP1g1 = T @ P1g1
    TP1g2 = T @ P1g2
    Te4   = T @ e4

    # Row 1: Y''(4) = 0
    A[1, 0] = e3 @ TP1g1
    A[1, 1] = e3 @ TP1g2
    A[1, 2] = e3 @ Te4

    # Row 2: Y'''(4) = 0
    A[2, 0] = e4 @ TP1g1
    A[2, 1] = e4 @ TP1g2
    A[2, 2] = e4 @ Te4

    return A


def det_A(beta):
    """计算 det A(β)"""
    if beta < 1e-13:
        return 0.0
    return np.linalg.det(build_A(beta))


def find_roots(beta_min=0.01, beta_max=20.0, n_scan=8000, n_roots=6):
    """
    扫描求 det A(β)=0 的根。
    返回 beta_n 列表。
    """
    betas = np.linspace(beta_min, beta_max, n_scan)
    dets = np.array([det_A(b) for b in betas])

    # 找符号变化
    roots = []
    for i in range(len(betas) - 1):
        if dets[i] * dets[i+1] < 0:
            try:
                root = bisect(det_A, betas[i], betas[i+1],
                              xtol=1e-10, maxiter=100)
                # 去重
                if not roots or abs(root - roots[-1]) > 1e-4:
                    roots.append(root)
            except Exception:
                pass
        if len(roots) >= n_roots:
            break

    return np.array(roots)


print("正在扫描频率特征根 β_n ...")
beta_roots = find_roots(beta_max=20.0, n_scan=10000)
print(f"  找到 {len(beta_roots)} 个特征根:")
for i, bn in enumerate(beta_roots):
    omega_n = bn**2 / l**2 * np.sqrt(EJ / mu)
    f_n = omega_n / (2 * np.pi)
    print(f"    β_{i+1} = {bn:.6f}  →  ω_{i+1} = {omega_n:.4f} rad/s  "
          f"→  f_{i+1} = {f_n:.4f} Hz")
print()


# ============================================================
# 4. 求解振型系数并计算振型
# ============================================================
def solve_mode_coeffs(beta):
    """对给定 β，从 A(β)·c=0 中求解 c=[a,b,R]^T (零空间)"""
    A = build_A(beta)
    # SVD 求零空间
    U, S, Vh = np.linalg.svd(A)
    c = Vh[-1, :]  # 最小奇异值对应的右奇异向量
    # 归一化: 令 max|Y(ξ)| = 1
    return c


def compute_mode_shape(beta, c, n_pts_per_seg=100):
    """
    从系数 c=[a,b,R]^T 计算全梁振型 Y(ξ), ξ∈[0,4]。
    返回 (xi, Y, Ypp) 其中 Ypp = Y''(ξ) (弯矩形状)。
    """
    a, b_coeff, R = c

    # 初始状态 z(0)
    z0 = np.array([a, b_coeff, 0.0, 0.0])

    P1 = transfer_matrix(1.0, beta)
    Jm = jump_matrix_mass(beta)
    Jk = jump_matrix_spring()

    # B 点前后
    z1m = P1 @ z0                      # 1⁻
    z1p = z1m + R * e4                 # 1⁺

    # C 点前后
    z2m = P1 @ z1p                     # 2⁻
    z2p = Jm @ z2m                     # 2⁺

    # D 点前后
    z3m = P1 @ z2p                     # 3⁻
    z3p = Jk @ z3m                     # 3⁺

    # E 点
    z4  = P1 @ z3p                     # 4

    # --- 各段插值 ---
    # AB: ξ∈[0,1]
    xi_ab  = np.linspace(0, 1, n_pts_per_seg, endpoint=False)
    Y_ab   = np.array([(transfer_matrix(xi, beta) @ z0)[0] for xi in xi_ab])
    Ypp_ab = np.array([(transfer_matrix(xi, beta) @ z0)[2] for xi in xi_ab])

    # BC: ξ∈[1,2]
    xi_bc  = np.linspace(0, 1, n_pts_per_seg, endpoint=False)
    Y_bc   = np.array([(transfer_matrix(xi, beta) @ z1p)[0] for xi in xi_bc])
    Ypp_bc = np.array([(transfer_matrix(xi, beta) @ z1p)[2] for xi in xi_bc])

    # CD: ξ∈[2,3]
    xi_cd  = np.linspace(0, 1, n_pts_per_seg, endpoint=False)
    Y_cd   = np.array([(transfer_matrix(xi, beta) @ z2p)[0] for xi in xi_cd])
    Ypp_cd = np.array([(transfer_matrix(xi, beta) @ z2p)[2] for xi in xi_cd])

    # DE: ξ∈[3,4]
    xi_de  = np.linspace(0, 1, n_pts_per_seg, endpoint=True)
    Y_de   = np.array([(transfer_matrix(xi, beta) @ z3p)[0] for xi in xi_de])
    Ypp_de = np.array([(transfer_matrix(xi, beta) @ z3p)[2] for xi in xi_de])

    # 拼接
    xi  = np.concatenate([xi_ab, xi_bc + 1, xi_cd + 2, xi_de + 3])
    Y   = np.concatenate([Y_ab, Y_bc, Y_cd, Y_de])
    Ypp = np.concatenate([Ypp_ab, Ypp_bc, Ypp_cd, Ypp_de])

    # 归一化: max|Y| = 1
    scale = np.max(np.abs(Y))
    if scale > 1e-14:
        Y /= scale
        Ypp /= scale

    return xi, Y, Ypp


# ============================================================
# 5. 模态质量与正交性验证
# ============================================================
def compute_modal_mass(beta, c, n_pts=800):
    """计算模态质量 M_n = ∫ μ φ² dx + m φ²(2l)"""
    xi, Y, _ = compute_mode_shape(beta, c, n_pts_per_seg=n_pts // 4 + 1)
    x = xi * l
    # ∫ μ φ² dx
    integral = simpson(Y**2, x)
    # 集中质量贡献 (ξ=2 → C 点)
    Y_C = Y[np.argmin(np.abs(xi - 2.0))]
    M_n = mu * integral + m_mass * Y_C**2
    return M_n


def check_orthogonality(betas, coeffs, n_pts=400):
    """验证前几阶模态的正交性"""
    print("正交性验证 (∫ μ φᵢ φⱼ dx + m φᵢ(2l) φⱼ(2l)):")
    N = len(betas)
    max_off = 0.0
    for i in range(N):
        xi_i, Y_i, _ = compute_mode_shape(betas[i], coeffs[i],
                                          n_pts_per_seg=n_pts // 4 + 1)
        x = xi_i * l
        Y_i_C = Y_i[np.argmin(np.abs(xi_i - 2.0))]
        for j in range(i + 1, N):
            xi_j, Y_j, _ = compute_mode_shape(betas[j], coeffs[j],
                                              n_pts_per_seg=n_pts // 4 + 1)
            Y_j_C = Y_j[np.argmin(np.abs(xi_j - 2.0))]
            # 使用较粗网格上的公共点
            x_common = x
            Y_i_interp = np.interp(x_common, xi_j * l, Y_j) if i != j else Y_j
            # 统一用 xi_i 的网格
            integral_ij = simpson(Y_i * np.interp(xi_i * l, xi_j * l, Y_j), x_common)
            ortho = mu * integral_ij + m_mass * Y_i_C * Y_j_C
            rel = abs(ortho) / (np.sqrt(compute_modal_mass(betas[i], coeffs[i],
                                       n_pts // 4 + 1) *
                               compute_modal_mass(betas[j], coeffs[j],
                                       n_pts // 4 + 1)) + 1e-30)
            if rel > max_off:
                max_off = rel
            if rel > 0.01:
                print(f"  φ_{i+1} · φ_{j+1} : 相对 {rel:.2e}  ⚠️")
    print(f"  最大非对角相对值: {max_off:.2e}")
    if max_off < 0.02:
        print("  正交性良好 ✓")
    print()


# ============================================================
# 6. 动力学响应
# ============================================================
def dynamic_response(betas, coeffs, t_max=0.5, n_t=2000, n_modes=None):
    """
    计算 C 点 (ξ=2) 的动力学响应 w(2l, t)。
    返回 (t, w_C, v_C, a_C)
    """
    if n_modes is None:
        n_modes = len(betas)
    use_modes = min(n_modes, len(betas))

    t = np.linspace(0, t_max, n_t)
    w_C = np.zeros_like(t)
    v_C = np.zeros_like(t)
    a_C = np.zeros_like(t)

    for n in range(use_modes):
        beta = betas[n]
        c = coeffs[n]
        omega_n = beta**2 / l**2 * np.sqrt(EJ / mu)
        M_n = compute_modal_mass(beta, c)
        xi, Y, _ = compute_mode_shape(beta, c, n_pts_per_seg=100)
        phi_C = Y[np.argmin(np.abs(xi - 2.0))]

        coeff_n = m_mass * v0 * phi_C / (M_n * omega_n)
        w_C += phi_C * coeff_n * np.sin(omega_n * t)
        v_C += phi_C * coeff_n * omega_n * np.cos(omega_n * t)
        a_C += -phi_C * coeff_n * omega_n**2 * np.sin(omega_n * t)

    return t, w_C, v_C, a_C


# ============================================================
# 7. 计算所有模态数据
# ============================================================
print("计算振型和模态质量...")
coeffs = []
mode_shapes = []
modal_masses = []
omegas = []
freqs_hz = []

for i, beta in enumerate(beta_roots):
    c = solve_mode_coeffs(beta)
    coeffs.append(c)
    xi, Y, Ypp = compute_mode_shape(beta, c, n_pts_per_seg=120)
    mode_shapes.append((xi, Y, Ypp))
    M_n = compute_modal_mass(beta, c)
    modal_masses.append(M_n)
    omega_n = beta**2 / l**2 * np.sqrt(EJ / mu)
    omegas.append(omega_n)
    freqs_hz.append(omega_n / (2 * np.pi))
    print(f"  模态 {i+1}: M_{i+1} = {M_n:.4f}, ω_{i+1} = {omega_n:.4f} rad/s")

print()

# 正交性验证
if len(beta_roots) >= 2:
    check_orthogonality(beta_roots, coeffs)


# ============================================================
# 8. 绘图 — 使用 scienceplots 样式
# ============================================================
plt.style.use(['science', 'no-latex', 'bright'])
# 设置中英文兼容字体
plt.rcParams['font.family'] = 'sans-serif'
# 若系统有 Times New Roman / Arial，优先使用
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.unicode_minus'] = False

# 颜色循环 (Nature 风格常用色)
colors = ['#0C5DA5', '#00B945', '#FF9500', '#FF2C00', '#845B97',
          '#474747', '#9e9e9e']


def save_figure(fig, name):
    """保存 PNG 和 PDF"""
    fig.savefig(f"{name}.png", dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    fig.savefig(f"{name}.pdf", bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"  已保存: {name}.png, {name}.pdf")


# ---- 图 1: 频率特征函数 det A(β) ----
print("绘制图 1: 频率特征函数...")
fig1, ax1 = plt.subplots(1, 1, figsize=(7, 4.5))
beta_scan = np.linspace(0.02, beta_roots[-1] * 1.15, 3000)
det_scan = np.array([det_A(b) for b in beta_scan])
ax1.plot(beta_scan, det_scan, color=colors[0], linewidth=0.9, label=r'$\det\mathbf{A}(\beta)$')
ax1.axhline(y=0, color='black', linewidth=0.5, linestyle='--')
for bn in beta_roots:
    ax1.axvline(x=bn, color=colors[3], linewidth=0.7, linestyle=':', alpha=0.7)
ax1.set_xlabel(r'$\beta$', fontsize=12)
ax1.set_ylabel(r'$\det\mathbf{A}(\beta)$', fontsize=12)
ax1.set_title('Frequency Characteristic Function', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9, frameon=True)
ax1.set_xlim(0, beta_scan[-1])
# 截断 y 轴以突出零点
ym = np.percentile(np.abs(det_scan), 95) * 2
ax1.set_ylim(-ym, ym)
ax1.tick_params(labelsize=10)
fig1.tight_layout()
save_figure(fig1, 'fig1_det_A_beta')
plt.close(fig1)


# ---- 图 2: 前 4 阶振型 (位移 + 弯矩) ----
print("绘制图 2: 振型...")
n_plot = min(4, len(beta_roots))
fig2, axes2 = plt.subplots(n_plot, 1, figsize=(8, 2.8 * n_plot), sharex=True)
if n_plot == 1:
    axes2 = [axes2]

for idx in range(n_plot):
    ax = axes2[idx]
    xi, Y, Ypp = mode_shapes[idx]
    x_phys = xi * l
    beta_n = beta_roots[idx]
    omega_n = omegas[idx]

    # 位移振型
    ax.plot(x_phys, Y, color=colors[0], linewidth=1.6,
            label=r'$\phi_{%d}$ (displacement)' % (idx + 1))
    # 弯矩振型 (归一化到合理范围)
    scale_pp = np.max(np.abs(Y)) / max(np.max(np.abs(Ypp)), 1e-12)
    # 实际用一个独立轴画弯矩
    ax2_m = ax.twinx()
    ax2_m.plot(x_phys, Ypp, color=colors[3], linewidth=1.2, linestyle='--',
               label=r"$\phi_{%d}''$ (moment)" % (idx + 1))
    ax2_m.set_ylabel(r"$\phi''$ (moment shape)", fontsize=9, color=colors[3])
    ax2_m.tick_params(axis='y', labelcolor=colors[3], labelsize=8)

    # 标注特殊点
    for xc, label in [(0, 'A'), (l, 'B'), (2*l, 'C'), (3*l, 'D'), (4*l, 'E')]:
        ax.axvline(x=xc, color='gray', linewidth=0.4, linestyle=':')
    ax.text(0, ax.get_ylim()[1] * 0.88, 'A', fontsize=8, ha='center', color='gray')
    ax.text(l, ax.get_ylim()[1] * 0.88, 'B', fontsize=8, ha='center', color='gray')
    ax.text(2*l, ax.get_ylim()[1] * 0.88, 'C', fontsize=8, ha='center', color='gray')
    ax.text(3*l, ax.get_ylim()[1] * 0.88, 'D', fontsize=8, ha='center', color='gray')
    ax.text(4*l, ax.get_ylim()[1] * 0.88, 'E', fontsize=8, ha='center', color='gray')

    ax.set_ylabel(r'$\phi_{%d}(\xi)$' % (idx + 1), fontsize=10)
    ax.set_title(
        f'Mode {idx + 1}:  '
        rf'$\beta_{idx+1}={beta_n:.3f}$,  '
        rf'$\omega_{idx+1}={omega_n:.2f}$ rad/s,  '
        rf'$f_{idx+1}={freqs_hz[idx]:.2f}$ Hz',
        fontsize=10, fontweight='bold')
    ax.tick_params(labelsize=8)
    # 合并图例
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_m.get_legend_handles_labels()
    if idx == 0:
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7.5,
                  loc='upper right', frameon=True)

ax.set_xlabel(r'$x$ [m]', fontsize=11)
fig2.tight_layout()
save_figure(fig2, 'fig2_mode_shapes')
plt.close(fig2)


# ---- 图 3: 模态质量分布 ----
print("绘制图 3: 模态质量...")
fig3, ax3 = plt.subplots(1, 1, figsize=(6.5, 4))
n_bars = min(6, len(modal_masses))
x_bar = np.arange(1, n_bars + 1)
ax3.bar(x_bar, modal_masses[:n_bars], color=colors[:n_bars], edgecolor='white',
        linewidth=0.8, alpha=0.85)
ax3.set_xlabel('Mode number $n$', fontsize=11)
ax3.set_ylabel(r'Modal mass $M_n$ [kg]', fontsize=11)
ax3.set_title('Modal Masses', fontsize=13, fontweight='bold')
ax3.set_xticks(x_bar)
ax3.tick_params(labelsize=10)
ax3.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
fig3.tight_layout()
save_figure(fig3, 'fig3_modal_masses')
plt.close(fig3)


# ---- 图 4: C 点动力学响应 ----
print("绘制图 4: C 点时程响应...")
n_modes_resp = min(len(beta_roots), 5)
t_max_resp = 0.2   # [s]
t, w_C, v_C, a_C = dynamic_response(beta_roots, coeffs,
                                     t_max=t_max_resp, n_t=3000,
                                     n_modes=n_modes_resp)

fig4, axes4 = plt.subplots(3, 1, figsize=(8, 6.5), sharex=True)

# 位移
axes4[0].plot(t, w_C * 1000, color=colors[0], linewidth=1.2)
axes4[0].set_ylabel(r'$w(2l,t)$ [mm]', fontsize=10)
axes4[0].set_title('Dynamic Response at Point C ($x = 2l$)', fontsize=12,
                   fontweight='bold')
axes4[0].tick_params(labelsize=9)
axes4[0].grid(True, alpha=0.3)

# 速度
axes4[1].plot(t, v_C, color=colors[1], linewidth=1.2)
axes4[1].set_ylabel(r'$\dot{w}(2l,t)$ [m/s]', fontsize=10)
axes4[1].tick_params(labelsize=9)
axes4[1].grid(True, alpha=0.3)

# 加速度
axes4[2].plot(t, a_C, color=colors[2], linewidth=1.2)
axes4[2].set_xlabel(r'$t$ [s]', fontsize=11)
axes4[2].set_ylabel(r'$\ddot{w}(2l,t)$ [m/s²]', fontsize=10)
axes4[2].tick_params(labelsize=9)
axes4[2].grid(True, alpha=0.3)

fig4.tight_layout()
save_figure(fig4, 'fig4_response_C')
plt.close(fig4)


# ---- 图 5: 各阶模态对响应的贡献 (频谱) ----
print("绘制图 5: 频谱贡献...")
fig5, ax5 = plt.subplots(1, 1, figsize=(7, 3.8))
n_spec = min(len(beta_roots), 8)
contribs = []
labels_spec = []
for n in range(n_spec):
    beta = beta_roots[n]
    c = coeffs[n]
    omega_n = omegas[n]
    M_n = modal_masses[n]
    xi, Y, _ = compute_mode_shape(beta, c, n_pts_per_seg=100)
    phi_C = Y[np.argmin(np.abs(xi - 2.0))]
    coeff_n = m_mass * v0 * phi_C / (M_n * omega_n)
    amp = abs(phi_C * coeff_n)
    contribs.append(amp)
    labels_spec.append(f'Mode {n+1}\n{freqs_hz[n]:.2f} Hz')

contribs_arr = np.array(contribs) * 1000  # mm
markerline, stemlines, baseline = ax5.stem(np.arange(1, n_spec + 1), contribs_arr,
         basefmt=' ', linefmt=colors[0], markerfmt='o')
markerline.set_markersize(8)
ax5.set_xticks(np.arange(1, n_spec + 1))
ax5.set_xticklabels(labels_spec, fontsize=8)
ax5.set_ylabel(r'$|w_n(2l)|_{\max}$ [mm]', fontsize=11)
ax5.set_title('Modal Contribution to Displacement at Point C',
              fontsize=12, fontweight='bold')
ax5.tick_params(labelsize=9)
ax5.grid(axis='y', alpha=0.3)
fig5.tight_layout()
save_figure(fig5, 'fig5_modal_contribution')
plt.close(fig5)


# ---- 图 6: 梁全貌振型三维示意 (x-t 瀑布图) ----
print("绘制图 6: 梁位移时空图...")
n_modes_waterfall = min(len(beta_roots), 4)
t_wf = np.linspace(0, 0.15, 60)
x_wf = np.linspace(0, 4 * l, 400)

W_wf = np.zeros((len(t_wf), len(x_wf)))
for n in range(n_modes_waterfall):
    beta = beta_roots[n]
    c = coeffs[n]
    omega_n = omegas[n]
    M_n = modal_masses[n]
    xi, Y, _ = compute_mode_shape(beta, c, n_pts_per_seg=250)
    phi_C = Y[np.argmin(np.abs(xi - 2.0))]
    coeff_n = m_mass * v0 * phi_C / (M_n * omega_n)
    # 插值到统一网格
    phi_interp = np.interp(x_wf / l, xi, Y)
    for j, tt in enumerate(t_wf):
        W_wf[j, :] += phi_interp * coeff_n * np.sin(omega_n * tt)

fig6, ax6 = plt.subplots(1, 1, figsize=(9, 5))
T_wf, X_wf = np.meshgrid(t_wf, x_wf, indexing='ij')
cont = ax6.contourf(T_wf * 1000, X_wf, W_wf * 1000, levels=50,
                     cmap='RdBu_r')
cbar = fig6.colorbar(cont, ax=ax6, label=r'$w(x,t)$ [mm]')
cbar.ax.tick_params(labelsize=8)
ax6.set_xlabel(r'$t$ [ms]', fontsize=11)
ax6.set_ylabel(r'$x$ [m]', fontsize=11)
ax6.set_title('Spatio-Temporal Displacement $w(x,t)$', fontsize=12,
              fontweight='bold')
# 标注特殊位置
for xc, lab in [(0, 'A'), (l, 'B'), (2*l, 'C'), (3*l, 'D'), (4*l, 'E')]:
    ax6.axhline(y=xc, color='black', linewidth=0.4, linestyle=':')
    ax6.text(t_wf[-1] * 1000 + 2, xc, lab, fontsize=8, va='center', color='black')
ax6.tick_params(labelsize=9)
fig6.tight_layout()
save_figure(fig6, 'fig6_spacetime')
plt.close(fig6)


# ============================================================
# 9. 输出数值摘要
# ============================================================
print("\n" + "=" * 65)
print("数值结果摘要")
print("=" * 65)
print(f"{'n':>3s}  {'β_n':>10s}  {'ω_n [rad/s]':>14s}  {'f_n [Hz]':>10s}  "
      f"{'M_n [kg]':>10s}  {'|φ_n(2l)|':>10s}")
print("-" * 65)
for i in range(min(len(beta_roots), 8)):
    xi, Y, _ = mode_shapes[i]
    phi_C = Y[np.argmin(np.abs(xi - 2.0))]
    print(f"{i+1:3d}  {beta_roots[i]:10.5f}  {omegas[i]:14.4f}  "
          f"{freqs_hz[i]:10.4f}  {modal_masses[i]:10.4f}  {abs(phi_C):10.5f}")

print("\n✓ 所有图片已保存。")
print("文件列表: fig1_det_A_beta.png/.pdf, fig2_mode_shapes.png/.pdf,")
print("          fig3_modal_masses.png/.pdf, fig4_response_C.png/.pdf,")
print("          fig5_modal_contribution.png/.pdf, fig6_spacetime.png/.pdf")
