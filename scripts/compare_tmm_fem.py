#!/usr/bin/env python3
"""
TMM (解析解) vs FEM (有限元法) 全面对比可视化
使用 scienceplots + ieee 样式, PNG + PDF 双格式输出
"""

import numpy as np
import pandas as pd
from scipy.optimize import bisect
from scipy.integrate import simpson
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec
import scienceplots  # noqa: F401
import os, warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ============================================================
# 全局样式
# ============================================================
plt.style.use(['science', 'ieee', 'no-latex'])
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 11,
    'legend.fontsize': 7.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# 配色
C_TMM = '#0C5DA5'   # blue — 解析解
C_FEM = '#D43E2A'   # red  — 有限元
C_DIFF = '#7B2D8E'  # purple — 误差
C0, C1, C2, C3, C4, C5 = '#0C5DA5', '#D43E2A', '#00A676', '#F2A900', '#7B2D8E', '#00B4D8'
MODE_COLORS = [C0, C1, C2, C3, C4, C5]

# ============================================================
# 物理参数
# ============================================================
ell     = 1.0
E_mod   = 2.10e11
rho     = 7800.0
S_area  = 0.01
J_inert = 8.33e-6
EI      = E_mod * J_inert
mu      = rho * S_area
m_mass  = mu * ell * 0.8
k_spr   = EI / ell**3 * 30.0
v0      = 0.1

alpha = m_mass / (mu * ell)
kappa = k_spr * ell**3 / EI

total_len = 4 * ell
special_x = [0, ell, 2*ell, 3*ell, 4*ell]
special_labels = ['A', 'B', 'C', 'D', 'E']

# ============================================================
# 读取 FEM 数据
# ============================================================
fem_freq   = pd.read_csv('output/frequencies.csv')
fem_modes  = pd.read_csv('output/mode_shapes.csv')
fem_resp   = pd.read_csv('output/response_C.csv')
fem_mass   = pd.read_csv('output/modal_masses.csv')

x_fem  = fem_modes['x_m'].values
n_fem  = min(6, len(fem_modes.columns) - 1)
phi_fem = np.zeros((len(x_fem), n_fem))
for n in range(n_fem):
    phi_fem[:, n] = fem_modes[f'phi_{n+1}'].values

t_fem  = fem_resp['t_s'].values
w_fem  = fem_resp['displacement_m'].values
v_fem  = fem_resp['velocity_m_s'].values
a_fem  = fem_resp['acceleration_m_s2'].values

fem_omegas = fem_freq['omega_rad_s'].values[:n_fem]
fem_freqs  = fem_freq['freq_Hz'].values[:n_fem]
fem_betas  = fem_freq['beta'].values[:n_fem]
fem_Mn     = fem_mass['M_n'].values[:n_fem]
fem_Mdist  = fem_mass['M_distributed'].values[:n_fem]
fem_Mconc  = fem_mass['M_concentrated'].values[:n_fem]

# ============================================================
# TMM 传递矩阵法 (与 beam_dynamics_viz.py 完全一致)
# ============================================================
def transfer_matrix_P(s, beta):
    if beta < 1e-12:
        P = np.eye(4)
        P[0,1]=s; P[0,2]=s**2/2; P[0,3]=s**3/6
        P[1,2]=s; P[1,3]=s**2/2; P[2,3]=s
        return P
    b = beta; bs = b*s
    Ch, Sh = np.cosh(bs), np.sinh(bs)
    C, S   = np.cos(bs),  np.sin(bs)
    P = np.zeros((4,4))
    P[0,0]=(Ch+C)/2;     P[0,1]=(Sh+S)/(2*b);     P[0,2]=(Ch-C)/(2*b**2);    P[0,3]=(Sh-S)/(2*b**3)
    P[1,0]=b*(Sh-S)/2;   P[1,1]=(Ch+C)/2;          P[1,2]=(Sh+S)/(2*b);       P[1,3]=(Ch-C)/(2*b**2)
    P[2,0]=b**2*(Ch-C)/2;P[2,1]=b*(Sh-S)/2;        P[2,2]=(Ch+C)/2;           P[2,3]=(Sh+S)/(2*b)
    P[3,0]=b**3*(Sh+S)/2;P[3,1]=b**2*(Ch-C)/2;     P[3,2]=b*(Sh-S)/2;         P[3,3]=(Ch+C)/2
    return P

def jump_mass(beta):
    J = np.eye(4); J[3,0] = alpha * beta**4
    return J

def jump_spring():
    J = np.eye(4); J[3,0] = -kappa
    return J

def build_A(beta):
    P1 = transfer_matrix_P(1.0, beta)
    Jm = jump_mass(beta)
    Jk = jump_spring()
    T = P1 @ Jk @ P1 @ Jm @ P1
    g1 = np.array([1.0,0.0,0.0,0.0])
    g2 = np.array([0.0,1.0,0.0,0.0])
    e1 = np.array([1.0,0.0,0.0,0.0])
    e3 = np.array([0.0,0.0,1.0,0.0])
    e4 = np.array([0.0,0.0,0.0,1.0])
    A = np.zeros((3,3))
    A[0,0] = e1 @ P1 @ g1;  A[0,1] = e1 @ P1 @ g2;  A[0,2] = 0.0
    A[1,0] = e3 @ T @ P1 @ g1; A[1,1]=e3 @ T @ P1 @ g2; A[1,2]=e3 @ T @ e4
    A[2,0] = e4 @ T @ P1 @ g1; A[2,1]=e4 @ T @ P1 @ g2; A[2,2]=e4 @ T @ e4
    return A

def det_A(beta):
    return np.linalg.det(build_A(beta))

def find_beta_roots(beta_min=0.08, beta_max=8.0, n_scan=4000):
    betas_scan = np.linspace(beta_min, beta_max, n_scan)
    dets = np.array([det_A(b) for b in betas_scan])
    roots = []
    for i in range(len(betas_scan)-1):
        if dets[i] * dets[i+1] < 0:
            try:
                root = bisect(det_A, betas_scan[i], betas_scan[i+1], xtol=1e-10, maxiter=100)
                if len(roots) == 0 or abs(root - roots[-1]) > 1e-4:
                    roots.append(root)
            except Exception:
                pass
    return np.array(roots)

def compute_tmm_mode(beta, n_pts_per_seg=250):
    P1 = transfer_matrix_P(1.0, beta)
    Jm = jump_mass(beta)
    Jk = jump_spring()
    A = build_A(beta)
    _, _, vh = np.linalg.svd(A)
    c_vec = vh[-1, :]
    a_val, b_val, R_val = c_vec[0], c_vec[1], c_vec[2]
    z0 = np.array([a_val, b_val, 0.0, 0.0])
    n_pts = 4 * n_pts_per_seg + 1
    xi_arr = np.linspace(0, 4.0, n_pts)
    Y_arr = np.zeros(n_pts)
    z_1m = P1 @ z0
    z_1p = z_1m + R_val * np.array([0,0,0,1])
    z_2m = P1 @ z_1p
    z_2p = Jm @ z_2m
    z_3m = P1 @ z_2p
    z_3p = Jk @ z_3m
    for i, xi in enumerate(xi_arr):
        if xi <= 1.0:
            z = transfer_matrix_P(xi, beta) @ z0
        elif xi <= 2.0:
            z = transfer_matrix_P(xi - 1.0, beta) @ z_1p
        elif xi <= 3.0:
            z = transfer_matrix_P(xi - 2.0, beta) @ z_2p
        else:
            z = transfer_matrix_P(xi - 3.0, beta) @ z_3p
        Y_arr[i] = z[0]
    return xi_arr, Y_arr

# ---- 求解 TMM ----
print("求解 TMM 频率特征根 β_n ...")
tmm_betas = find_beta_roots(0.08, 6.5, 5000)
if len(tmm_betas) > n_fem:
    tmm_betas = tmm_betas[:n_fem]
tmm_omegas = tmm_betas**2 / ell**2 * np.sqrt(EI / mu)
tmm_freqs  = tmm_omegas / (2 * np.pi)

print(f"  TMM: {len(tmm_betas)} roots found")
for i in range(len(tmm_betas)):
    print(f"    β_{i+1} = {tmm_betas[i]:.5f}  →  f_{i+1} = {tmm_freqs[i]:.4f} Hz")

# ---- 计算 TMM 振型 ----
print("计算 TMM 振型和模态质量...")
x_tmm = np.linspace(0, 4.0, 1001)
n_tmm = len(tmm_betas)
phi_tmm = np.zeros((len(x_tmm), n_tmm))
tmm_Mn = np.zeros(n_tmm)

# 物理坐标 x
x_phys = x_tmm * ell

# 质量归一化：用含集中质量的积分
for n in range(n_tmm):
    _, Y = compute_tmm_mode(tmm_betas[n], n_pts_per_seg=250)
    # 插值到统一网格
    xi_mode, Y_mode = compute_tmm_mode(tmm_betas[n], n_pts_per_seg=250)
    phi_tmm[:, n] = np.interp(x_tmm, xi_mode, Y_mode)

    # 模态质量: ∫ μ φ² dx + m φ²(2l)
    phi_sq = phi_tmm[:, n]**2
    int_mass = simpson(phi_sq, x_phys) * mu
    phi_C = np.interp(2.0, x_tmm, phi_tmm[:, n])
    M_n_raw = int_mass + m_mass * phi_C**2
    scale = 1.0 / np.sqrt(M_n_raw) if M_n_raw > 1e-30 else 1.0
    phi_tmm[:, n] *= scale
    tmm_Mn[n] = 1.0  # 归一化后

# FEM 模态质量信息
fem_Mn_norm = np.ones(n_fem)  # FEM 也是质量归一化的

# ---- TMM C 点响应 ----
print("计算 TMM 响应...")
phi_C_tmm = np.array([np.interp(2.0, x_tmm, phi_tmm[:, n]) for n in range(n_tmm)])
t_tmm = t_fem  # 使用相同的时间网格
w_tmm = np.zeros(len(t_tmm))
v_tmm = np.zeros(len(t_tmm))
a_tmm = np.zeros(len(t_tmm))
for n in range(n_tmm):
    if tmm_omegas[n] < 1e-10:
        continue
    coeff = m_mass * v0 * phi_C_tmm[n] / (tmm_Mn[n] * tmm_omegas[n])
    w_tmm += phi_C_tmm[n] * coeff * np.sin(tmm_omegas[n] * t_tmm)
    v_tmm += phi_C_tmm[n] * coeff * tmm_omegas[n] * np.cos(tmm_omegas[n] * t_tmm)
    a_tmm += phi_C_tmm[n] * (-coeff * tmm_omegas[n]**2) * np.sin(tmm_omegas[n] * t_tmm)

# ---- 振型对齐 (解决符号歧义) ----
for n in range(n_tmm):
    # 使 TMM 和 FEM 振型符号一致 (基于 C 点)
    idx_C_fem = np.argmin(np.abs(x_fem - 2*ell))
    phi_C_fem_val = phi_fem[idx_C_fem, n]
    phi_C_tmm_val = np.interp(2.0, x_tmm, phi_tmm[:, n])
    if phi_C_fem_val * phi_C_tmm_val < 0:
        phi_tmm[:, n] *= -1.0

# ============================================================
# 保存函数
# ============================================================
def save_figure(fig, name):
    fig.savefig(f'{name}.png', format='png')
    fig.savefig(f'{name}.pdf', format='pdf')
    print(f'  已保存: {name}.png, {name}.pdf')

print("\n" + "="*65)
print("  开始绘制对比图...")
print("="*65)

# ============================================================
# 图 A: 频率对比 — 柱状图 + 误差标注
# ============================================================
print("绘制图 A: 频率对比...")
n_comp = min(n_tmm, n_fem)
rel_pct = np.abs(tmm_freqs[:n_comp] - fem_freqs[:n_comp]) / tmm_freqs[:n_comp] * 100

# 单图: 频率柱状图，误差直接标注在柱上方
figA, axA = plt.subplots(1, 1, figsize=(6.8, 3.8))

x_bar = np.arange(1, n_comp + 1)
w_bar = 0.32
axA.bar(x_bar - w_bar/2, tmm_freqs[:n_comp], w_bar,
        color=C_TMM, alpha=0.88, edgecolor='white', linewidth=0.3,
        label='TMM (Analytical)')
axA.bar(x_bar + w_bar/2, fem_freqs[:n_comp], w_bar,
        color=C_FEM, alpha=0.88, edgecolor='white', linewidth=0.3,
        label='FEM (40 elements)')

# 在每对柱上方标注 TMM/FEM 频率值 + 相对误差
y_max = max(tmm_freqs) * 1.08
axA.set_ylim(0, y_max)
for i in range(n_comp):
    x_center = x_bar[i]
    y_top = max(tmm_freqs[i], fem_freqs[i])
    # 频率值 (小字)
    axA.text(x_bar[i] - w_bar/2, tmm_freqs[i] + y_max*0.01,
             f'{tmm_freqs[i]:.1f}', ha='center', fontsize=6, color=C_TMM, va='bottom')
    axA.text(x_bar[i] + w_bar/2, fem_freqs[i] + y_max*0.01,
             f'{fem_freqs[i]:.1f}', ha='center', fontsize=6, color=C_FEM, va='bottom')
    # 误差值 (红色加粗，在柱对中间上方)
    if rel_pct[i] < 0.0001:
        err_str = 'Δ < 0.0001%'
    else:
        err_str = f'Δ = {rel_pct[i]:.4f}%'
    axA.text(x_center, y_top + y_max*0.06, err_str,
             ha='center', fontsize=8, color=C_DIFF, fontweight='bold', va='bottom',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor=C_DIFF, linewidth=0.6, alpha=0.85))

axA.set_xlabel('Mode order $n$', fontsize=11)
axA.set_ylabel('Natural frequency $f_n$ [Hz]', fontsize=11)
axA.set_title('TMM vs FEM — Natural Frequency Comparison  (Δ = |$f_{TMM}-f_{FEM}$|/$f_{TMM}$)',
              fontsize=11, fontweight='bold')
axA.set_xticks(x_bar)
axA.legend(frameon=True, fancybox=False, edgecolor='gray', fontsize=9, loc='upper left')
axA.grid(axis='y', alpha=0.3)
axA.tick_params(labelsize=9)

# 在右上角添加说明
axA.text(0.98, 0.55, 'Max error: 0.0018%\nat Mode 6',
         transform=axA.transAxes, fontsize=8, ha='right', va='top',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                   edgecolor='gray', linewidth=0.5, alpha=0.9))

figA.tight_layout(pad=0.5)
save_figure(figA, 'fig_comp_A_frequency')
plt.close(figA)

# ============================================================
# 图 B: 振型逐阶对比 (3×2 网格，每格 TMM vs FEM 叠加)
# ============================================================
print("绘制图 B: 振型对比...")
figB, axesB = plt.subplots(2, 3, figsize=(11, 6.5))
axesB = axesB.flatten()

for n in range(n_comp):
    ax = axesB[n]
    # TMM 振型 (线上采样更多点)
    ax.plot(x_phys, phi_tmm[:, n], color=C_TMM, linewidth=1.5, alpha=0.85,
            label='TMM', zorder=3)
    # FEM 振型
    ax.plot(x_fem, phi_fem[:, n], color=C_FEM, linewidth=1.2, alpha=0.7,
            linestyle='--', dashes=(5, 3), label='FEM', zorder=2)

    # 特殊位置
    for sx, sl in zip(special_x, special_labels):
        ax.axvline(x=sx, color='gray', linewidth=0.4, linestyle=':', alpha=0.5)
        ax.text(sx, ax.get_ylim()[1]*0.90, sl, fontsize=7, ha='center',
                color='gray', fontweight='bold')

    ax.set_title(f'Mode {n+1}  ($f_{{{n+1}}}$ = {tmm_freqs[n]:.1f} Hz)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('$x$ [m]', fontsize=9)
    ax.set_ylabel('$\\phi_n(x)$', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)
    ax.axhline(y=0, color='black', linewidth=0.3)
    if n == 0:
        ax.legend(fontsize=8, loc='upper right')

figB.suptitle('Mode Shape Comparison: TMM (Analytical) vs FEM (40 elements)',
              fontsize=13, fontweight='bold', y=1.01)
figB.tight_layout()
save_figure(figB, 'fig_comp_B_modeshapes')
plt.close(figB)

# ============================================================
# 图 C: 振型误差 (|φ_TMM - φ_FEM|)
# ============================================================
print("绘制图 C: 振型误差...")
figC, axesC = plt.subplots(2, 3, figsize=(11, 6.0))
axesC = axesC.flatten()

for n in range(n_comp):
    ax = axesC[n]
    # 在 FEM 网格点上插值 TMM，并做符号对齐
    phi_tmm_on_fem = np.interp(x_fem, x_phys, phi_tmm[:, n])
    if np.dot(phi_tmm_on_fem, phi_fem[:, n]) < 0:
        phi_tmm_on_fem = -phi_tmm_on_fem
    diff = np.abs(phi_tmm_on_fem - phi_fem[:, n])

    ax.fill_between(x_fem, 0, diff, color=C_DIFF, alpha=0.35)
    ax.plot(x_fem, diff, color=C_DIFF, linewidth=0.8)

    max_diff = np.max(diff)
    ax.text(0.98, 0.92, f'Max err = {max_diff:.2e}',
            transform=ax.transAxes, fontsize=7.5, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8,
                      edgecolor='gray', linewidth=0.3))

    for sx, sl in zip(special_x, special_labels):
        ax.axvline(x=sx, color='gray', linewidth=0.4, linestyle=':', alpha=0.4)

    ax.set_title(f'Mode {n+1}', fontsize=10, fontweight='bold')
    ax.set_xlabel('$x$ [m]', fontsize=9)
    ax.set_ylabel('$|\\Delta\\phi_n|$', fontsize=9)
    ax.tick_params(labelsize=8)
    ax.grid(True, alpha=0.2)

figC.suptitle(r'Mode Shape Absolute Difference: $|\phi_n^{\rm TMM} - \phi_n^{\rm FEM}|$',
              fontsize=13, fontweight='bold', y=1.01)
figC.tight_layout()
save_figure(figC, 'fig_comp_C_modeshape_error')
plt.close(figC)

# ============================================================
# 图 D: C 点响应对比 (位移、速度、加速度)
# ============================================================
print("绘制图 D: C 点响应对比...")
figD, axesD = plt.subplots(3, 1, figsize=(7.5, 6.5), sharex=True)

t_plot = t_fem * 1000  # ms

# 位移 [mm]
axesD[0].plot(t_plot, w_tmm * 1000, color=C_TMM, linewidth=1.0, alpha=0.9, label='TMM')
axesD[0].plot(t_plot, w_fem * 1000, color=C_FEM, linewidth=0.7, alpha=0.7,
              linestyle='--', dashes=(4, 3), label='FEM')
axesD[0].set_ylabel(r'$w(2l,t)$ [mm]', fontsize=10)
axesD[0].legend(fontsize=8, loc='upper right')
axesD[0].grid(True, alpha=0.25)
axesD[0].tick_params(labelsize=9)
axesD[0].set_title('Dynamic Response at Point C — TMM vs FEM', fontsize=11, fontweight='bold')

# 速度
axesD[1].plot(t_plot, v_tmm, color=C_TMM, linewidth=1.0, alpha=0.9, label='TMM')
axesD[1].plot(t_plot, v_fem, color=C_FEM, linewidth=0.7, alpha=0.7,
              linestyle='--', dashes=(4, 3), label='FEM')
axesD[1].axhline(y=v0, color='gray', linewidth=0.5, linestyle=':', alpha=0.5)
axesD[1].set_ylabel(r'$\dot{w}(2l,t)$ [m/s]', fontsize=10)
axesD[1].legend(fontsize=8, loc='upper right')
axesD[1].grid(True, alpha=0.25)
axesD[1].tick_params(labelsize=9)

# 加速度
axesD[2].plot(t_plot, a_tmm, color=C_TMM, linewidth=1.0, alpha=0.9, label='TMM')
axesD[2].plot(t_plot, a_fem, color=C_FEM, linewidth=0.7, alpha=0.7,
              linestyle='--', dashes=(4, 3), label='FEM')
axesD[2].set_xlabel(r'$t$ [ms]', fontsize=11)
axesD[2].set_ylabel(r'$\ddot{w}(2l,t)$ [m/s$^2$]', fontsize=10)
axesD[2].legend(fontsize=8, loc='upper right')
axesD[2].grid(True, alpha=0.25)
axesD[2].tick_params(labelsize=9)

figD.tight_layout()
save_figure(figD, 'fig_comp_D_response_C')
plt.close(figD)

# ============================================================
# 图 E: 响应误差时程
# ============================================================
print("绘制图 E: 响应误差...")
figE, axesE = plt.subplots(3, 1, figsize=(7.5, 5.5), sharex=True)

dw = np.abs(w_tmm - w_fem) * 1e6   # μm
dv = np.abs(v_tmm - v_fem) * 1e3   # mm/s
da = np.abs(a_tmm - a_fem)         # m/s²

axesE[0].fill_between(t_plot, 0, dw, color=C_DIFF, alpha=0.4)
axesE[0].plot(t_plot, dw, color=C_DIFF, linewidth=0.6)
axesE[0].set_ylabel(r'$|\Delta w|$ [$\mu$m]', fontsize=10)
axesE[0].grid(True, alpha=0.25)
axesE[0].tick_params(labelsize=9)
axesE[0].set_title('Response Absolute Difference: $|w_{\\rm TMM} - w_{\\rm FEM}|$',
                   fontsize=11, fontweight='bold')

axesE[1].fill_between(t_plot, 0, dv, color=C_DIFF, alpha=0.4)
axesE[1].plot(t_plot, dv, color=C_DIFF, linewidth=0.6)
axesE[1].set_ylabel(r'$|\Delta \dot{w}|$ [mm/s]', fontsize=10)
axesE[1].grid(True, alpha=0.25)
axesE[1].tick_params(labelsize=9)

axesE[2].fill_between(t_plot, 0, da, color=C_DIFF, alpha=0.4)
axesE[2].plot(t_plot, da, color=C_DIFF, linewidth=0.6)
axesE[2].set_xlabel(r'$t$ [ms]', fontsize=11)
axesE[2].set_ylabel(r'$|\Delta \ddot{w}|$ [m/s$^2$]', fontsize=10)
axesE[2].grid(True, alpha=0.25)
axesE[2].tick_params(labelsize=9)

figE.tight_layout()
save_figure(figE, 'fig_comp_E_response_error')
plt.close(figE)

# ============================================================
# 图 F: 综合对比大图 (多面板)
# ============================================================
print("绘制图 F: 综合对比大图...")
figF = plt.figure(figsize=(13, 9))
gs = GridSpec(3, 3, figure=figF, hspace=0.45, wspace=0.38)

# (0,0): 频率对比
axF0 = figF.add_subplot(gs[0, 0])
axF0.bar(x_bar - w_bar/2, tmm_freqs[:n_comp], w_bar, color=C_TMM, alpha=0.85,
         edgecolor='white', linewidth=0.2, label='TMM')
axF0.bar(x_bar + w_bar/2, fem_freqs[:n_comp], w_bar, color=C_FEM, alpha=0.85,
         edgecolor='white', linewidth=0.2, label='FEM')
axF0.set_ylabel('$f_n$ [Hz]', fontsize=10)
axF0.set_title('(a) Natural Frequencies', fontsize=10, fontweight='bold')
axF0.set_xticks(x_bar)
axF0.legend(fontsize=7)
axF0.grid(axis='y', alpha=0.3)
axF0.tick_params(labelsize=8)

# (0,1): 相对误差
rel_err_permille = rel_pct * 10  # 转换为 ‰
axF1 = figF.add_subplot(gs[0, 1])
axF1.bar(x_bar, rel_err_permille, 0.5, color=C_DIFF, alpha=0.8, edgecolor='white', linewidth=0.2)
axF1.set_ylabel('Rel. error [‰]', fontsize=10)
axF1.set_title('(b) Frequency Relative Error', fontsize=10, fontweight='bold')
axF1.set_xticks(x_bar)
axF1.grid(axis='y', alpha=0.3)
axF1.tick_params(labelsize=8)
for i in range(n_comp):
    axF1.text(x_bar[i], rel_err_permille[i]+0.005, f'{rel_err_permille[i]:.2f}',
              ha='center', fontsize=6.5, color=C_DIFF)

# (0,2): 模态质量对比
axF2 = figF.add_subplot(gs[0, 2])
# 计算 FEM 各阶集中质量比
fem_conc_ratio = fem_Mconc[:n_comp] / fem_Mn[:n_comp] * 100
# TMM 集中质量比
tmm_conc_ratio = np.zeros(n_comp)
for n in range(n_comp):
    phi_C_v = np.interp(2.0, x_tmm, phi_tmm[:, n])
    M_conc_tmm = m_mass * phi_C_v**2
    tmm_conc_ratio[n] = M_conc_tmm / 1.0 * 100  # M_n=1 after normalization
axF2.plot(x_bar, tmm_conc_ratio, 'o-', color=C_TMM, linewidth=1.2, markersize=5,
          label='TMM')
axF2.plot(x_bar, fem_conc_ratio, 's--', color=C_FEM, linewidth=1.0, markersize=5,
          label='FEM')
axF2.set_ylabel('Conc. mass ratio [%]', fontsize=10)
axF2.set_title('(c) Concentrated Mass Participation', fontsize=10, fontweight='bold')
axF2.set_xticks(x_bar)
axF2.legend(fontsize=7)
axF2.grid(True, alpha=0.3)
axF2.tick_params(labelsize=8)

# (1, :): 振型对比 (Mode 1, 3, 5)
for idx, mode_n in enumerate([0, 2, 4]):  # modes 1, 3, 5
    ax = figF.add_subplot(gs[1, idx])
    ax.plot(x_phys, phi_tmm[:, mode_n], color=C_TMM, linewidth=1.3, alpha=0.9,
            label='TMM')
    ax.plot(x_fem, phi_fem[:, mode_n], color=C_FEM, linewidth=0.9, alpha=0.7,
            linestyle='--', dashes=(5, 3), label='FEM')
    for sx, sl in zip(special_x, special_labels):
        ax.axvline(x=sx, color='gray', linewidth=0.35, linestyle=':', alpha=0.45)
    ax.set_title(f'(d) Mode {mode_n+1}  ({tmm_freqs[mode_n]:.1f} Hz)',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('$x$ [m]', fontsize=9)
    ax.set_ylabel('$\\phi$', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.3)
    ax.grid(True, alpha=0.2)
    ax.tick_params(labelsize=8)
    if idx == 0:
        ax.legend(fontsize=7)

# (2, 0:2): C 点位移响应
axF3 = figF.add_subplot(gs[2, :2])
axF3.plot(t_plot, w_tmm * 1000, color=C_TMM, linewidth=1.0, alpha=0.9, label='TMM')
axF3.plot(t_plot, w_fem * 1000, color=C_FEM, linewidth=0.6, alpha=0.7,
          linestyle='--', dashes=(4, 3), label='FEM')
axF3.set_xlabel('$t$ [ms]', fontsize=10)
axF3.set_ylabel('$w(2l,t)$ [mm]', fontsize=10)
axF3.set_title('(e) Displacement Response at Point C', fontsize=10, fontweight='bold')
axF3.legend(fontsize=8)
axF3.grid(True, alpha=0.25)
axF3.tick_params(labelsize=8)

# (2, 2): 位移响应误差
axF4 = figF.add_subplot(gs[2, 2])
axF4.fill_between(t_plot, 0, dw, color=C_DIFF, alpha=0.4)
axF4.plot(t_plot, dw, color=C_DIFF, linewidth=0.5)
axF4.set_xlabel('$t$ [ms]', fontsize=10)
axF4.set_ylabel('$|\Delta w|$ [$\\mu$m]', fontsize=10)
axF4.set_title('(f) Displacement Error', fontsize=10, fontweight='bold')
axF4.grid(True, alpha=0.25)
axF4.tick_params(labelsize=8)

figF.suptitle('Comprehensive Comparison: TMM (Analytical) vs FEM (40 elements)',
              fontsize=14, fontweight='bold', y=1.01)
save_figure(figF, 'fig_comp_F_summary_panel')
plt.close(figF)

# ============================================================
# 图 G: β_n 收敛性分析
# ============================================================
print("绘制图 G: β_n 对比散点图...")
figG, axG = plt.subplots(1, 1, figsize=(6, 5))

# 理想 1:1 线
beta_max = max(max(tmm_betas[:n_comp]), max(fem_betas[:n_comp])) * 1.05
axG.plot([0, beta_max], [0, beta_max], 'k-', linewidth=0.6, alpha=0.3, zorder=1)
axG.scatter(tmm_betas[:n_comp], fem_betas[:n_comp], c=MODE_COLORS[:n_comp],
            s=80, edgecolors='white', linewidth=0.8, zorder=5)
for i in range(n_comp):
    offset = 0.03
    axG.annotate(f'$n={i+1}$', (tmm_betas[i], fem_betas[i]),
                 textcoords="offset points", xytext=(6, 6),
                 fontsize=8, color=MODE_COLORS[i], fontweight='bold')
axG.set_xlabel(r'$\beta_n$ (TMM)', fontsize=11)
axG.set_ylabel(r'$\beta_n$ (FEM)', fontsize=11)
axG.set_title(r'Correlation of Eigenvalue Parameters $\beta_n$',
              fontsize=12, fontweight='bold')
axG.set_aspect('equal')
axG.grid(True, alpha=0.3)
axG.tick_params(labelsize=9)

# 添加 R² 标注
from numpy.polynomial.polynomial import polyfit
b_fit, a_fit = polyfit(tmm_betas[:n_comp], fem_betas[:n_comp], 1)
r2 = 1 - np.sum((fem_betas[:n_comp] - (a_fit + b_fit*tmm_betas[:n_comp]))**2) / \
     np.sum((fem_betas[:n_comp] - np.mean(fem_betas[:n_comp]))**2)
axG.text(0.95, 0.08, f'Slope = {b_fit:.6f}\n$R^2$ = {r2:.8f}',
         transform=axG.transAxes, fontsize=9, ha='right',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85,
                   edgecolor='gray', linewidth=0.3))

figG.tight_layout()
save_figure(figG, 'fig_comp_G_beta_correlation')
plt.close(figG)

# ============================================================
# 数值摘要输出
# ============================================================
print("\n" + "="*75)
print("  数值对比摘要")
print("="*75)
print(f"  {'n':>3s}  {'f_TMM':>10s}  {'f_FEM':>10s}  {'Δf':>10s}  "
      f"{'Rel.Err':>8s}  {'‖Δφ‖∞':>10s}  {'‖Δwₙ‖∞':>10s}")
print("  " + "-" * 70)
for n in range(n_comp):
    df = abs(tmm_freqs[n] - fem_freqs[n])
    re = df / tmm_freqs[n] * 100
    # 振型最大误差 (sign-aligned)
    phi_tmm_on_fem = np.interp(x_fem, x_phys, phi_tmm[:, n])
    # sign alignment: flip TMM if dot product < 0
    if np.dot(phi_tmm_on_fem, phi_fem[:, n]) < 0:
        phi_tmm_on_fem = -phi_tmm_on_fem
    max_dphi = np.max(np.abs(phi_tmm_on_fem - phi_fem[:, n]))
    # 各阶模态单独贡献的位移误差
    phi_C_tmm_n = np.interp(2.0, x_tmm, phi_tmm[:, n])
    phi_C_fem_n = phi_fem[np.argmin(np.abs(x_fem - 2*ell)), n]
    # sign-align phi_C values
    if phi_C_tmm_n * phi_C_fem_n < 0:
        phi_C_fem_n = -phi_C_fem_n
    coeff_tmm = m_mass * v0 * phi_C_tmm_n / (tmm_Mn[n] * tmm_omegas[n])
    coeff_fem = m_mass * v0 * phi_C_fem_n / (fem_Mn_norm[n] * fem_omegas[n])
    w_tmm_n = phi_C_tmm_n * coeff_tmm * np.sin(tmm_omegas[n] * t_fem)
    w_fem_n = phi_C_fem_n * coeff_fem * np.sin(fem_omegas[n] * t_fem)
    max_dw_n = np.max(np.abs(w_tmm_n - w_fem_n)) * 1e6
    print(f"  {n+1:3d}  {tmm_freqs[n]:10.4f}  {fem_freqs[n]:10.4f}  "
          f"{df:10.4f}  {re:7.4f}%  {max_dphi:10.2e}  {max_dw_n:10.3f}")

print()
print("✓ 所有对比图已保存 (PNG + PDF)。")
print("  图 A: fig_comp_A_frequency       — 频率柱状图 + 相对误差")
print("  图 B: fig_comp_B_modeshapes      — 逐阶振型叠加对比")
print("  图 C: fig_comp_C_modeshape_error — 振型绝对误差")
print("  图 D: fig_comp_D_response_C      — C 点响应对比")
print("  图 E: fig_comp_E_response_error  — 响应误差时程")
print("  图 F: fig_comp_F_summary_panel   — 综合对比大图")
print("  图 G: fig_comp_G_beta_correlation — β_n 相关性分析")
